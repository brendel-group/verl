# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import numpy as np
import os
import logging
import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from verl.utils.debug import log_gpu_memory_usage

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)                

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating

                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                
                print(f'Response length: {response_length}')
                print(f'Logits.shape: {logits.shape}')
                
                log_gpu_memory_usage("After logit computation", logger=logger)

                # TODO: perhaps have to make this more computationally feasible. Still: I do not understand why we want
                # to allocate 12 GiB with these operation...
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])

                print(f"Log probs.shape: {log_probs.shape}")
                # TODO: maybe use an approximation to the entropy instead...
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                log_gpu_memory_usage("After entropy computation", logger=logger)

                print(f"Optimizer: {type(self.actor_optimizer)}, Found DEV_ESTIMATE_ENTROPY_DELTA: {self.config.get('DEV_ESTIMATE_ENTROPY_DELTA', False)}")
                if self.config.get('DEV_ESTIMATE_ENTROPY_DELTA', False) and self.actor_module.training:
                    """
                    Entropy Delta Estimation using Policy Gradient Loss Approximation

                    We compute the gradient of policy loss w.r.t. logits to estimate entropy changes.

                    For PSR (advantage > 0) and NSR (advantage < 0):

                    -dL_PSR/dz_v ∝ {  π(y_t)(1-π(y_t))                  if v = y_t
                                     -π(y_v)π(y_t)                      if v ≠ y_t }

                    -dL_NSR/dz_v ∝ { -π(y_t)(1-π(y_t))                  if v = y_t
                                      π(y_v)π(y_t)                      if v ≠ y_t }

                    Scaling: advantage / sequence_length
                    """

                    ALPHA, APPROX_K = 1e-6, 100

                    # 1) Top-K over vocabulary
                    # logits: (bsz, response_length, vocab)
                    topk_logits, topk_indices = torch.topk(input=logits, k=APPROX_K, dim=-1)  # (B, L, K), (B, L, K)

                    # 2) Force-include sampled tokens in top-k (overwrite the last slot if missing)
                    sampled_tokens = micro_batch['responses']                                     # (B, L) token ids
                    sampled_logits_full = torch.gather(logits, -1, sampled_tokens.unsqueeze(-1))  # (B, L, 1)

                    # whether sampled token already in top-k
                    sampled_in_topk_mask = (topk_indices == sampled_tokens.unsqueeze(-1))         # (B, L, K)
                    missing_mask = ~sampled_in_topk_mask.any(dim=-1, keepdim=True)                # (B, L, 1)

                    # overwrite the last position (index K-1) where missing
                    last_col = APPROX_K - 1
                    # logits
                    topk_logits[..., last_col:last_col+1] = torch.where(
                        missing_mask, sampled_logits_full, topk_logits[..., last_col:last_col+1]
                    )
                    # indices
                    topk_indices[..., last_col:last_col+1] = torch.where(
                        missing_mask, sampled_tokens.unsqueeze(-1), topk_indices[..., last_col:last_col+1]
                    )

                    # 3) Recompute probs over the augmented top-k
                    topk_log_probs = torch.nn.functional.log_softmax(topk_logits, dim=-1)         # (B, L, K)
                    topk_probs = topk_log_probs.exp()                                             # (B, L, K)

                    # 4) Entropy gradient term (∂H/∂z) restricted to top-k
                    # entropy: (B, L)
                    dH_dz = topk_probs * (topk_log_probs - entropy.unsqueeze(-1))                 # (B, L, K)

                    advantages = micro_batch['advantages']                                        # (B, L)
                    advantage_scale = (advantages / advantages.size(1)).unsqueeze(-1)             # (B, L, 1)

                    print(f"topk_logits.shape: {topk_logits.shape}")
                    print(f"topk_indices.shape: {topk_indices.shape}")
                    print(f"advantages.shape: {advantages.shape}")

                    # 5) Locate sampled positions in the (augmented) top-k and gather π(y_t)
                    sampled_mask = (topk_indices == sampled_tokens.unsqueeze(-1))                 # (B, L, K)
                    # Argmax is safe because sampled is guaranteed to be present now.
                    sampled_pos = sampled_mask.to(torch.int32).argmax(dim=-1, keepdim=True)       # (B, L, 1) in [0..K-1]
                    pi_t = torch.take_along_dim(topk_probs, sampled_pos, dim=-1)                  # (B, L, 1)
                    print(f'pi_t.shape: {pi_t.shape}')

                    # 6) Build -∂L/∂z over the K entries (vectorized, no flattening)
                    # For v = y_t:  ± π_t (1 - π_t)
                    # For v ≠ y_t:  ∓ π_v π_t
                    pi_t_expanded = pi_t.expand_as(topk_probs)                                    # (B, L, K)

                    neg_dL_dz = torch.zeros_like(topk_probs)
                    # sampled entries
                    neg_dL_dz = neg_dL_dz + sampled_mask * (pi_t_expanded * (1.0 - pi_t_expanded))
                    # unsampled entries
                    neg_dL_dz = neg_dL_dz + (~sampled_mask) * (-topk_probs * pi_t_expanded)

                    # Apply scaling by advantage / sequence_length (handles PSR/NSR sign automatically)
                    neg_dL_dz = neg_dL_dz * advantage_scale                                      # (B, L, K)

                    # Convert to positive gradient dL/dz
                    dL_dz = -neg_dL_dz                                                           # (B, L, K)
                    print(f'dL_dz.shape: {dL_dz.shape}')

                    # 7) Entropy change estimate
                    Delta_z = ALPHA * dL_dz                                                      # (B, L, K)
                    H_t = (dH_dz * Delta_z).sum(dim=-1)

                    log_gpu_memory_usage("After entropy log computation", logger=logger)
                    return entropy, log_probs, H_t.detach().cpu().numpy()

                return entropy, log_probs


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        entropy_deltas = []
        advantage_buffer = []

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    entropy_coeff = self.config.entropy_coeff
                    use_token_level_loss = self.config.use_token_level_loss
                    clipping_mode = self.config.clipping_mode

                    # all return: (bsz, response_length)
                    out = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    if self.config.get('DEV_ESTIMATE_ENTROPY_DELTA', False):
                        entropy, log_prob, H_t = out
                        entropy_deltas.append(H_t)
                        advantage_buffer.append(advantages.detach().clone().cpu().numpy())
                    else:
                        entropy, log_prob = out

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        use_token_level_loss=use_token_level_loss,
                        clipping_mode=clipping_mode)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()

        # NOTE: DEV_ESTIMATE_ENTROPY_DELTA
        deltas : np.ndarray = np.concatenate(entropy_deltas)
        advs : np.ndarray = np.concatenate(advantage_buffer)
        metrics.update({'actor/H_t' : deltas, 'actor/advantages' : advs})
        
        return metrics
