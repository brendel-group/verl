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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import shutil
import time
import uuid
import pathlib
import pickle

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import json
from verl.utils import hdfs_io


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size) # update statistics like \beta of the adaptive KL controller
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        # Initialize advantage tracking storage
        self.advantage_tracking_enabled = getattr(self.config.trainer, 'track_advantages', False)
        self.advantage_tracking_path = getattr(self.config.trainer, 'track_advantages_path', 
                                                os.path.join(self.config.trainer.default_local_dir, 'advantage_tracking'))
        if self.advantage_tracking_enabled or self.config.trainer.get('advantage_tracking_only', False):
             os.makedirs(self.advantage_tracking_path, exist_ok=True)


        if self.advantage_tracking_enabled:
            self.advantage_tracking_freq = getattr(self.config.trainer, 'track_advantages_freq', 1)  # Default: every step

        if self.config.trainer.get('use_ref_for_generation', False):
            assert self.use_reference_policy, "Cannot use reference policy for generation if no reference policy worker is configured (Role.RefPolicy needed)."

        # Initialize early stopping variables
        # To enable early stopping, add the following to your trainer config:
        # trainer:
        #   early_stopping_enabled: true
        #   early_stopping_patience: 5  # Number of validation checks without improvement
        #   early_stopping_min_delta: 0.001  # Minimum improvement to be considered significant
        #   save_best_checkpoint: true  # Whether to save best checkpoint based on mean eval metrics
        self.early_stopping_enabled = getattr(self.config.trainer, 'early_stopping_enabled', False)
        if self.early_stopping_enabled:
            self.early_stopping_patience = getattr(self.config.trainer, 'early_stopping_patience', 10)
            self.early_stopping_min_delta = getattr(self.config.trainer, 'early_stopping_min_delta', 0.0)
            self.save_best_checkpoint = getattr(self.config.trainer, 'save_best_checkpoint', True)
            
            # Initialize tracking variables for mean of all eval metrics
            self.best_mean_metric_value = float('-inf')  # Always maximize mean of eval metrics
            self.best_step = 0
            self.patience_counter = 0
            self.early_stopped = False
            self.best_checkpoint_path = None
            
            print(f"Early stopping enabled: patience={self.early_stopping_patience}, "
                  f"min_delta={self.early_stopping_min_delta}, save_best={self.save_best_checkpoint}")

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # GRPO validation: ensure rollout.n is properly set
        if config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
            if not hasattr(config.actor_rollout_ref.rollout, 'n') or config.actor_rollout_ref.rollout.n is None:
                # Set default value for GRPO using open_dict to allow modification
                with open_dict(config):
                    config.actor_rollout_ref.rollout.n = 5
                print(f"WARNING: rollout.n not set for GRPO. Setting default value: {config.actor_rollout_ref.rollout.n}")
            elif config.actor_rollout_ref.rollout.n <= 1:
                print(f"WARNING: rollout.n={config.actor_rollout_ref.rollout.n} is too small for GRPO. GRPO requires n > 1 to generate multiple responses per prompt for advantage estimation.")
                # Set to minimum recommended value using open_dict to allow modification
                with open_dict(config):
                    config.actor_rollout_ref.rollout.n = 5
                print(f"Setting rollout.n to recommended value: {config.actor_rollout_ref.rollout.n}")

        if not config.algorithm.filter_groups.enable:
            assert config.data.train_batch_size == config.data.gen_batch_size, \
                f"train_batch_size must be equal to gen_batch_size when filter_groups.enable is False, but got {config.data.train_batch_size =} and {config.data.gen_batch_size =}"

        overlong_buffer_cfg = config.custom_reward_function.overlong_buffer
        if overlong_buffer_cfg.enable:
            assert config.data.max_response_length >= overlong_buffer_cfg.len > 0, \
                f"{config.data.max_response_length=} / {overlong_buffer_cfg.len=}"

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, set_to_self=True, shuffle=None):
        shuffle = self.config.data.shuffle if shuffle is None else shuffle
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.truncation,
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        # use sampler for better ckpt resume
        if shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.gen_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.truncation,
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        
        if self.config.data.val_batch_size is None:
            self.config.data.val_batch_size = len(self.val_dataset)


        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=self.config.data.val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

        if set_to_self:
            self.train_dataloader = self.train_dataloader
            self.val_dataloader = self.val_dataloader
            self.train_dataset = self.train_dataset
            self.val_dataset = self.val_dataset
        else:
            return self.train_dataloader, self.val_dataloader

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (csv, wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return
        if generations_to_log == -1:
            generations_to_log = len(inputs)

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text TODO: why?

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        backends = self.config.trainer.get('val_logger_backends', self.config.trainer.logger)
        self.validation_generations_logger.log(backends, samples, self.global_steps)

    def _validate(self):
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_sources = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size * self.config.actor_rollout_ref.rollout.val_kwargs.n)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            sample_sources.extend(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(input_ids)))
            
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            # if "reward_extra_info" in result:
            #     for key, lst in result["reward_extra_info"].items():
            #         reward_extra_infos_dict[key].extend(lst)

            # Reshape from [batch_size*n_responses, seq_len] to [num_prompts, n_responses, seq_len]
            reward_tensor = reward_tensor.view(len(input_ids), self.config.actor_rollout_ref.rollout.val_kwargs.n, -1)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().numpy()
            sample_scores.append(scores)

            # sample_sources.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        
        sample_scores = np.concatenate(sample_scores, axis=0)
        print(f'sample_scores: {sample_scores.shape}')

        # repeat to match shape of outputs, then log
        sample_inputs_repeated = np.repeat(sample_inputs, self.config.actor_rollout_ref.rollout.val_kwargs.n).tolist()
        self._maybe_log_val_generations(inputs=sample_inputs_repeated, outputs=sample_outputs, scores=sample_scores)

        metric_dict = {}
        for data_source in set(sample_sources):
            source_mask = np.array(sample_sources) == data_source
            source_scores = sample_scores[source_mask]
            metric_dict[f'val/{data_source}/pass@1/mean@{sample_scores.shape[1]}'] = np.mean(source_scores)
            metric_dict[f'val/{data_source}/pass@1/std@{sample_scores.shape[1]}'] = np.std(np.mean(source_scores, axis=1))
            metric_dict[f'val/{data_source}/pass@{sample_scores.shape[1]}/mean'] = np.mean(np.any(source_scores > 0, axis=1))
            metric_dict[f'val/{data_source}/pass@{sample_scores.shape[1]}/std'] = np.std(np.any(source_scores > 0, axis=1))
        return metric_dict
            
            
            

        # for lst in reward_extra_infos_dict.values():
        #     assert len(lst) == 0 or len(lst) == len(sample_scores)


        # src2metric2val = {}
        # for data_source in data_sources:
        #     if data_source not in src2metric2val:
        #         src2metric2val[data_source] = {}
            

        # data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # for sample_idx, data_source in enumerate(data_sources):
        #     prompt = sample_inputs[sample_idx]

        #     var2vals = data_src2prompt2var2vals[data_source][prompt]
        #     var2vals["final_reward"].append(sample_scores[sample_idx])
        #     for metric_name, metric_vals in reward_extra_infos_dict.items():
        #         var2vals[metric_name].append(metric_vals[sample_idx])

        # data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        #     for prompt, var2vals in prompt2var2vals.items():
        #         n_resps = len(var2vals["final_reward"])
        #         preds = var2vals["pred"]
        #         for var_name, var_vals in var2vals.items():
        #             if var_name in ["pred"]:
        #                 continue
        #             metric = {}

        #             metric[f"mean@{n_resps}"] = np.mean(var_vals)
        #             metric[f"std@{n_resps}"] = np.std(var_vals)
        #             metric[f"pass@{n_resps}"] = float(np.any(np.array(var_vals) > 0))
        #             # if n_resps > 1:
        #             # if n_resps > 1:
        #             #     ns = []
        #             #     n = 2
        #             #     while n < n_resps:
        #             #         ns.append(n)
        #             #         n *= 2
        #             #     ns.append(n_resps)

        #             #     if preds is not None:
        #             #         data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, preds)]
        #             #     else:
        #             #         data = [{"val": val} for val in var_vals]

        #             #     for n in ns:

        #             #         (bon_mean, bon_std), (won_mean, won_std), (maj_n_mean, maj_n_std) = bootstrap_metric(
        #             #             data,
        #             #             subset_size=n,
        #             #             reduce_fns=[
        #             #                 lambda arr: np.max([d["val"] for d in arr]),
        #             #                 lambda arr: np.min([d["val"] for d in arr]),
        #             #                 partial(calc_maj_val, vote_key="pred", val_key="val")
        #             #             ])
        #             #         metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
        #             #         metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
        #             #         metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

        #             data_src2prompt2var2metric[data_source][prompt][var_name] = metric

        # data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        #     for prompt, var2metric in prompt2var2metric.items():
        #         for var_name, metric in var2metric.items():
        #             for metric_name, metric_val in metric.items():
        #                 data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

        # metric_dict = {}
        # for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        #     for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
        #         for metric_name, prompt_vals in metric2prompt_vals.items():
        #             pfx = f"{data_source}/{var_name}/{metric_name}"
        #             metric_dict[pfx] = np.mean(prompt_vals)

        # val_metric_dict = {f"val/{key}": value for key, value in metric_dict.items()}
        # return val_metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # Determine the role based on whether it will be used for generation
            ref_role = 'ref'
            if self.config.trainer.get('use_ref_for_generation', False):
                 # If used for generation, it needs rollout capabilities.
                 # Assign 'actor_rollout_ref' role to ensure rollout components are initialized.
                 ref_role = 'actor_rollout_ref'
                 print(f"Initializing reference policy worker with role '{ref_role}' because use_ref_for_generation is True.")

            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role=ref_role) # Use the determined role
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            # Init model for the reference policy worker (now potentially with role 'actor_rollout_ref')
            self.ref_policy_wg.init_model() # Ensure init_model is called

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # Note: self.actor_rollout_wg is distinct from self.ref_policy_wg
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
    
    def _compute_and_save_dataset_advantages(self, step, dataset_type='train', get_gt_log_prob=False, return_entropy=False):
        """Compute and save advantages for the entire dataset. This function is also used  to compute advantage variance for sampler weights
        
        Args:
            step: Current step number (-1 means before training)
            dataset_type: 'train' or 'val'
            save: Whether to save the advantages
        
        Returns (optional): advantage variance for sampler weights
        """
        # TODO: this probably doesnt work for PPO or the likes. Need to uncomment few lines below to potentially fix (also need to change scores, etc).
        
        print(f"Computing advantages for entire {dataset_type} dataset (step {step})...")

        # important args for advantage tracking
        if self.advantage_tracking_enabled:
            n_rollouts = self.config.actor_rollout_ref.rollout.n if self.config.actor_rollout_ref.rollout.n_advantage_tracking is None else self.config.actor_rollout_ref.rollout.n_advantage_tracking
        else:
            n_rollouts = self.config.actor_rollout_ref.rollout.n
        
        shuffle = False
        # Fix the division by zero error by checking if n_rollouts equals rollout.n
        if n_rollouts == self.config.actor_rollout_ref.rollout.n:
            batch_size = self.config.data.train_batch_size
        else:
            batch_size = self.config.data.train_batch_size // (n_rollouts // self.config.actor_rollout_ref.rollout.n)
        
        # TODO creating dataloader again is inefficient. But doing it anyway because I want advantage computation for same samples. 
        # Create and choose the appropriate dataloader
        train_dataloader, val_dataloader = self._create_dataloader(set_to_self=False, shuffle=shuffle)
        dataloader = train_dataloader if dataset_type == 'train' else val_dataloader
        
        # Storage for all advantages
        all_advantages = defaultdict(list)
        
        # Get world size for batch size adjustment
        world_size = self.actor_rollout_wg.world_size
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc=f"Computing {dataset_type} advantages")):
                # Convert to DataProto
                batch = DataProto.from_single_dict(batch_dict)
                
                # Skip empty batches
                if len(batch.batch) == 0:
                    continue
                
                # Ensure batch size is divisible by world_size by truncating extra samples
                batch_size = len(batch.batch)
                if batch_size % world_size != 0:
                    # Calculate how many samples to keep (truncate the rest)
                    keep_size = (batch_size // world_size) * world_size
                    
                    # Use the reorder method to keep only the first keep_size samples
                    indices = torch.arange(keep_size)
                    batch.reorder(indices)
                    
                    print(f"Truncated batch from {batch_size} to {keep_size} samples to ensure divisibility by {world_size}")
                
                # apparently this is needed for the advantage computation

                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            dtype=object)
                # Add unique indices are not present
                if 'index' not in batch.non_tensor_batch:
                    batch.non_tensor_batch['index'] = np.array([f"{dataset_type}_{batch_idx}_{i}" 
                                                             for i in range(len(batch.batch))], dtype=object)
                
                # get ground truth log prob
                if get_gt_log_prob:
                    with torch.no_grad():
                        # Clone the original data to avoid interfering with advantage logic
                        gt_data = deepcopy(batch)
                        gt_answers = []
                        gt_data.batch['input_ids'] = torch.zeros_like(gt_data.batch['input_ids'])
                        # We'll replace "responses" in gt_data with the ground-truth answer per sample
                        for i, info in enumerate(gt_data.non_tensor_batch['extra_info']):
                            ground_truth_answer = info['answer']
                            # Encode the ground-truth answer (strip special tokens as needed)
                            answer_ids = self.tokenizer.encode(ground_truth_answer, add_special_tokens=False)
                            # Convert to tensor and place it into the same shape as "responses"
                            ans_t = torch.tensor(answer_ids, dtype=torch.long, device=gt_data.batch['input_ids'].device)
                            gt_data.batch['input_ids'][i, :len(ans_t)] = ans_t
                            # Update attention_mask for the newly assigned tokens
                            gt_data.batch['attention_mask'][i, -(len(ans_t)):] = 1
                            gt_answers.append(ground_truth_answer)

                        # Now compute the log_prob of these ground-truth answers with the current policy
                        # "actor" or "actor_rollout_ref" might differ based on your usage
                        gt_data.batch['responses'] = gt_data.batch['input_ids']
                        if return_entropy:
                            entropy, gt_log_prob = self.actor_rollout_wg._forward_micro_batch(gt_data)
                            batch.batch['gt_entropy'] = entropy
                        else:
                            gt_log_prob = self.actor_rollout_wg.compute_log_prob(gt_data)

                        # Save it alongside the rest of the data
                        batch.batch['answer_ids'] = gt_data.batch['input_ids']
                        batch.non_tensor_batch['gt_answers'] = np.array(gt_answers, dtype=object)
                        batch.batch['gt_log_prob'] = gt_log_prob.batch['old_log_probs']
                
                # Generate responses using the current policy
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                if n_rollouts // self.config.actor_rollout_ref.rollout.n > 1: # we need this only for saving and not for sampler weights
                    #TODO: This works but it's super ugly. Better to recreate a actor with larger n and load checkpoint
                    gen_batch_output = []
                    for i in range(n_rollouts // self.config.actor_rollout_ref.rollout.n):
                        gen_batch_output.append(self.actor_rollout_wg.generate_sequences(gen_batch))
                    gen_batch_output = DataProto.concat(gen_batch_output)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.repeat(repeat_times=n_rollouts // self.config.actor_rollout_ref.rollout.n, interleave=False)
                    batch = batch.union(gen_batch_output)
                else:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    batch = batch.repeat(repeat_times=n_rollouts, interleave=True)
                    batch = batch.union(gen_batch_output)
              
                # Compute rewards
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)
                
                reward_tensor = self.reward_fn(batch)
                batch.batch['token_level_scores'] = reward_tensor
                
                # Apply KL penalty if needed
                if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False) and self.use_reference_policy:
                    batch, _ = apply_kl_penalty(batch,
                                              kl_ctrl=self.kl_ctrl,
                                              kl_penalty=self.config.algorithm.kl_penalty)
                else:
                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                
                # Compute advantages
                batch = compute_advantage(batch,
                                        adv_estimator=self.config.algorithm.adv_estimator,
                                        gamma=self.config.algorithm.gamma,
                                        lam=self.config.algorithm.lam,
                                        num_repeat=1)

                # get log prob for rollouts
                if return_entropy:
                    with torch.no_grad():
                        entropy, rollout_log_prob = self.actor_rollout_wg._forward_micro_batch(batch)
                        batch.batch['rollout_entropy'] = entropy
                        batch.batch['rollout_log_prob'] = rollout_log_prob
                else:
                    rollout_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch.batch['rollout_log_prob'] = rollout_log_prob.batch['old_log_probs']
                
                
                # Extract and store the data
                advantages = batch.batch['advantages'].detach().cpu()
                sample_ids = batch.non_tensor_batch['index']
                input_ids = batch.batch['input_ids'].detach().cpu()
                response_ids = batch.batch['responses'].detach().cpu()
                rollout_log_probs = batch.batch['rollout_log_prob'].detach().cpu()
                rollout_entropies = batch.batch['rollout_entropy'].detach().cpu() if return_entropy else None
                # get ground truth log prob
                if get_gt_log_prob:
                    answer_ids = batch.batch['answer_ids'].detach().cpu()
                    gt_log_probs = batch.batch['gt_log_prob'].detach().cpu()
                    gt_answers = batch.non_tensor_batch['gt_answers']
                # get token level scores (this also might change for something that is not grpo)
                if dataset_type == 'train':
                    batch.batch['token_level_rewards'] = self.reward_fn(batch)
                else:
                    batch.batch['token_level_rewards'] = self.val_reward_fn(batch)
                token_level_rewards = batch.batch['token_level_rewards'].detach().cpu()

                # not computing this for now
                # attention_mask = batch.batch['attention_mask'].detach().cpu()
                # response_length = response_ids.size(1)

    
                #response_mask = attention_mask[:, -response_length:]
                #token_rewards = batch.batch['token_level_rewards'].detach().cpu() if 'token_level_rewards' in batch.batch else None
                
                # Store data for each sample
                for i in range(len(sample_ids)):
                    sample_id = sample_ids[i]
                    
                    # Decode text for better analysis
                    prompt_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    response_text = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)

                    if get_gt_log_prob:
                        gt_answer = gt_answers[i]
                        gt_log_prob = gt_log_probs[i]
                        gt_answer_ids = answer_ids[i]
                        rollout_log_prob = rollout_log_probs[i]

                    
                        all_advantages[sample_id].append({
                            'prompt': prompt_text,
                            'response': response_text,
                            'gt_answer': gt_answer,
                            'gt_log_prob': gt_log_prob,
                            'gt_answer_ids': gt_answer_ids,
                            'advantage': advantages[i].numpy()[0],
                            'score': token_level_rewards[i].numpy().sum(),
                            'rollout_log_prob': rollout_log_prob,
                            'rollout_entropy': rollout_entropies[i] if return_entropy else None
                        })
                    else:
                        all_advantages[sample_id].append({
                            'prompt': prompt_text,
                            'response': response_text,
                            'advantage': advantages[i].numpy()[0],
                            'score': token_level_rewards[i].numpy().sum()
                        })
                        
                    # computing entire response mask and token rewards and advantage is unnecessary and only useful if we have token level / process level rewards
                    # all_advantages[sample_id].append({
                    #     'prompt': prompt_text,
                    #     'response': response_text,
                    #     'advantage': advantages[i].numpy(),
                    #     'response_mask': response_mask[i].numpy(),
                    #     'token_reward': token_rewards[i].numpy() if token_rewards is not None else None
                    # })
                    # Save the collected advantages
        step_label = "pre_training" if step == -1 else f"step_{step}"
        filename = f'advantages_{dataset_type}_{step_label}.pt'
        filepath = os.path.join(self.advantage_tracking_path, filename)
        advantage_data = {
                'step': step,
                'dataset_type': dataset_type,
                'samples': dict(all_advantages)
            }
            
        torch.save(advantage_data, filepath)
        print(f"Saved {dataset_type} advantage data to {filepath}")
        
    def _log_metrics_to_jsonl(self, filepath: str, step: int, metrics: dict):
        """Appends a step's metrics to a JSON Lines file."""
        if self.config.actor_rollout_ref.actor.DEV_ESTIMATE_ENTROPY_DELTA and metrics.get('actor/H_t') is not None:
            deltas, advs = metrics.pop('actor/H_t'), metrics.pop('actor/advantages')

            print(f"LOG_METRICS, deltas: {deltas}")

            path = pathlib.Path(filepath).parent / 'entropy_dumps'
            path.mkdir(exist_ok=True)
            with open(path / f'dump-{step}.pickle', 'wb') as file:
                pickle.dump({
                    'deltas' : deltas,
                    'advantages' : advs
                }, file)


        def convert_value(v):
            if isinstance(v, torch.Tensor):
                return v.item() # Use .item() to get standard Python type from tensor
            elif isinstance(v, np.floating): # Check for any numpy float type
                return float(v)
            elif isinstance(v, np.integer): # Check for any numpy integer type
                 return int(v)
            # Add checks for other non-serializable types if needed
            return v

        try:
            # Ensure all values are JSON serializable
            log_entry = {"step": step, **{k: convert_value(v) for k, v in metrics.items()}}
            with open(filepath, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write metrics to {filepath} for step {step}. Error: {e}")

    def _check_early_stopping(self, val_metrics: dict, current_step: int) -> bool:
        """
        Check if early stopping criteria are met based on the mean of validation benchmark metrics.
        
        Args:
            val_metrics: Dictionary of validation metrics
            current_step: Current training step
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if not self.early_stopping_enabled:
            return False
            
        if not val_metrics:
            print("Warning: No validation metrics provided for early stopping check")
            return False
        
        # Calculate mean of validation benchmark mean metrics only
        # Filter for keys that contain "val/" and "mean" but exclude "std" and exclude duplicates
        metric_values = []
        used_metrics = []
        for key, value in val_metrics.items():
            if (isinstance(value, (int, float)) and 
                key.startswith('val/') and 
                '/mean' in key and 
                '/std' not in key and
                'time' not in key.lower()):
                metric_values.append(float(value))
                used_metrics.append(key)
        
        if not metric_values:
            print("Warning: No validation benchmark mean metrics found for early stopping")
            print(f"Available metrics: {list(val_metrics.keys())}")
            return False
            
        current_mean_metric = sum(metric_values) / len(metric_values)
        
        # Check if current mean metric is better than best (always maximize mean)
        is_better = current_mean_metric > self.best_mean_metric_value + self.early_stopping_min_delta
            
        if is_better:
            # New best metric found
            self.best_mean_metric_value = current_mean_metric
            self.best_step = current_step
            self.patience_counter = 0
            
            # Save checkpoint as best if enabled
            if self.save_best_checkpoint:
                self._save_best_checkpoint(current_step)
                
            print(f"New best mean eval metric: {current_mean_metric:.6f} at step {current_step}")
            print(f"  Based on {len(metric_values)} metrics: {used_metrics}")
            return False
        else:
            # No improvement
            self.patience_counter += 1
            print(f"No improvement in mean eval metric for {self.patience_counter}/{self.early_stopping_patience} steps "
                  f"(current: {current_mean_metric:.6f}, best: {self.best_mean_metric_value:.6f} at step {self.best_step})")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered! No improvement in mean eval metric "
                      f"for {self.early_stopping_patience} validation checks.")
                print(f"Best mean eval metric: {self.best_mean_metric_value:.6f} at step {self.best_step}")
                
                self.early_stopped = True
                return True
                
        return False

    def _save_best_checkpoint(self, step: int):
        """Save the current model as the best checkpoint based on mean eval metrics."""
        best_checkpoint_folder = os.path.join(self.config.trainer.default_local_dir, 'best_checkpoint')
        
        # Clean up previous best checkpoint if it exists
        if os.path.exists(best_checkpoint_folder):
            shutil.rmtree(best_checkpoint_folder)
            print(f"Removed previous best checkpoint directory: {best_checkpoint_folder}")
        
        os.makedirs(best_checkpoint_folder, exist_ok=True)
        
        try:
            actor_local_path = os.path.join(best_checkpoint_folder, 'actor')
            
            # Save actor checkpoint without updating the previous_save_local_path
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                None,
                step,
                remove_previous_ckpt=False,  # Don't interfere with regular checkpoint cleanup
                update_previous_path=False   # Don't update the checkpoint manager's tracking
            )
            
            # Save critic checkpoint if using critic
            if self.use_critic:
                critic_local_path = os.path.join(best_checkpoint_folder, 'critic')
                self.critic_wg.save_checkpoint(
                    critic_local_path,
                    None,
                    step,
                    remove_previous_ckpt=False,  # Don't interfere with regular checkpoint cleanup
                    update_previous_path=False   # Don't update the checkpoint manager's tracking
                )
            
            # Save metadata about the best checkpoint
            metadata = {
                'step': step,
                'mean_eval_metric': self.best_mean_metric_value,
                'timestamp': time.time(),
                'early_stopping_enabled': self.early_stopping_enabled,
                'patience': self.early_stopping_patience,
                'min_delta': self.early_stopping_min_delta
            }
            
            metadata_path = os.path.join(best_checkpoint_folder, 'best_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Also save dataloader state for complete restoration if needed
            dataloader_local_path = os.path.join(best_checkpoint_folder, 'data.pt')
            try:
                dataloader_state_dict = self.train_dataloader.state_dict()
                torch.save(dataloader_state_dict, dataloader_local_path)
            except Exception as e:
                print(f"Warning: Could not save dataloader state for best checkpoint: {e}")
                
            self.best_checkpoint_path = best_checkpoint_folder
            print(f"Saved best checkpoint at step {step} with mean eval metric={self.best_mean_metric_value:.6f}")
            
        except Exception as e:
            print(f"Error saving best checkpoint: {e}")
            # Don't fail training if checkpoint saving fails
            import traceback
            traceback.print_exc()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        import json # Added import for json
        from verl.utils import hdfs_io # Added import for hdfs_io

        # Save config before initializing tracking
        base_save_dir = self.config.trainer.default_local_dir
        os.makedirs(base_save_dir, exist_ok=True)

        # Define eval log file path
        eval_log_path = os.path.join(base_save_dir, 'eval.jsonl')
        print(f"Logging validation metrics to: {eval_log_path}")
        # Define training metrics log file path
        train_metrics_log_path = os.path.join(base_save_dir, 'metrics.jsonl')
        print(f"Logging training metrics to: {train_metrics_log_path}")


        config_path = os.path.join(base_save_dir, 'config.json')
        config_dict = OmegaConf.to_container(self.config, resolve=True) # Convert OmegaConf to dict
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(f"Configuration saved to {config_path}")

        # Optionally copy config to HDFS once
        if self.config.trainer.default_hdfs_dir:
            hdfs_base_dir = self.config.trainer.default_hdfs_dir
            hdfs_io.makedirs(hdfs_base_dir, exist_ok=True)
            hdfs_config_path = os.path.join(hdfs_base_dir, 'config.json')
            try:
                # Use put to copy the single file
                hdfs_io.put(src=config_path, dst=hdfs_config_path)
                print(f"Configuration copied to HDFS: {hdfs_config_path}")
            except Exception as e:
                print(f"Failed to copy config to HDFS: {e}")


        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=config_dict) # Use the saved config dict for tracking

        self.global_steps = 0

        # load checkpoint before doing anything
        print(f'Loading checkpoint. Configured resume_mode: {self.config.trainer.resume_mode}, resume_from_path: {self.config.trainer.resume_from_path}')
        self._load_checkpoint() # This sets self.global_steps. If resume_mode='disable', self.global_steps remains 0 or its initial value.


        # Determine the step for validation logging
        # If evaluation_step is explicitly passed (e.g., for val_only from a specific model folder), use it.
        # Otherwise, use the global_steps determined by checkpoint loading.
        effective_eval_step = self.global_steps 
        if hasattr(self.config.trainer, 'evaluation_step') and self.config.trainer.evaluation_step is not None:
            try:
                passed_eval_step = int(self.config.trainer.evaluation_step)
                effective_eval_step = passed_eval_step # Prioritize if passed
                print(f"Using trainer.evaluation_step ({effective_eval_step}) for current operation (validation/advantage tracking step).")
            except ValueError:
                print(f"Warning: Could not parse trainer.evaluation_step ('{self.config.trainer.evaluation_step}') as int. Using loaded global_steps: {self.global_steps} for current operation.")
        
        advantage_tracking_step_for_saving = effective_eval_step # Step to use for naming advantage files

        if self.config.trainer.get('advantage_tracking_only', False):
            print(f"Advantage tracking only mode enabled. Tracking for step: {advantage_tracking_step_for_saving}")
            
            # Ensure advantage_tracking_path is created (already handled in __init__ modification)

            if not self.advantage_tracking_enabled and not self.config.trainer.get('track_advantages', False):
                # This condition implies trainer.track_advantages was initially false.
                # The __init__ modification for os.makedirs might not have run if only advantage_tracking_only was true but track_advantages was false.
                # Let's ensure it's created.
                print("Ensuring advantage_tracking_path exists as 'advantage_tracking_only' is True.")
                os.makedirs(self.advantage_tracking_path, exist_ok=True)


            self._compute_and_save_dataset_advantages(
                step=advantage_tracking_step_for_saving, 
                dataset_type='train',
                get_gt_log_prob=self.config.trainer.get('get_gt_log_prob', False),
                return_entropy=self.config.trainer.get('return_entropy', False)
            )
            self._compute_and_save_dataset_advantages(
                step=advantage_tracking_step_for_saving, 
                dataset_type='val',
                get_gt_log_prob=self.config.trainer.get('get_gt_log_prob', False),
                return_entropy=self.config.trainer.get('return_entropy', False)
            )
            print(f"Advantage tracking complete for step {advantage_tracking_step_for_saving}. Exiting.")
            return # Exit after tracking
        
        initial_val_log_step = effective_eval_step # Use this for the initial validation log if not in advantage_tracking_only mode

        # perform validation before training
        print(f'Performing validation (logging as step {initial_val_log_step})')
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            if self.config.trainer.get('skip_val', False):
                print("Skipping validation")
            else:
                val_metrics = self._validate()
                pprint(f'Initial validation metrics (step {initial_val_log_step}): {val_metrics}')
                logger.log(data=val_metrics, step=initial_val_log_step)
                # Log initial metrics to jsonl
                if val_metrics: # Ensure metrics are not empty
                    self._log_metrics_to_jsonl(eval_log_path, initial_val_log_step, val_metrics)
                
                # Initialize early stopping with initial validation metrics
                if self.early_stopping_enabled and val_metrics:
                    # Calculate mean of validation benchmark mean metrics for initialization
                    metric_values = []
                    used_metrics = []
                    for key, value in val_metrics.items():
                        if (isinstance(value, (int, float)) and 
                            key.startswith('val/') and 
                            '/mean' in key and 
                            '/std' not in key and
                            'time' not in key.lower()):
                            metric_values.append(float(value))
                            used_metrics.append(key)
                    
                    if metric_values:
                        initial_mean_metric = sum(metric_values) / len(metric_values)
                        self.best_mean_metric_value = initial_mean_metric
                        self.best_step = initial_val_log_step
                        print(f"Initialized early stopping with mean eval metric={initial_mean_metric:.6f} at step {initial_val_log_step}")
                        print(f"  Based on {len(metric_values)} metrics: {used_metrics}")
                        
                        # Save initial checkpoint as best if enabled
                        if self.save_best_checkpoint:
                            self._save_best_checkpoint(initial_val_log_step)

            if self.advantage_tracking_enabled:
                #step_init = self.config.trainer.get('step_init', -1)
                # For advantage tracking, use the initial_val_log_step which reflects evaluation_step if provided
                step_init_adv_track = initial_val_log_step
                self._compute_and_save_dataset_advantages(step=step_init_adv_track, dataset_type='val', get_gt_log_prob=self.config.trainer.get('get_gt_log_prob', False), return_entropy=self.config.trainer.get('return_entropy', False))
                self._compute_and_save_dataset_advantages(step=step_init_adv_track, dataset_type='train', get_gt_log_prob=self.config.trainer.get('get_gt_log_prob', False), return_entropy=self.config.trainer.get('return_entropy', False))

            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1 (or the step after load)
        # If resuming, global_steps is already set. If not (e.g. resume_mode='disable'), it's 0.
        # Increment only if we are actually starting training steps.
        if not self.config.trainer.get('val_only', False) and self.global_steps == 0 : # check global_steps too if it was loaded as >0
             self.global_steps += 1 # Start from step 1 if not resuming and not val_only

        total_seen_samples = 0
        last_val_metrics = None
        steps_per_epoch = self.total_training_steps // self.config.trainer.total_epochs

        while self.global_steps < self.total_training_steps:
            epoch = self.global_steps // steps_per_epoch
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        if self.config.trainer.get('use_ref_for_generation', False):
                            gen_batch_output = self.ref_policy_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['extra_info']
                        except Exception as e:
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch['token_level_scores'] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(reward_extra_infos_dict)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                    # TODO: this should probably be skipped when algorithm.only_positive_advantages.enable is True
                    if self.config.algorithm.filter_groups.enable:
                        filter_metric_dict = {}
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            batch.non_tensor_batch["seq_final_reward"] = batch.batch['token_level_scores'].sum(
                                dim=-1).tolist()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(batch.non_tensor_batch['uid'], batch.non_tensor_batch[metric_name]):
                        #for uid, metric_val in zip(batch.non_tensor_batch['uid'], batch.batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0]
                        filter_metric_dict[f"qualified_prompt_ratio/{metric_name}"] = len(kept_prompt_uids) / len(
                            prompt_uid2metric_vals)
                        filter_metric_dict[f"qualified_prompt_bsz/{metric_name}"] = len(kept_prompt_uids)

                        train_prompt_bsz = self.config.data.train_batch_size
                        fill_to_train_bsz = self.config.algorithm.filter_groups.fill_to_train_bsz
                        if len(kept_prompt_uids) > train_prompt_bsz or not fill_to_train_bsz:
                            kept_prompt_uids = kept_prompt_uids[:train_prompt_bsz]
                        else:
                            for prompt_uid in prompt_uid2metric_std.keys():
                                if prompt_uid not in kept_prompt_uids:
                                    kept_prompt_uids.append(prompt_uid)
                                if len(kept_prompt_uids) == train_prompt_bsz:
                                    break

                        kept_traj_idxs = []
                        for traj_idx, traj_prompt_uid in enumerate(batch.non_tensor_batch['uid']):
                            if traj_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(traj_idx)
                        filter_metric_dict[f"qualified_traj_bsz/{metric_name}"] = len(kept_traj_idxs)

                        world_size = self.actor_rollout_wg.world_size
                        kept_traj_idxs = kept_traj_idxs[:len(kept_traj_idxs) // world_size * world_size]
                        # TODO: what if len(kept_traj_idxs) < world_size?
                        if self.config.algorithm.filter_groups.drop_last_mini_batch:
                            train_traj_mini_bsz = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
                            if len(kept_traj_idxs) > train_traj_mini_bsz:
                                kept_traj_idxs = kept_traj_idxs[:len(kept_traj_idxs) // train_traj_mini_bsz *
                                                                train_traj_mini_bsz]
                            else:
                                print(f'[WARNING] {len(kept_traj_idxs)=} < {train_traj_mini_bsz=}')

                        filter_metric_dict["final_traj_ratio"] = len(kept_traj_idxs) / len(batch.batch)
                        filter_metric_dict["final_traj_bsz"] = len(kept_traj_idxs)

                        metrics.update({f"train/filter/{k}": v for k, v in filter_metric_dict.items()})
                        kept_traj_idxs = np.array(kept_traj_idxs)
                        batch = batch.select_idxs(kept_traj_idxs)

                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        if self.config.trainer.get('use_ref_for_generation', False):
                            # Generate with ref, so "old" log prob IS the ref log prob
                            with _timer('ref_log_prob', timing_raw):
                                # Ensure ref_policy_wg exists due to __init__ check
                                ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob_output)
                            # Copy ref_log_prob to old_log_probs for PPO update
                            batch.batch['old_log_probs'] = batch.batch['ref_log_prob']
                            # Actor's log prob is not needed as 'old_log_probs' here
                        else:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                            if self.use_reference_policy:
                                # compute reference log_prob (only needed if KL penalty/loss is active, but compute anyway for consistency)
                                with _timer('ref_log_prob', timing_raw):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)


                    total_seen_samples += len(batch.batch)
                    metrics['total_seen_samples'] = total_seen_samples

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        
                        # actor_output_metrics already contains 'perf/mfu/actor' and potentially raw FLOPs counts
                        # if added in the worker. Let's ensure we log everything it returns.
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            # For periodic validation during training, log with current self.global_steps
                            # unless evaluation_step is meant to override all val logs (less likely for periodic)
                            # The effective_eval_step logic above was more for initial/val_only.
                            # For periodic validation, self.global_steps is the most relevant.
                            current_periodic_val_log_step = self.global_steps
                            if hasattr(self.config.trainer, 'evaluation_step') and self.config.trainer.evaluation_step is not None and self.config.trainer.get('val_only', False):
                                # If val_only and evaluation_step is set, periodic validation doesn't occur, but if it did, it should use evaluation_step
                                current_periodic_val_log_step = int(self.config.trainer.evaluation_step)

                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        # Log periodic/final metrics to jsonl
                        if val_metrics: # Ensure metrics are not empty
                            self._log_metrics_to_jsonl(eval_log_path, current_periodic_val_log_step, val_metrics)
                            logger.log(data=val_metrics, step=current_periodic_val_log_step) # also log to wandb etc.
                        
                        # Check early stopping criteria
                        if self.early_stopping_enabled and val_metrics and not is_last_step:
                            should_stop = self._check_early_stopping(val_metrics, current_periodic_val_log_step)
                            if should_stop:
                                print("Early stopping triggered, ending training...")
                                return

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Log all collected metrics, including FLOPs/MFU from workers
                logger.log(data=metrics, step=self.global_steps)
                # Log training metrics to JSONL file
                self._log_metrics_to_jsonl(train_metrics_log_path, self.global_steps, metrics)


                # Compute advantages at the end of the epoch if needed
                if self.advantage_tracking_enabled and (
                    epoch == self.config.trainer.total_epochs - 1 or  # Last epoch
                    self.advantage_tracking_freq > 0 and (self.global_steps % self.advantage_tracking_freq == 0)  # By frequency
                ):
                    self._compute_and_save_dataset_advantages(step=self.global_steps, dataset_type='train', get_gt_log_prob=self.config.trainer.get('get_gt_log_prob', False), return_entropy=self.config.trainer.get('return_entropy', False))
                    self._compute_and_save_dataset_advantages(step=self.global_steps, dataset_type='val', get_gt_log_prob=self.config.trainer.get('get_gt_log_prob', False), return_entropy=self.config.trainer.get('return_entropy', False))


                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    if self.early_stopping_enabled and not self.early_stopped:
                        print("Training completed normally (no early stopping triggered)")
                    return

                self.global_steps += 1
        
        # Training completed after reaching total_training_steps
        if self.early_stopping_enabled and not self.early_stopped:
            print("Training completed normally after reaching total_training_steps")
