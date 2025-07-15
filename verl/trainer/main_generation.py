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
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
import logging

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from typing import List, Callable
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.reward_score.math import compute_score as math_compute_score


logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


def select_reward_fn(data_source):
    if data_source == 'DigitalLearningGmbH/MATH-lighteval' or data_source == 'lighteval/MATH':
        return math_compute_score
    else:
        raise NotImplementedError

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))

def add_score_column_to_dataset(output_list : List[List[str]], dataset: pd.DataFrame, reward_fn : Callable, config : OmegaConf) -> pd.DataFrame:
    try:   
        ground_truths = dataset['reward_model'].apply(lambda d: d['ground_truth'])
        ground_truths = ground_truths.apply(lambda entry: [entry] * config.data.n_samples) # broadcast ground truth to match n_samples

        # take responses and ground truths and map them to rewards
        rewards = [np.array(list(map(reward_fn, outs, gts))) for outs, gts in zip(output_list, ground_truths)]
        dataset['rewards'] = rewards
    except Exception as e:
        logger.warning('Encountered exception during reward evaluation. Continuing...')

def dump_parquet(filename : str, dataset : pd.DataFrame):
    output_dir = os.path.dirname(filename)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(filename)
    print(f'Output saved to {filename}')

@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    
    compute_scores = config.data.get('compute_scores', False) # whether to compute the reward scores of the rollouts
    dump_parts = config.data.get('dump_parts', False) # whether to dump the intermediate batches instead of the full list

    
    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    # only take first n samples.
    dataset = pd.read_parquet(config.data.path)
    dataset = dataset.head(config.data.take_first_n)  \
        if config.data.get('take_first_n', -1) > 0 else dataset
    
    if compute_scores:
        data_sources = dataset['data_source']
        if len(set(data_sources)) > 1:
            raise RuntimeError("Mixed data source. Currently not supported.")
        reward_fn = select_reward_fn(data_source=data_sources.iloc[0])

    prompts = dataset[config.data.prompt_key].apply(lambda arr : arr[0]['content']).tolist()
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size)
    output_accu = [[] for _ in range(config.data.n_samples)]


    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_start_idx, batch_end_idx = batch_idx * config_batch_size, (batch_idx + 1) * config_batch_size
        batch = prompts[batch_start_idx: batch_end_idx]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors='pt',
            return_attention_mask=True,
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        # Repeat the batch for n_samples, interleaved, as in RayPPOTrainer
        n_samples = config.data.n_samples
        data_repeated = data.repeat(repeat_times=n_samples, interleave=True)
        output = wg.generate_sequences(data_repeated)  # shape [batch_size * n_samples, ...]

        # Remove dummy data (only keep batch_size * n_samples)
        total_real = real_batch_size * n_samples
        output = output[:total_real]

        # Reshape output: group by original prompt, each with n_samples
        # output.batch['input_ids'] shape: [total_real, seq_len]
        output_ids = output.batch['input_ids'][:, -config.rollout.response_length:]
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        pad_token = tokenizer.pad_token
        output_text_unpad = [text.replace(pad_token, '') for text in output_text]

        # Group outputs: [batch_size, n_samples]
        grouped_outputs = [
            output_text_unpad[i * n_samples:(i + 1) * n_samples]
            for i in range(real_batch_size)
        ]

        if dump_parts:
            sub_df = dataset.iloc[batch_start_idx:batch_end_idx]
            sub_df = sub_df.copy()
            sub_df['responses'] = grouped_outputs

            if compute_scores:
                add_score_column_to_dataset(output_list=grouped_outputs, dataset=sub_df, reward_fn=reward_fn, config=config)

            part_path = config.data.output_path + f'-part{batch_idx}'
            dump_parquet(filename=part_path, dataset=sub_df)
        else:
            # extend along the second (n_data) dimension
            for i, l in enumerate(output_accu):
                l.extend([group[i] for group in grouped_outputs])

    if not dump_parts: # dump whole list at the end
        # convert output_accu from (n_batch, n_samples, n_data) to (n_data, n_sampels)
        output_accu = np.array(output_accu, dtype=object)
        output_accu = np.transpose(output_accu, axes=(1, 0)).tolist()
        # add to the data frame
        dataset[f'responses'] = output_accu

        if compute_scores:
            add_score_column_to_dataset(output_list=output_accu, dataset=dataset, reward_fn=reward_fn, config=config)

        dump_parquet(filename=config.data.output_path, dataset=dataset)

if __name__ == '__main__':
    main()
