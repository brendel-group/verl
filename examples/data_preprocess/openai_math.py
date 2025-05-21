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
Preprocess the OpenAI Math dataset to parquet format
"""

import os
import datasets

# Replace verl.utils.hdfs_io import with direct implementations
# from verl.utils.hdfs_io import copy, makedirs
import shutil
import os

def copy(src, dst):
    """Simple replacement for hdfs copy using shutil"""
    shutil.copytree(src, dst, dirs_exist_ok=True)

def makedirs(dir_path):
    """Simple replacement for hdfs makedirs using os"""
    os.makedirs(dir_path, exist_ok=True)

import argparse

# Remove import for math utility functions that we won't use
# from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


# Remove this function as we don't need it with the new dataset
# def extract_solution(solution_str):
#    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/fast/pmayilvahanan/datasets/openai_math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--follow_instruction', action='store_true')

    args = parser.parse_args()

    # Use the new simplescaling/openaimath dataset
    data_source = 'simplescaling/openaimath'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    if args.follow_instruction:
        instruction_following = " Let's think step by step and output the final answer within \\boxed{}."
    else:
        instruction_following = ""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')
            question = question + instruction_following

            # Get answer directly from the answer field 
            answer = example.pop('answer')
            
            # Extract other fields to store in extra_info
            solution = example.pop('solution', None)
            subject = example.pop('subject', None)
            level = example.pop('level', None)
            unique_id = example.pop('unique_id', None)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'solution': solution,
                    'subject': subject,
                    'level': level,
                    'unique_id': unique_id
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.follow_instruction:
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    else:
        train_dataset.to_parquet(os.path.join(local_dir, 'train_no_instruction.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, 'test_no_instruction.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)