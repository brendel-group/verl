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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                                 axis=1)]

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        prompt_with_chat_template = None # Initialize

        # Try applying the chat template; handle missing template and ensure string output
        try:
            # Attempt to apply the chat template
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            # Ensure the template output is actually a string (should be, but good practice)
            if not isinstance(prompt_with_chat_template, str):
                 print(f"Warning: tokenizer.apply_chat_template did not return str. Type: {type(prompt_with_chat_template)}. Converting.")
                 prompt_with_chat_template = str(prompt_with_chat_template)

        except ValueError as e:
            if "Cannot use chat template functions" in str(e):
                print("Warning: Chat template not found or applicable. Using fallback formatting.")
                # Fallback: Convert chat to a string representation reliably
                if isinstance(chat, list):
                    # Format list of messages into a single string
                    formatted_chat = ""
                    for message in chat:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        # Ensure content is string before appending
                        formatted_chat += f"{role}: {str(content)}\n"
                    # Manually add the equivalent of add_generation_prompt=True if needed
                    # This assumes the last role is user and expects an assistant response
                    if formatted_chat and not formatted_chat.strip().endswith("assistant:"):
                         formatted_chat += "assistant: " # Adjust based on expected format
                    prompt_with_chat_template = formatted_chat
                elif isinstance(chat, str):
                    # If it's already a string, use it directly
                     prompt_with_chat_template = chat
                     # Optionally ensure assistant prompt if needed for consistency
                     # if not prompt_with_chat_template.strip().endswith("assistant: "):
                     #    prompt_with_chat_template += " assistant: "
                else:
                    # Force conversion for any other type
                    print(f"Warning: Converting non-list/non-string chat data of type {type(chat)} to string.")
                    prompt_with_chat_template = str(chat)
                    # Potentially add assistant prompt here too if structure is unknown
                    # if not str(prompt_with_chat_template).strip().endswith("assistant:"):
                    #    prompt_with_chat_template += " assistant: "

                # Final check: Ensure the fallback produced a string
                if not isinstance(prompt_with_chat_template, str):
                     # This should ideally not happen with the logic above
                     raise TypeError(f"Fallback formatting failed to produce a string from chat data. Got type: {type(prompt_with_chat_template)}")
            else:
                # Re-raise any other ValueError not related to chat templates
                raise
        except Exception as e:
            # Catch any other unexpected errors during template application/fallback
            print(f"Error processing chat data: {e}")
            raise

        # Ensure prompt is not None before proceeding (safety check)
        if prompt_with_chat_template is None:
            raise ValueError("Failed to obtain a valid string prompt from the chat data.")


        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            # Ensure multimodal prompt replacement happens on the string we just prepared
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs.get('image_grid_thw') # Use .get for safety
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None and image_grid_thw.numel() > 0: # Check if not None and not empty
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                # Use a temporary variable for replacement to avoid modifying the loop condition source unexpectedly
                processed_prompt = prompt_with_chat_template
                while '<image>' in processed_prompt and index < len(image_grid_thw): # Add index boundary check
                    processed_prompt = processed_prompt.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod().item() // merge_length) + # Use .item()
                        '<|vision_end|>',
                        1,
                    )
                    index += 1
                if '<image>' in processed_prompt: # Check if there were leftover placeholders
                     print("Warning: More <image> tags found than images provided or processed.")
                # Replace placeholders with the actual image token
                prompt_with_chat_template = processed_prompt.replace('<|placeholder|>', self.processor.image_token)
            elif '<image>' in prompt_with_chat_template:
                 print("Warning: <image> tag found in prompt but no valid image data (image_grid_thw) available.")
                 # Decide how to handle this: remove tag, raise error, etc.
                 # Option: Remove the tag if no image is present
                 # prompt_with_chat_template = prompt_with_chat_template.replace('<image>', '')


        else: # Not multi-modal
            raw_prompt = prompt_with_chat_template

        # Tokenize the finalized prompt string
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template, # Pass the guaranteed string
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            # Check if get_rope_index is available and processor exists
            if hasattr(self.processor, 'image_processor') and 'qwen2_vl' in str(type(self.processor)).lower():
                try:
                    from verl.models.transformers.qwen2_vl import get_rope_index
                    position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=image_grid_thw,
                        attention_mask=attention_mask[0],
                    )
                except ImportError:
                     print("Warning: qwen2_vl specific function not found. Using default position_ids.")
                     position_ids = compute_position_id_with_mask(attention_mask)
                except Exception as e:
                     print(f"Error computing rope_index: {e}. Using default position_ids.")
                     position_ids = compute_position_id_with_mask(attention_mask)
            else:
                 print("Warning: Multimodal detected but processor doesn't seem to support get_rope_index. Using default position_ids.")
                 position_ids = compute_position_id_with_mask(attention_mask)


        else: # Not multi-modal
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        # Ensure position_ids has the expected shape (it might be (3, seq_len) or (1, seq_len))
        row_dict['position_ids'] = position_ids[0] if position_ids.ndim > 1 else position_ids

        # Encode raw_prompt safely
        try:
            row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        except Exception as e:
            print(f"Error encoding raw_prompt: {e}. raw_prompt: {raw_prompt[:100]}...")
            row_dict['raw_prompt_ids'] = [] # Assign empty list on error


        # encode prompts without chat template
        if self.return_raw_chat:
            # Ensure 'chat' is serializable (e.g., convert numpy arrays if they exist)
            if isinstance(chat, np.ndarray):
                 row_dict['raw_prompt'] = chat.tolist()
            elif isinstance(chat, list):
                 # Ensure elements within the list are basic types if necessary
                 row_dict['raw_prompt'] = [str(item) if not isinstance(item, (str, int, float, dict, list)) else item for item in chat]
            else:
                 row_dict['raw_prompt'] = chat # Assume other types are directly usable


        # add index for each prompt
        # Use .get() with default values for safer access
        index = row_dict.get("extra_info", {}).get("index", item) # Use item as fallback index
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
