import os
import numpy
import random
import torch
import json
import pandas as pd

# compute pass@k


def compute_pass_at_k(advantages):
    """
    Compute the pass@k for a given list of advantages.
    """
    pass_at_k = 0
    pass_at_1 = 0
    total_groups = 0
    for sample_id in advantages['samples']:
        group = advantages['samples'][sample_id]
        total_groups += 1

        # compute pass@k for each group
        for sample in group:
            if sample['score'] == 1.0:
                pass_at_k += 1
                break
        
        # compute pass@1 for each group
        random.shuffle(group)
        sample = group[0]
        if sample['score'] == 1.0:
            pass_at_1 += 1

    return pass_at_1 / total_groups, pass_at_k / total_groups, total_groups


def load_advantages(checkpoints, split='train', stage='pre_training', epoch=None):
    """
    Load advantages for all checkpoints for a specific data split and training stage.
    
    Args:
        split (str): Data split - 'train' or 'val'
        stage (str): Training stage - 'pre_training' or specific epoch
        epoch (int, optional): If loading advantages for a specific epoch
    
    Returns:
        dict: Dictionary mapping checkpoint names to their advantage tensors
    """
    advantages = {}
    
    # Construct filename based on parameters
    if epoch is not None:
        filename = f'advantages_{split}_step_{epoch}.pt'
    else:
        filename = f'advantages_{split}_{stage}.pt'
    
    for name, checkpoint_dir in checkpoints.items():
        advantage_path = os.path.join(checkpoint_dir, 'advantage_tracking', filename)
        try:
            # Use weights_only=True to address the FutureWarning
            advantages[name] = torch.load(advantage_path)
            print(f"Loaded {split} advantages for {name}")
        except Exception as e:
            print(f"Error loading advantages for {name}: {e}")
    
    return advantages

# load the tokenizer
import warnings

__all__ = ['hf_tokenizer', 'hf_processor']


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def process_training_results(train_results):
    """
    Process training results to extract per-step, per-sample statistics.
    
    Returns a DataFrame with columns:
    - step: training step
    - sample_id: unique identifier for each sample
    - pass@1mean@32: average pass@1 across 32 trials
    - pass@32: whether any of the 32 trials passed
    - correct_resp_length: average length of correct responses (-1 if none)
    - incorrect_resp_length: average length of incorrect responses (-1 if none)
    """
    import pandas as pd
    import numpy as np
    
    rows = []
    
    for step in sorted(train_results.keys()):
        step_data = train_results[step]
        
        for sample_id, data in step_data.items():
            scores = data.get('scores', [])
            lengths = data.get('response_lengths', [])
            
            # Skip if no data
            if not scores or not lengths:
                continue
                
            # Calculate metrics
            pass_at_1_mean = np.mean(scores) if scores else 0
            pass_at_32 = 1.0 if any(score == 1.0 for score in scores) else 0.0
            
            # Get correct and incorrect lengths
            correct_lengths = [lengths[i] for i, score in enumerate(scores) if score == 1.0]
            incorrect_lengths = [lengths[i] for i, score in enumerate(scores) if score == 0.0]
            
            # Calculate average lengths (-1 if no data)
            correct_avg_length = np.mean(correct_lengths) if correct_lengths else -1
            incorrect_avg_length = np.mean(incorrect_lengths) if incorrect_lengths else -1
            
            # Add row to results
            rows.append({
                'step': step,
                'sample_id': sample_id,
                'pass@1mean@32': pass_at_1_mean,
                'pass@32': pass_at_32,
                'correct_resp_length': correct_avg_length,
                'incorrect_resp_length': incorrect_avg_length
            })
    
    # Create DataFrame and ensure step is treated as numeric for proper sorting
    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df
        
    results_df['step'] = pd.to_numeric(results_df['step'])
    return results_df.sort_values(['step', 'sample_id']).reset_index(drop=True)


import pandas as pd

def transform_r1_to_qwen_format(row):
    try:
        # Ensure 'prompt' is a list and has at least one element (dictionary)
        # if not isinstance(row.get('prompt'), list) or not row['prompt']:
        #     return row # Return row unchanged if prompt format is unexpected

        original_prompt_dict = row['prompt'][0]
        original_content = original_prompt_dict.get('content')
        #print(original_content)

        # Ensure content is a string
        if not isinstance(original_content, str):
            return row # Return row unchanged if content is not a string

        user_marker = " "
        assistant_marker = " "

        start_user_idx = original_content.find(user_marker)
        
        # Find the first occurrence of assistant_marker *after* user_marker
        start_assistant_idx = -1
        if start_user_idx != -1:
            # Search for assistant_marker starting after the user_marker
            start_assistant_idx = original_content.find(assistant_marker, start_user_idx + len(user_marker))

        if start_user_idx != -1 and start_assistant_idx != -1:
            # Both markers found in the correct order
            question_start_pos = start_user_idx + len(user_marker)
            question = original_content[question_start_pos:start_assistant_idx].strip()
            
            # Construct the new Qwen-style prompt content
            new_content = f"system\nYou are a helpful assistant.\nuser\n{question}\nassistant\n"
            
            # Update the prompt in the row. The prompt is a list containing one dictionary.
            # The role should be 'user' as per Qwen format examples.
            row['prompt'] = [{'content': new_content, 'role': 'user'}]
        else:
            # Markers not found in the expected order or original_content was not a string.
            # In this case, we return the row unchanged to avoid processing errors
            # or incorrectly formatting data that doesn't match the R1 style.
            # If logging is set up, a warning could be logged here.
            pass

    except Exception as e:
        print(f"Error transforming row: {e}")
        # In case of any other unexpected error during processing,
        # return the row unchanged to be safe.
        # If logging is set up: import logging; logging.exception(f"Error transforming row: {e}")
        pass
        
    return row

def _get_nested_key(data_dict, keys, default=None):
    """Safely retrieve a nested key from a dictionary."""
    temp_dict = data_dict
    for key in keys:
        if isinstance(temp_dict, dict) and key in temp_dict:
            temp_dict = temp_dict[key]
        else:
            return default
    return temp_dict

def _extract_training_data_name(file_path_str):
    """Extracts the filename without extension, e.g., /path/to/train_data.parquet -> train_data."""
    if not file_path_str or not isinstance(file_path_str, str):
        return None
    base_name = os.path.basename(file_path_str)
    name_part = base_name.split('.')[0]
    return name_part

def collate_experiment_results(base_results_dir):
    """
    Collates experiment results from subdirectories of base_results_dir.
    Reads eval.jsonl for metrics and config.json for parameters.

    Args:
        base_results_dir (str): The path to the base directory containing experiment folders.

    Returns:
        pandas.DataFrame: A DataFrame with all collated results.
    """
    all_results_list = []

    if not os.path.isdir(base_results_dir):
        print(f"Error: Base directory '{base_results_dir}' not found.")
        return pd.DataFrame()

    for exp_name in os.listdir(base_results_dir):
        exp_dir = os.path.join(base_results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        # Initialize config parameters
        config_model = None
        config_lr = None
        config_training_data = None
        config_epochs = None
        config_total_training_steps = None
        config_train_batch_size = None
        config_micro_batch_size = None
        
        raw_config = {}
        config_path = os.path.join(exp_dir, 'config.json')

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    raw_config = json.load(f)
                
                is_sft_config = False
                is_rl_config = False

                # Determine config type based on characteristic keys
                if _get_nested_key(raw_config, ['model', 'partial_pretrain']) is not None:
                    is_sft_config = True
                elif _get_nested_key(raw_config, ['actor_rollout_ref', 'model', 'path']) is not None:
                    is_rl_config = True

                if is_sft_config:
                    config_model = _get_nested_key(raw_config, ['model', 'partial_pretrain'])
                    config_lr = _get_nested_key(raw_config, ['optim', 'lr'])
                    train_files = _get_nested_key(raw_config, ['data', 'train_files'])
                    config_training_data = _extract_training_data_name(train_files)
                    config_epochs = _get_nested_key(raw_config, ['trainer', 'total_epochs'])
                    config_total_training_steps = _get_nested_key(raw_config, ['trainer', 'total_training_steps'])
                    config_train_batch_size = _get_nested_key(raw_config, ['data', 'train_batch_size'])
                    config_micro_batch_size = _get_nested_key(raw_config, ['data', 'micro_batch_size'])
                elif is_rl_config:
                    config_model = _get_nested_key(raw_config, ['actor_rollout_ref', 'model', 'path'])
                    config_lr = _get_nested_key(raw_config, ['actor_rollout_ref', 'actor', 'optim', 'lr'])
                    train_files = _get_nested_key(raw_config, ['data', 'train_files'])
                    config_training_data = _extract_training_data_name(train_files)
                    config_epochs = _get_nested_key(raw_config, ['trainer', 'total_epochs'])
                    config_total_training_steps = _get_nested_key(raw_config, ['actor_rollout_ref', 'actor', 'optim', 'total_training_steps'])
                    config_train_batch_size = _get_nested_key(raw_config, ['data', 'train_batch_size'])
                    config_micro_batch_size = _get_nested_key(raw_config, ['actor_rollout_ref', 'actor', 'ppo_mini_batch_size'])
                else:
                    if raw_config: # Config loaded but not recognized
                        print(f"Warning: Unrecognized config.json structure in {exp_dir}. Config parameters may be incomplete.")
            
            except json.JSONDecodeError:
                print(f"Warning: Could not parse config.json in {exp_dir}.")
            except Exception as e:
                print(f"Warning: Error processing config.json in {exp_dir}: {e}")
        
        eval_data_points = []
        eval_jsonl_path = os.path.join(exp_dir, 'eval.jsonl')
        if os.path.exists(eval_jsonl_path):
            try:
                with open(eval_jsonl_path, 'r') as f:
                    for line_number, line in enumerate(f, 1):
                        try:
                            record = json.loads(line)
                            eval_data_points.append(record)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping malformed JSON line {line_number} in {eval_jsonl_path}")
            except Exception as e:
                print(f"Warning: Could not read or process {eval_jsonl_path}: {e}")

        # Determine training_type based on user's rules
        num_eval_steps = len(eval_data_points)
        training_type = 'Base' 
        if 'sft' in exp_name.lower():
            training_type = 'SFT'
        elif num_eval_steps > 3:
            training_type = 'RL'

        if not eval_data_points:
            if raw_config : # If config exists but no eval data, create one row with config info
                row_data = {
                    'step': None,
                    'exp_name': exp_name,
                    'training_type': training_type,
                    'model': config_model,
                    'lr': config_lr,
                    'training_data': config_training_data,
                    'epochs': config_epochs,
                    'total_training_steps': config_total_training_steps,
                    'train_batch_size': config_train_batch_size,
                    'micro_batch_size': config_micro_batch_size,
                }
                all_results_list.append(row_data)
            else:
                 print(f"Info: Skipping {exp_name} as it has no eval.jsonl and no parsable config.json for basic info.")
            continue # Move to the next experiment folder

        for record in eval_data_points:
            step_data = {'step': record.get('step')}
            for key, value in record.items():
                if key != 'step' and ('/mean' in key or key.endswith('/mean@8')):
                    step_data[key] = value
            
            step_data.update({
                'exp_name': exp_name,
                'training_type': training_type,
                'model': config_model,
                'lr': config_lr,
                'training_data': config_training_data,
                'epochs': config_epochs,
                'total_training_steps': config_total_training_steps,
                'train_batch_size': config_train_batch_size,
                'micro_batch_size': config_micro_batch_size,
            })
            all_results_list.append(step_data)

    if not all_results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results_list)
    
    # Define preferred column order
    id_cols = [
        'exp_name', 'training_type', 'step', 'model', 'lr', 
        'training_data', 'epochs', 'total_training_steps', 
        'train_batch_size', 'micro_batch_size'
    ]
    
    present_id_cols = [col for col in id_cols if col in results_df.columns]
    metric_cols = sorted([col for col in results_df.columns if col not in present_id_cols and ('/' in col or '@' in col)])
    other_cols = sorted([col for col in results_df.columns if col not in present_id_cols and col not in metric_cols])

    final_column_order = present_id_cols + metric_cols + other_cols
    results_df = results_df[final_column_order]

    return results_df