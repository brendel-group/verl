import numpy as np
import os
import torch
from utils import compute_pass_at_k

# Define checkpoint directories
base_dir = '/fast/pmayilvahanan/post_training/verl_checkpoints'

# Model checkpoints with different training steps
checkpoints = {
    'qwen2-7b_step_1': os.path.join(base_dir, 'qqwen2-7b_sft_gsm8k_epochs_1_lr_1e-5_rl_step_1'),
    'qwen2-7b_step_3': os.path.join(base_dir, 'qqwen2-7b_sft_gsm8k_epochs_1_lr_1e-5_rl_step_3'),
    'qwen2-7b_step_6': os.path.join(base_dir, 'qwen2-7b_function_rm_sft_init_gsm8k_steps_6'),
    'qwen2-7b_step_12': os.path.join(base_dir, 'qwen2-7b_function_rm_sft_init_gsm8k_steps_12'),
    'qwen2-7b_step_24': os.path.join(base_dir, 'qwen2-7b_function_rm_sft_init_gsm8k_steps_24'),
    'qwen2-7b_step_30': os.path.join(base_dir, 'qwen2-7b_function_rm_sft_init_gsm8k_steps_30'),
    'qwen2-1.5b_step_12': os.path.join(base_dir, 'qwen2-1.5b_sft_gsm8k_epochs_5_lr_1e-5_rl_step_12')
}

def load_advantages(split='train', stage='pre_training', epoch=None):
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
            advantages[name] = torch.load(advantage_path, weights_only=True)
            print(f"Loaded {split} advantages for {name}")
        except FileNotFoundError:
            print(f"Warning: Could not find advantage file for {name} at {advantage_path}")
        except Exception as e:
            print(f"Error loading advantages for {name}: {e}")
    
    return advantages

def analyze_advantages(advantages_dict):
    """
    Analyze advantages by computing pass@k metrics for each checkpoint.
    
    Args:
        advantages_dict (dict): Dictionary of advantages from load_advantages()
    
    Returns:
        dict: Dictionary of pass@k metrics for each checkpoint
    """
    results = {}
    for name, advantage in advantages_dict.items():
        try:
            results[name] = compute_pass_at_k(advantage)
            print(f"{name}: {results[name]}")
        except Exception as e:
            print(f"Error computing pass@k for {name}: {e}")
    
    return results

def compare_train_val_advantages():
    """
    Load and compare advantages between training and validation sets.
    """
    train_advantages = load_advantages(split='train', stage='pre_training')
    val_advantages = load_advantages(split='val', stage='pre_training')
    
    print("\nAnalyzing training advantages:")
    train_results = analyze_advantages(train_advantages)
    
    print("\nAnalyzing validation advantages:")
    val_results = analyze_advantages(val_advantages)
    
    return train_results, val_results

def compare_epoch_advantages(epochs=[1, 2, 3]):
    """
    Compare advantages across different epochs.
    
    Args:
        epochs (list): List of epoch numbers to analyze
    """
    epoch_results = {}
    
    for epoch in epochs:
        print(f"\nLoading advantages for epoch {epoch}:")
        advantages = load_advantages(split='train', epoch=epoch)
        
        print(f"\nAnalyzing epoch {epoch} advantages:")
        results = analyze_advantages(advantages)
        epoch_results[epoch] = results
    
    return epoch_results

# Example usage
if __name__ == "__main__":
    # Compare train and validation advantages
    train_results, val_results = compare_train_val_advantages()
    
    # Uncomment to compare advantages across epochs
    # epoch_results = compare_epoch_advantages(epochs=[1, 2, 3]) 