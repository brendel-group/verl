"""
Simple checkpoint validator for VERL rollouts.
Validates that checkpoint sharding matches the target rollout configuration.
"""

import os
from typing import Dict, Any


def detect_checkpoint_sharding(checkpoint_dir: str) -> int:
    """Detect how many GPUs the checkpoint was sharded across."""
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('model_world_size_') and filename.endswith('.pt'):
            parts = filename.split('_')
            if len(parts) >= 5:
                return int(parts[3])
    raise ValueError(f"Could not detect checkpoint sharding in {checkpoint_dir}")


def validate_checkpoint_path(checkpoint: str) -> str:
    """Validate and convert checkpoint path to absolute path."""
    if not checkpoint:
        return None
        
    abs_checkpoint = os.path.abspath(checkpoint)
    if not (os.path.basename(abs_checkpoint) == 'actor' and os.path.isdir(abs_checkpoint)):
        raise ValueError(
            "Please pass valid '.../actor' directory. This should be located within "
            "one of the global_step_X directories."
        )
    return abs_checkpoint


def get_checkpoint_gpu_count(checkpoint_dir: str) -> int:
    """Get the number of GPUs that a checkpoint was trained with."""
    if not checkpoint_dir:
        return 4  # Default to 4 GPUs if no checkpoint
    
    abs_checkpoint = validate_checkpoint_path(checkpoint_dir)
    return detect_checkpoint_sharding(abs_checkpoint)


def handle_checkpoint_validation(checkpoint_dir: str, target_gpus: int) -> str:
    """
    Validate checkpoint sharding matches target configuration.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        target_gpus: Number of target GPUs for rollouts
        
    Returns:
        str: Validated absolute checkpoint path
        
    Raises:
        ValueError: If checkpoint validation fails or sharding mismatch detected
    """
    if not checkpoint_dir:
        return None
    
    # Validate checkpoint path
    abs_checkpoint = validate_checkpoint_path(checkpoint_dir)
    
    # Get checkpoint world size
    checkpoint_gpus = detect_checkpoint_sharding(abs_checkpoint)
    
    # Validate sharding matches target - throw error if mismatch
    if checkpoint_gpus != target_gpus:
        raise ValueError(
            f"Checkpoint world size mismatch: checkpoint was saved with {checkpoint_gpus} GPUs "
            f"but target rollout configuration requires {target_gpus} GPUs. "
            f"Please retrain the model with the correct world size ({target_gpus} GPUs) "
            f"or adjust your rollout configuration to match the checkpoint ({checkpoint_gpus} GPUs)."
        )
    
    print(f"✅ Checkpoint sharding validated ({checkpoint_gpus} GPUs matches target)")
    
    return abs_checkpoint


def get_checkpoint_info(checkpoint_dir: str) -> dict:
    """Get detailed information about a checkpoint."""
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return {"exists": False}
    
    try:
        gpus = detect_checkpoint_sharding(checkpoint_dir)
        return {
            "exists": True,
            "path": os.path.abspath(checkpoint_dir),
            "gpus": gpus,
            "is_valid_actor": os.path.basename(checkpoint_dir) == 'actor' and os.path.isdir(checkpoint_dir)
        }
    except Exception as e:
        return {
            "exists": True,
            "path": os.path.abspath(checkpoint_dir),
            "error": str(e),
            "is_valid_actor": False
        }


def print_checkpoint_config(model: str, checkpoint_path: str = None):
    """Print checkpoint configuration information for debugging."""
    print(f"Model: {model}")
    if checkpoint_path:
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        if checkpoint_info["exists"]:
            if checkpoint_info.get("error"):
                print(f"Checkpoint: {checkpoint_path} (ERROR: {checkpoint_info['error']})")
            else:
                print(f"Checkpoint: {checkpoint_path} ({checkpoint_info['gpus']} GPUs)")
        else:
            print(f"Checkpoint: {checkpoint_path} (NOT FOUND)")
    print()
