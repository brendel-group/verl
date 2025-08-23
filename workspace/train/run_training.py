import os
import subprocess
import argparse
from datetime import datetime

def main(model: str, entropy_coef: float, kl_loss_coef: float, project_name: str, nodes: int, identifier: str = None, seed: int = 42, 
         mask_positive_entropy_change: bool = False, mask_negative_entropy_change: bool = False):
    """
    Launch a SLURM training job with the specified parameters.
    """
    # Set default paths
    train_files = "data/math/train.parquet"
    
    # Convert paths to absolute and check validity
    abs_train_files = os.path.abspath(train_files)
    
    if not os.path.isfile(abs_train_files):
        raise FileNotFoundError(f"Training dataset file not found: {abs_train_files}")
    
    # Generate timestamp for consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Training files: {abs_train_files}")
    print(f"Model: {model}")
    print(f"Entropy coefficient: {entropy_coef}")
    print(f"KL loss coefficient: {kl_loss_coef}")
    print(f"Project name: {project_name}")
    print(f"Nodes: {nodes}")
    print(f"Timestamp: {timestamp}")
    print(f"Seed: {seed}")
    if identifier:
        print(f"Identifier: {identifier}")
    print("=" * 60)

    # Choose the appropriate entrypoint script based on number of nodes
    if nodes == 4:
        entrypoint_script = "verl/workspace/train/train_entrypoint_4nodes.sh"
    elif nodes == 2:
        entrypoint_script = "verl/workspace/train/train_entrypoint.sh"
    else:
        raise ValueError(f"Unsupported number of nodes: {nodes}. Only 2 and 4 nodes are supported.")

    export_vars = [
        f"TRAIN_FILES={abs_train_files}",
        f"MODEL_PATH={model}",
        f"ENTROPY_COEF={entropy_coef}",
        f"KL_LOSS_COEF={kl_loss_coef}",
        f"PROJECT_NAME={project_name}",
        f"TIMESTAMP={timestamp}",
        f"IDENTIFIER={identifier if identifier else ''}",
        f"DATA_SEED={seed}",
        f"MASK_POSITIVE_ENTROPY_CHANGE={'1' if mask_positive_entropy_change else '0'}",
        f"MASK_NEGATIVE_ENTROPY_CHANGE={'1' if mask_negative_entropy_change else '0'}"
    ]

    sbatch_cmd = [
        "sbatch",
        f"--export={','.join(export_vars)}",
        os.path.abspath(entrypoint_script)
    ]
    
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        print(f"Job submitted successfully!")
        print(f"SLURM output: {result.stdout.strip()}")
        if result.stderr:
            print(f"SLURM stderr: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

        
if __name__ == "__main__":
    """
    Sample usage:
        # Basic usage
        python run_training.py --model Qwen/Qwen2.5-7B \
                              --entropy-coef 0.001 \
                              --kl-loss-coef 0.001 \
                              --project-name my_training \
                              --nodes 4 \
                              --seed 42
        
        # With identifier for better tracking
        python run_training.py --model Qwen/Qwen2.5-7B \
                              --entropy-coef 0.01 \
                              --kl-loss-coef 0.001 \
                              --project-name entropy_experiments \
                              --identifier high_entropy \
                              --nodes 2 \
                              --seed 123
    """
    parser = argparse.ArgumentParser(description="Launch SLURM training job for VERL PPO training.")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                       help="Model identifier/path for training (default: Qwen/Qwen2.5-7B).")
    
    # Training hyperparameters
    parser.add_argument("--entropy-coef", type=float, default=0.001,
                       help="Entropy coefficient for training (default: 0.001).")
    parser.add_argument("--kl-loss-coef", type=float, default=0.001,
                       help="KL loss coefficient (default: 0.001).")
    
    # Project configuration
    parser.add_argument("--project-name", type=str, default="entropy_training_runs",
                       help="Project name for logging and checkpointing (default: verl_training).")
    
    # Compute configuration
    parser.add_argument("--nodes", type=int, default=2, choices=[2, 4],
                       help="Number of nodes to use for training (default: 2, choices: 2 or 4).")
    
    # Optional identifier for run tracking
    parser.add_argument("--identifier", type=str, default=None,
                       help="Optional string identifier to help identify the training run (e.g., 'baseline', 'high_entropy', etc.).")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for data shuffling and reproducibility (default: 42).")
    # Entropy change options
    parser.add_argument("--mask-pos", action="store_true", default=False,
                       help="Enable positive entropy change masking (default: False).")
    parser.add_argument("--mask-neg", action="store_true", default=False,
                       help="Enable negative entropy change masking (default: False).")
    args = parser.parse_args()

    # Change to home directory for consistent paths
    os.chdir(os.getenv("HOME"))
    print('WARNING CURRENTLY RUNNING IN DEV MODE -> ENTROPY DELTA ANALYSIS')
    main(
        model=args.model,
        entropy_coef=args.entropy_coef,
        kl_loss_coef=args.kl_loss_coef,
        project_name=args.project_name,
        nodes=args.nodes,
        identifier=args.identifier,
        seed=args.seed,
        mask_positive_entropy_change=args.mask_pos,
        mask_negative_entropy_change=args.mask_neg
    )
