import os
import subprocess
import argparse

def main(model: str, entropy_coef: float, kl_loss_coef: float, project_name: str, nodes: int):
    """
    Launch a SLURM training job with the specified parameters.
    """
    # Set default paths
    train_files = "data/math/train.parquet"
    
    # Convert paths to absolute and check validity
    abs_train_files = os.path.abspath(train_files)
    
    if not os.path.isfile(abs_train_files):
        raise FileNotFoundError(f"Training dataset file not found: {abs_train_files}")    
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
        f"PROJECT_NAME={project_name}"
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
        python run.py --model Qwen/Qwen2.5-7B \
                      --entropy-coef 0.001 \
                      --kl-loss-coef 0.001 \
                      --project-name my_training \
                      --nodes 4
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

    args = parser.parse_args()

    # Change to home directory for consistent paths
    os.chdir(os.getenv("HOME"))
    
    main(
        model=args.model,
        entropy_coef=args.entropy_coef,
        kl_loss_coef=args.kl_loss_coef,
        project_name=args.project_name,
        nodes=args.nodes
    )
