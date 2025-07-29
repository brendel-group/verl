
import pandas as pd
import os
import subprocess
import argparse
import pathlib
from datetime import datetime
from checkpoint_utils import handle_checkpoint_validation, print_checkpoint_config, get_checkpoint_gpu_count

MAX_N_CHUNKS = 10

def main(path: str, chunksize: int, rollouts : int, model : str, checkpoint: str, identifier: str = None, allocate_nodes: bool = False):
    # Convert paths to absolute and check validity
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Dataset file not found: {abs_path}")
    df = pd.read_parquet(abs_path)
    n_rows = len(df)
    n_chunks = (n_rows + chunksize - 1) // chunksize
    dataset_name = pathlib.Path(abs_path).parent.name
    filename = pathlib.Path(abs_path).name
    joined_name = f"{dataset_name}/{filename}"

    # Determine target GPUs and entrypoint script based on allocate_nodes flag
    if allocate_nodes and checkpoint:
        # Infer GPU count from checkpoint and use 2-node setup
        target_gpus = get_checkpoint_gpu_count(checkpoint)
        entrypoint_script = "verl/workspace/rollouts/rollouts_entrypoint_2node.sh"
        print(f"🔧 Auto-allocating nodes: detected {target_gpus} GPUs from checkpoint, using 2-node setup")
    else:
        # Use default 4 GPU setup with 1 node
        target_gpus = 4
        entrypoint_script = "verl/workspace/rollouts/rollouts_entrypoint_1node.sh"
        if allocate_nodes:
            print("⚠️  --allocate-nodes specified but no checkpoint provided, using default 4-GPU 1-node setup")

    # Validate checkpoint sharding matches target configuration
    abs_checkpoint = handle_checkpoint_validation(checkpoint, target_gpus) if checkpoint else None

    assert n_chunks <= MAX_N_CHUNKS, f"n_chunks were larger than MAX_N_CHUNKS: {MAX_N_CHUNKS} < n_chunks: {n_chunks}. Adjust limit or chunksize to not overwhelm scheduler."

    print(f"\n📊 Rollout Configuration:")
    print(f"Dataset name: {dataset_name}")
    print(f"Total rows: {n_rows}")
    print(f"Chunk size: {chunksize}")
    print(f"Rollouts per question: {rollouts}")
    print(f"Target GPUs: {target_gpus}")
    print(f"Entrypoint script: {entrypoint_script}")
    print_checkpoint_config(model, abs_checkpoint)
    if identifier:
        print(f"Identifier: {identifier}")
    print(f"Number of chunks (array jobs): {n_chunks}")
    print()

    # Generate timestamp for consistent naming across all chunks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Using timestamp: {timestamp}")

    # Launch sbatch calls asynchronously, print errors if any.
    for slurm_idx in range(n_chunks):
        export_vars = [
            f"DATASET_NAME={dataset_name}",
            f"DATASET_PATH={joined_name}",
            f"MODEL_PATH={model}",
            f"CHECKPOINT={abs_checkpoint if abs_checkpoint else ''}",
            f"CHUNK_SIZE={chunksize}",
            f"ROLLOUTS={rollouts}",
            f"SLURM_IDX={slurm_idx}",
            f"TIMESTAMP={timestamp}",
            f"IDENTIFIER={identifier if identifier else ''}"
        ]
        _ = subprocess.Popen([
            "sbatch",
            f"--export={','.join(export_vars)}",
            os.path.abspath(entrypoint_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f'Launched: {slurm_idx}')

        
if __name__ == "__main__":
    """
    Sample usage:
        # Basic usage with HuggingFace model (1 node, 4 GPUs)
        python run_rollouts.py --file data/math/test.parquet --chunksize 1024 --model Qwen/Qwen2.5-7B --identifier baseline
        
        # Using a checkpoint (must match 4 GPU sharding for rollouts)
        python run_rollouts.py --file data/math/test.parquet --chunksize 1024 \
               --checkpoint /path/to/global_step_280/actor \
               --model Qwen/Qwen2.5-7B --identifier entropy0.01
        
        # Auto-allocate nodes based on checkpoint GPU count (uses 2-node setup)
        python run_rollouts.py --file data/math/test.parquet --chunksize 1024 \
               --checkpoint /path/to/global_step_280/actor \
               --model Qwen/Qwen2.5-7B --identifier entropy0.01 --allocate-nodes
    """
    parser = argparse.ArgumentParser(description="Proxy for multiple SLURM job submissions based on dataset size.")
    parser.add_argument("--file", type=str, required=True, help="Path to the dataset parquet file.")
    parser.add_argument("--chunksize", type=int, default=1024, help="Chunk size for splitting the dataset.")
    parser.add_argument("--rollouts", type=int, default=1024, help="Number of rollouts to run for each question.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model Identifier. Used to instantiate the model. NOTE: In case you specify a checkpoint, Model families have to match.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to ../actor directory of an actor checkpoint to generate rollouts from. Must be sharded for 4 GPUs to match rollout configuration.")
    parser.add_argument("--identifier", type=str, default=None, help="Optional string identifier to help identify the type of rollout (e.g., 'entropy0.01', 'baseline', etc.). Required when using --checkpoint.")

    parser.add_argument("--allocate-nodes", action="store_true", default=False, help="Automatically allocate appropriate number of nodes based on checkpoint GPU count. Uses 2-node setup for multi-GPU checkpoints.")

    args = parser.parse_args()

    # Validate that if checkpoint is provided, identifier must also be provided
    if args.checkpoint and not args.identifier:
        parser.error("--identifier is required when --checkpoint is specified to avoid confusion.")

    os.chdir(os.getenv("HOME"))
    main(path=args.file, chunksize=args.chunksize, rollouts=args.rollouts, model = args.model, checkpoint=args.checkpoint, identifier=args.identifier, allocate_nodes=args.allocate_nodes)
