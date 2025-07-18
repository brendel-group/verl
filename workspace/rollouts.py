
import pandas as pd
import os
import subprocess
import argparse
import pathlib

MAX_N_CHUNKS = 10
valid_datasets = {'math', 'gsm8k'}

def main(path: str, chunksize: int, model : str, checkpoint: str):
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

    assert dataset_name in valid_datasets, f"Dataset {dataset_name} couldn't be found in valid datasets, perhaps invalid or not supported."
    abs_checkpoint = None
    if checkpoint:
        abs_checkpoint = os.path.abspath(checkpoint)
        if not (pathlib.Path(abs_checkpoint).name == 'agent' and os.path.isdir(abs_checkpoint)):
            raise ValueError("Please pass valid '.../agent' directory. This should be located within one of the global_step_X directories.")

    assert n_chunks <= MAX_N_CHUNKS, f"n_chunks were larger than MAX_N_CHUNKS: {MAX_N_CHUNKS} < n_chunks: {n_chunks}. Adjust limit or chunksize to not overwhelm scheduler."

    print(f"Dataset name: {dataset_name}")
    print(f"Total rows: {n_rows}")
    print(f"Chunk size: {chunksize}")
    print(f"Model: {model}")
    if abs_checkpoint:
        print(f"Loading checkpoint: {abs_checkpoint}")
    print(f"Number of chunks (array jobs): {n_chunks}")

    # Launch sbatch calls asynchronously, print errors if any.
    for slurm_idx in range(n_chunks):
        export_vars = [
            f"DATASET_NAME={dataset_name}",
            f"DATASET_PATH={joined_name}",
            f"MODEL_PATH={model}",
            f"CHECKPOINT={abs_checkpoint if abs_checkpoint else ''}",
            f"CHUNK_SIZE={chunksize}",
            f"SLURM_IDX={slurm_idx}"
        ]
        _ = subprocess.Popen([
            "sbatch",
            f"--export={','.join(export_vars)}",
            os.path.abspath("verl/workspace/job_array_rollouts.sbatch")
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f'Launched: {slurm_idx}')

        
if __name__ == "__main__":
    """
        Sample usage:
            python rollouts.py --file data/math/test.parquet --chunksize 1024 --model Qwen/Qwen2.5_7B
    """
    parser = argparse.ArgumentParser(description="Proxy for multiple SLURM job submissions based on dataset size.")
    parser.add_argument("--file", type=str, required=True, help="Path to the dataset parquet file.")
    parser.add_argument("--chunksize", type=int, required=True, help="Chunk size for splitting the dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5_7B", help="Model Identifier. Used to instantiate the model. NOTE: In case you specify a checkpoint, Model families have to match.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to ../actor directory of a actor checkpoint to generate the rollouts from.")
    args = parser.parse_args()

    os.chdir(os.getenv("HOME"))
    main(path=args.file, chunksize=args.chunksize, model = args.model, checkpoint=args.checkpoint)
