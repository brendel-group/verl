#!/usr/bin/env python3
"""
Generic script to grade rollouts for any dataset.

This script:
1. Discovers available rollout datasets in the specified directory
2. Maps dataset names to their corresponding data sources for reward computation
3. Stitches together chunked rollout files
4. Grades the rollouts using the appropriate reward function
5. Saves the graded results

Usage:
    python grade_generic.py --rollouts_dir /u/rfechner/out/rollouts/Qwen --output_dir ./graded_rollouts
    python grade_generic.py --dataset aime24 --rollouts_dir /u/rfechner/out/rollouts/Qwen
    python grade_generic.py --list-datasets --rollouts_dir /u/rfechner/out/rollouts/Qwen
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from verl.utils.reward_score import _default_compute_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset name to data_source mapping based on verl/utils/reward_score/__init__.py
DATASET_MAPPINGS = {
    # Math-related datasets using math_verify
    'math': 'DigitalLearningGmbH/MATH-lighteval',
    'math500': 'HuggingFaceH4/MATH-500', 
    'aime24': 'math-ai/aime24',
    'aime25': 'math-ai/aime25',
    'olympiadbench': 'math-ai/olympiadbench', 
    'minervamath': 'math-ai/minervamath',
    'gpqa': 'math-ai/gpqa',
    'amc23': 'math-ai/amc23',
    
    # GSM8K dataset
    'gsm8k': 'openai/gsm8k',
    
    # Other potential mappings (add as needed)
    'geometry3k': 'hiyouga/geometry3k',
}

def find_parquet_parts(directory: str) -> List[str]:
    """Find all parquet-partX files and return list of filepaths."""
    parquet_files = []
    pattern = re.compile(r".*\.parquet-part\d+$")

    for root, _, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                full_path = os.path.join(root, file)
                parquet_files.append(full_path)
    return sorted(parquet_files)

def stitch_rollout_chunks(rollout_dir: str, output_file: str) -> bool:
    """Stitch together chunked rollout files into a single parquet file."""
    logger.info(f"Stitching rollout chunks from {rollout_dir}")
    
    files = find_parquet_parts(rollout_dir)
    if not files:
        logger.warning(f"No parquet-partX files found in {rollout_dir}")
        return False

    dataframes = []
    for filepath in files:
        logger.info(f"Reading: {filepath}")
        df = pd.read_parquet(filepath)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_parquet(output_file)
    logger.info(f"Stitched {len(files)} files into {output_file} with {len(combined_df)} rows")
    return True

def get_dataset_name_from_path(rollout_path: str) -> str:
    """Extract dataset name from rollout directory path."""
    # Extract from patterns like "Qwen2.5-7B_aime24_nrollouts_1024_20250718_203841"
    dir_name = os.path.basename(rollout_path.rstrip('/'))
    parts = dir_name.split('_')
    
    # Look for dataset name (usually the second part after model name)
    if len(parts) >= 2:
        return parts[1]
    else:
        logger.warning(f"Could not extract dataset name from {dir_name}")
        return dir_name

def discover_rollout_datasets(rollouts_dir: str) -> List[Tuple[str, str]]:
    """Discover all available rollout datasets."""
    datasets = []
    
    if not os.path.exists(rollouts_dir):
        logger.error(f"Rollouts directory does not exist: {rollouts_dir}")
        return datasets
    
    for item in os.listdir(rollouts_dir):
        item_path = os.path.join(rollouts_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item != 'old':
            dataset_name = get_dataset_name_from_path(item)
            datasets.append((dataset_name, item_path))
    
    return sorted(datasets)

def grade_rollouts(stitched_file: str, dataset_name: str, output_file: str) -> bool:
    """Grade rollouts using the appropriate reward function."""
    logger.info(f"Grading rollouts for dataset: {dataset_name}")
    
    # Get data source for this dataset
    data_source = DATASET_MAPPINGS.get(dataset_name)
    if not data_source:
        logger.error(f"No data source mapping found for dataset: {dataset_name}")
        logger.error(f"Available mappings: {list(DATASET_MAPPINGS.keys())}")
        return False
    
    logger.info(f"Using data source: {data_source}")
    
    # Load the stitched rollout data
    try:
        dataset = pd.read_parquet(stitched_file)
        logger.info(f"Loaded {len(dataset)} rows from {stitched_file}")
    except Exception as e:
        logger.error(f"Failed to load stitched file {stitched_file}: {e}")
        return False
    
    # Check required columns
    required_columns = ['reward_model', 'responses']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.error(f"Available columns: {dataset.columns.tolist()}")
        return False
    
    # Extract ground truths
    logger.info("Extracting ground truths from reward_model column")
    try:
        ground_truths = dataset['reward_model'].apply(lambda d: d['ground_truth'])
        n_responses = len(dataset['responses'].iloc[0]) if len(dataset) > 0 else 0
        logger.info(f"Found {n_responses} responses per question")
        
        # Expand ground truths to match number of responses
        ground_truths = ground_truths.apply(lambda entry: [entry] * n_responses)
    except Exception as e:
        logger.error(f"Failed to extract ground truths: {e}")
        return False
    
    # Create reward function
    reward_fn = lambda solution_str, ground_truth: _default_compute_score(
        data_source, solution_str=solution_str, ground_truth=ground_truth
    )
    
    # Compute rewards
    logger.info("Computing rewards...")
    try:
        rewards = []
        for i, (responses, gts) in enumerate(zip(dataset['responses'], ground_truths)):
            if i % 100 == 0:
                logger.info(f"Processing row {i}/{len(dataset)}")
            
            row_rewards = np.array([reward_fn(response, gt) for response, gt in zip(responses, gts)])
            rewards.append(row_rewards)
        
        dataset['rewards'] = rewards
        logger.info(f"Computed rewards for {len(rewards)} questions")
        
    except Exception as e:
        logger.error(f"Failed to compute rewards: {e}")
        return False
    
    # Save graded results
    try:
        dataset.to_parquet(output_file)
        logger.info(f"Saved graded results to {output_file}")
        
        # Print some statistics
        mean_rewards = [np.mean(r) for r in rewards]
        logger.info(f"Reward statistics - Mean: {np.mean(mean_rewards):.4f}, Std: {np.std(mean_rewards):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to save graded results: {e}")
        return False
    
    return True

def process_dataset(dataset_name: str, rollout_path: str, output_dir: str) -> bool:
    """Process a single dataset: stitch chunks and grade rollouts."""
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Files for this dataset
    stitched_file = os.path.join(output_dir, f"{dataset_name}_stitched.parquet")
    graded_file = os.path.join(output_dir, f"{dataset_name}_graded.parquet")
    
    # Skip if already graded
    if os.path.exists(graded_file):
        logger.info(f"Graded file already exists: {graded_file}")
        return True
    
    # Stitch chunks if needed
    if not os.path.exists(stitched_file):
        if not stitch_rollout_chunks(rollout_path, stitched_file):
            return False
    else:
        logger.info(f"Stitched file already exists: {stitched_file}")
    
    # Grade rollouts
    return grade_rollouts(stitched_file, dataset_name, graded_file)

def main():
    parser = argparse.ArgumentParser(description="Grade rollouts for datasets")
    parser.add_argument("--rollouts_dir", type=str, required=True,
                       help="Directory containing rollout datasets")
    parser.add_argument("--output_dir", type=str, default="./graded_rollouts",
                       help="Output directory for graded results")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Process only this specific dataset")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    
    args = parser.parse_args()
    
    # Discover available datasets
    datasets = discover_rollout_datasets(args.rollouts_dir)
    
    if args.list_datasets:
        print("Available datasets:")
        for dataset_name, path in datasets:
            mapped_source = DATASET_MAPPINGS.get(dataset_name, "NO MAPPING")
            print(f"  {dataset_name:15} -> {mapped_source}")
        return
    
    if not datasets:
        logger.error("No datasets found")
        return
    
    # Filter to specific dataset if requested
    if args.dataset:
        datasets = [(name, path) for name, path in datasets if name == args.dataset]
        if not datasets:
            logger.error(f"Dataset '{args.dataset}' not found")
            return
    
    # Process datasets
    success_count = 0
    for dataset_name, rollout_path in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"Path: {rollout_path}")
        
        if process_dataset(dataset_name, rollout_path, args.output_dir):
            success_count += 1
            logger.info(f"✓ Successfully processed {dataset_name}")
        else:
            logger.error(f"✗ Failed to process {dataset_name}")
    
    logger.info(f"\nCompleted: {success_count}/{len(datasets)} datasets processed successfully")

if __name__ == "__main__":
    main()
