#!/usr/bin/env python3
"""
Meta-script to stitch all rollout experiments in a directory structure.

This script walks through a directory containing multiple experiments,
finds parquet files (including chunked ones) for each experiment,
and creates a clone directory structure with consolidated results.parquet files.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

# Import the concatenation logic from stitch.py
from stitch import concatenate_parquet_parts, find_parquet_parts

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_experiment_directories(root_dir):
    """
    Find all directories that contain parquet files (experiments).
    Returns a list of tuples: (relative_path, absolute_path)
    """
    experiment_dirs = []
    root_path = Path(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if this directory contains any parquet files
        parquet_files = find_parquet_parts(dirpath)
        if parquet_files:
            rel_path = Path(dirpath).relative_to(root_path)
            experiment_dirs.append((str(rel_path), dirpath))
            logger.info(f"Found experiment directory: {rel_path} ({len(parquet_files)} parquet files)")
    
    return experiment_dirs

def create_clone_structure(source_dir, output_dir):
    """
    Create the clone directory structure without copying files.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create the root output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Walk through source and create directory structure
    for dirpath, dirnames, filenames in os.walk(source_dir):
        rel_path = Path(dirpath).relative_to(source_path)
        target_dir = output_path / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)

def stitch_experiment(exp_rel_path, exp_abs_path, output_root, results_filename="results.parquet"):
    """
    Stitch all parquet files in an experiment directory into a single results.parquet file.
    """
    output_exp_dir = Path(output_root) / exp_rel_path
    output_file = output_exp_dir / results_filename
    
    # Check if output file already exists
    if output_file.exists():
        logger.warning(f"Output file {output_file} already exists. Skipping {exp_rel_path}")
        return False
    
    # Find parquet files in the experiment directory
    parquet_files = find_parquet_parts(exp_abs_path)
    if not parquet_files:
        logger.warning(f"No parquet files found in {exp_rel_path}")
        return False
    
    try:
        logger.info(f"Stitching {len(parquet_files)} files in {exp_rel_path} -> {output_file}")
        concatenate_parquet_parts(exp_abs_path, str(output_file))
        return True
    except Exception as e:
        logger.error(f"Failed to stitch {exp_rel_path}: {e}")
        return False

def copy_non_parquet_files(source_dir, output_dir, experiment_dirs):
    """
    Copy non-parquet files from source to output directory.
    Skip directories that are experiment directories (contain parquet files).
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Get set of experiment directory paths for quick lookup
    exp_dirs_set = {exp_path for _, exp_path in experiment_dirs}
    
    for dirpath, dirnames, filenames in os.walk(source_dir):
        # Skip if this is an experiment directory
        if dirpath in exp_dirs_set:
            continue
            
        rel_path = Path(dirpath).relative_to(source_path)
        target_dir = output_path / rel_path
        
        # Copy non-parquet files
        for filename in filenames:
            if not filename.endswith('.parquet'):
                source_file = Path(dirpath) / filename
                target_file = target_dir / filename
                try:
                    shutil.copy2(source_file, target_file)
                    logger.debug(f"Copied {source_file} -> {target_file}")
                except Exception as e:
                    logger.warning(f"Failed to copy {source_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Stitch all rollout experiments in a directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stitch all experiments and create clone structure
  python stitch_all.py -i /u/rfechner/out/fixed_rollouts/Qwen -o /u/rfechner/out/stitched_rollouts/Qwen
  
  # Use custom filename for results
  python stitch_all.py -i /path/to/experiments -o /path/to/output --results-file combined_results.parquet
  
  # Force overwrite existing files
  python stitch_all.py -i /path/to/experiments -o /path/to/output --force
        """
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True, 
        help='Input directory containing experiment subdirectories with parquet files'
    )
    parser.add_argument(
        '--output', '-o', 
        required=True, 
        help='Output directory where cloned structure with stitched results will be created'
    )
    parser.add_argument(
        '--results-file', 
        default='results.parquet',
        help='Name for the consolidated parquet file in each experiment directory (default: results.parquet)'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--copy-other-files', 
        action='store_true',
        help='Copy non-parquet files to the output directory structure'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not os.path.exists(args.input):
        logger.error(f"Input directory '{args.input}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(args.input):
        logger.error(f"Input path '{args.input}' is not a directory")
        sys.exit(1)
    
    # Find all experiment directories
    logger.info(f"Scanning for experiment directories in: {args.input}")
    experiment_dirs = find_experiment_directories(args.input)
    
    if not experiment_dirs:
        logger.warning("No experiment directories with parquet files found")
        sys.exit(0)
    
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    if args.dry_run:
        logger.info("DRY RUN - showing what would be done:")
        for exp_rel_path, exp_abs_path in experiment_dirs:
            parquet_files = find_parquet_parts(exp_abs_path)
            output_file = Path(args.output) / exp_rel_path / args.results_file
            logger.info(f"  {exp_rel_path}: {len(parquet_files)} files -> {output_file}")
        sys.exit(0)
    
    # Create clone directory structure
    logger.info(f"Creating clone directory structure at: {args.output}")
    create_clone_structure(args.input, args.output)
    
    # Process each experiment directory
    success_count = 0
    failed_count = 0
    
    for exp_rel_path, exp_abs_path in experiment_dirs:
        output_file = Path(args.output) / exp_rel_path / args.results_file
        
        # Check if we should skip existing files
        if output_file.exists() and not args.force:
            logger.info(f"Skipping {exp_rel_path} (output exists, use --force to overwrite)")
            continue
        
        success = stitch_experiment(exp_rel_path, exp_abs_path, args.output, args.results_file)
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Copy non-parquet files if requested
    if args.copy_other_files:
        logger.info("Copying non-parquet files...")
        copy_non_parquet_files(args.input, args.output, experiment_dirs)
    
    # Summary
    logger.info(f"Stitching complete: {success_count} successful, {failed_count} failed")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
