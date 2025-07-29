import os
import re
import pandas as pd
import argparse
import logging

# Set up logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def find_parquet_parts(directory):
    """Find all parquet-partX files and return list of (filepath, X) tuples."""
    parquet_files = []
    pattern = re.compile(r".*\.parquet-part\d+$")

    for root, _, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                full_path = os.path.join(root, file)
                parquet_files.append(full_path)
    return parquet_files

def concatenate_parquet_parts(input_dir, output_file="combined.parquet"):
    files = find_parquet_parts(input_dir)
    if not files:
        logger.warning("No matching parquet-partX files found.")
        return

    dataframes = []
    for filepath in files:
        logger.info(f"Reading: {filepath}")
        df = pd.read_parquet(filepath)
        dataframes.append(df)

    combined_df = pd.concat(dataframes)
    combined_df.to_parquet(output_file)
    logger.info(f"Concatenated {len(files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate parquet-partX files into a single parquet file.")
    parser.add_argument('--input', '-i', required=True, help='Input directory containing parquet-partX files')
    parser.add_argument('--output', '-o', default='./', help='Output directory or output .parquet file (default: ./)')
    args = parser.parse_args()

    # Check input directory exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory '{args.input}' does not exist.")
    if not os.path.isdir(args.input):
        raise NotADirectoryError(f"Input path '{args.input}' is not a directory.")

    # Determine output path
    output_path = args.output
    # Check if output_path has a file extension
    if os.path.splitext(output_path)[1]:
        # Treat as file path
        output_dir = os.path.dirname(output_path) or '.'
        if not os.path.exists(output_dir):
            logger.info(f"Output directory '{output_dir}' does not exist. Creating it.")
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Output is a directory, ensure it exists
        if not os.path.exists(output_path):
            logger.info(f"Output directory '{output_path}' does not exist. Creating it.")
            os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "combined.parquet")

    # Check if output_path already exists and is a file
    if os.path.exists(output_path):
        if os.path.isfile(output_path):
            logger.warning(f"Output file '{output_path}' already exists and will be overwritten.")
        else:
            raise FileExistsError(f"Output path '{output_path}' exists and is not a file.")

    concatenate_parquet_parts(args.input, output_file=output_path)
    
