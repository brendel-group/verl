"""
Data Processor for Self-Distilled Data

This script processes self-distilled data from reinforcement learning training and converts it
to a structured DataFrame, which is then saved as parquet files. The script is designed
to handle self-distilled data saved as PyTorch (.pt) files, extracting samples with scores
of 1.0 (positive) and optionally 0.0 (negative), and formatting them into a standardized 
structure suitable for further training.

The script takes the following command-line arguments:
- adv_path: Path to the advantage file (.pt)
- save_path: Base directory to save the parquet file
- data_source: Data source identifier for the DataFrame
- ability: Ability identifier for the DataFrame
- select_all: Optional flag to select all elements with score 1.0 instead of just one
- test_split: Percentage (0-100) of data to use for test set
- include_negatives: Include samples with score 0.0 as negative examples
- neg_weight: Weight value for negative samples (score 0.0)
- neg_ratio: Ratio of negative to positive samples (0.0-x.0)

The output parquet files will be saved at:
  {save_path}/{model_name_from_adv_path}_train_step_X/train.parquet
  {save_path}/{model_name_from_adv_path}_train_step_X/test.parquet (if test_split > 0)

Example usage:
python wrap_selfdistilled_data.py \
  --adv_path /path/to/advantages_train_step_6.pt \
  --save_path /path/to/output \
  --data_source my_dataset \
  --ability reasoning \
  --test_split 10 \
  --include_negatives \
  --neg_weight -1.0 \
  --neg_ratio 0.5
"""

import pandas as pd
import random
import os
import argparse
import torch
from pathlib import Path


def create_dataframe_from_advantage(advantage, data_source, ability, select_all=False, include_negatives=False, neg_weight=-1.0, neg_ratio=0.5):
    """
    Creates a dataframe from advantage data with the specified structure.
    
    Parameters:
    -----------
    advantage : dict
        Dictionary containing dataset information and samples
    data_source : str
        Value to set for 'data_source' column
    ability : str
        Value to set for 'ability' column
    select_all : bool, optional (default=False)
        If True, create a row for each element with score 1.0 or 0.0.
        If False, select one random element with score 1.0 per sample.
    include_negatives : bool, optional (default=False)
        If True, include samples with score 0.0 as negative examples
    neg_weight : float, optional (default=-1.0)
        Weight value for negative samples (score 0.0)
    neg_ratio : float, optional (default=0.5)
        Ratio of negative to positive samples (0.0-1.0)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: 'data_source', 'prompt', 'ability', 'extra_info', 'original_idx'
        and optionally 'weight' if include_negatives=True
    """
    import pandas as pd
    import random
    
    dataset_type = advantage.get('dataset_type', '')
    samples = advantage.get('samples', {})
    
    positive_data = []
    negative_data = []
    
    for sample_idx, sample_list in samples.items():
        # Filter elements with score 1.0 (positive examples)
        score_1_elements = [elem for elem in sample_list if elem.get('score', 0) == 1.0]
        
        # Process positive examples
        if score_1_elements:
            if select_all:
                elements_to_process = score_1_elements
            else:
                # Select one random element with score 1.0
                elements_to_process = [random.choice(score_1_elements)]
            
            for elem in elements_to_process:
                if include_negatives:
                    # Include weight only if we have negative examples
                    row = create_row(elem, data_source, ability, dataset_type, sample_idx, weight=1.0)
                else:
                    row = create_row(elem, data_source, ability, dataset_type, sample_idx)
                positive_data.append(row)
        
        # Process negative examples (score 0.0) if requested
        if include_negatives:
            score_0_elements = [elem for elem in sample_list if elem.get('score', 0) == 0.0]
            if score_0_elements:
                if select_all:
                    # Apply select_all to negative examples as well
                    elements_to_process = score_0_elements
                else:
                    # Only select one random negative example per sample
                    elements_to_process = [random.choice(score_0_elements)] if score_0_elements else []
                
                for elem in elements_to_process:
                    row = create_row(elem, data_source, ability, dataset_type, sample_idx, weight=neg_weight)
                    negative_data.append(row)
    
    # Calculate how many negative examples to include based on the ratio
    if include_negatives and negative_data:
        num_positives = len(positive_data)
        num_negatives_to_include = int(num_positives * neg_ratio)
        
        # Randomly sample negative examples if we have more than needed
        if num_negatives_to_include < len(negative_data):
            negative_data = random.sample(negative_data, num_negatives_to_include)
    
    # Combine positive and negative examples
    all_data = positive_data + negative_data
    
    # Shuffle the combined data
    random.shuffle(all_data)
    
    return pd.DataFrame(all_data)


def create_row(elem, data_source, ability, dataset_type, sample_idx, weight=None):
    """
    Helper function to create a row for the DataFrame.
    
    Parameters:
    -----------
    elem : dict
        Dictionary containing element data
    data_source : str
        Value for 'data_source' column
    ability : str
        Value for 'ability' column
    dataset_type : str
        Value for 'split' in 'extra_info'
    sample_idx : str
        Value for 'original_idx' column
    weight : float, optional
        Value for 'weight' column, if None, weight is not included
        
    Returns:
    --------
    dict
        Row data for DataFrame
    """
    prompt_full = elem.get('prompt', '')
    
    # Extract the part before '\nassistant\n'
    prompt_user_part = prompt_full.split('\nassistant\n')[0] if '\nassistant\n' in prompt_full else prompt_full
    prompt_user_part += '\nassistant\n'
    
    # Create formatted prompt
    formatted_prompt = [{'content': prompt_user_part, 'role': 'user'}]
    
    # Get response
    response = elem.get('response', '')
    
    # Create row data
    row = {
        'data_source': data_source,
        'prompt': formatted_prompt,
        'ability': ability,
        'extra_info': {'answer': response, 'split': dataset_type},
        'original_idx': sample_idx
    }
    
    # Add weight only if specified
    if weight is not None:
        row['weight'] = weight
    
    return row


def extract_output_name(adv_path):
    """
    Extract output directory name from advantage path.
    
    Parameters:
    -----------
    adv_path : str
        Path to the advantage file
    
    Returns:
    --------
    str
        Directory name for the output
    """
    # Extract the parent directory name and filename
    path = Path(adv_path)
    parent_dir = path.parts[-3] if len(path.parts) > 1 else ""
    filename = path.stem
    
    # Extract step number if present
    step_info = ""
    if "step" in filename:
        step_info = "_step_" + filename.split("step_")[1].split(".pt")[0]
    else:
        step_info = "_step_0"
    
    # Construct output directory name
    if parent_dir:
        return f"{parent_dir}{step_info}"
    else:
        return f"{filename}"


def split_train_test(df, test_proportion):
    """
    Split dataframe into training and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to split
    test_percentage : float
        Percentage of data to use for test set (0-100)
    
    Returns:
    --------
    tuple
        (train_df, test_df) - Training and test DataFrames
    """
    if test_proportion <= 0:
        return df, None
    
    # Convert percentage to proportion    
    # Shuffle the indices
    indices = df.index.tolist()
    random.shuffle(indices)
    
    # Calculate split point
    test_size = int(len(indices) * test_proportion)
    
    # Split the indices
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Create train and test dataframes
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Process advantage data and save as parquet")
    parser.add_argument("--adv_path", type=str, default=None, help="Path to the advantage file (.pt)")
    parser.add_argument("--save_path", type=str, default='/fast/pmayilvahanan/post_training/self_distilled_datasets/', help="Base directory to save the parquet file")
    parser.add_argument("--data_source", type=str, default='openai/gsm8k', help="Data source identifier")
    parser.add_argument("--ability", type=str, default='math', help="Ability identifier")
    parser.add_argument("--select_all", action="store_true", help="Select all elements with score 1.0 instead of just one")
    parser.add_argument("--test_split", type=float, default=0.1, help="Percentage of data to use for test set (0-100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--include_negatives", action="store_true", help="Include samples with score 0.0 as negative examples")
    parser.add_argument("--neg_weight", type=float, default=-1.0, help="Weight value for negative samples (score 0.0)")
    parser.add_argument("--neg_ratio", type=float, default=0.5, help="Ratio of negative to positive samples (0.0-x.0)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load the advantage data
    print(f"Loading advantage data from {args.adv_path}")
    advantage = torch.load(args.adv_path)
    
    # Create the DataFrame
    print(f"Creating DataFrame with data_source={args.data_source}, ability={args.ability}")
    if args.include_negatives:
        print(f"Including negative samples with weight={args.neg_weight}, ratio={args.neg_ratio}")
    
    df = create_dataframe_from_advantage(
        advantage=advantage,
        data_source=args.data_source,
        ability=args.ability,
        select_all=args.select_all,
        include_negatives=args.include_negatives,
        neg_weight=args.neg_weight,
        neg_ratio=args.neg_ratio
    )
    
    # Extract output directory name
    output_dir_name = extract_output_name(args.adv_path)

    if args.select_all:
        output_dir_name = output_dir_name + "_all"
    
    if args.include_negatives:
        output_dir_name = output_dir_name + "_with_neg" + f"_neg_ratio_{args.neg_ratio}_seed_{args.seed}"
    
    # Create output directory
    output_dir = os.path.join(args.save_path, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train and test sets if requested
    if args.test_split > 0:
        print(f"Splitting data: {(1 - args.test_split)*100}% train, {args.test_split*100}% test")
        train_df, test_df = split_train_test(df, args.test_split)
        
        # Save train and test DataFrames
        train_file = os.path.join(output_dir, "train.parquet")
        test_file = os.path.join(output_dir, "test.parquet")
        
        print(f"Saving train DataFrame with {len(train_df)} rows to {train_file}")
        train_df.to_parquet(train_file, index=False)
        
        print(f"Saving test DataFrame with {len(test_df)} rows to {test_file}")
        test_df.to_parquet(test_file, index=False)
        
        print(f"Successfully saved train and test datasets")
    else:
        # Save the entire DataFrame as train
        output_file = os.path.join(output_dir, "train.parquet")
        print(f"Saving DataFrame with {len(df)} rows to {output_file}")
        df.to_parquet(output_file, index=False)
        print(f"Successfully saved to {output_file}")


if __name__ == "__main__":
    main()