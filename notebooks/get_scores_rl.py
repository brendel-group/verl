import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm # Optional: for progress bar
import sys # To handle potential recursion depth issues with deep data structures
import os # Added for path manipulation and directory creation
import pickle # Added for saving results
import re  # For extracting step number from filenames

# Increase recursion depth limit if needed for deep data structures in .pt files
# sys.setrecursionlimit(2000) # Uncomment and adjust if you encounter recursion errors

# --- Configuration ---
# IMPORTANT: Replace 'gpt2' with the actual tokenizer you need (e.g., 'Qwen/Qwen2.5-1.5B')
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"
# Set the number of parallel workers (adjust based on your CPU cores)
MAX_WORKERS = os.cpu_count() // 4 if os.cpu_count() else 4 # Default to quarter of cores or 4

# --- Initialize Tokenizer ---
# Load tokenizer globally once to potentially be reused by child processes (depends on OS/fork method)
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Tokenizer '{TOKENIZER_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer '{TOKENIZER_NAME}': {e}")
    print("Please ensure the tokenizer name is correct and you have internet access or the model cached.")
    exit(1) # Exit if tokenizer fails to load

def get_token_length(text):
    """Tokenizes text and returns its length. Handles None or non-string input."""
    if not isinstance(text, str):
        return 0
    # Use encode to get token IDs, then count them. Add special tokens if desired.
    return len(tokenizer.encode(text, add_special_tokens=False)) # Set add_special_tokens as needed

def extract_step_from_filename(filename):
    """Extract the step number from the filename using regex"""
    match = re.search(r'step_(\d+)', str(filename))
    if match:
        return int(match.group(1))
    return 0  # Return 0 (pretraining) if no step number found

def process_file(file_path: Path):
    """
    Loads a single .pt file, extracts scores and response lengths per prompt ID.

    Args:
        file_path: Path object pointing to the .pt file.

    Returns:
        A tuple: (dataset_type, step, dict_of_results) or None if an error occurs.
        dict_of_results maps prompt_id -> {'scores': [...], 'response_lengths': [...]}
    """
    try:
        # Extract step from filename
        step = extract_step_from_filename(file_path.name)

            
        # Load data onto CPU to avoid potential GPU memory issues in parallel processes
        data = torch.load(file_path, map_location='cpu')

        dataset_type = data.get('dataset_type')
        samples = data.get('samples')

        if not dataset_type or dataset_type not in ['train', 'val']:
            print(f"Warning: Invalid or missing 'dataset_type' in {file_path.name}. Skipping.")
            return None
        if not samples or not isinstance(samples, dict):
            print(f"Warning: Invalid or missing 'samples' dict in {file_path.name}. Skipping.")
            return None

        # Use defaultdict for easy appending within this file's processing
        file_results = defaultdict(lambda: {'scores': [], 'response_lengths': []})

        for prompt_id, responses_list in samples.items():
            if not isinstance(responses_list, list):
                # print(f"Warning: Expected list for prompt_id {prompt_id} in {file_path.name}, got {type(responses_list)}. Skipping entry.")
                continue # Skip this prompt_id if data format is wrong

            for response_data in responses_list:
                if not isinstance(response_data, dict):
                    # print(f"Warning: Expected dict for response data under prompt_id {prompt_id} in {file_path.name}. Skipping response.")
                    continue # Skip this malformed response entry

                score = response_data.get('score')
                response_text = response_data.get('response')

                # Only process if both score and response are found
                if score is not None and response_text is not None:
                    try:
                        # Ensure score is a float
                        score_float = float(score)
                        token_length = get_token_length(response_text)

                        file_results[prompt_id]['scores'].append(score_float)
                        file_results[prompt_id]['response_lengths'].append(token_length)
                    except ValueError:
                         print(f"Warning: Could not convert score '{score}' to float for prompt {prompt_id} in {file_path.name}. Skipping response score.")
                    except Exception as token_err:
                         print(f"Warning: Error tokenizing response for prompt {prompt_id} in {file_path.name}: {token_err}. Skipping response length.")
                # else: # Optional: Warn about missing score/response
                    # print(f"Warning: Missing 'score' or 'response' for an entry under prompt {prompt_id} in {file_path.name}.")

        return dataset_type, step, dict(file_results) # Return step along with other data

    except FileNotFoundError:
        print(f"Error: File not found {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing file {file_path.name}: {e.__class__.__name__}: {e}")
        # Consider logging the full traceback here for debugging if needed
        # import traceback
        # print(traceback.format_exc())
        return None

def main(data_dir: str):
    """
    Finds .pt files, processes them in parallel, aggregates results,
    and saves them to a structured output directory.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Error: Directory not found: {data_dir}")
        return

    pt_files = list(data_path.glob("advantages_*.pt")) # Adjust glob pattern if needed
    if not pt_files:
        print(f"No matching '.pt' files found in {data_dir}")
        return

    # --- Determine Output Directory ---
    # Go up one level from data_dir and append the new folder name
    parent_dir = data_path.parent
    output_dir_name = "scores_response_length"
    output_dir = parent_dir / output_dir_name

    print(f"Found {len(pt_files)} files matching pattern 'advantages_*.pt'.")
    print(f"Output will be saved to: {output_dir}")
    print(f"Starting processing with {MAX_WORKERS} workers...")

    # New hierarchical structure: dataset_type -> step -> prompt_id -> data
    train_results = {}  # {step: {prompt_id: {'scores': [], 'response_lengths': []}}}
    val_results = {}  # {step: {prompt_id: {'scores': [], 'response_lengths': []}}}

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create futures for all file processing tasks
        future_to_file = {executor.submit(process_file, f): f for f in pt_files}

        # Process completed tasks as they finish, with a progress bar
        for future in tqdm(as_completed(future_to_file), total=len(pt_files), desc="Processing files"):
            result = future.result()
            if result:
                dataset_type, step, file_data = result
                
                # Select the appropriate dictionary based on dataset_type
                target_dict = train_results if dataset_type == 'train' else val_results
                
                # Initialize step dictionary if it doesn't exist
                if step not in target_dict:
                    target_dict[step] = {}
                
                # Add data for each prompt_id at this step
                for prompt_id, data in file_data.items():
                    target_dict[step][prompt_id] = data

    print("\n--- Processing Complete ---")
    print(f"Total steps in train data: {len(train_results)}")
    print(f"Total steps in validation data: {len(val_results)}")

    # --- Example: Print info for one step and one prompt ID if available ---
    if train_results:
        first_step = next(iter(train_results))
        prompt_ids = list(train_results[first_step].keys())
        if prompt_ids:
            first_prompt_id = prompt_ids[0]
            print(f"\nExample Train Data (Step: {first_step}, Prompt ID: {first_prompt_id}):")
            print(f"  Number of scores: {len(train_results[first_step][first_prompt_id]['scores'])}")
            print(f"  Number of response lengths: {len(train_results[first_step][first_prompt_id]['response_lengths'])}")
    
    if val_results:
        first_step = next(iter(val_results))
        prompt_ids = list(val_results[first_step].keys())
        if prompt_ids:
            first_prompt_id = prompt_ids[0]
            print(f"\nExample Validation Data (Step: {first_step}, Prompt ID: {first_prompt_id}):")
            print(f"  Number of scores: {len(val_results[first_step][first_prompt_id]['scores'])}")
            print(f"  Number of response lengths: {len(val_results[first_step][first_prompt_id]['response_lengths'])}")

    # --- Save results to the calculated output directory ---
    try:
        # Create the output directory, including parent directories if needed.
        # exist_ok=True prevents an error if the directory already exists.
        output_dir.mkdir(parents=True, exist_ok=True)

        train_output_path = output_dir / 'train_aggregated_results.pkl'
        val_output_path = output_dir / 'val_aggregated_results.pkl'

        print(f"\nSaving train results to {train_output_path}")
        with open(train_output_path, 'wb') as f:
            pickle.dump(train_results, f)

        print(f"Saving validation results to {val_output_path}")
        with open(val_output_path, 'wb') as f:
            pickle.dump(val_results, f)

        print("\nResults saved successfully.")
    except Exception as save_err:
        print(f"\nError saving results: {save_err}")

    # Return the dictionaries if needed for further processing in a larger script
    return train_results, val_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Efficiently process .pt files in parallel to aggregate scores and response lengths per prompt ID."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the .pt files (e.g., 'advantages_train_step_*.pt', 'advantages_val_step_*.pt')."
    )

    # Optional: Add arguments for tokenizer name, workers, output file paths etc.
    # parser.add_argument("--tokenizer", default=TOKENIZER_NAME, help="Hugging Face tokenizer name.")
    # parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers.")

    args = parser.parse_args()

    # Update config if args are provided (example for tokenizer/workers if added)
    # TOKENIZER_NAME = args.tokenizer
    # MAX_WORKERS = args.workers
    # Need to re-initialize tokenizer if name changes via args

    main(args.data_dir)