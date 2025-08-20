import pandas as pd
import numpy as np
from pathlib import Path

def load_and_preprocess_data():
    """
    Load and preprocess the dataframes similar to extract.py, 
    focusing on questions that were degraded during training.
    
    Returns:
        dict: Dictionary containing processed dataframes and degraded question subsets
    """
    
    # Load the three parquet files
    base_path = Path('/u/rfechner/verl/workspace/rollouts/results')
    
    print("Loading parquet files...")
    base_df = pd.read_parquet(base_path / 'math500_base.parquet')
    entropy0_df = pd.read_parquet(base_path / 'math500_entropy0.0.parquet')
    entropy01_df = pd.read_parquet(base_path / 'math500_entropy0.01.parquet')
    
    print(f"Loaded data:")
    print(f"  Base model: {len(base_df)} questions")
    print(f"  Entropy 0.0: {len(entropy0_df)} questions") 
    print(f"  Entropy 0.01: {len(entropy01_df)} questions")
    
    # Extract prompt text and compute mean rewards for each dataframe
    def process_dataframe(df, suffix):
        """Extract prompt text, responses, answers and compute mean rewards."""
        df_processed = df.copy()
        
        # Extract prompt text from the nested structure
        df_processed['prompt_text'] = df_processed['prompt'].apply(
            lambda x: x[0]['content'] if len(x) > 0 and 'content' in x[0] else ''
        )
        
        # Compute mean rewards
        df_processed[f'rewards_mean{suffix}'] = df_processed['rewards'].apply(np.mean)
        
        # Rename key columns with suffix to avoid conflicts
        df_processed[f'responses{suffix}'] = df_processed['responses']
        df_processed[f'answer{suffix}'] = df_processed['answer']
        df_processed[f'rewards{suffix}'] = df_processed['rewards']
        
        return df_processed
    
    print("\nProcessing dataframes...")
    base_processed = process_dataframe(base_df, '_base')
    entropy0_processed = process_dataframe(entropy0_df, '_entropy0')
    entropy01_processed = process_dataframe(entropy01_df, '_entropy01')
    
    # For merging, we need to be more careful about column selection
    # Keep only the essential columns for merging to avoid conflicts
    
    def select_merge_columns(df, suffix):
        """Select only the necessary columns for merging."""
        keep_cols = ['prompt_text', f'rewards_mean{suffix}', f'responses{suffix}', f'answer{suffix}', f'rewards{suffix}']
        return df[keep_cols]
    
    base_merge = select_merge_columns(base_processed, '_base')
    entropy0_merge = select_merge_columns(entropy0_processed, '_entropy0') 
    entropy01_merge = select_merge_columns(entropy01_processed, '_entropy01')
    
    # Merge dataframes on 'prompt_text' column
    print("Merging dataframes...")
    merged = base_merge.merge(entropy0_merge, on='prompt_text', how='inner')
    merged = merged.merge(entropy01_merge, on='prompt_text', how='inner')
    
    print(f"Merged dataset: {len(merged)} questions")
    
    # Calculate performance deltas
    merged['delta_entropy0'] = merged['rewards_mean_entropy0'] - merged['rewards_mean_base']
    merged['delta_entropy01'] = merged['rewards_mean_entropy01'] - merged['rewards_mean_base']
    
    # Define degradation thresholds
    degradation_threshold = -0.05  # 5% drop in performance
    
    # Identify degraded questions for each training method
    degraded_entropy0 = merged[merged['delta_entropy0'] < degradation_threshold].copy()
    degraded_entropy01 = merged[merged['delta_entropy01'] < degradation_threshold].copy()
    
    # Questions degraded in both methods
    degraded_both = merged[
        (merged['delta_entropy0'] < degradation_threshold) & 
        (merged['delta_entropy01'] < degradation_threshold)
    ].copy()
    
    # High-confidence base questions that were degraded (high passrate threshold)
    high_passrate_threshold = 0.7
    
    high_base_degraded_entropy0 = merged[
        (merged['rewards_mean_base'] >= high_passrate_threshold) & 
        (merged['delta_entropy0'] < degradation_threshold)
    ].copy()
    
    high_base_degraded_entropy01 = merged[
        (merged['rewards_mean_base'] >= high_passrate_threshold) & 
        (merged['delta_entropy01'] < degradation_threshold)
    ].copy()
    
    high_base_degraded_both = merged[
        (merged['rewards_mean_base'] >= high_passrate_threshold) & 
        (merged['delta_entropy0'] < degradation_threshold) & 
        (merged['delta_entropy01'] < degradation_threshold)
    ].copy()
    
    # Sort degraded questions by severity
    degraded_entropy0_sorted = degraded_entropy0.sort_values('delta_entropy0')
    degraded_entropy01_sorted = degraded_entropy01.sort_values('delta_entropy01')
    degraded_both_sorted = degraded_both.sort_values(['delta_entropy0', 'delta_entropy01'])
    high_base_degraded_both_sorted = high_base_degraded_both.sort_values(['delta_entropy0', 'delta_entropy01'])
    
    print(f"\nDegradation Analysis:")
    print(f"  Questions degraded in entropy0.0: {len(degraded_entropy0)} ({len(degraded_entropy0)/len(merged)*100:.1f}%)")
    print(f"  Questions degraded in entropy0.01: {len(degraded_entropy01)} ({len(degraded_entropy01)/len(merged)*100:.1f}%)")
    print(f"  Questions degraded in both: {len(degraded_both)} ({len(degraded_both)/len(merged)*100:.1f}%)")
    print(f"  High-confidence base degraded in entropy0.0: {len(high_base_degraded_entropy0)}")
    print(f"  High-confidence base degraded in entropy0.01: {len(high_base_degraded_entropy01)}")
    print(f"  High-confidence base degraded in both: {len(high_base_degraded_both)}")
    
    # Package results
    results = {
        'merged_df': merged,
        'degraded_entropy0': degraded_entropy0_sorted,
        'degraded_entropy01': degraded_entropy01_sorted,
        'degraded_both': degraded_both_sorted,
        'high_base_degraded_entropy0': high_base_degraded_entropy0.sort_values('delta_entropy0'),
        'high_base_degraded_entropy01': high_base_degraded_entropy01.sort_values('delta_entropy01'),
        'high_base_degraded_both': high_base_degraded_both_sorted,
        'stats': {
            'total_questions': len(merged),
            'degradation_threshold': degradation_threshold,
            'high_passrate_threshold': high_passrate_threshold,
            'degraded_entropy0_count': len(degraded_entropy0),
            'degraded_entropy01_count': len(degraded_entropy01),
            'degraded_both_count': len(degraded_both),
            'high_base_degraded_entropy0_count': len(high_base_degraded_entropy0),
            'high_base_degraded_entropy01_count': len(high_base_degraded_entropy01),
            'high_base_degraded_both_count': len(high_base_degraded_both)
        }
    }
    
    return results

def preview_degraded_questions(data, n_examples=5):
    """
    Print a preview of the most degraded questions for each category.
    
    Args:
        data (dict): Results from load_and_preprocess_data()
        n_examples (int): Number of examples to show for each category
    """
    
    print(f"\n{'='*80}")
    print("PREVIEW OF MOST DEGRADED QUESTIONS")
    print(f"{'='*80}")
    
    # Most degraded in entropy0.0
    print(f"\nWORST {n_examples} DEGRADATIONS - ENTROPY 0.0:")
    print("-" * 60)
    for i, (_, row) in enumerate(data['degraded_entropy0'].head(n_examples).iterrows(), 1):
        print(f"{i}. Degradation: {row['delta_entropy0']:.4f}")
        print(f"   Base: {row['rewards_mean_base']:.4f} → Entropy0.0: {row['rewards_mean_entropy0']:.4f}")
        print(f"   Problem: {row['prompt_text'][:100]}...")
        print()
    
    # Most degraded in entropy0.01
    print(f"\nWORST {n_examples} DEGRADATIONS - ENTROPY 0.01:")
    print("-" * 60)
    for i, (_, row) in enumerate(data['degraded_entropy01'].head(n_examples).iterrows(), 1):
        print(f"{i}. Degradation: {row['delta_entropy01']:.4f}")
        print(f"   Base: {row['rewards_mean_base']:.4f} → Entropy0.01: {row['rewards_mean_entropy01']:.4f}")
        print(f"   Problem: {row['prompt_text'][:100]}...")
        print()
    
    # High-confidence base questions degraded in both
    if len(data['high_base_degraded_both']) > 0:
        print(f"\nHIGH-CONFIDENCE BASE DEGRADED IN BOTH METHODS:")
        print("-" * 60)
        for i, (_, row) in enumerate(data['high_base_degraded_both'].iterrows(), 1):
            print(f"{i}. Base: {row['rewards_mean_base']:.4f}")
            print(f"   → Entropy0.0: {row['rewards_mean_entropy0']:.4f} (Δ: {row['delta_entropy0']:.4f})")
            print(f"   → Entropy0.01: {row['rewards_mean_entropy01']:.4f} (Δ: {row['delta_entropy01']:.4f})")
            print(f"   Problem: {row['prompt_text'][:100]}...")
            print()

def analyze_responses_for_question(row, question_idx=None):
    """
    Analyze responses for a specific degraded question to understand failure modes.
    
    Args:
        row: A row from the merged dataframe containing a degraded question
        question_idx: Optional index for display purposes
    """
    
    title = f"QUESTION {question_idx}" if question_idx else "QUESTION ANALYSIS"
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    
    print(f"Problem: {row['prompt_text']}")
    print(f"Ground Truth Answer: {row['answer_base']}")
    print()
    
    print(f"Performance Summary:")
    print(f"  Base Model: {row['rewards_mean_base']:.4f}")
    print(f"  Entropy 0.0: {row['rewards_mean_entropy0']:.4f} (Δ: {row['delta_entropy0']:.4f})")
    print(f"  Entropy 0.01: {row['rewards_mean_entropy01']:.4f} (Δ: {row['delta_entropy01']:.4f})")
    print()
    
    # Analyze responses for each model
    models = [
        ('Base', row['responses_base'], row['rewards_base']),
        ('Entropy 0.0', row['responses_entropy0'], row['rewards_entropy0']),
        ('Entropy 0.01', row['responses_entropy01'], row['rewards_entropy01'])
    ]
    
    for model_name, responses, rewards in models:
        print(f"{model_name} Model Analysis:")
        print(f"  Number of responses: {len(responses)}")
        print(f"  Success rate: {np.mean(rewards):.4f}")
        
        # Show sample successful and failed responses
        rewards_array = np.array(rewards)
        successful_indices = np.where(rewards_array == 1.0)[0]
        failed_indices = np.where(rewards_array == 0.0)[0]
        
        if len(successful_indices) > 0:
            print(f"  Sample successful response (#{successful_indices[0]}):")
            print(f"    {responses[successful_indices[0]][:300]}...")
            print()
        
        if len(failed_indices) > 0:
            print(f"  Sample failed response (#{failed_indices[0]}):")
            print(f"    {responses[failed_indices[0]][:300]}...")
            print()
        
        print()

def compare_successful_vs_failed_responses(data, model_comparison='entropy0', n_examples=3):
    """
    Compare successful vs failed responses for degraded questions.
    
    Args:
        data: Result from load_and_preprocess_data()
        model_comparison: 'entropy0' or 'entropy01' 
        n_examples: Number of questions to analyze in detail
    """
    
    print(f"\n{'='*100}")
    print(f"DETAILED RESPONSE ANALYSIS - {model_comparison.upper()} DEGRADATIONS")
    print(f"{'='*100}")
    
    # Select the appropriate degraded dataset
    if model_comparison == 'entropy0':
        degraded_questions = data['high_base_degraded_entropy0']
    else:
        degraded_questions = data['high_base_degraded_entropy01']
    
    print(f"Analyzing top {n_examples} degraded questions...")
    
    for i, (_, row) in enumerate(degraded_questions.head(n_examples).iterrows(), 1):
        analyze_responses_for_question(row, question_idx=i)
        
        if i < n_examples:
            print(f"\n{'-'*100}\n")

def extract_response_patterns(data, save_to_file=True):
    """
    Extract and categorize common response patterns in degraded questions.
    
    Args:
        data: Result from load_and_preprocess_data()
        save_to_file: Whether to save results to a JSON file
    
    Returns:
        dict: Analysis of response patterns
    """
    
    print("\nExtracting response patterns from degraded questions...")
    
    patterns = {
        'high_base_degraded_both': [],
        'entropy0_specific_failures': [],
        'entropy01_specific_failures': [],
        'summary_statistics': {}
    }
    
    # Analyze questions degraded in both methods (most concerning)
    for _, row in data['high_base_degraded_both'].iterrows():
        question_analysis = {
            'prompt': row['prompt_text'],
            'ground_truth': row['answer_base'],
            'base_performance': float(row['rewards_mean_base']),
            'entropy0_performance': float(row['rewards_mean_entropy0']),
            'entropy01_performance': float(row['rewards_mean_entropy01']),
            'base_sample_response': row['responses_base'][0] if len(row['responses_base']) > 0 else '',
            'entropy0_sample_response': row['responses_entropy0'][0] if len(row['responses_entropy0']) > 0 else '',
            'entropy01_sample_response': row['responses_entropy01'][0] if len(row['responses_entropy01']) > 0 else '',
            'entropy0_degradation': float(row['delta_entropy0']),
            'entropy01_degradation': float(row['delta_entropy01'])
        }
        patterns['high_base_degraded_both'].append(question_analysis)
    
    # Add summary statistics
    patterns['summary_statistics'] = {
        'total_questions_analyzed': len(data['merged_df']),
        'high_base_degraded_both_count': len(data['high_base_degraded_both']),
        'entropy0_degraded_count': len(data['degraded_entropy0']),
        'entropy01_degraded_count': len(data['degraded_entropy01']),
        'mean_base_performance_degraded_both': float(data['high_base_degraded_both']['rewards_mean_base'].mean()),
        'mean_entropy0_performance_degraded_both': float(data['high_base_degraded_both']['rewards_mean_entropy0'].mean()),
        'mean_entropy01_performance_degraded_both': float(data['high_base_degraded_both']['rewards_mean_entropy01'].mean())
    }
    
    if save_to_file:
        import json
        output_file = '/u/rfechner/verl/workspace/judge/migration_analysis/degraded_answers/response_patterns_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
        print(f"Response patterns saved to: {output_file}")
    
    return patterns

if __name__ == "__main__":
    # Load and preprocess the data
    print("Loading and preprocessing degraded answer data...")
    data = load_and_preprocess_data()
    
    # Show preview of most degraded questions
    preview_degraded_questions(data, n_examples=3)
    
    print(f"\n{'='*80}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*80}")
    print("Available datasets in 'data' dictionary:")
    for key in data.keys():
        if key != 'stats':
            print(f"  - {key}: {len(data[key])} questions" if hasattr(data[key], '__len__') else f"  - {key}")
        else:
            print(f"  - {key}: summary statistics")
    
    # Demonstrate enhanced analysis capabilities
    print(f"\n{'='*80}")
    print("ENHANCED ANALYSIS CAPABILITIES DEMONSTRATION")
    print(f"{'='*80}")
    
    # Show detailed response analysis for the worst degraded question
    if len(data['high_base_degraded_both']) > 0:
        print("\nDemonstrating detailed response analysis for worst degraded question...")
        worst_question = data['high_base_degraded_both'].iloc[0]
        analyze_responses_for_question(worst_question, question_idx=1)
    
    # Extract and save response patterns
    print("\nExtracting response patterns for further analysis...")
    patterns = extract_response_patterns(data, save_to_file=True)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE - READY FOR FURTHER INVESTIGATION")
    print(f"{'='*80}")
    print("The enhanced script now provides:")
    print("  ✓ Full access to prompts, answers, and responses")
    print("  ✓ Detailed response analysis functions") 
    print("  ✓ Response pattern extraction")
    print("  ✓ JSON export for further analysis")
    print("  ✓ Sample successful vs failed response comparison")
    print("\nNext steps:")
    print("  1. Use compare_successful_vs_failed_responses() for detailed analysis")
    print("  2. Analyze response_patterns_analysis.json for common failure modes")
    print("  3. Implement topic-specific analysis based on prompt content")
