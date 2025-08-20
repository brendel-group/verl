#!/usr/bin/env python3
"""
Example usage script demonstrating the enhanced degraded answers analysis capabilities.

This script shows how to use the enhanced load_degraded.py functionality for detailed analysis
of questions that were degraded during training.
"""

from load_degraded import (
    load_and_preprocess_data, 
    preview_degraded_questions,
    analyze_responses_for_question,
    compare_successful_vs_failed_responses,
    extract_response_patterns
)
import json
import pandas as pd

def main():
    """Demonstrate the enhanced analysis capabilities."""
    
    print("="*80)
    print("DEGRADED ANSWERS ANALYSIS - USAGE EXAMPLE")
    print("="*80)
    
    # 1. Load the data with full response information
    print("\n1. Loading data with full response information...")
    data = load_and_preprocess_data()
    
    # 2. Quick overview
    print(f"\n2. Quick overview:")
    print(f"   Total questions: {len(data['merged_df'])}")
    print(f"   Degraded in entropy0.0: {len(data['degraded_entropy0'])}")
    print(f"   Degraded in entropy0.01: {len(data['degraded_entropy01'])}")
    print(f"   High-confidence failures in both: {len(data['high_base_degraded_both'])}")
    
    # 3. Analyze specific degraded questions in detail
    print(f"\n3. Detailed analysis of most concerning questions...")
    if len(data['high_base_degraded_both']) > 0:
        print(f"   Analyzing the worst question that failed in both methods:")
        worst_question = data['high_base_degraded_both'].iloc[0]
        analyze_responses_for_question(worst_question, question_idx=1)
    
    # 4. Compare successful vs failed responses for entropy0.0 degradations
    print(f"\n4. Comparing successful vs failed responses (entropy0.0)...")
    compare_successful_vs_failed_responses(data, model_comparison='entropy0', n_examples=2)
    
    # 5. Extract and examine response patterns
    print(f"\n5. Extracting response patterns...")
    patterns = extract_response_patterns(data, save_to_file=True)
    
    # 6. Demonstrate data access for custom analysis
    print(f"\n6. Custom analysis example - Topic categorization:")
    
    # Example: Categorize degraded questions by mathematical topic
    topic_keywords = {
        'algebra': ['equation', 'solve', 'variable', 'polynomial'],
        'geometry': ['triangle', 'circle', 'angle', 'area', 'volume'],
        'calculus': ['derivative', 'integral', 'limit', 'differential'],
        'number_theory': ['prime', 'gcd', 'modular', 'divisible'],
        'probability': ['probability', 'permutation', 'combination', 'random'],
        'trigonometry': ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']
    }
    
    topic_counts = {topic: 0 for topic in topic_keywords.keys()}
    
    for _, row in data['high_base_degraded_both'].iterrows():
        prompt = row['prompt_text'].lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                topic_counts[topic] += 1
                break
    
    print("   Topic distribution of high-confidence failures:")
    for topic, count in topic_counts.items():
        if count > 0:
            percentage = count / len(data['high_base_degraded_both']) * 100
            print(f"     {topic.title()}: {count} questions ({percentage:.1f}%)")
    
    # 7. Show how to access individual responses for analysis
    print(f"\n7. Accessing individual responses for detailed analysis:")
    if len(data['high_base_degraded_both']) > 0:
        sample_question = data['high_base_degraded_both'].iloc[0]
        print(f"   Sample question: {sample_question['prompt_text'][:100]}...")
        print(f"   Base model responses available: {len(sample_question['responses_base'])}")
        print(f"   Entropy0.0 responses available: {len(sample_question['responses_entropy0'])}")
        print(f"   Entropy0.01 responses available: {len(sample_question['responses_entropy01'])}")
        
        # Show sample responses
        print(f"\n   Sample base model response:")
        print(f"     {sample_question['responses_base'][0][:200]}...")
        print(f"\n   Sample entropy0.0 response:")
        print(f"     {sample_question['responses_entropy0'][0][:200]}...")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Key data access patterns:")
    print("  • data['merged_df'] - Complete dataset with all responses")
    print("  • data['high_base_degraded_both'] - Most concerning failures")
    print("  • row['responses_base'] - List of base model responses for a question")
    print("  • row['rewards_base'] - Individual reward scores for each response")
    print("  • response_patterns_analysis.json - Exported patterns for further analysis")
    
    print(f"\nReady for advanced analysis:")
    print("  1. Response content analysis (common error patterns)")
    print("  2. Mathematical reasoning breakdown analysis")
    print("  3. Training intervention design based on failure modes")

if __name__ == "__main__":
    main()
