"""
Large-Scale Error Categorization with Majority Voting

Runs error categorization on many samples and uses majority voting for final results.
"""

import pandas as pd
import numpy as np
import torch
import json
import re
import datetime
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

# Error categories
ERROR_CATEGORIES = [
    "Computational Error",
    "Conceptual Misunderstanding", 
    "Incomplete Solution",
    "Tool misuse",
    "Logic Error",
    "Mislabeling"
]

def get_system_prompt() -> str:
    """System prompt for the AI."""
    return """You are an expert mathematics educator analyzing student errors. Your task is to categorize why a student's solution is incorrect."""

def get_best_prompt() -> callable:
    """Get the best performing prompt (update based on optimization results)."""
    return lambda q, a, g: f"""QUESTION: {q}

STUDENT SOLUTION (INCORRECT): {a}
CORRECT ANSWER: {g}

TASK: Categorize the primary error type.

ERROR CATEGORIES:
1. Computational Error
2. Conceptual Misunderstanding
3. Incomplete Solution
4. Tool misuse - inappropriate use of Python/programming to solve a math problem
5. Logic Error
6. Mislabeling

INSTRUCTIONS: Analyze step by step and provide your final answer as \\boxed{{Category Name}}

Analysis:"""

def parse_category(text: str) -> Optional[str]:
    """Extract error category from response text."""
    
    # Look for boxed format
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\\(\s*\\boxed\{([^}]+)\}\s*\\\)',
        r'\$\s*\\boxed\{([^}]+)\}\s*\$',
    ]
    
    for pattern in boxed_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            category = match.group(1).strip()
            for valid_cat in ERROR_CATEGORIES:
                if valid_cat.lower() in category.lower():
                    return valid_cat
    
    # Look for explicit mentions
    for category in ERROR_CATEGORIES:
        if category.lower() in text.lower():
            return category
    
    return None

def load_dataset(num_samples: int) -> List[Dict]:
    """Load dataset samples (focusing on negative/lowest reward samples first)."""
    
    print(f"📂 Loading {num_samples} samples (prioritizing negative rewards)...")
    dataset = pd.read_parquet('/u/rfechner/verl/mathlighteval_chunk0_graded.parquet')
    dataset['mean_reward'] = dataset['rewards'].apply(np.mean)
    
    # Sort by mean reward ascending to get lowest/most negative rewards first
    dataset = dataset.sort_values(by='mean_reward', ascending=True).head(num_samples)
    
    entries = []
    for i, row in dataset.iterrows():
        question = row['prompt'][0]['content']
        answer = row['responses'][0]
        ground_truth = row['reward_model']['ground_truth']
        reward = row['mean_reward']
        
        entries.append({
            'index': i,
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'reward': reward
        })
    
    print(f"✅ Loaded {len(entries)} entries (lowest reward samples)")
    print(f"   Reward range: {min(e['reward'] for e in entries):.3f} to {max(e['reward'] for e in entries):.3f}")
    
    # Show count of negative vs positive rewards
    negative_count = sum(1 for e in entries if e['reward'] < 0)
    print(f"   Negative rewards: {negative_count}/{len(entries)} ({negative_count/len(entries)*100:.1f}%)")
    
    return entries

def annotate_samples(entries: List[Dict], responses_per_sample: int = 5, model_key : str = "Qwen/Qwen2.5-7B-Instruct") -> List[Dict]:
    """Generate annotations for all samples."""
    
    print(f"🤖 Generating annotations...")
    print(f"   {len(entries)} samples × {responses_per_sample} responses = {len(entries) * responses_per_sample} total")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_key,
        device_map="auto",
        torch_dtype="auto"
    )
    
    template_func = get_best_prompt()
    annotated_entries = []
    
    for i, entry in enumerate(entries):
        print(f"\n--- Sample {i+1}/{len(entries)} ---")
        print(f"Reward: {entry['reward']:.3f}")
        
        user_msg = template_func(entry['question'], entry['answer'], entry['ground_truth'])
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_msg}
        ]
        
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        
        inputs = encoded['input_ids'].to(model.device)
        attention_mask = encoded['attention_mask'].to(model.device)
        
        responses = []
        for j in range(responses_per_sample):
            with torch.no_grad():
                outputs = model.generate(
                    inputs, attention_mask=attention_mask,
                    max_new_tokens=400, temperature=0.7,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
            
            response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            category = parse_category(response_text)
            
            responses.append({
                'response_id': j + 1,
                'text': response_text,
                'parsed_category': category
            })
        
        annotated_entries.append({
            **entry,
            'responses': responses
        })
    
    return annotated_entries

def compute_majority_votes(annotated_entries: List[Dict]) -> Dict:
    """Compute majority vote results."""
    
    print(f"\n🗳️  Computing majority votes...")
    
    results = []
    total_responses = 0
    total_parsed = 0
    final_categories = []
    
    for entry in annotated_entries:
        responses = entry['responses']
        parsed_categories = [r['parsed_category'] for r in responses if r['parsed_category'] is not None]
        
        total_responses += len(responses)
        total_parsed += len(parsed_categories)
        
        if parsed_categories:
            # Get majority vote
            category_counts = Counter(parsed_categories)
            majority_category = category_counts.most_common(1)[0][0]
            majority_count = category_counts.most_common(1)[0][1]
            
            confidence = majority_count / len(parsed_categories)
            is_unanimous = len(set(parsed_categories)) == 1
            
            final_categories.append(majority_category)
            
            results.append({
                'index': entry['index'],
                'reward': entry['reward'],
                'question': entry['question'][:100] + "..." if len(entry['question']) > 100 else entry['question'],
                'majority_category': majority_category,
                'confidence': confidence,
                'majority_count': majority_count,
                'total_parsed': len(parsed_categories),
                'total_responses': len(responses),
                'is_unanimous': is_unanimous,
                'category_distribution': dict(category_counts)
            })
        else:
            # No valid categories
            results.append({
                'index': entry['index'],
                'reward': entry['reward'],
                'question': entry['question'][:100] + "..." if len(entry['question']) > 100 else entry['question'],
                'majority_category': None,
                'confidence': 0.0,
                'majority_count': 0,
                'total_parsed': 0,
                'total_responses': len(responses),
                'is_unanimous': False,
                'category_distribution': {}
            })
    
    # Overall statistics
    parsing_rate = total_parsed / total_responses if total_responses > 0 else 0
    successful_entries = len([r for r in results if r['majority_category'] is not None])
    success_rate = successful_entries / len(results) if results else 0
    
    high_confidence = len([r for r in results if r['confidence'] >= 0.8])
    unanimous = len([r for r in results if r['is_unanimous']])
    
    final_distribution = dict(Counter(final_categories))
    
    stats = {
        'total_entries': len(results),
        'total_responses': total_responses,
        'parsing_rate': parsing_rate,
        'success_rate': success_rate,
        'high_confidence_entries': high_confidence,
        'unanimous_entries': unanimous,
        'final_category_distribution': final_distribution
    }
    
    print(f"✅ Majority voting completed:")
    print(f"   Parsing rate: {parsing_rate:.1%}")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   High confidence (≥80%): {high_confidence}/{len(results)}")
    print(f"   Unanimous: {unanimous}/{len(results)}")
    
    # Print category distribution with percentages
    if final_distribution:
        print(f"\n📊 Final Category Distribution (Majority Votes):")
        total_categorized = sum(final_distribution.values())
        for cat, count in sorted(final_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_categorized * 100 if total_categorized > 0 else 0
            print(f"   {cat}: {count} ({percentage:.1f}%)")
    
    return {
        'results': results,
        'statistics': stats,
        'raw_data': annotated_entries
    }

def run_large_scale(num_samples: int = 100, responses_per_sample: int = 5, model_key : str = "Qwen/Qwen2.5-7B-Instruct"):
    """Run the complete large-scale annotation pipeline."""
    
    model_subkey = model_key.split('/')[1]

    print(f"🚀 LARGE-SCALE ERROR CATEGORIZATION")
    print(f"Samples: {num_samples}, Responses per sample: {responses_per_sample}")
    print("=" * 60)
    
    # Load data
    entries = load_dataset(num_samples)
    
    # Generate annotations
    annotated_entries = annotate_samples(entries, responses_per_sample, model_key=model_key)
    
    # Compute majority votes
    majority_results = compute_majority_votes(annotated_entries)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results
    results_file = f"large_scale_results_{model_subkey}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'num_samples': num_samples,
                'responses_per_sample': responses_per_sample
            },
            **majority_results
        }, f, indent=2)
    
    # Summary CSV
    summary_file = f"large_scale_summary_{model_subkey}_{timestamp}.csv"
    df = pd.DataFrame(majority_results['results'])
    df.to_csv(summary_file, index=False)
    
    # Print summary
    stats = majority_results['statistics']
    print(f"\n📊 FINAL RESULTS (Majority Vote Classification):")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Parsing rate: {stats['parsing_rate']:.1%}")
    print(f"   High confidence: {stats['high_confidence_entries']}/{stats['total_entries']}")
    
    if stats['final_category_distribution']:
        print(f"\n📈 Category Distribution (Majority Vote):")
        total_categorized = sum(stats['final_category_distribution'].values())
        for cat, count in sorted(stats['final_category_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_categorized * 100 if total_categorized > 0 else 0
            print(f"   {cat}: {count} ({percentage:.1f}%)")
    
    print(f"\n📁 Files saved:")
    print(f"   Full results: {results_file}")
    print(f"   Summary CSV: {summary_file}")
    
    return majority_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--responses', type=int, default=10)
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    
    run_large_scale(args.samples, args.responses, args.model)
