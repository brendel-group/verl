"""
Simple Prompt Optimization for Mathematical Error Categorization

Tests different prompt templates on a fixed set of samples to find the best performing one.
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

def get_prompt_templates() -> Dict[str, callable]:
    """Get all prompt templates to test."""
    
    templates = {}
    
    templates["baseline"] = lambda q, a, g: f"""Analyze why this student solution is INCORRECT and categorize the error.

Question: {q}

Student Answer (INCORRECT): {a}
Correct Answer: {g}

Error Types:
1. Computational Error - arithmetic/algebraic mistakes
2. Conceptual Misunderstanding - wrong approach
3. Incomplete Solution - didn't finish
4. Tool misuse - inappropriate use of Python/programming to solve a math problem
5. Logic Error - flawed reasoning
6. Mislabeling - student was actually correct

Put your final categorization in \\boxed{{}}."""

    templates["structured"] = lambda q, a, g: f"""QUESTION: {q}

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

    templates["concise"] = lambda q, a, g: f"""Math Error Analysis:

Q: {q}
Student (WRONG): {a}
Correct: {g}

Categories: Computational Error, Conceptual Misunderstanding, Incomplete Solution, Tool misuse (inappropriate Python/programming use), Logic Error, Mislabeling

Answer with: \\boxed{{Category}}"""

    templates["guided"] = lambda q, a, g: f"""Step-by-step error analysis:

PROBLEM: {q}
STUDENT WORK: {a}
SOLUTION: {g}

STEPS:
1. Compare student answer to correct answer
2. Find where reasoning went wrong
3. Classify using: Computational Error, Conceptual Misunderstanding, Incomplete Solution, Tool misuse (inappropriate Python/programming use), Logic Error, Mislabeling

Result: \\boxed{{Category}}"""

    return templates

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

def load_test_data(num_samples: int = 100) -> List[Dict]:
    """Load test data from dataset (focusing on negative reward samples)."""
    print(f"📂 Loading {num_samples} samples (prioritizing negative rewards)...")
    dataset = pd.read_parquet('/u/rfechner/verl/mathlighteval_chunk0_graded.parquet')
    dataset['mean_reward'] = dataset['rewards'].apply(np.mean)
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

def test_prompt(template_func, entries: List[Dict], num_responses: int = 10) -> Dict:
    """Test a single prompt template with majority voting."""
    
    print(f"   🤖 Generating {len(entries)} × {num_responses} = {len(entries) * num_responses} responses...")
    
    # Load model
    model_key = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_key,
        device_map="auto",
        torch_dtype="auto"
    )
    
    total_responses = 0
    total_parsed = 0
    majority_votes = []
    entry_results = []
    
    for i, entry in enumerate(entries):
        if (i + 1) % 10 == 0:
            print(f"   Processing sample {i + 1}/{len(entries)}...")
            
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
        
        # Generate multiple responses for this entry
        responses = []
        parsed_categories = []
        
        for j in range(num_responses):
            with torch.no_grad():
                outputs = model.generate(
                    inputs, attention_mask=attention_mask,
                    max_new_tokens=300, temperature=0.7,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            total_responses += 1
            
            category = parse_category(response)
            responses.append({
                'text': response,
                'parsed_category': category
            })
            
            if category:
                total_parsed += 1
                parsed_categories.append(category)
        
        # Compute majority vote for this entry
        if parsed_categories:
            category_counts = Counter(parsed_categories)
            majority_category = category_counts.most_common(1)[0][0]
            majority_count = category_counts.most_common(1)[0][1]
            confidence = majority_count / len(parsed_categories)
            is_unanimous = len(set(parsed_categories)) == 1
            
            majority_votes.append(majority_category)
            
            entry_results.append({
                'index': entry['index'],
                'reward': entry['reward'],
                'majority_category': majority_category,
                'confidence': confidence,
                'majority_count': majority_count,
                'total_parsed': len(parsed_categories),
                'total_responses': len(responses),
                'is_unanimous': is_unanimous,
                'category_distribution': dict(category_counts)
            })
        else:
            entry_results.append({
                'index': entry['index'],
                'reward': entry['reward'],
                'majority_category': None,
                'confidence': 0.0,
                'majority_count': 0,
                'total_parsed': 0,
                'total_responses': len(responses),
                'is_unanimous': False,
                'category_distribution': {}
            })
    
    # Calculate overall metrics
    parsing_rate = total_parsed / total_responses if total_responses > 0 else 0
    successful_entries = len([r for r in entry_results if r['majority_category'] is not None])
    success_rate = successful_entries / len(entries) if entries else 0
    
    high_confidence = len([r for r in entry_results if r['confidence'] >= 0.8])
    unanimous = len([r for r in entry_results if r['is_unanimous']])
    
    majority_category_dist = dict(Counter(majority_votes))
    
    return {
        'success_rate': success_rate,
        'parsing_rate': parsing_rate,
        'total_responses': total_responses,
        'total_parsed': total_parsed,
        'successful_entries': successful_entries,
        'high_confidence_entries': high_confidence,
        'unanimous_entries': unanimous,
        'majority_category_distribution': majority_category_dist,
        'entry_results': entry_results
    }

def optimize_prompts(num_samples: int = 100, num_responses: int = 10):
    """Run prompt optimization experiment with majority voting."""
    
    print(f"🔬 PROMPT OPTIMIZATION (EXPANDED)")
    print(f"Samples: {num_samples}, Responses per sample: {num_responses}")
    print(f"Total responses to generate: {num_samples * num_responses * 4}")  # 4 prompt templates
    print("=" * 60)
    
    # Load test data
    entries = load_test_data(num_samples)
    templates = get_prompt_templates()
    
    results = {}
    
    for name, template_func in templates.items():
        print(f"\n🧪 Testing prompt: {name}")
        result = test_prompt(template_func, entries, num_responses)
        results[name] = result
        
        print(f"   ✅ Results:")
        print(f"      Success rate (majority vote): {result['success_rate']:.1%}")
        print(f"      Parsing rate: {result['parsing_rate']:.1%}")
        print(f"      High confidence (≥80%): {result['high_confidence_entries']}/{len(entries)}")
        print(f"      Unanimous votes: {result['unanimous_entries']}/{len(entries)}")
        
        # Show category distribution
        if result['majority_category_distribution']:
            print(f"      Category distribution (majority votes):")
            total_categorized = sum(result['majority_category_distribution'].values())
            for cat, count in sorted(result['majority_category_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_categorized * 100 if total_categorized > 0 else 0
                print(f"         {cat}: {count} ({percentage:.1f}%)")
    
    # Find best prompt based on success rate
    best_prompt = max(results.items(), key=lambda x: x[1]['success_rate'])
    
    print(f"\n🏆 BEST PROMPT: {best_prompt[0]}")
    print(f"   Success rate: {best_prompt[1]['success_rate']:.1%}")
    print(f"   High confidence: {best_prompt[1]['high_confidence_entries']}/{num_samples}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prompt_optimization_expanded_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'num_samples': num_samples, 
                'num_responses': num_responses,
                'total_responses_generated': sum(r['total_responses'] for r in results.values())
            },
            'results': results,
            'best_prompt': best_prompt[0]
        }, f, indent=2)
    
    # Save summary CSV
    summary_file = f"prompt_optimization_summary_{timestamp}.csv"
    summary_data = []
    for prompt_name, result in results.items():
        summary_data.append({
            'prompt_name': prompt_name,
            'success_rate': result['success_rate'],
            'parsing_rate': result['parsing_rate'],
            'high_confidence_entries': result['high_confidence_entries'],
            'unanimous_entries': result['unanimous_entries'],
            'total_responses': result['total_responses'],
            'total_parsed': result['total_parsed']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n📁 Results saved to:")
    print(f"   Full results: {output_file}")
    print(f"   Summary CSV: {summary_file}")
    
    return best_prompt[0], results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to test (default: 100)')
    parser.add_argument('--responses', type=int, default=10, help='Number of responses per sample (default: 10)')
    args = parser.parse_args()
    
    optimize_prompts(args.samples, args.responses)
