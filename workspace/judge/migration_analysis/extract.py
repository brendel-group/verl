import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_pass_at_1_changes(merged_df, model1_name, model2_name, low_prob_threshold=0.2):
    """
    Analyze per-question changes in pass@1 rates, focusing on low initial probability questions.
    
    Args:
        merged_df: DataFrame with merged results from both models
        model1_name: Name of the first model (baseline)
        model2_name: Name of the second model (trained)
        low_prob_threshold: Threshold for considering a question as having low initial probability
    
    Returns:
        dict: Analysis results for pass@1 changes
    """
    # Calculate pass@1 rates (assuming rewards_mean represents pass@1)
    model1_pass = merged_df[f'rewards_mean_{model1_name}']
    model2_pass = merged_df[f'rewards_mean_{model2_name}']
    
    # Calculate absolute change in pass@1 rate
    pass_rate_change = model2_pass - model1_pass
    
    # Identify low initial probability questions
    low_initial_prob = merged_df[model1_pass <= low_prob_threshold]
    
    # Categorize questions by initial probability and training effect
    categories = {
        'low_prob_no_help': low_initial_prob[
            (low_initial_prob[f'rewards_mean_{model2_name}'] <= low_initial_prob[f'rewards_mean_{model1_name}'])
        ],
        'low_prob_helped': low_initial_prob[
            (low_initial_prob[f'rewards_mean_{model2_name}'] > low_initial_prob[f'rewards_mean_{model1_name}'])
        ],
        'low_prob_made_worse': low_initial_prob[
            (low_initial_prob[f'rewards_mean_{model2_name}'] < low_initial_prob[f'rewards_mean_{model1_name}'])
        ],
        'high_prob_maintained': merged_df[
            (model1_pass > low_prob_threshold) & 
            (abs(pass_rate_change) <= 0.05)  # Small change threshold
        ],
        'high_prob_improved': merged_df[
            (model1_pass > low_prob_threshold) & 
            (pass_rate_change > 0.05)
        ],
        'high_prob_degraded': merged_df[
            (model1_pass > low_prob_threshold) & 
            (pass_rate_change < -0.05)
        ]
    }
    
    # Calculate statistics for each category
    analysis = {}
    total_questions = len(merged_df)
    
    for category_name, category_df in categories.items():
        if len(category_df) > 0:
            analysis[category_name] = {
                'count': len(category_df),
                'percentage': len(category_df) / total_questions * 100,
                'mean_initial_pass_rate': float(category_df[f'rewards_mean_{model1_name}'].mean()),
                'mean_final_pass_rate': float(category_df[f'rewards_mean_{model2_name}'].mean()),
                'mean_change': float((category_df[f'rewards_mean_{model2_name}'] - category_df[f'rewards_mean_{model1_name}']).mean()),
                'examples': []
            }
            
            # Add examples for problematic categories
            if category_name in ['low_prob_no_help', 'low_prob_made_worse', 'high_prob_degraded']:
                # Sort by how much worse they got (or stayed bad)
                if category_name == 'low_prob_no_help':
                    sorted_examples = category_df.nsmallest(3, f'rewards_mean_{model2_name}')
                else:
                    # Calculate the difference column for sorting
                    category_df_copy = category_df.copy()
                    category_df_copy['reward_diff'] = category_df_copy[f'rewards_mean_{model2_name}'] - category_df_copy[f'rewards_mean_{model1_name}']
                    sorted_examples = category_df_copy.nsmallest(3, 'reward_diff')
                
                analysis[category_name]['examples'] = [
                    {
                        'prompt': row['prompt'][:200] + '...',
                        f'{model1_name}_pass_rate': float(row[f'rewards_mean_{model1_name}']),
                        f'{model2_name}_pass_rate': float(row[f'rewards_mean_{model2_name}']),
                        'change': float(row[f'rewards_mean_{model2_name}'] - row[f'rewards_mean_{model1_name}'])
                    }
                    for _, row in sorted_examples.iterrows()
                ]
    
    # Overall statistics
    low_prob_questions = len(low_initial_prob)
    analysis['summary'] = {
        'total_questions': total_questions,
        'low_initial_prob_count': low_prob_questions,
        'low_initial_prob_percentage': low_prob_questions / total_questions * 100,
        'low_prob_threshold': low_prob_threshold,
        'overall_pass_rate_change': float(pass_rate_change.mean()),
        'questions_where_training_hurt': int((pass_rate_change < 0).sum()),
        'questions_where_training_helped': int((pass_rate_change > 0).sum()),
        'questions_unchanged': int((pass_rate_change == 0).sum())
    }
    
    return analysis

def extract_raw_responses_for_regression(merged_df, base_model_name, finetuned_model_name, 
                                        min_base_reward=0.5, max_finetuned_reward=0.2, max_samples=10):
    """
    Extract raw responses for questions where base model did well but fine-tuned model failed.
    
    Args:
        merged_df: DataFrame with merged results from both models
        base_model_name: Name of the base model
        finetuned_model_name: Name of the fine-tuned model
        min_base_reward: Minimum reward threshold for base model to be considered "good"
        max_finetuned_reward: Maximum reward threshold for fine-tuned model to be considered "bad"
        max_samples: Maximum number of samples to extract
    
    Returns:
        dict: Raw responses for regression cases
    """
    # Find questions where base model did well but fine-tuned model failed
    regression_mask = (
        (merged_df[f'rewards_mean_{base_model_name}'] >= min_base_reward) &
        (merged_df[f'rewards_mean_{finetuned_model_name}'] <= max_finetuned_reward)
    )
    
    regression_df = merged_df[regression_mask]
    
    if len(regression_df) == 0:
        return {
            'regression_cases': [],
            'total_regression_questions': 0,
            'criteria': {
                'min_base_reward': min_base_reward,
                'max_finetuned_reward': max_finetuned_reward
            }
        }
    
    # Sort by the difference to get the worst regressions first
    regression_df = regression_df.copy()
    regression_df['regression_severity'] = (
        regression_df[f'rewards_mean_{base_model_name}'] - 
        regression_df[f'rewards_mean_{finetuned_model_name}']
    )
    regression_df = regression_df.nlargest(max_samples, 'regression_severity')
    
    raw_responses = []
    
    for _, row in regression_df.iterrows():
        # Get raw responses for both models
        base_responses = row.get(f'response_{base_model_name}', [])
        finetuned_responses = row.get(f'response_{finetuned_model_name}', [])
        base_rewards = row.get(f'rewards_{base_model_name}', [])
        finetuned_rewards = row.get(f'rewards_{finetuned_model_name}', [])
        
        # Convert to lists if needed
        for var_name, var_value in [('base_responses', base_responses), ('finetuned_responses', finetuned_responses),
                                   ('base_rewards', base_rewards), ('finetuned_rewards', finetuned_rewards)]:
            if not isinstance(var_value, list):
                if hasattr(var_value, '__iter__') and not isinstance(var_value, str):
                    locals()[var_name] = list(var_value) if var_value is not None else []
                else:
                    locals()[var_name] = [var_value] if var_value is not None else []
        
        # Create sample with raw responses
        sample = {
            'prompt': row['prompt'],
            'base_model_mean_reward': float(row[f'rewards_mean_{base_model_name}']),
            'finetuned_model_mean_reward': float(row[f'rewards_mean_{finetuned_model_name}']),
            'regression_severity': float(row['regression_severity']),
            'base_model_responses': [],
            'finetuned_model_responses': []
        }
        
        # Add all base model responses with their rewards
        for i, (response, reward) in enumerate(zip(base_responses, base_rewards)):
            sample['base_model_responses'].append({
                'response_index': i,
                'response_text': str(response),
                'reward': float(reward)
            })
        
        # Add all fine-tuned model responses with their rewards
        for i, (response, reward) in enumerate(zip(finetuned_responses, finetuned_rewards)):
            sample['finetuned_model_responses'].append({
                'response_index': i,
                'response_text': str(response),
                'reward': float(reward)
            })
        
        raw_responses.append(sample)
    
    return {
        'regression_cases': raw_responses,
        'total_regression_questions': len(regression_df),
        'criteria': {
            'min_base_reward': min_base_reward,
            'max_finetuned_reward': max_finetuned_reward
        },
        'statistics': {
            'mean_base_reward': float(regression_df[f'rewards_mean_{base_model_name}'].mean()),
            'mean_finetuned_reward': float(regression_df[f'rewards_mean_{finetuned_model_name}'].mean()),
            'mean_regression_severity': float(regression_df['regression_severity'].mean())
        }
    }

def extract_rewarded_samples(merged_df, model1_name, model2_name, categories_of_interest=None, max_samples_per_category=5):
    """
    Extract positively and negatively rewarded samples from questions of interest.
    
    Args:
        merged_df: DataFrame with merged results from both models
        model1_name: Name of the first model (baseline)
        model2_name: Name of the second model (trained)
        categories_of_interest: List of categories to focus on. If None, focuses on problematic categories
        max_samples_per_category: Maximum number of samples to extract per category
    
    Returns:
        dict: Extracted samples organized by category and reward type
    """
    if categories_of_interest is None:
        # Focus on problematic categories by default
        categories_of_interest = ['failed_both', 'solved_to_failed', 'low_prob_no_help', 'training_regression']
    
    # Define reward thresholds
    positive_reward_threshold = 0.7  # High reward
    negative_reward_threshold = 0.3  # Low reward
    
    extracted_samples = {}
    
    # Define category filters
    category_filters = {
        'failed_both': (
            (merged_df[f'rewards_mean_{model1_name}'] == 0) & 
            (merged_df[f'rewards_mean_{model2_name}'] == 0)
        ),
        'solved_to_failed': (
            (merged_df[f'rewards_mean_{model1_name}'] > 0) & 
            (merged_df[f'rewards_mean_{model2_name}'] == 0)
        ),
        'low_prob_no_help': (
            (merged_df[f'rewards_mean_{model1_name}'] <= 0.2) & 
            (merged_df[f'rewards_mean_{model2_name}'] <= merged_df[f'rewards_mean_{model1_name}'])
        ),
        'training_regression': (
            merged_df[f'rewards_mean_{model2_name}'] < merged_df[f'rewards_mean_{model1_name}']
        ),
        'significant_improvement': (
            (merged_df[f'rewards_mean_{model2_name}'] - merged_df[f'rewards_mean_{model1_name}']) > 0.3
        ),
        'partial_success': (
            (merged_df[f'rewards_mean_{model1_name}'] > 0) & 
            (merged_df[f'rewards_mean_{model1_name}'] < 1.0) &
            (merged_df[f'rewards_mean_{model2_name}'] > 0) & 
            (merged_df[f'rewards_mean_{model2_name}'] < 1.0)
        )
    }
    
    for category in categories_of_interest:
        if category not in category_filters:
            continue
            
        category_df = merged_df[category_filters[category]]
        
        if len(category_df) == 0:
            continue
            
        extracted_samples[category] = {
            'category_description': f'Samples from {category.replace("_", " ")} questions',
            'total_questions_in_category': len(category_df),
            'samples': {
                'positive_rewards': [],
                'negative_rewards': [],
                'mixed_rewards': []
            }
        }
        
        # Extract samples with different reward patterns
        for _, row in category_df.head(max_samples_per_category * 2).iterrows():  # Get more to have variety
            # Get the actual response samples if available
            model1_responses = row.get(f'response_{model1_name}', row.get('response', []))
            model2_responses = row.get(f'response_{model2_name}', row.get('response', []))
            model1_rewards = row.get(f'rewards_{model1_name}', row.get('rewards', []))
            model2_rewards = row.get(f'rewards_{model2_name}', row.get('rewards', []))
            
            # Handle case where responses/rewards might be in different format
            if not isinstance(model1_responses, list):
                if hasattr(model1_responses, '__iter__') and not isinstance(model1_responses, str):
                    model1_responses = list(model1_responses) if model1_responses is not None else []
                else:
                    model1_responses = [model1_responses] if model1_responses is not None else []
            
            if not isinstance(model2_responses, list):
                if hasattr(model2_responses, '__iter__') and not isinstance(model2_responses, str):
                    model2_responses = list(model2_responses) if model2_responses is not None else []
                else:
                    model2_responses = [model2_responses] if model2_responses is not None else []
            
            if not isinstance(model1_rewards, list):
                if hasattr(model1_rewards, '__iter__') and not isinstance(model1_rewards, str):
                    model1_rewards = list(model1_rewards) if model1_rewards is not None else []
                else:
                    model1_rewards = [model1_rewards] if model1_rewards is not None else []
            
            if not isinstance(model2_rewards, list):
                if hasattr(model2_rewards, '__iter__') and not isinstance(model2_rewards, str):
                    model2_rewards = list(model2_rewards) if model2_rewards is not None else []
                else:
                    model2_rewards = [model2_rewards] if model2_rewards is not None else []
            
            sample_data = {
                'prompt': row['prompt'][:300] + '...' if len(row['prompt']) > 300 else row['prompt'],
                f'{model1_name}_mean_reward': float(row[f'rewards_mean_{model1_name}']),
                f'{model2_name}_mean_reward': float(row[f'rewards_mean_{model2_name}']),
                'reward_change': float(row[f'rewards_mean_{model2_name}'] - row[f'rewards_mean_{model1_name}']),
                'samples': []
            }
            
            # Extract individual response-reward pairs
            for i, (resp1, resp2, rew1, rew2) in enumerate(zip(
                model1_responses[:3], model2_responses[:3], model1_rewards[:3], model2_rewards[:3]
            )):
                if i >= 3:  # Limit to first 3 samples per question
                    break
                    
                sample_data['samples'].append({
                    f'{model1_name}_response': resp1[:200] + '...' if len(str(resp1)) > 200 else str(resp1),
                    f'{model1_name}_reward': float(rew1),
                    f'{model2_name}_response': resp2[:200] + '...' if len(str(resp2)) > 200 else str(resp2),
                    f'{model2_name}_reward': float(rew2)
                })
            
            # Categorize by reward pattern
            avg_model1_reward = np.mean(model1_rewards) if model1_rewards else 0
            avg_model2_reward = np.mean(model2_rewards) if model2_rewards else 0
            
            if avg_model1_reward >= positive_reward_threshold or avg_model2_reward >= positive_reward_threshold:
                if len(extracted_samples[category]['samples']['positive_rewards']) < max_samples_per_category:
                    extracted_samples[category]['samples']['positive_rewards'].append(sample_data)
            elif avg_model1_reward <= negative_reward_threshold and avg_model2_reward <= negative_reward_threshold:
                if len(extracted_samples[category]['samples']['negative_rewards']) < max_samples_per_category:
                    extracted_samples[category]['samples']['negative_rewards'].append(sample_data)
            else:
                if len(extracted_samples[category]['samples']['mixed_rewards']) < max_samples_per_category:
                    extracted_samples[category]['samples']['mixed_rewards'].append(sample_data)
    
    return extracted_samples

def comprehensive_model_comparison(dataset_name, model1_name, model2_name, tolerance=1e-6, low_prob_threshold=0.2):
    """
    Compare two models and provide a comprehensive report on their performance differences.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'math500', 'aime25', etc.)
        model1_name (str): First model name ('base', 'entropy0.0', 'entropy0.01')
        model2_name (str): Second model name ('base', 'entropy0.0', 'entropy0.01')
        tolerance (float): Tolerance for considering rewards as equal (for stagnation detection)
        low_prob_threshold (float): Threshold for considering a question as having low initial probability
    
    Returns:
        dict: Comprehensive report with statistics and examples
    """
    base_path = Path('/u/rfechner/verl/workspace/rollouts/results')
    
    # Load data for both models
    model1_path = base_path / f'{dataset_name}_{model1_name}.parquet'
    model2_path = base_path / f'{dataset_name}_{model2_name}.parquet'
    
    if not model1_path.exists():
        raise FileNotFoundError(f"File not found: {model1_path}")
    if not model2_path.exists():
        raise FileNotFoundError(f"File not found: {model2_path}")
    
    df1 = pd.read_parquet(model1_path)
    df2 = pd.read_parquet(model2_path)
    
    # Calculate rewards mean for both models
    df1['rewards_mean'] = df1['rewards'].apply(np.mean)
    df2['rewards_mean'] = df2['rewards'].apply(np.mean)
    df1['prompt'] = df1['prompt'].apply(lambda x: x[0]['content'])
    df2['prompt'] = df2['prompt'].apply(lambda x: x[0]['content'])

    # Merge on prompt to compare same questions
    merged = pd.merge(df1, df2, on='prompt', suffixes=(f'_{model1_name}', f'_{model2_name}'))
    
    # Calculate performance difference
    merged['reward_diff'] = merged[f'rewards_mean_{model2_name}'] - merged[f'rewards_mean_{model1_name}']
    
    # Categorize questions
    failed_both = merged[
        (merged[f'rewards_mean_{model1_name}'] == 0) & 
        (merged[f'rewards_mean_{model2_name}'] == 0)
    ]
    
    improved = merged[merged['reward_diff'] > tolerance]
    got_worse = merged[merged['reward_diff'] < -tolerance]
    stagnated = merged[abs(merged['reward_diff']) <= tolerance]
    
    # Additional specific categories
    failed_to_solved = merged[
        (merged[f'rewards_mean_{model1_name}'] == 0) & 
        (merged[f'rewards_mean_{model2_name}'] > 0)
    ]
    
    solved_to_failed = merged[
        (merged[f'rewards_mean_{model1_name}'] > 0) & 
        (merged[f'rewards_mean_{model2_name}'] == 0)
    ]
    
    both_solved_improved = merged[
        (merged[f'rewards_mean_{model1_name}'] > 0) & 
        (merged[f'rewards_mean_{model2_name}'] > 0) & 
        (merged['reward_diff'] > tolerance)
    ]
    
    both_solved_worse = merged[
        (merged[f'rewards_mean_{model1_name}'] > 0) & 
        (merged[f'rewards_mean_{model2_name}'] > 0) & 
        (merged['reward_diff'] < -tolerance)
    ]
    
    # Calculate summary statistics
    total_questions = len(merged)
    
    # Create safe examples by selecting only the needed columns and ensuring they're serializable
    def create_safe_examples(df, columns):
        if len(df) == 0:
            return []
        # Only select the specific columns we need and convert to basic Python types
        safe_df = df[columns].copy()
        return safe_df.to_dict('records')
    
    example_columns_improved = ['prompt', f'rewards_mean_{model1_name}', f'rewards_mean_{model2_name}', 'reward_diff']
    example_columns_failed = ['prompt', f'rewards_mean_{model1_name}', f'rewards_mean_{model2_name}']
    
    # Analyze pass@1 rate changes
    pass_at_1_analysis = analyze_pass_at_1_changes(merged, model1_name, model2_name, low_prob_threshold)
    
    # Extract rewarded samples from questions of interest
    rewarded_samples = extract_rewarded_samples(merged, model1_name, model2_name)
    
    # Extract raw responses for regression cases (where base model did well but fine-tuned model failed)
    # This is especially relevant when model1 is 'base' and model2 is a fine-tuned version
    raw_regression_responses = extract_raw_responses_for_regression(
        merged, model1_name, model2_name, 
        min_base_reward=0.5, max_finetuned_reward=0.2, max_samples=15
    )
    
    report = {
        'dataset': dataset_name,
        'model1': model1_name,
        'model2': model2_name,
        'total_questions': total_questions,
        'summary_stats': {
            'model1_mean_reward': float(merged[f'rewards_mean_{model1_name}'].mean()),
            'model2_mean_reward': float(merged[f'rewards_mean_{model2_name}'].mean()),
            'mean_improvement': float(merged['reward_diff'].mean()),
            'std_improvement': float(merged['reward_diff'].std()),
        },
        'pass_at_1_analysis': pass_at_1_analysis,
        'rewarded_samples': rewarded_samples,
        'raw_regression_responses': raw_regression_responses,
        'categories': {
            'failed_both': {
                'count': len(failed_both),
                'percentage': len(failed_both) / total_questions * 100,
                'description': f'Questions where both {model1_name} and {model2_name} got 0 reward'
            },
            'improved': {
                'count': len(improved),
                'percentage': len(improved) / total_questions * 100,
                'description': f'Questions where {model2_name} performed better than {model1_name}',
                'mean_improvement': float(improved['reward_diff'].mean()) if len(improved) > 0 else 0.0
            },
            'got_worse': {
                'count': len(got_worse),
                'percentage': len(got_worse) / total_questions * 100,
                'description': f'Questions where {model2_name} performed worse than {model1_name}',
                'mean_degradation': float(got_worse['reward_diff'].mean()) if len(got_worse) > 0 else 0.0
            },
            'stagnated': {
                'count': len(stagnated),
                'percentage': len(stagnated) / total_questions * 100,
                'description': f'Questions where performance remained essentially the same'
            },
            'failed_to_solved': {
                'count': len(failed_to_solved),
                'percentage': len(failed_to_solved) / total_questions * 100,
                'description': f'Questions where {model1_name} failed (0 reward) but {model2_name} succeeded',
                'mean_reward_gained': float(failed_to_solved[f'rewards_mean_{model2_name}'].mean()) if len(failed_to_solved) > 0 else 0.0
            },
            'solved_to_failed': {
                'count': len(solved_to_failed),
                'percentage': len(solved_to_failed) / total_questions * 100,
                'description': f'Questions where {model1_name} succeeded but {model2_name} failed (0 reward)',
                'mean_reward_lost': float(solved_to_failed[f'rewards_mean_{model1_name}'].mean()) if len(solved_to_failed) > 0 else 0.0
            },
            'both_solved_improved': {
                'count': len(both_solved_improved),
                'percentage': len(both_solved_improved) / total_questions * 100,
                'description': f'Questions where both models succeeded but {model2_name} did better',
                'mean_improvement': float(both_solved_improved['reward_diff'].mean()) if len(both_solved_improved) > 0 else 0.0
            },
            'both_solved_worse': {
                'count': len(both_solved_worse),
                'percentage': len(both_solved_worse) / total_questions * 100,
                'description': f'Questions where both models succeeded but {model2_name} did worse',
                'mean_degradation': float(both_solved_worse['reward_diff'].mean()) if len(both_solved_worse) > 0 else 0.0
            }
        },
        'reward_distribution': {
            f'{model1_name}_zero_reward': int((merged[f'rewards_mean_{model1_name}'] == 0).sum()),
            f'{model2_name}_zero_reward': int((merged[f'rewards_mean_{model2_name}'] == 0).sum()),
            f'{model1_name}_perfect_reward': int((merged[f'rewards_mean_{model1_name}'] == 1.0).sum()),
            f'{model2_name}_perfect_reward': int((merged[f'rewards_mean_{model2_name}'] == 1.0).sum()),
        },
        'examples': {
            'biggest_improvements': create_safe_examples(improved.nlargest(3, 'reward_diff'), example_columns_improved),
            'biggest_degradations': create_safe_examples(got_worse.nsmallest(3, 'reward_diff'), example_columns_improved),
            'failed_both_examples': create_safe_examples(failed_both.head(3), example_columns_failed)
        }
    }
    
    return report

def print_comparison_report(report):
    """
    Print a nicely formatted comparison report.
    
    Args:
        report (dict): Report from comprehensive_model_comparison
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"Dataset: {report['dataset']}")
    print(f"Model 1: {report['model1']}")
    print(f"Model 2: {report['model2']}")
    print(f"Total Questions: {report['total_questions']}")
    
    print(f"\n{'SUMMARY STATISTICS':<30}")
    print(f"{'-'*50}")
    stats = report['summary_stats']
    print(f"Model 1 ({report['model1']}) Mean Reward: {stats['model1_mean_reward']:.4f}")
    print(f"Model 2 ({report['model2']}) Mean Reward: {stats['model2_mean_reward']:.4f}")
    print(f"Mean Improvement (Model 2 - Model 1): {stats['mean_improvement']:.4f}")
    print(f"Std Dev of Improvement: {stats['std_improvement']:.4f}")
    
    print(f"\n{'PERFORMANCE CATEGORIES':<30}")
    print(f"{'-'*80}")
    for category, info in report['categories'].items():
        print(f"{info['description']}")
        print(f"  Count: {info['count']} ({info['percentage']:.1f}%)")
        if 'mean_improvement' in info and info['mean_improvement'] != 0:
            print(f"  Mean Improvement: {info['mean_improvement']:.4f}")
        if 'mean_degradation' in info and info['mean_degradation'] != 0:
            print(f"  Mean Degradation: {info['mean_degradation']:.4f}")
        if 'mean_reward_gained' in info and info['mean_reward_gained'] != 0:
            print(f"  Mean Reward Gained: {info['mean_reward_gained']:.4f}")
        if 'mean_reward_lost' in info and info['mean_reward_lost'] != 0:
            print(f"  Mean Reward Lost: {info['mean_reward_lost']:.4f}")
        print()
    
    print(f"{'REWARD DISTRIBUTION':<30}")
    print(f"{'-'*50}")
    dist = report['reward_distribution']
    model1_name = report['model1']
    model2_name = report['model2']
    print(f"Model 1 ({model1_name}) - Zero Rewards: {dist[f'{model1_name}_zero_reward']}")
    print(f"Model 1 ({model1_name}) - Perfect Rewards: {dist[f'{model1_name}_perfect_reward']}")
    print(f"Model 2 ({model2_name}) - Zero Rewards: {dist[f'{model2_name}_zero_reward']}")
    print(f"Model 2 ({model2_name}) - Perfect Rewards: {dist[f'{model2_name}_perfect_reward']}")
    
    # Print pass@1 analysis
    if 'pass_at_1_analysis' in report:
        print(f"\n{'PASS@1 RATE ANALYSIS':<30}")
        print(f"{'-'*80}")
        pass_analysis = report['pass_at_1_analysis']
        
        if 'summary' in pass_analysis:
            summary = pass_analysis['summary']
            print(f"Low Initial Probability Questions: {summary['low_initial_prob_count']} ({summary['low_initial_prob_percentage']:.1f}%)")
            print(f"Threshold for Low Probability: {summary['low_prob_threshold']}")
            print(f"Overall Pass Rate Change: {summary['overall_pass_rate_change']:.4f}")
            print(f"Questions Where Training Helped: {summary['questions_where_training_helped']}")
            print(f"Questions Where Training Hurt: {summary['questions_where_training_hurt']}")
            print(f"Questions Unchanged: {summary['questions_unchanged']}")
            print()
        
        # Print category details
        for category, info in pass_analysis.items():
            if category != 'summary' and isinstance(info, dict):
                print(f"{category.replace('_', ' ').title()}:")
                print(f"  Count: {info['count']} ({info['percentage']:.1f}%)")
                print(f"  Mean Initial Pass Rate: {info['mean_initial_pass_rate']:.4f}")
                print(f"  Mean Final Pass Rate: {info['mean_final_pass_rate']:.4f}")
                print(f"  Mean Change: {info['mean_change']:.4f}")
                
                # Show examples for problematic categories
                if info['examples'] and category in ['low_prob_no_help', 'low_prob_made_worse', 'high_prob_degraded']:
                    print(f"  Examples:")
                    for i, ex in enumerate(info['examples'][:2], 1):  # Show top 2 examples
                        model1_name = report['model1']
                        model2_name = report['model2']
                        print(f"    {i}. Change: {ex['change']:.4f}")
                        print(f"       {model1_name}: {ex[f'{model1_name}_pass_rate']:.4f} → {model2_name}: {ex[f'{model2_name}_pass_rate']:.4f}")
                        print(f"       Prompt: {ex['prompt']}")
                print()
    
    # Print rewarded samples analysis
    if 'rewarded_samples' in report:
        print(f"\n{'REWARDED SAMPLES ANALYSIS':<30}")
        print(f"{'-'*80}")
        rewarded_samples = report['rewarded_samples']
        
        for category, data in rewarded_samples.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  {data['category_description']}")
            print(f"  Total questions in category: {data['total_questions_in_category']}")
            
            for reward_type, samples in data['samples'].items():
                if samples:
                    print(f"\n  {reward_type.replace('_', ' ').title()} ({len(samples)} samples):")
                    for i, sample in enumerate(samples[:2], 1):  # Show first 2 samples
                        model1_name = report['model1']
                        model2_name = report['model2']
                        print(f"    Sample {i}:")
                        print(f"      Reward Change: {sample['reward_change']:.4f}")
                        print(f"      {model1_name}: {sample[f'{model1_name}_mean_reward']:.4f} → {model2_name}: {sample[f'{model2_name}_mean_reward']:.4f}")
                        print(f"      Prompt: {sample['prompt']}")
                        
                        # Show individual response-reward pairs if available
                        if sample['samples']:
                            print(f"      Individual Responses:")
                            for j, resp_sample in enumerate(sample['samples'][:2], 1):  # First 2 response pairs
                                print(f"        Response {j}:")
                                print(f"          {model1_name} (reward: {resp_sample[f'{model1_name}_reward']:.2f}): {resp_sample[f'{model1_name}_response']}")
                                print(f"          {model2_name} (reward: {resp_sample[f'{model2_name}_reward']:.2f}): {resp_sample[f'{model2_name}_response']}")
                        print()
    
    # Print examples if available
    examples = report['examples']
    model1_name = report['model1']
    model2_name = report['model2']
    
    if examples['biggest_improvements']:
        print(f"\n{'BIGGEST IMPROVEMENTS (Top 3)':<30}")
        print(f"{'-'*80}")
        for i, ex in enumerate(examples['biggest_improvements'], 1):
            print(f"{i}. Improvement: {ex['reward_diff']:.4f}")
            print(f"   {model1_name}: {ex[f'rewards_mean_{model1_name}']:.4f} → {model2_name}: {ex[f'rewards_mean_{model2_name}']:.4f}")
            print(f"   Prompt: {ex['prompt'][:100]}...")
            print()
    
    if examples['biggest_degradations']:
        print(f"\n{'BIGGEST DEGRADATIONS (Top 3)':<30}")
        print(f"{'-'*80}")
        for i, ex in enumerate(examples['biggest_degradations'], 1):
            print(f"{i}. Degradation: {ex['reward_diff']:.4f}")
            print(f"   {model1_name}: {ex[f'rewards_mean_{model1_name}']:.4f} → {model2_name}: {ex[f'rewards_mean_{model2_name}']:.4f}")
            print(f"   Prompt: {ex['prompt'][:100]}...")
            print()
    
    # Print raw regression responses summary
    if 'raw_regression_responses' in report:
        print(f"\n{'RAW REGRESSION RESPONSES SUMMARY':<30}")
        print(f"{'-'*80}")
        regression_data = report['raw_regression_responses']
        print(f"Total regression questions: {regression_data['total_regression_questions']}")
        print(f"Criteria - Min {model1_name} reward: {regression_data['criteria']['min_base_reward']}")
        print(f"Criteria - Max {model2_name} reward: {regression_data['criteria']['max_finetuned_reward']}")
        
        if regression_data['regression_cases']:
            stats = regression_data['statistics']
            print(f"Mean {model1_name} reward: {stats['mean_base_reward']:.4f}")
            print(f"Mean {model2_name} reward: {stats['mean_finetuned_reward']:.4f}")
            print(f"Mean regression severity: {stats['mean_regression_severity']:.4f}")
            print(f"Raw response samples extracted: {len(regression_data['regression_cases'])}")
            
            # Show one example
            if regression_data['regression_cases']:
                print(f"\nExample regression case:")
                example = regression_data['regression_cases'][0]
                print(f"  Regression severity: {example['regression_severity']:.4f}")
                print(f"  {model1_name} reward: {example['base_model_mean_reward']:.4f}")
                print(f"  {model2_name} reward: {example['finetuned_model_mean_reward']:.4f}")
                print(f"  Prompt: {example['prompt'][:150]}...")
                print(f"  (Full responses saved to JSON file)")
        else:
            print(f"No regression cases found with the specified criteria")
        print()
            
    print(f"{'='*80}")

def compare_multiple_models(dataset_name, models=None):
    """
    Compare multiple models against the base model for a given dataset.
    Focus on base → entropy comparisons rather than cross-comparisons.
    
    Args:
        dataset_name (str): Name of the dataset
        models (list): List of model names to compare against base. If None, uses ['entropy0.0', 'entropy0.01']
    
    Returns:
        dict: Dictionary with base → model comparison reports
    """
    if models is None:
        models = ['entropy0.0', 'entropy0.01']
    
    results = {}
    base_model = 'base'
    
    # Perform base → fine-tuned model comparisons
    for model in models:
        comparison_key = f"{base_model}_to_{model}"
        try:
            report = comprehensive_model_comparison(dataset_name, base_model, model)
            results[comparison_key] = report
            print(f"✓ Completed comparison: {base_model} → {model}")
        except FileNotFoundError as e:
            print(f"✗ Skipped comparison {base_model} → {model}: {e}")
                    
    return results

if __name__ == "__main__":
    # Focus on base → entropy model comparisons for understanding training effects
    dataset = 'math500'
    
    print("="*80)
    print("BASE MODEL → ENTROPY REGULARIZED MODEL COMPARISONS")
    print("="*80)
    
    # Compare base → entropy0.0 and base → entropy0.01
    entropy_models = ['entropy0.0', 'entropy0.01']
    all_reports = {}
    
    for entropy_model in entropy_models:
        print(f"\n{'-'*60}")
        print(f"COMPARING: base → {entropy_model}")
        print(f"{'-'*60}")
        
        try:
            report = comprehensive_model_comparison(dataset, 'base', entropy_model, low_prob_threshold=0.2)
            all_reports[f'base_to_{entropy_model}'] = report
            print_comparison_report(report)
            print(f"✓ Completed comparison: base → {entropy_model}")
        except FileNotFoundError as e:
            print(f"✗ Skipped comparison base → {entropy_model}: {e}")
    
    # Save comprehensive results to JSON
    print("\n" + "="*80)
    print("SAVING COMPREHENSIVE RESULTS")
    print("="*80)
    
    # Combine all comparisons with metadata
    all_results = {
        'comparison_type': 'base_to_entropy_regularized',
        'dataset': dataset,
        'base_model': 'base',
        'entropy_models': entropy_models,
        'comparisons': all_reports,
        'metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_comparisons': len(all_reports),
            'focus': 'questions_where_base_succeeded_but_finetuned_failed',
            'raw_responses_included': True
        }
    }
    
    # Save to file
    output_file = Path('/u/rfechner/verl/workspace/judge/migration_analysis/base_to_entropy_comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary of regression cases
    print(f"\n{'REGRESSION CASES SUMMARY':<30}")
    print(f"{'-'*80}")
    
    for comparison_name, report in all_reports.items():
        if 'raw_regression_responses' in report:
            regression_data = report['raw_regression_responses']
            print(f"\n{comparison_name}:")
            print(f"  Total regression questions: {regression_data['total_regression_questions']}")
            print(f"  Criteria - Min base reward: {regression_data['criteria']['min_base_reward']}")
            print(f"  Criteria - Max finetuned reward: {regression_data['criteria']['max_finetuned_reward']}")
            
            if regression_data['regression_cases']:
                stats = regression_data['statistics']
                print(f"  Mean base reward: {stats['mean_base_reward']:.4f}")
                print(f"  Mean finetuned reward: {stats['mean_finetuned_reward']:.4f}")
                print(f"  Mean regression severity: {stats['mean_regression_severity']:.4f}")
                print(f"  Raw response samples extracted: {len(regression_data['regression_cases'])}")
            else:
                print(f"  No regression cases found with the specified criteria")
    
    print(f"\n{'='*80}")
    print("Analysis complete! Check the JSON file for detailed raw responses.")
    print(f"{'='*80}")