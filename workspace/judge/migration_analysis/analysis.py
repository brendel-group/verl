#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_passrate_migrations():
    """
    Analyze how questions migrate between solvable/unsolvable states across training methods.
    """
    # Load the results
    with open('/u/rfechner/verl/workspace/judge/migration_analysis/base_to_entropy_comparison_results.json', 'r') as f:
        data = json.load(f)
    
    # Extract the key data
    base_entropy0 = data['comparisons']['base_to_entropy0.0']
    base_entropy01 = data['comparisons']['base_to_entropy0.01']
    
    # Load raw data for detailed analysis
    base_path = Path('/u/rfechner/verl/workspace/rollouts/results')
    
    df_base = pd.read_parquet(base_path / 'math500_base.parquet')
    df_entropy0 = pd.read_parquet(base_path / 'math500_entropy0.0.parquet')
    df_entropy01 = pd.read_parquet(base_path / 'math500_entropy0.01.parquet')
    
    # Process data
    for df in [df_base, df_entropy0, df_entropy01]:
        df['rewards_mean'] = df['rewards'].apply(np.mean)
        df['prompt'] = df['prompt'].apply(lambda x: x[0]['content'])
    
    # Merge all three datasets
    merged = df_base.merge(df_entropy0, on='prompt', suffixes=('_base', '_entropy0'))
    merged = merged.merge(df_entropy01, on='prompt')
    merged = merged.rename(columns={
        'rewards_mean': 'rewards_mean_entropy01',
        'response': 'response_entropy01',
        'rewards': 'rewards_entropy01'
    })
    
    # Define high pass rate threshold
    high_passrate_threshold = 0.7
    
    analysis = {}
    
    # 1. High passrate in base, low in entropy0.0
    high_base_low_entropy0 = merged[
        (merged['rewards_mean_base'] >= high_passrate_threshold) & 
        (merged['rewards_mean_entropy0'] < high_passrate_threshold)
    ].copy()
    
    # 2. High passrate in base, low in entropy0.01
    high_base_low_entropy01 = merged[
        (merged['rewards_mean_base'] >= high_passrate_threshold) & 
        (merged['rewards_mean_entropy01'] < high_passrate_threshold)
    ].copy()
    
    # 3. Calculate passrate deltas
    merged['delta_entropy0'] = merged['rewards_mean_entropy0'] - merged['rewards_mean_base']
    merged['delta_entropy01'] = merged['rewards_mean_entropy01'] - merged['rewards_mean_base']
    
    # Also add deltas to the filtered dataframes
    high_base_low_entropy0['delta_entropy0'] = high_base_low_entropy0['rewards_mean_entropy0'] - high_base_low_entropy0['rewards_mean_base']
    high_base_low_entropy01['delta_entropy01'] = high_base_low_entropy01['rewards_mean_entropy01'] - high_base_low_entropy01['rewards_mean_base']
    
    # Top 10 improvements and degradations for each method
    top_10_improvements_entropy0 = merged.nlargest(10, 'delta_entropy0')
    bottom_10_degradations_entropy0 = merged.nsmallest(10, 'delta_entropy0')
    
    top_10_improvements_entropy01 = merged.nlargest(10, 'delta_entropy01')
    bottom_10_degradations_entropy01 = merged.nsmallest(10, 'delta_entropy01')
    
    # 4. Questions hard in both GRPO methods but easy in base
    easy_base_hard_both = merged[
        (merged['rewards_mean_base'] >= high_passrate_threshold) & 
        (merged['rewards_mean_entropy0'] < high_passrate_threshold) & 
        (merged['rewards_mean_entropy01'] < high_passrate_threshold)
    ].copy()
    
    # 5. Migration analysis - categorize questions by their journey
    def categorize_migration(row):
        base_high = row['rewards_mean_base'] >= high_passrate_threshold
        entropy0_high = row['rewards_mean_entropy0'] >= high_passrate_threshold
        entropy01_high = row['rewards_mean_entropy01'] >= high_passrate_threshold
        
        if base_high and entropy0_high and entropy01_high:
            return "stayed_solvable"
        elif not base_high and not entropy0_high and not entropy01_high:
            return "stayed_unsolvable"
        elif base_high and not entropy0_high and not entropy01_high:
            return "became_unsolvable_both"
        elif not base_high and entropy0_high and entropy01_high:
            return "became_solvable_both"
        elif base_high and entropy0_high and not entropy01_high:
            return "entropy01_only_failed"
        elif base_high and not entropy0_high and entropy01_high:
            return "entropy0_only_failed"
        elif not base_high and entropy0_high and not entropy01_high:
            return "entropy0_only_solved"
        elif not base_high and not entropy0_high and entropy01_high:
            return "entropy01_only_solved"
        else:
            return "mixed_pattern"
    
    merged['migration_category'] = merged.apply(categorize_migration, axis=1)
    migration_counts = merged['migration_category'].value_counts()
    
    # Compile results
    analysis = {
        'high_base_low_entropy0': {
            'count': len(high_base_low_entropy0),
            'percentage': len(high_base_low_entropy0) / len(merged) * 100,
            'mean_base_passrate': float(high_base_low_entropy0['rewards_mean_base'].mean()),
            'mean_entropy0_passrate': float(high_base_low_entropy0['rewards_mean_entropy0'].mean()),
            'mean_degradation': float((high_base_low_entropy0['rewards_mean_entropy0'] - high_base_low_entropy0['rewards_mean_base']).mean()),
            'examples': [
                {
                    'prompt': row['prompt'][:150] + '...',
                    'base_passrate': float(row['rewards_mean_base']),
                    'entropy0_passrate': float(row['rewards_mean_entropy0']),
                    'degradation': float(row['rewards_mean_entropy0'] - row['rewards_mean_base'])
                }
                for _, row in high_base_low_entropy0.nsmallest(5, 'delta_entropy0').iterrows()
            ]
        },
        
        'high_base_low_entropy01': {
            'count': len(high_base_low_entropy01),
            'percentage': len(high_base_low_entropy01) / len(merged) * 100,
            'mean_base_passrate': float(high_base_low_entropy01['rewards_mean_base'].mean()),
            'mean_entropy01_passrate': float(high_base_low_entropy01['rewards_mean_entropy01'].mean()),
            'mean_degradation': float((high_base_low_entropy01['rewards_mean_entropy01'] - high_base_low_entropy01['rewards_mean_base']).mean()),
            'examples': [
                {
                    'prompt': row['prompt'][:150] + '...',
                    'base_passrate': float(row['rewards_mean_base']),
                    'entropy01_passrate': float(row['rewards_mean_entropy01']),
                    'degradation': float(row['rewards_mean_entropy01'] - row['rewards_mean_base'])
                }
                for _, row in high_base_low_entropy01.nsmallest(5, 'delta_entropy01').iterrows()
            ]
        },
        
        'top_improvements_entropy0': [
            {
                'prompt': row['prompt'][:150] + '...',
                'base_passrate': float(row['rewards_mean_base']),
                'entropy0_passrate': float(row['rewards_mean_entropy0']),
                'improvement': float(row['delta_entropy0'])
            }
            for _, row in top_10_improvements_entropy0.iterrows()
        ],
        
        'top_degradations_entropy0': [
            {
                'prompt': row['prompt'][:150] + '...',
                'base_passrate': float(row['rewards_mean_base']),
                'entropy0_passrate': float(row['rewards_mean_entropy0']),
                'degradation': float(row['delta_entropy0'])
            }
            for _, row in bottom_10_degradations_entropy0.iterrows()
        ],
        
        'top_improvements_entropy01': [
            {
                'prompt': row['prompt'][:150] + '...',
                'base_passrate': float(row['rewards_mean_base']),
                'entropy01_passrate': float(row['rewards_mean_entropy01']),
                'improvement': float(row['delta_entropy01'])
            }
            for _, row in top_10_improvements_entropy01.iterrows()
        ],
        
        'top_degradations_entropy01': [
            {
                'prompt': row['prompt'][:150] + '...',
                'base_passrate': float(row['rewards_mean_base']),
                'entropy01_passrate': float(row['rewards_mean_entropy01']),
                'degradation': float(row['delta_entropy01'])
            }
            for _, row in bottom_10_degradations_entropy01.iterrows()
        ],
        
        'easy_base_hard_both': {
            'count': len(easy_base_hard_both),
            'percentage': len(easy_base_hard_both) / len(merged) * 100,
            'mean_base_passrate': float(easy_base_hard_both['rewards_mean_base'].mean()),
            'mean_entropy0_passrate': float(easy_base_hard_both['rewards_mean_entropy0'].mean()),
            'mean_entropy01_passrate': float(easy_base_hard_both['rewards_mean_entropy01'].mean()),
            'examples': [
                {
                    'prompt': row['prompt'][:150] + '...',
                    'base_passrate': float(row['rewards_mean_base']),
                    'entropy0_passrate': float(row['rewards_mean_entropy0']),
                    'entropy01_passrate': float(row['rewards_mean_entropy01']),
                    'entropy0_degradation': float(row['delta_entropy0']),
                    'entropy01_degradation': float(row['delta_entropy01'])
                }
                for _, row in easy_base_hard_both.iterrows()
            ]
        },
        
        'migration_analysis': {
            'categories': {cat: int(count) for cat, count in migration_counts.items()},
            'total_questions': len(merged),
            'high_passrate_threshold': high_passrate_threshold
        }
    }
    
    return analysis

def print_detailed_analysis():
    """Print the detailed passrate migration analysis."""
    
    print("="*100)
    print("DETAILED PASSRATE MIGRATION ANALYSIS")
    print("="*100)
    
    analysis = analyze_passrate_migrations()
    
    print(f"\nAnalysis based on high passrate threshold: {analysis['migration_analysis']['high_passrate_threshold']}")
    print(f"Total questions analyzed: {analysis['migration_analysis']['total_questions']}")
    
    # Requirement 1: High passrate in base, low in GRPO (entropy0.0)
    print(f"\n{'-'*80}")
    print("1. QUESTIONS HIGH PASSRATE IN BASE, LOW IN GRPO (entropy0.0)")
    print(f"{'-'*80}")
    
    hb_le0 = analysis['high_base_low_entropy0']
    print(f"Count: {hb_le0['count']} ({hb_le0['percentage']:.1f}% of all questions)")
    print(f"Mean base passrate: {hb_le0['mean_base_passrate']:.4f}")
    print(f"Mean entropy0.0 passrate: {hb_le0['mean_entropy0_passrate']:.4f}")
    print(f"Mean degradation: {hb_le0['mean_degradation']:.4f}")
    
    print("\nWorst 5 degradations:")
    for i, ex in enumerate(hb_le0['examples'][:5], 1):
        print(f"  {i}. Degradation: {ex['degradation']:.4f}")
        print(f"     Base: {ex['base_passrate']:.4f} → Entropy0.0: {ex['entropy0_passrate']:.4f}")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    # Requirement 2: High passrate in base, low in entropy-regularized GRPO (entropy0.01)
    print(f"\n{'-'*80}")
    print("2. QUESTIONS HIGH PASSRATE IN BASE, LOW IN ENTROPY-REG GRPO (entropy0.01)")
    print(f"{'-'*80}")
    
    hb_le01 = analysis['high_base_low_entropy01']
    print(f"Count: {hb_le01['count']} ({hb_le01['percentage']:.1f}% of all questions)")
    print(f"Mean base passrate: {hb_le01['mean_base_passrate']:.4f}")
    print(f"Mean entropy0.01 passrate: {hb_le01['mean_entropy01_passrate']:.4f}")
    print(f"Mean degradation: {hb_le01['mean_degradation']:.4f}")
    
    print("\nWorst 5 degradations:")
    for i, ex in enumerate(hb_le01['examples'][:5], 1):
        print(f"  {i}. Degradation: {ex['degradation']:.4f}")
        print(f"     Base: {ex['base_passrate']:.4f} → Entropy0.01: {ex['entropy01_passrate']:.4f}")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    # Requirement 3: Migration analysis - top/bottom 10 changes
    print(f"\n{'-'*80}")
    print("3. PASSRATE MIGRATION ANALYSIS - TOP/BOTTOM 10 CHANGES")
    print(f"{'-'*80}")
    
    print("\n3a. GRPO (entropy0.0) - TOP 10 IMPROVEMENTS:")
    for i, ex in enumerate(analysis['top_improvements_entropy0'][:10], 1):
        print(f"  {i}. Improvement: {ex['improvement']:.4f}")
        print(f"     Base: {ex['base_passrate']:.4f} → Entropy0.0: {ex['entropy0_passrate']:.4f}")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    print("\n3b. GRPO (entropy0.0) - BOTTOM 10 DEGRADATIONS:")
    for i, ex in enumerate(analysis['top_degradations_entropy0'][:10], 1):
        print(f"  {i}. Degradation: {ex['degradation']:.4f}")
        print(f"     Base: {ex['base_passrate']:.4f} → Entropy0.0: {ex['entropy0_passrate']:.4f}")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    print("\n3c. ENTROPY-REG GRPO (entropy0.01) - TOP 10 IMPROVEMENTS:")
    for i, ex in enumerate(analysis['top_improvements_entropy01'][:10], 1):
        print(f"  {i}. Improvement: {ex['improvement']:.4f}")
        print(f"     Base: {ex['base_passrate']:.4f} → Entropy0.01: {ex['entropy01_passrate']:.4f}")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    print("\n3d. ENTROPY-REG GRPO (entropy0.01) - BOTTOM 10 DEGRADATIONS:")
    for i, ex in enumerate(analysis['top_degradations_entropy01'][:10], 1):
        print(f"  {i}. Degradation: {ex['degradation']:.4f}")
        print(f"     Base: {ex['base_passrate']:.4f} → Entropy0.01: {ex['entropy01_passrate']:.4f}")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    # Requirement 4: Questions hard in both GRPO methods but easy in base
    print(f"\n{'-'*80}")
    print("4. QUESTIONS EASY IN BASE BUT HARD IN BOTH GRPO METHODS")
    print(f"{'-'*80}")
    
    eb_hb = analysis['easy_base_hard_both']
    print(f"Count: {eb_hb['count']} ({eb_hb['percentage']:.1f}% of all questions)")
    print(f"Mean base passrate: {eb_hb['mean_base_passrate']:.4f}")
    print(f"Mean entropy0.0 passrate: {eb_hb['mean_entropy0_passrate']:.4f}")
    print(f"Mean entropy0.01 passrate: {eb_hb['mean_entropy01_passrate']:.4f}")
    
    print(f"\nAll {eb_hb['count']} questions that regressed in both methods:")
    for i, ex in enumerate(eb_hb['examples'], 1):
        print(f"  {i}. Base: {ex['base_passrate']:.4f}")
        print(f"     → Entropy0.0: {ex['entropy0_passrate']:.4f} (Δ: {ex['entropy0_degradation']:.4f})")
        print(f"     → Entropy0.01: {ex['entropy01_passrate']:.4f} (Δ: {ex['entropy01_degradation']:.4f})")
        print(f"     Problem: {ex['prompt']}")
        print()
    
    # Migration summary
    print(f"\n{'-'*80}")
    print("5. OVERALL MIGRATION SUMMARY")
    print(f"{'-'*80}")
    
    migration = analysis['migration_analysis']['categories']
    total = analysis['migration_analysis']['total_questions']
    
    print("Question migration patterns:")
    for category, count in migration.items():
        percentage = count / total * 100
        print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\n{'='*100}")
    print("DETAILED ANALYSIS COMPLETE")
    print(f"{'='*100}")
    
    return analysis

if __name__ == "__main__":
    # Run the detailed analysis
    analysis_results = print_detailed_analysis()
    
    # Save the detailed analysis
    # Save results
    output_file = '/u/rfechner/verl/workspace/judge/migration_analysis/detailed_passrate_migration_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to: {output_file}")
