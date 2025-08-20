#!/usr/bin/env python3
"""
Deep Analysis Report Generator for Degraded Questions

This script generates a comprehensive report analyzing failed questions in detail,
including response analysis, mathematical reasoning breakdown, and failure patterns.
"""

import pandas as pd
import numpy as np
import json
import re
from collections import Counter, defaultdict
from load_degraded import load_and_preprocess_data
from pathlib import Path

def extract_mathematical_concepts(prompt_text):
    """Extract mathematical concepts and topics from prompt text."""
    
    # Define concept patterns
    concepts = {
        'algebra': [
            r'equation', r'solve', r'variable', r'polynomial', r'factor', r'expand',
            r'quadratic', r'linear', r'system', r'inequality'
        ],
        'geometry': [
            r'triangle', r'circle', r'angle', r'area', r'volume', r'perimeter',
            r'radius', r'diameter', r'polygon', r'coordinate', r'distance'
        ],
        'number_theory': [
            r'prime', r'gcd', r'lcm', r'modular', r'divisible', r'remainder',
            r'congruent', r'factor', r'integer', r'digit'
        ],
        'calculus': [
            r'derivative', r'integral', r'limit', r'differential', r'continuous',
            r'function', r'rate', r'maximum', r'minimum'
        ],
        'probability': [
            r'probability', r'permutation', r'combination', r'random', r'expected',
            r'variance', r'distribution', r'outcome', r'sample'
        ],
        'trigonometry': [
            r'sin', r'cos', r'tan', r'sec', r'csc', r'cot', r'angle', r'radian',
            r'degree', r'triangle'
        ],
        'combinatorics': [
            r'permutation', r'combination', r'arrangement', r'counting', r'ways',
            r'choose', r'selection', r'order'
        ],
        'sequences_series': [
            r'sequence', r'series', r'arithmetic', r'geometric', r'sum', r'term',
            r'progression', r'recursive'
        ]
    }
    
    detected_concepts = []
    prompt_lower = prompt_text.lower()
    
    for concept, patterns in concepts.items():
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                detected_concepts.append(concept)
                break  # Only add concept once
    
    return detected_concepts

def analyze_response_quality(response, ground_truth):
    """Analyze the quality and characteristics of a response."""
    
    analysis = {
        'length': len(response),
        'has_boxed_answer': bool(re.search(r'\\boxed\{[^}]*\}', response)),
        'has_step_by_step': 'step' in response.lower(),
        'has_mathematical_notation': bool(re.search(r'\\[a-zA-Z]', response)),
        'mentions_theorem': any(word in response.lower() for word in [
            'theorem', 'lemma', 'property', 'formula', 'identity'
        ]),
        'shows_calculation': bool(re.search(r'=|\\equiv|\\approx', response)),
        'has_error_indicators': any(phrase in response.lower() for phrase in [
            'error', 'mistake', 'wrong', 'incorrect', 'invalid'
        ])
    }
    
    # Extract potential final answer
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', response)
    if boxed_match:
        analysis['extracted_answer'] = boxed_match.group(1)
    else:
        # Try to find answer at the end
        lines = response.strip().split('\n')
        if lines:
            analysis['extracted_answer'] = lines[-1][:50]  # Last line, truncated
        else:
            analysis['extracted_answer'] = 'No clear answer'
    
    return analysis

def categorize_failure_modes(base_response, failed_response, ground_truth):
    """Categorize the type of failure by comparing successful and failed responses."""
    
    base_analysis = analyze_response_quality(base_response, ground_truth)
    failed_analysis = analyze_response_quality(failed_response, ground_truth)
    
    failure_modes = []
    
    # Compare key characteristics
    if base_analysis['has_boxed_answer'] and not failed_analysis['has_boxed_answer']:
        failure_modes.append('missing_final_answer')
    
    if base_analysis['shows_calculation'] and not failed_analysis['shows_calculation']:
        failure_modes.append('incomplete_calculation')
    
    if failed_analysis['has_error_indicators']:
        failure_modes.append('explicit_error_recognition')
    
    if failed_analysis['length'] < base_analysis['length'] * 0.5:
        failure_modes.append('truncated_response')
    elif failed_analysis['length'] > base_analysis['length'] * 2:
        failure_modes.append('overly_verbose')
    
    # Check for conceptual differences
    if base_analysis['mentions_theorem'] != failed_analysis['mentions_theorem']:
        failure_modes.append('different_approach')
    
    return failure_modes

def deep_analyze_degraded_questions(data):
    """Perform deep analysis of degraded questions."""
    
    print("🔍 DEEP ANALYSIS OF DEGRADED QUESTIONS")
    print("=" * 80)
    
    analysis_results = {
        'summary_statistics': {},
        'conceptual_analysis': {},
        'failure_mode_analysis': {},
        'detailed_question_breakdowns': [],
        'response_quality_comparison': {},
        'recommendations': []
    }
    
    # Focus on the most critical failures
    critical_failures = data['high_base_degraded_both']
    
    print(f"\n📊 Analyzing {len(critical_failures)} critical failure questions...")
    
    # Conceptual analysis
    concept_distribution = defaultdict(int)
    all_concepts = []
    
    for _, row in critical_failures.iterrows():
        concepts = extract_mathematical_concepts(row['prompt_text'])
        all_concepts.extend(concepts)
        for concept in concepts:
            concept_distribution[concept] += 1
    
    analysis_results['conceptual_analysis'] = {
        'concept_distribution': dict(concept_distribution),
        'most_vulnerable_concepts': sorted(concept_distribution.items(), 
                                         key=lambda x: x[1], reverse=True)[:5],
        'total_concepts_affected': len(concept_distribution)
    }
    
    # Failure mode analysis
    all_failure_modes = []
    
    print(f"\n🔬 Analyzing failure modes for each question...")
    
    for idx, (_, row) in enumerate(critical_failures.iterrows()):
        print(f"  Analyzing question {idx + 1}/{len(critical_failures)}...")
        
        # Get sample responses
        base_responses = row['responses_base']
        entropy0_responses = row['responses_entropy0']
        entropy01_responses = row['responses_entropy01']
        
        # Find successful base response and failed responses
        base_rewards = row['rewards_base']
        entropy0_rewards = row['rewards_entropy0']
        entropy01_rewards = row['rewards_entropy01']
        
        successful_base_idx = np.where(np.array(base_rewards) == 1.0)[0]
        failed_entropy0_idx = np.where(np.array(entropy0_rewards) == 0.0)[0]
        failed_entropy01_idx = np.where(np.array(entropy01_rewards) == 0.0)[0]
        
        question_analysis = {
            'question_idx': idx,
            'prompt': row['prompt_text'][:200] + '...',
            'ground_truth': row['answer_base'],
            'concepts': extract_mathematical_concepts(row['prompt_text']),
            'performance': {
                'base': float(row['rewards_mean_base']),
                'entropy0': float(row['rewards_mean_entropy0']),
                'entropy01': float(row['rewards_mean_entropy01'])
            },
            'degradation': {
                'entropy0': float(row['delta_entropy0']),
                'entropy01': float(row['delta_entropy01'])
            }
        }
        
        # Analyze failure modes if we have examples
        if len(successful_base_idx) > 0 and len(failed_entropy0_idx) > 0:
            base_response = base_responses[successful_base_idx[0]]
            failed_response = entropy0_responses[failed_entropy0_idx[0]]
            
            failure_modes = categorize_failure_modes(base_response, failed_response, row['answer_base'])
            question_analysis['entropy0_failure_modes'] = failure_modes
            all_failure_modes.extend(failure_modes)
            
            # Store sample responses for detailed analysis
            question_analysis['sample_responses'] = {
                'successful_base': base_response[:500] + '...' if len(base_response) > 500 else base_response,
                'failed_entropy0': failed_response[:500] + '...' if len(failed_response) > 500 else failed_response
            }
        
        if len(successful_base_idx) > 0 and len(failed_entropy01_idx) > 0:
            base_response = base_responses[successful_base_idx[0]]
            failed_response = entropy01_responses[failed_entropy01_idx[0]]
            
            failure_modes = categorize_failure_modes(base_response, failed_response, row['answer_base'])
            question_analysis['entropy01_failure_modes'] = failure_modes
            
            if 'sample_responses' not in question_analysis:
                question_analysis['sample_responses'] = {}
            question_analysis['sample_responses']['failed_entropy01'] = (
                failed_response[:500] + '...' if len(failed_response) > 500 else failed_response
            )
        
        analysis_results['detailed_question_breakdowns'].append(question_analysis)
    
    # Failure mode distribution
    failure_mode_counts = Counter(all_failure_modes)
    analysis_results['failure_mode_analysis'] = {
        'mode_distribution': dict(failure_mode_counts),
        'most_common_failures': failure_mode_counts.most_common(5),
        'total_failure_instances': len(all_failure_modes)
    }
    
    # Response quality comparison
    base_quality_scores = []
    entropy0_quality_scores = []
    entropy01_quality_scores = []
    
    for question in analysis_results['detailed_question_breakdowns']:
        if 'sample_responses' in question:
            # Simple quality scoring based on characteristics
            if 'successful_base' in question['sample_responses']:
                base_response = question['sample_responses']['successful_base']
                base_analysis = analyze_response_quality(base_response, question['ground_truth'])
                base_score = sum([
                    base_analysis['has_boxed_answer'],
                    base_analysis['has_step_by_step'],
                    base_analysis['shows_calculation'],
                    base_analysis['mentions_theorem']
                ])
                base_quality_scores.append(base_score)
            
            if 'failed_entropy0' in question['sample_responses']:
                failed_response = question['sample_responses']['failed_entropy0']
                failed_analysis = analyze_response_quality(failed_response, question['ground_truth'])
                failed_score = sum([
                    failed_analysis['has_boxed_answer'],
                    failed_analysis['has_step_by_step'],
                    failed_analysis['shows_calculation'],
                    failed_analysis['mentions_theorem']
                ])
                entropy0_quality_scores.append(failed_score)
    
    analysis_results['response_quality_comparison'] = {
        'base_mean_quality': np.mean(base_quality_scores) if base_quality_scores else 0,
        'entropy0_mean_quality': np.mean(entropy0_quality_scores) if entropy0_quality_scores else 0,
        'quality_degradation': (np.mean(base_quality_scores) - np.mean(entropy0_quality_scores)) 
                             if base_quality_scores and entropy0_quality_scores else 0
    }
    
    # Generate recommendations
    analysis_results['recommendations'] = generate_recommendations(analysis_results)
    
    # Summary statistics
    analysis_results['summary_statistics'] = {
        'total_critical_failures': len(critical_failures),
        'mean_base_performance': float(critical_failures['rewards_mean_base'].mean()),
        'mean_entropy0_performance': float(critical_failures['rewards_mean_entropy0'].mean()),
        'mean_entropy01_performance': float(critical_failures['rewards_mean_entropy01'].mean()),
        'worst_entropy0_degradation': float(critical_failures['delta_entropy0'].min()),
        'worst_entropy01_degradation': float(critical_failures['delta_entropy01'].min()),
        'concepts_analyzed': len(concept_distribution),
        'failure_modes_identified': len(failure_mode_counts)
    }
    
    return analysis_results

def generate_recommendations(analysis_results):
    """Generate actionable recommendations based on the analysis."""
    
    recommendations = []
    
    # Based on conceptual vulnerabilities
    vulnerable_concepts = analysis_results['conceptual_analysis']['most_vulnerable_concepts']
    if vulnerable_concepts:
        top_concept = vulnerable_concepts[0][0]
        recommendations.append({
            'category': 'curriculum_design',
            'priority': 'high',
            'recommendation': f"Focus training on {top_concept} problems - most vulnerable concept with {vulnerable_concepts[0][1]} failures",
            'implementation': f"Increase {top_concept} problem diversity in training data and add concept-specific validation"
        })
    
    # Based on failure modes
    common_failures = analysis_results['failure_mode_analysis']['most_common_failures']
    if common_failures:
        top_failure = common_failures[0][0]
        recommendations.append({
            'category': 'training_methodology',
            'priority': 'high',
            'recommendation': f"Address {top_failure} - most common failure mode ({common_failures[0][1]} instances)",
            'implementation': f"Implement specific training objectives to prevent {top_failure.replace('_', ' ')}"
        })
    
    # Based on response quality
    quality_degradation = analysis_results['response_quality_comparison']['quality_degradation']
    if quality_degradation > 1:
        recommendations.append({
            'category': 'response_generation',
            'priority': 'medium', 
            'recommendation': f"Significant response quality degradation detected ({quality_degradation:.2f} points)",
            'implementation': "Implement response quality metrics in training objectives and validation"
        })
    
    # General recommendations
    recommendations.extend([
        {
            'category': 'monitoring',
            'priority': 'medium',
            'recommendation': "Implement real-time degradation monitoring during training",
            'implementation': "Track performance on high-confidence base questions as early stopping criterion"
        },
        {
            'category': 'data_curation',
            'priority': 'low',
            'recommendation': "Curate training data to avoid systematic biases",
            'implementation': "Filter training examples that show similar failure patterns to degraded questions"
        }
    ])
    
    return recommendations

def generate_deep_analysis_report(analysis_results, save_path):
    """Generate a comprehensive markdown report."""
    
    report = []
    report.append("# 🔍 DEEP ANALYSIS REPORT: DEGRADED QUESTIONS")
    report.append("=" * 80)
    report.append("")
    
    # Executive Summary
    report.append("## 📋 Executive Summary")
    report.append("")
    stats = analysis_results['summary_statistics']
    report.append(f"**Critical Failures Analyzed:** {stats['total_critical_failures']} questions")
    report.append(f"**Performance Degradation:** {stats['mean_base_performance']:.3f} → {stats['mean_entropy0_performance']:.3f} (Δ: {stats['mean_base_performance'] - stats['mean_entropy0_performance']:.3f})")
    report.append(f"**Worst Single Degradation:** {abs(stats['worst_entropy0_degradation']):.3f} performance drop")
    report.append(f"**Mathematical Concepts Affected:** {stats['concepts_analyzed']} different areas")
    report.append(f"**Failure Modes Identified:** {stats['failure_modes_identified']} distinct patterns")
    report.append("")
    
    # Conceptual Vulnerability Analysis
    report.append("## 🎯 Mathematical Concept Vulnerability Analysis")
    report.append("")
    concept_analysis = analysis_results['conceptual_analysis']
    report.append("### Most Vulnerable Mathematical Areas:")
    for concept, count in concept_analysis['most_vulnerable_concepts']:
        percentage = count / stats['total_critical_failures'] * 100
        report.append(f"- **{concept.title().replace('_', ' ')}**: {count} failures ({percentage:.1f}%)")
    report.append("")
    
    # Failure Mode Analysis
    report.append("## ⚠️ Failure Mode Analysis")
    report.append("")
    failure_analysis = analysis_results['failure_mode_analysis']
    report.append("### Most Common Failure Patterns:")
    for mode, count in failure_analysis['most_common_failures']:
        report.append(f"- **{mode.replace('_', ' ').title()}**: {count} instances")
    report.append("")
    
    # Response Quality Analysis
    report.append("## 📊 Response Quality Comparison")
    report.append("")
    quality = analysis_results['response_quality_comparison']
    report.append(f"**Base Model Quality Score:** {quality['base_mean_quality']:.2f}/4.0")
    report.append(f"**Trained Model Quality Score:** {quality['entropy0_mean_quality']:.2f}/4.0")
    report.append(f"**Quality Degradation:** {quality['quality_degradation']:.2f} points")
    report.append("")
    
    # Detailed Question Analysis
    report.append("## 🔬 Detailed Question-by-Question Analysis")
    report.append("")
    
    for i, question in enumerate(analysis_results['detailed_question_breakdowns'][:5], 1):
        report.append(f"### Question {i}: {question['concepts'][0].title() if question['concepts'] else 'Mixed'} Problem")
        report.append("")
        report.append(f"**Problem:** {question['prompt']}")
        report.append(f"**Ground Truth:** {question['ground_truth']}")
        report.append(f"**Concepts:** {', '.join(question['concepts'])}")
        report.append("")
        
        perf = question['performance']
        report.append(f"**Performance:**")
        report.append(f"- Base Model: {perf['base']:.3f}")
        report.append(f"- Entropy 0.0: {perf['entropy0']:.3f} (Δ: {question['degradation']['entropy0']:.3f})")
        report.append(f"- Entropy 0.01: {perf['entropy01']:.3f} (Δ: {question['degradation']['entropy01']:.3f})")
        report.append("")
        
        if 'entropy0_failure_modes' in question:
            report.append(f"**Failure Modes:** {', '.join(question['entropy0_failure_modes'])}")
            report.append("")
        
        if 'sample_responses' in question:
            if 'successful_base' in question['sample_responses']:
                report.append("**Successful Base Response:**")
                report.append("```")
                report.append(question['sample_responses']['successful_base'])
                report.append("```")
                report.append("")
            
            if 'failed_entropy0' in question['sample_responses']:
                report.append("**Failed Entropy 0.0 Response:**")
                report.append("```")
                report.append(question['sample_responses']['failed_entropy0'])
                report.append("```")
                report.append("")
        
        report.append("---")
        report.append("")
    
    # Recommendations
    report.append("## 💡 Actionable Recommendations")
    report.append("")
    
    recommendations = analysis_results['recommendations']
    high_priority = [r for r in recommendations if r['priority'] == 'high']
    medium_priority = [r for r in recommendations if r['priority'] == 'medium']
    
    if high_priority:
        report.append("### 🚨 High Priority Actions")
        for i, rec in enumerate(high_priority, 1):
            report.append(f"{i}. **{rec['category'].title().replace('_', ' ')}**")
            report.append(f"   - **Issue:** {rec['recommendation']}")
            report.append(f"   - **Action:** {rec['implementation']}")
            report.append("")
    
    if medium_priority:
        report.append("### ⚡ Medium Priority Actions")
        for i, rec in enumerate(medium_priority, 1):
            report.append(f"{i}. **{rec['category'].title().replace('_', ' ')}**")
            report.append(f"   - **Issue:** {rec['recommendation']}")
            report.append(f"   - **Action:** {rec['implementation']}")
            report.append("")
    
    # Conclusion
    report.append("## 🎯 Key Takeaways")
    report.append("")
    report.append("1. **Systematic degradation patterns** suggest training artifacts rather than random failures")
    report.append("2. **Specific mathematical concepts** are disproportionately vulnerable")
    report.append("3. **Response quality degradation** indicates fundamental reasoning changes")
    report.append("4. **Targeted interventions** can address identified failure modes")
    report.append("")
    report.append("---")
    report.append(f"*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    """Main execution function."""
    
    print("🚀 STARTING DEEP ANALYSIS OF DEGRADED QUESTIONS")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading degraded questions data...")
    data = load_and_preprocess_data()
    
    # Perform deep analysis
    print("\n2. Performing deep analysis...")
    analysis_results = deep_analyze_degraded_questions(data)
    
    # Generate reports
    print("\n3. Generating comprehensive report...")
    
    # Save detailed JSON analysis
    json_path = '/u/rfechner/verl/workspace/judge/migration_analysis/degraded_answers/deep_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Detailed JSON analysis saved: {json_path}")
    
    # Generate markdown report
    report_path = '/u/rfechner/verl/workspace/judge/migration_analysis/degraded_answers/DEEP_ANALYSIS_REPORT.md'
    report_content = generate_deep_analysis_report(analysis_results, report_path)
    print(f"   ✓ Comprehensive report saved: {report_path}")
    
    # Print summary
    print("\n4. Analysis Summary:")
    print("=" * 50)
    stats = analysis_results['summary_statistics']
    print(f"   📊 Questions analyzed: {stats['total_critical_failures']}")
    print(f"   📉 Performance drop: {stats['mean_base_performance'] - stats['mean_entropy0_performance']:.3f}")
    print(f"   🎯 Concepts affected: {stats['concepts_analyzed']}")
    print(f"   ⚠️  Failure modes: {stats['failure_modes_identified']}")
    
    concept_analysis = analysis_results['conceptual_analysis']
    if concept_analysis['most_vulnerable_concepts']:
        top_concept = concept_analysis['most_vulnerable_concepts'][0]
        print(f"   🔥 Most vulnerable: {top_concept[0]} ({top_concept[1]} failures)")
    
    failure_analysis = analysis_results['failure_mode_analysis']
    if failure_analysis['most_common_failures']:
        top_failure = failure_analysis['most_common_failures'][0]
        print(f"   💥 Common failure: {top_failure[0]} ({top_failure[1]} instances)")
    
    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"📄 View full report: {report_path}")
    print(f"📊 View detailed data: {json_path}")

if __name__ == "__main__":
    main()
