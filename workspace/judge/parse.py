"""
Simplified Response Parser for Error Categorization

This module parses and evaluates responses from a language model tasked with 
categorizing mathematical errors. It handles both single questions and multiple entries.

Key Functions:
- parse_error_category(): Extracts error categories from response text
- analyze_single_entry(): Evaluates responses for a single question
- analyze_multiple_entries(): Evaluates responses for multiple questions  
- auto_analyze(): Auto-detects format and runs appropriate analysis

Error Categories:
1. Computational Error - arithmetic mistakes, algebraic manipulation errors
2. Conceptual Misunderstanding - wrong approach, misunderstanding the problem  
3. Incomplete Solution - stopped too early, didn't finish the problem
4. Wrong Method - used inappropriate technique or formula
5. Logic Error - flawed reasoning or invalid steps

Usage Examples:
    # Auto-detect and analyze any file
    result = auto_analyze("generated_responses.json")
    
    # Manual analysis
    result = analyze_single_entry("generated_responses_20.json")
    result = analyze_multiple_entries("generated_responses_100.json")
"""

import json
import re
from typing import Dict, List, Tuple, Optional
import os

def parse_error_category(response_text: str) -> Optional[str]:
    """
    Parse the error category from a response text.
    
    Args:
        response_text: The generated text from the model
        
    Returns:
        The error category if found, None otherwise
    """
    # Define the valid error categories
    error_categories = [
        "Computational Error",
        "Conceptual Misunderstanding", 
        "Incomplete Solution",
        "Wrong Method",
        "Logic Error",
        "Mislabeling"
    ]
    
    # Look for boxed format first - handle both \\boxed{} and \(\boxed{}\) formats
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',           # Standard \boxed{} format
        r'\\\(\s*\\boxed\{([^}]+)\}\s*\\\)',  # LaTeX math mode \(\boxed{}\)
        r'\$\s*\\boxed\{([^}]+)\}\s*\$',      # Dollar sign math mode $\boxed{}$
    ]
    
    for pattern in boxed_patterns:
        boxed_match = re.search(pattern, response_text, re.IGNORECASE)
        if boxed_match:
            category = boxed_match.group(1).strip()
            # Clean up the category text
            category = re.sub(r'\\text\{([^}]+)\}', r'\1', category)
            category = category.strip()
            
            # Check if the cleaned category matches one of our valid categories
            for valid_category in error_categories:
                if valid_category.lower() in category.lower():
                    return valid_category
            
            # If it's a number (like "8"), it's likely the correct answer, not a category
            if category.isdigit():
                return None
                
            return category
    
    # Look for explicit categorization statements
    for category in error_categories:
        # Look for patterns like "Final Categorization: X" or "Error Type: X"
        patterns = [
            rf"Final Categorization[:\s]*[\*]*{re.escape(category)}",
            rf"Error Type[:\s]*[\*]*{re.escape(category)}",
            rf"categorization[:\s]*[\*]*{re.escape(category)}",
            rf"main category[:\s]*[\*]*{re.escape(category)}",
            rf"primary[:\s]+category[:\s]*[\*]*{re.escape(category)}",
            rf"The[:\s]+final[:\s]+categorization[:\s]*[\*]*{re.escape(category)}",
            rf"categorization.*?{re.escape(category)}",
            rf"The.*?categorization.*?{re.escape(category)}",
            rf"\*\*{re.escape(category)}\*\*",  # **Category Name**
            rf"1\.\s*\*\*{re.escape(category)}\*\*",  # 1. **Category Name**
            rf"Final Categorization:\s*\n\s*1\.\s*\*\*{re.escape(category)}\*\*",  # Multi-line format
        ]
        
        for pattern in patterns:
            if re.search(pattern, response_text, re.IGNORECASE | re.DOTALL):
                return category
    
    # Look for standalone category mentions in specific contexts
    for category in error_categories:
        # Look for patterns where the category is mentioned as a final answer
        final_patterns = [
            rf"Final.*?{re.escape(category)}",
            rf"Thus.*?{re.escape(category)}",
            rf"Therefore.*?{re.escape(category)}",
            rf"The.*?error.*?{re.escape(category)}",
            rf"primary.*?{re.escape(category)}",
        ]
        
        for pattern in final_patterns:
            if re.search(pattern, response_text, re.IGNORECASE | re.DOTALL):
                return category
    
    # Look for the category mentioned at the end of the response (last 300 chars)
    for category in error_categories:
        if category.lower() in response_text[-300:].lower():
            return category
    
    # Fallback: For incomplete responses, count category mentions and return the most frequent
    category_counts = {}
    for category in error_categories:
        count = len(re.findall(re.escape(category.lower()), response_text.lower()))
        if count > 0:
            category_counts[category] = count
    
    if category_counts:
        # Return the most mentioned category
        most_mentioned = max(category_counts.items(), key=lambda x: x[1])
        if most_mentioned[1] >= 1:  # At least one mention
            return most_mentioned[0]
    
    return None



def evaluate_single_entry(json_file: str) -> Dict:
    """
    Evaluate responses for a single question (original format with 'responses' key).
    
    Args:
        json_file: Path to the JSON file containing responses for one question
        
    Returns:
        Dictionary with evaluation results
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    responses = data['responses']
    total_responses = len(responses)
    
    parsed_categories = []
    parsable_count = 0
    category_counts = {}
    
    for response in responses:
        response_text = response['generated_text']
        category = parse_error_category(response_text)
        
        if category:
            parsable_count += 1
            parsed_categories.append({
                'response_id': response['response_id'],
                'category': category
            })
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            parsed_categories.append({
                'response_id': response['response_id'],
                'category': None
            })
    
    parsable_rate = parsable_count / total_responses if total_responses > 0 else 0
    
    return {
        'analysis_type': 'single_entry',
        'total_responses': total_responses,
        'parsable_responses': parsable_count,
        'parsable_rate': parsable_rate,
        'category_distribution': category_counts,
        'parsed_categories': parsed_categories,
        'metadata': data['metadata']
    }

def evaluate_multiple_entries(json_file: str) -> Dict:
    """
    Evaluate responses for multiple questions (new format with 'entries' key).
    
    Args:
        json_file: Path to the JSON file containing responses for multiple questions
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = data['entries']
    total_entries = len(entries)
    
    # Overall statistics
    total_responses = 0
    total_parsable = 0
    overall_category_counts = {}
    
    # Per-entry results
    entry_results = []
    
    for entry in entries:
        entry_idx = entry['entry_index']
        responses = entry['responses']
        entry_total = len(responses)
        total_responses += entry_total
        
        # Analyze this entry's responses
        entry_parsable = 0
        entry_category_counts = {}
        entry_parsed_categories = []
        
        for response in responses:
            response_text = response['generated_text']
            category = parse_error_category(response_text)
            
            if category:
                entry_parsable += 1
                total_parsable += 1
                entry_parsed_categories.append({
                    'response_id': response['response_id'],
                    'category': category
                })
                entry_category_counts[category] = entry_category_counts.get(category, 0) + 1
                overall_category_counts[category] = overall_category_counts.get(category, 0) + 1
            else:
                entry_parsed_categories.append({
                    'response_id': response['response_id'],
                    'category': None
                })
        
        # Calculate entry rates
        entry_parsable_rate = entry_parsable / entry_total if entry_total > 0 else 0
        
        entry_results.append({
            'entry_index': entry_idx,
            'metadata': entry['metadata'],
            'total_responses': entry_total,
            'parsable_responses': entry_parsable,
            'parsable_rate': entry_parsable_rate,
            'category_distribution': entry_category_counts,
            'parsed_categories': entry_parsed_categories
        })
    
    # Calculate overall rates
    overall_parsable_rate = total_parsable / total_responses if total_responses > 0 else 0
    
    return {
        'analysis_type': 'multiple_entries',
        'total_entries': total_entries,
        'total_responses': total_responses,
        'total_parsable_responses': total_parsable,
        'overall_parsable_rate': overall_parsable_rate,
        'overall_category_distribution': overall_category_counts,
        'entry_results': entry_results,
        'metadata': data['metadata']
    }

def print_analysis(evaluation_result: Dict):
    """Print analysis of the evaluation results."""
    
    print("="*60)
    print("ERROR CATEGORIZATION ANALYSIS")
    print("="*60)
    
    if evaluation_result['analysis_type'] == 'single_entry':
        print_single_entry_analysis(evaluation_result)
    else:
        print_multiple_entries_analysis(evaluation_result)

def print_single_entry_analysis(result: Dict):
    """Print analysis for single entry results."""
    
    print(f"SINGLE QUESTION ANALYSIS")
    print("-" * 40)
    print(f"Total Responses: {result['total_responses']}")
    print(f"Parsable Error Categories: {result['parsable_responses']}")
    print(f"Parsable Rate: {result['parsable_rate']:.2%}")
    
    if result['category_distribution']:
        print("\nError Category Distribution:")
        for category, count in result['category_distribution'].items():
            percentage = count / result['total_responses'] * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Show unparsable responses
    unparsable = [item for item in result['parsed_categories'] if item['category'] is None]
    
    print(f"\nUnparsable Responses: {len(unparsable)}")
    if unparsable and len(unparsable) <= 10:
        print(f"  Response IDs: {[item['response_id'] for item in unparsable]}")

def print_multiple_entries_analysis(result: Dict):
    """Print analysis for multiple entries results."""
    
    print(f"MULTIPLE QUESTIONS ANALYSIS")
    print("-" * 40)
    print(f"Total Entries: {result['total_entries']}")
    print(f"Total Responses: {result['total_responses']}")
    print(f"Responses per Entry: {result['total_responses'] // result['total_entries']}")
    print(f"Total Parsable Error Categories: {result['total_parsable_responses']}")
    print(f"Overall Parsable Rate: {result['overall_parsable_rate']:.2%}")
    
    if result['overall_category_distribution']:
        print("\nOverall Error Category Distribution:")
        for category, count in result['overall_category_distribution'].items():
            percentage = count / result['total_responses'] * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\nPer-Entry Analysis:")
    for entry_result in result['entry_results']:
        print(f"\n  Entry {entry_result['entry_index']}:")
        print(f"    Question: {entry_result['metadata']['question'][:60]}...")
        print(f"    Parsable: {entry_result['parsable_responses']}/{entry_result['total_responses']} ({entry_result['parsable_rate']:.1%})")
        if entry_result['category_distribution']:
            top_categories = sorted(entry_result['category_distribution'].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"    Top Categories: {', '.join([f'{cat} ({count})' for cat, count in top_categories])}")

def analyze_single_entry(json_file: str) -> Dict:
    """
    Quick analysis function for single entry files.
    """
    result = evaluate_single_entry(json_file)
    print_analysis(result)
    return result

def analyze_multiple_entries(json_file: str) -> Dict:
    """
    Quick analysis function for multiple entries files.
    """
    result = evaluate_multiple_entries(json_file)
    print_analysis(result)
    return result

def auto_analyze(json_file: str) -> Dict:
    """
    Automatically detect file type and run appropriate analysis.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'entries' in data:
        print("📊 Detected multiple entries format")
        return analyze_multiple_entries(json_file)
    elif 'responses' in data:
        print("📊 Detected single entry format")
        return analyze_single_entry(json_file)
    else:
        raise ValueError("Unrecognized JSON format. Expected 'entries' or 'responses' key.")

if __name__ == "__main__":
    # Analyze only the 10-response file generated by annotate.py
    json_10 = "/u/rfechner/verl/generated_responses_10_Qw7B.json"
    
    print("RESPONSE PARSER ANALYSIS - GENERATED_RESPONSES_10.JSON")
    print("="*60)
    
    if os.path.exists(json_10):
        print(f"\n🔍 Analyzing generated_responses_10.json file:")
        print("-" * 50)
        result_10 = auto_analyze(json_10)
        
        print(f"\n📊 PARSING RESULTS:")
        print(f"   SUCCESS RATE: {result_10['overall_parsable_rate']:.2%}")
        print(f"   PARSABLE RESPONSES: {result_10['total_parsable_responses']}/{result_10['total_responses']}")
        print(f"   TOTAL ENTRIES: {result_10['total_entries']}")
        print(f"   RESPONSES PER ENTRY: {result_10['total_responses'] // result_10['total_entries']}")
        
        # Show detailed breakdown by entry
        print(f"\n📋 DETAILED ENTRY BREAKDOWN:")
        for entry_result in result_10['entry_results']:
            entry_idx = entry_result['entry_index']
            parsable = entry_result['parsable_responses']
            total = entry_result['total_responses']
            rate = entry_result['parsable_rate']
            print(f"   Entry {entry_idx}: {parsable}/{total} ({rate:.1%}) parsable")
        
        # Show unparsable responses for debugging
        print(f"\n� UNPARSABLE RESPONSES ANALYSIS:")
        unparsable_total = result_10['total_responses'] - result_10['total_parsable_responses']
        if unparsable_total > 0:
            print(f"   Found {unparsable_total} unparsable responses")
            print(f"   Entry-level breakdown:")
            for entry_result in result_10['entry_results']:
                entry_idx = entry_result['entry_index']
                unparsable_count = entry_result['total_responses'] - entry_result['parsable_responses']
                if unparsable_count > 0:
                    unparsable_ids = [item['response_id'] for item in entry_result['parsed_categories'] if item['category'] is None]
                    print(f"     Entry {entry_idx}: {unparsable_count} unparsable (IDs: {unparsable_ids})")
        else:
            print(f"   ✅ All responses successfully parsed!")
            
    else:
        print(f"❌ File not found: {json_10}")
        print("   Make sure to run annotate.py first to generate the file.")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
