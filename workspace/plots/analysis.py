#!/usr/bin/env python3
"""
Analysis script for rollout data using Kernel Density Estimation and movement analysis.

This script loads rollout data from parquet files and performs:
1. Distribution fitting using Kernel Density Estimation (KDE) with boundary reflection 
   to analyze passrate patterns across different datasets and models
2. Movement analysis showing how individual questions change from base to trained models

Features:
- Fast JSON caching for iterative analysis
- KDE with boundary reflection for [0,1] bounded data
- Automatic bandwidth selection using Silverman's rule
- Passrate movement visualization (bump plots) showing training effects
- High-quality visualization output
- Support for nested directory format (standard) and legacy format (backward compatibility)
- Multi-chunk support: automatically detects and concatenates multiple chunk directories

Standard Data Format:
- Nested directories: 'Qwen2.5-{SIZE}_{DATASET}_nrollouts_1024_{TIMESTAMP}_{SIZE}-{MODEL}/chunk*/results.parquet'
- Supports multiple chunks (chunk0, chunk1, chunk2, etc.) which are automatically concatenated
- Example: 'Qwen2.5-1.5B_aime25_nrollouts_1024_20250811_132728_1.5b-entropy0.01/chunk0/results.parquet'
            'Qwen2.5-1.5B_aime25_nrollouts_1024_20250811_132728_1.5b-entropy0.01/chunk1/results.parquet'

Legacy Data Format (deprecated):
- Flat files: 'dataset_model.parquet' 
- Example: 'aime25_entropy0.01.parquet'
"""

import pandas as pd
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import norm
import re


def save_passrates_to_json(sample_passrates, cache_file='passrates_cache.json'):
    """Save extracted passrates to JSON for fast loading"""
    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for dataset_name, models in sample_passrates.items():
        json_data[dataset_name] = {}
        for model_name, passrates in models.items():
            json_data[dataset_name][model_name] = passrates.tolist()
    
    with open(cache_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Passrates cached: {cache_file}")


def load_passrates_from_json(cache_file='passrates_cache.json'):
    """Load passrates from JSON cache"""
    with open(cache_file, 'r') as f:
        json_data = json.load(f)
    
    # Convert lists back to numpy arrays
    sample_passrates = {}
    for dataset_name, models in json_data.items():
        sample_passrates[dataset_name] = {}
        for model_name, passrates in models.items():
            sample_passrates[dataset_name][model_name] = np.array(passrates)
    
    print(f"Passrates loaded: {cache_file}")
    return sample_passrates


def extract_passrates_from_dfs(dfs):
    """Extract individual sample passrates from dataframes"""
    sample_passrates = {}
    for dataset_name, df_dict in dfs.items():
        sample_passrates[dataset_name] = {}
        for model_name, df in df_dict.items():
            individual_passrates = df['rewards'].apply(np.mean).values
            sample_passrates[dataset_name][model_name] = individual_passrates
    return sample_passrates


# Removed BetaMixture class - simplified to use only KDE


class KernelDensityEstimator:
    """Simple KDE with reflection at boundaries for [0,1] data"""
    
    def __init__(self, bandwidth=0.05):
        self.bandwidth = bandwidth
        self.data = None
    
    def fit(self, X):
        self.data = X.flatten()
        return self
    
    def pdf(self, x):
        """Compute PDF using reflection method for boundaries"""
        if self.data is None:
            return np.zeros_like(x)
        
        pdf = np.zeros_like(x)
        h = self.bandwidth
        
        for xi in self.data:
            # Original kernel
            pdf += norm.pdf(x, xi, h)
            
            # Reflection at x=0
            pdf += norm.pdf(x, -xi, h)
            
            # Reflection at x=1
            pdf += norm.pdf(x, 2-xi, h)
        
        pdf /= len(self.data)
        return pdf


def fit_kde_distribution(data):
    """Fit KDE distribution and return results"""
    results = {}
    
    # Kernel Density Estimation with boundary reflection
    try:
        # Choose bandwidth using Silverman's rule of thumb
        n = len(data)
        std_data = np.std(data)
        bandwidth = 1.06 * std_data * n**(-1/5)
        bandwidth = max(min(bandwidth, 0.1), 0.01)  # Constrain bandwidth
        
        kde = KernelDensityEstimator(bandwidth=bandwidth)
        kde.fit(data)
        results['kde'] = {
            'model': kde,
            'name': f'KDE (h={bandwidth:.3f})',
            'bandwidth': bandwidth
        }
    except Exception as e:
        print(f"KDE failed: {e}")
        # Fallback to histogram if KDE fails
        try:
            hist, bin_edges = np.histogram(data, bins=15, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            results['histogram'] = {
                'hist': hist,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'name': 'Histogram (fallback)'
            }
        except Exception as e2:
            print(f"Histogram fallback also failed: {e2}")
    
    return results


def analyze_distributions_kde(sample_passrates, save_plot=False, plot_filename='kde_analysis.png'):
    """Analyze distributions using KDE only"""
    
    # Create plots for each dataset showing KDE fits
    n_datasets = len(sample_passrates)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]

    colors = ['steelblue', 'orange', 'green', 'red']
    model_names = ['entropy0.01', 'entropy0.0', 'base']

    for idx, (dataset_name, model_data_dict) in enumerate(sample_passrates.items()):
        ax = axes[idx]
        
        bins = np.linspace(0, 1, 16)
        x_range = np.linspace(0.001, 0.999, 300)
        
        # Plot histograms and fit KDE for each model
        for i, model_name in enumerate(model_names):
            if model_name in model_data_dict:
                data = model_data_dict[model_name]
                
                # Plot histogram of actual data
                counts, bin_edges = np.histogram(data, bins=bins, density=False)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]
                offset = (i - 1) * bin_width * 0.2
                
                ax.bar(bin_centers + offset, counts, width=bin_width*0.2, alpha=0.6, 
                       label=f'{model_name} (data)', color=colors[i], edgecolor='black', linewidth=0.5)
                
                # Fit KDE
                fitted_results = fit_kde_distribution(data)
                
                # Plot KDE fit
                if 'kde' in fitted_results:
                    kde_result = fitted_results['kde']
                    if 'model' in kde_result:
                        y = kde_result['model'].pdf(x_range)
                        # Scale KDE to match frequency scale (multiply by data length and bin width)
                        y_scaled = y * len(data) * bin_width
                        ax.plot(x_range, y_scaled, color=colors[i], linewidth=3, linestyle='--', alpha=0.9,
                               label=f'{model_name} {kde_result["name"]}')
                elif 'histogram' in fitted_results:
                    # Fallback to histogram display
                    hist_result = fitted_results['histogram']
                    # Scale histogram to frequency (multiply by data length and bin width)
                    hist_scaled = hist_result['hist'] * len(data) * bin_width
                    ax.plot(hist_result['bin_centers'], hist_scaled, 
                           color=colors[i], linewidth=2, linestyle=':', alpha=0.8,
                           label=f'{model_name} {hist_result["name"]}')
        
        ax.set_xlabel('Passrate')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dataset_name} - KDE Distribution Fitting')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_filename}")
    
    plt.close()  # Close the plot instead of showing it


def get_sample_passrates(cache_file='passrates_cache.json', force_reload=False, results_dir=None, model_filter=None, size_filter=None, use_legacy_format=False):
    """Get sample passrates either from cache or by loading from parquet files
    
    Args:
        cache_file: Path to JSON cache file
        force_reload: If True, reload from parquet files even if cache exists
        results_dir: Directory containing rollout data (defaults to standard Qwen directory)
        model_filter: List of model names to include 
        size_filter: List of model sizes to include (e.g., ['7b'], ['1.5b'], or ['7b', '1.5b'])
        use_legacy_format: If True, use legacy flat file format
    """
    if results_dir is None:
        results_dir = '/u/rfechner/out/fixed_rollouts/Qwen/'
    
    if not force_reload and os.path.exists(cache_file):
        return load_passrates_from_json(cache_file)
    else:
        print("Loading data...")
        dfs = load_data(results_dir, model_filter=model_filter, size_filter=size_filter, use_legacy_format=use_legacy_format)
        validate_data(dfs)
        sample_passrates = extract_passrates_from_dfs(dfs)
        save_passrates_to_json(sample_passrates, cache_file)
        return sample_passrates


# =============================================================================
# LEGACY FORMAT SUPPORT (DEPRECATED)
# =============================================================================

def load_data_legacy_format(results_dir, model_filter=None):
    """Load data from legacy parquet files in the results directory (DEPRECATED)
    
    Args:
        results_dir: Directory containing parquet files
        model_filter: List of model names to include (e.g., ['base', 'entropy0.01'])
                     If None, load all models
    """
    all_names = ['minervamath', 'olympiadbench', 'math500', 'aime25']
    dfs = {name: {} for name in all_names}
    basepath = pathlib.Path(results_dir)
    
    if not basepath.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' not found")
    
    for file in os.listdir(results_dir):
        if file.endswith('.parquet'):
            dataset_name, model = file.replace('.parquet', '').split('_')
            if dataset_name in dfs:
                # Apply model filter if specified
                if model_filter is None or model in model_filter:
                    dfs[dataset_name][model] = pd.read_parquet(basepath / file)
                    print(f"Loaded: {dataset_name} {model} (legacy)")
    
    return dfs


def create_data_sources_config(legacy_dirs=None, qwen_dirs=None, legacy_models=None, qwen_models=None):
    """Helper function to create data sources configuration (DEPRECATED - use load_data directly)
    
    Args:
        legacy_dirs: List of directories with legacy format (dataset_model.parquet)
        qwen_dirs: List of directories with standard format (subdirs with chunk*/results.parquet)
        legacy_models: List of model names to load from legacy directories (e.g., ['base'])
        qwen_models: List of model names to load from standard directories (e.g., ['entropy0.01', 'entropy0.0'])
    
    Returns:
        List of data source configurations
    """
    print("Warning: create_data_sources_config is deprecated. Use load_data() directly.")
    data_sources = []
    
    if legacy_dirs:
        for path in legacy_dirs:
            source_config = {'path': path, 'format': 'legacy'}
            if legacy_models is not None:
                source_config['model_filter'] = legacy_models
            data_sources.append(source_config)
    
    if qwen_dirs:
        for path in qwen_dirs:
            source_config = {'path': path, 'format': 'standard'}
            if qwen_models is not None:
                source_config['model_filter'] = qwen_models
            data_sources.append(source_config)
    
    return data_sources


def load_data_multi_source(data_sources):
    """Load data from multiple sources with different formats (DEPRECATED)
    
    Args:
        data_sources: List of dictionaries with 'path', 'format', and optional 'model_filter' keys
                     Format can be 'legacy' or 'standard'
                     model_filter is a list of model names to include
    """
    print("Warning: load_data_multi_source is deprecated. Use load_data() directly.")
    all_names = ['minervamath', 'olympiadbench', 'math500', 'aime25']
    dfs = {name: {} for name in all_names}
    
    for source in data_sources:
        source_path = source['path']
        source_format = source.get('format', 'standard')
        model_filter = source.get('model_filter', None)
        
        print(f"Loading data from {source_path} (format: {source_format}, filter: {model_filter})")
        
        if source_format == 'legacy':
            source_dfs = load_data_legacy_format(source_path, model_filter=model_filter)
        else:  # standard format (default)
            source_dfs = load_data_standard_format(source_path, model_filter=model_filter)
        
        # Merge source_dfs into main dfs
        for dataset_name in all_names:
            if dataset_name in source_dfs:
                for model_name, df in source_dfs[dataset_name].items():
                    if model_name in dfs[dataset_name]:
                        print(f"Warning: Overwriting {dataset_name} {model_name}")
                    dfs[dataset_name][model_name] = df
    
    return dfs


def parse_directory_name(dirname):
    """Parse directory name to extract dataset, model, and size info
    
    Example patterns:
    - Qwen2.5-1.5B_aime25_nrollouts_1024_20250811_132728_1.5b-entropy0.01 -> ('aime25', 'entropy0.01', '1.5b')
    - Qwen2.5-7B_math500_nrollouts_1024_20250811_000000_7b-base -> ('math500', 'base', '7b')
    
    Returns: (dataset_name, model_name, model_size)
    """
    parts = dirname.split('_')
    if len(parts) < 6:
        return None, None, None
    
    # Extract dataset name (should be parts[1])
    dataset_name = parts[1]
    
    # Extract model size from the Qwen2.5-XB prefix (parts[0])
    if parts[0].startswith('Qwen2.5-'):
        size_part = parts[0].replace('Qwen2.5-', '')  # e.g., "1.5B" or "7B"
        if size_part == '1.5B':
            model_size = '1.5b'
        elif size_part == '7B':
            model_size = '7b'
        else:
            model_size = size_part.lower()  # Handle other sizes if they exist
    else:
        model_size = 'unknown'
    
    # Extract model info from the last part
    last_part = parts[-1]  # e.g., "1.5b-entropy0.01" or "7b-base"
    
    if 'entropy' in last_part:
        # Handle entropy models: "1.5b-entropy0.01" -> "entropy0.01"
        entropy_part = last_part.split('-entropy')[-1]  # e.g., "0.01"
        model_name = f"entropy{entropy_part}"
    elif last_part.endswith('-base'):
        # Handle base models: "7b-base" -> "base"
        model_name = "base"
    else:
        # Fallback to using the last part as-is
        model_name = last_part
    
    return dataset_name, model_name, model_size


def find_parquet_files(directory):
    """Find all parquet files and parquet-part files in a directory"""
    parquet_files = []
    pattern = re.compile(r".*\.parquet(-part\d+)?$")
    
    if not os.path.exists(directory):
        return parquet_files
        
    for file in os.listdir(directory):
        if pattern.match(file):
            full_path = os.path.join(directory, file)
            parquet_files.append(full_path)
    
    return sorted(parquet_files)


def load_chunks_from_directory(subdir_path):
    """Load and concatenate data from all chunks in a directory
    
    Each chunk directory may contain:
    - A single results.parquet file, OR
    - Multiple results.parquet-partX files that need to be concatenated
    
    Args:
        subdir_path: Path to directory containing chunk subdirectories
        
    Returns:
        Concatenated DataFrame from all chunks, or None if no chunks found
    """
    all_dataframes = []
    chunk_dirs = []
    
    # Find all chunk directories
    for item in subdir_path.iterdir():
        if item.is_dir() and item.name.startswith('chunk'):
            chunk_dirs.append(item)
    
    if not chunk_dirs:
        return None
    
    # Sort chunk directories by name to ensure consistent order
    chunk_dirs.sort(key=lambda x: x.name)
    
    # Load data from each chunk
    for chunk_dir in chunk_dirs:
        chunk_dataframes = []
        
        # Find all parquet files in this chunk directory
        parquet_files = find_parquet_files(str(chunk_dir))
        
        if not parquet_files:
            print(f"  Warning: No parquet files found in {chunk_dir}")
            continue
        
        # Load all parquet files in this chunk
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                chunk_dataframes.append(df)
            except Exception as e:
                print(f"  Warning: Failed to load {parquet_file}: {e}")
        
        if chunk_dataframes:
            # Concatenate all parts within this chunk
            chunk_combined = pd.concat(chunk_dataframes, ignore_index=True)
            all_dataframes.append(chunk_combined)
            print(f"  Loaded chunk: {chunk_dir.name} ({len(parquet_files)} files, {len(chunk_combined)} rows)")
        else:
            print(f"  Warning: No valid data in {chunk_dir}")
    
    if not all_dataframes:
        return None
    
    # Concatenate all chunks
    final_combined_df = pd.concat(all_dataframes, ignore_index=True)
    return final_combined_df

def load_data_standard_format(results_dir, model_filter=None, size_filter=None, all_names=['minervamath', 'olympiadbench', 'math500', 'aime25', 'math']):
    """Load data from standard directory structure with support for multiple chunks
    
    Args:
        results_dir: Directory containing subdirectories  
        model_filter: List of model names to include (e.g., ['entropy0.01', 'entropy0.0', 'base'])
                     If None, load all models
        size_filter: List of model sizes to include (e.g., ['7b'], ['1.5b'], or ['7b', '1.5b'])
                    If None, load all sizes
    """
    dfs = {name: {} for name in all_names}
    basepath = pathlib.Path(results_dir)
    
    if not basepath.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' not found")
    
    # Look for subdirectories with parquet files
    for subdir in os.listdir(results_dir):
        subdir_path = basepath / subdir
        if subdir_path.is_dir():
            # Parse directory name to get dataset, model, and size
            dataset_name, model_name, model_size = parse_directory_name(subdir)
            
            if dataset_name in all_names and model_name:
                # Apply model filter if specified
                model_passes_filter = model_filter is None or model_name in model_filter
                # Apply size filter if specified
                size_passes_filter = size_filter is None or model_size in size_filter
                
                if model_passes_filter and size_passes_filter:
                    # Load and concatenate data from all chunks
                    combined_df = load_chunks_from_directory(subdir_path)
                    if combined_df is not None:
                        dfs[dataset_name][model_name] = combined_df
                        print(f"Loaded: {dataset_name} {model_name} ({model_size}) - {len(combined_df)} total rows")
                    else:
                        print(f"Warning: No chunk data found in {subdir_path}")
                elif not size_passes_filter:
                    print(f"Skipped: {dataset_name} {model_name} ({model_size})")
    
    return dfs


def load_data(results_dir, model_filter=None, size_filter=None, use_legacy_format=False):
    """Load data from parquet files
    
    Args:
        results_dir: Directory containing rollout data
        model_filter: List of model names to include (e.g., ['entropy0.01', 'entropy0.0', 'base'])
                     If None, load all models
        size_filter: List of model sizes to include (e.g., ['7b'], ['1.5b'], or ['7b', '1.5b'])
                    If None, load all sizes
        use_legacy_format: If True, use legacy flat file format instead of standard format
    
    Returns:
        Dictionary of dataframes organized by dataset and model
    """
    if use_legacy_format:
        return load_data_legacy_format(results_dir, model_filter)
    else:
        return load_data_standard_format(results_dir, model_filter, size_filter)


def validate_data(dfs):
    """Validate that all dataframes have the required columns"""
    # Check that all dfs have rewards column
    assert all(['rewards' in df.columns for model_name, df in entry.items()] 
               for entry in dfs.values()), "Missing 'rewards' column in some dataframes"
    
    # make it, s.t. indeces align
    for dataset_name, df_dict in dfs.items():
        for model_name, df in df_dict.items():
            df.sort_index(inplace=True)

    # Print dataset information
    print("Loaded datasets:")
    for dataset_name, df_dict in dfs.items():
        for model_name, df in df_dict.items():
            print(f'{dataset_name}: {model_name} ({len(df)} samples)')
    print()


def analyze_distributions(results_dir=None, model_filter=None, use_legacy_format=False, save_plot=False, plot_filename='distribution_analysis.png'):
    """Analyze distributions using KDE
    
    Args:
        results_dir: Directory containing rollout data (defaults to standard directory)
        model_filter: List of model names to include
        use_legacy_format: If True, use legacy flat file format
        save_plot: Whether to save the plot
        plot_filename: Name of the output plot file
    """
    if results_dir is None:
        results_dir = '/u/rfechner/out/fixed_rollouts/Qwen/'
    
    dfs = load_data(results_dir, model_filter=model_filter, use_legacy_format=use_legacy_format)
    validate_data(dfs)
    sample_passrates = extract_passrates_from_dfs(dfs)
    analyze_distributions_kde(sample_passrates, save_plot, plot_filename)


def analyze_distributions_legacy(dfs=None, save_plot=False, plot_filename='distribution_analysis_legacy.png', data_sources=None):
    """Legacy analysis function for backward compatibility (DEPRECATED)"""
    print("Warning: analyze_distributions_legacy is deprecated. Use analyze_distributions() instead.")
    if dfs is None:
        if data_sources:
            dfs = load_data_multi_source(data_sources)
        else:
            dfs = load_data('../rollouts/results', use_legacy_format=True)
        validate_data(dfs)
    
    sample_passrates = extract_passrates_from_dfs(dfs)
    analyze_distributions_kde(sample_passrates, save_plot, plot_filename)


def tmp_validate_data(dfs):
    """Temporary validate function that limits base dataframes to first 300 entries for math500"""
    # Check that all dfs have rewards column
    assert all(['rewards' in df.columns for model_name, df in entry.items()] 
               for entry in dfs.values()), "Missing 'rewards' column in some dataframes"
    
    # Make it, s.t. indices align and limit math500 base to 300 entries
    for dataset_name, df_dict in dfs.items():
        for model_name, df in df_dict.items():
            df.sort_index(inplace=True)
            
            # For math500 datasets, limit base model to first 300 entries
            if 'math500' in dataset_name.lower() and model_name == 'base':
                dfs[dataset_name][model_name] = df.iloc[:300].copy()
                print(f"Limited {dataset_name} {model_name} to first 300 entries")

    # Print dataset information
    print("Loaded datasets:")
    for dataset_name, df_dict in dfs.items():
        for model_name, df in df_dict.items():
            print(f'{dataset_name}: {model_name} ({len(df)} samples)')
    print()


def tmp_main():
    print('WARNING: USING TMP FUNCTION.')
    """Temporary main function that limits base dataframes to first 300 entries for math500"""
    print("Starting analysis (with math500 base limitation to 300 entries)...")
    
    # Change to the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Configuration
        results_dir = '/u/rfechner/out/fixed_rollouts/Qwen/'
        model_filter = ['entropy0.01', 'entropy0.0', 'base']  # Load all model types
        size_filter = ['7b']  # None -> Load all sizes by default. Set to ['7b'] to only load 7B models
        
        # Load data manually to apply temporary validation
        print("Loading data...")
        dfs = load_data(results_dir, model_filter=model_filter, size_filter=size_filter, use_legacy_format=False)
        tmp_validate_data(dfs)  # Use temporary validation that limits math500 base
        sample_passrates = extract_passrates_from_dfs(dfs)
        save_passrates_to_json(sample_passrates, 'passrates_cache.json')
        
        analyze_distributions_kde(sample_passrates, save_plot=True, 
                                 plot_filename=f'kde_distribution_analysis-{"-".join(size_filter)}.png')
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    print('WARNING: USING TMP FUNCTION.')

def main():
    """Main function to run the analysis"""
    print("Starting analysis...")
    
    # Change to the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Configuration
        results_dir = '/u/rfechner/out/fixed_rollouts/Qwen/'
        model_filter = ['entropy0.01', 'entropy0.0', 'base']  # Load all model types
        size_filter = ['7b']  # None -> Load all sizes by default. Set to ['7b'] to only load 7B models
        
        # Get sample passrates (from cache if available)
        sample_passrates = get_sample_passrates(
            cache_file='passrates_cache.json', 
            force_reload=True,
            results_dir=results_dir,
            model_filter=model_filter,
            size_filter=size_filter,
            use_legacy_format=False
        )
    
        analyze_distributions_kde(sample_passrates, save_plot=True, 
                                 plot_filename=f'kde_distribution_analysis-{"-".join(size_filter)}.png')
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def main_legacy():
    """Legacy main function for backward compatibility (DEPRECATED)"""
    print("Warning: main_legacy is deprecated. Use main() instead.")
    print("Starting rollout data analysis (legacy mode)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    try:
        results_dir = '/u/rfechner/verl/workspace/rollouts/results'
        model_filter = ['entropy0.01', 'entropy0.0', 'base']  # Load all model types
        size_filter = ['7b']  # None -> Load all sizes by default. Set to ['7b'] to only load 7B models
        
        # Get sample passrates (from cache if available)
        sample_passrates = get_sample_passrates(
            cache_file='passrates_cache_legacy.json', 
            force_reload=True,
            results_dir=results_dir,
            model_filter=model_filter,
            size_filter=size_filter,
            use_legacy_format=True
        )
        
        analyze_distributions_kde(sample_passrates, save_plot=True, 
                                 plot_filename='kde_distribution_analysis_legacy.png')
        
    except Exception as e:
        print(f"Error during legacy analysis: {e}")
        raise


if __name__ == "__main__":
    tmp_main()
