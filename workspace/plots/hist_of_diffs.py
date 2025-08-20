import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from analysis import load_passrates_from_json


def main(passrates : dict):
    
    """
        We want to plot the differences between passrates of the different benchmarks 
    """
    num_models, num_datasets = len(passrates[list(passrates.keys())[0]]) - 1, len(passrates)

    # Get all model names (excluding 'base') and sort them for consistent ordering
    all_model_names = set()
    for model_dict in passrates.values():
        all_model_names.update(name for name in model_dict.keys() if name != 'base')
    sorted_model_names = sorted(all_model_names)

    # we need num_models (models besides the "base" model) * num_datasets histograms
    fig, axes = plt.subplots(nrows=num_datasets, ncols=num_models, figsize=(num_datasets * 4, num_models * 6))
    axes = axes.flatten()
    counter = 0

    for dataset, model_dict in passrates.items():
        base_passrates = model_dict['base']
        for name in sorted_model_names:
            if name not in model_dict:
                # Skip if this model doesn't exist for this dataset
                counter += 1
                continue
            passrates_data = model_dict[name]
            diffs = np.array(passrates_data) - np.array(base_passrates)
            
            # Set color based on model name
            if 'entropy0.0' in name:
                color = 'orange'
            elif 'entropy0.01' in name:
                color = 'lightblue'
            else:
                color = 'blue'  # default color
            
            axes[counter].hist(diffs, color=color)
            axes[counter].set_title(f'Dataset: {dataset}; base -> {name}')
            mean_diff = np.mean(diffs)
            axes[counter].axvline(mean_diff, color='red')
            axes[counter].text(mean_diff + 0.04, axes[counter].get_ylim()[1] * 0.75, 
                             f"$\\Delta$: {mean_diff:.3f}", 
                             rotation=0, color='red', fontsize=20)

            counter += 1

    fig.tight_layout()
    plt.subplots_adjust(hspace = .5)
    fig.savefig(fname='difference_histogram.png', dpi=300)


def tmp_main(passrates : dict):
    
    """
        We want to plot the differences between passrates of the different benchmarks 
        For math500, we only look at the first 300 entries as base and entropy models have different lengths
    """
    num_models, num_datasets = len(passrates[list(passrates.keys())[0]]) - 1, len(passrates)

    # Get all model names (excluding 'base') and sort them for consistent ordering
    all_model_names = set()
    for model_dict in passrates.values():
        all_model_names.update(name for name in model_dict.keys() if name != 'base')
    sorted_model_names = sorted(all_model_names)

    # we need num_models (models besides the "base" model) * num_datasets histograms
    fig, axes = plt.subplots(nrows=num_datasets, ncols=num_models, figsize=(num_datasets * 3, num_models * 7))
    axes = axes.flatten()
    counter = 0

    for dataset, model_dict in passrates.items():
        base_passrates = model_dict['base']
        
        # For math500, limit to first 300 entries
        if 'math500' in dataset.lower():
            base_passrates = base_passrates[:300]
        
        for name in sorted_model_names:
            if name not in model_dict:
                # Skip if this model doesn't exist for this dataset
                counter += 1
                continue
            
            passrates_data = model_dict[name]
            
            # For math500, limit to first 300 entries
            if 'math500' in dataset.lower():
                passrates_data = passrates_data[:300]
            
            diffs = np.array(passrates_data) - np.array(base_passrates)
            
            # Set color based on model name
            if 'entropy0.01' in name:
                color = 'lightblue'
            elif 'entropy0.0' in name:
                color = 'orange'
            else:
                color = 'blue'  # default color
            
            axes[counter].hist(diffs, color=color, bins=20)
            axes[counter].set_title(f'Dataset: {dataset}; base -> {name}')
            mean_diff = np.mean(diffs)
            axes[counter].axvline(mean_diff, color='red')
            axes[counter].text(mean_diff + 0.04, axes[counter].get_ylim()[1] * 0.75, 
                             f"$\\Delta$: {mean_diff:.3f}", 
                             rotation=0, color='red', fontsize=20)

            counter += 1

    fig.tight_layout()
    plt.subplots_adjust(hspace = .5)
    fig.savefig(fname='difference_histogram.png', dpi=300)

if __name__ == "__main__":
    print("WARNING: USING TMP FUNCTION.")
    d = load_passrates_from_json('/u/rfechner/verl/workspace/plots/passrates_cache.json')
    tmp_main(d)