import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import binom
import json

from analysis import load_passrates_from_json, load_data


def calculate_passat(df : pd.DataFrame, k : int):
    """
        Calculates the pass@k metric for the current dataframe.

        pass@k = \mathbb E_{x \sim D}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]
    """
    n = len(df.iloc[0]['rewards'])
    def apply_fn(rew):
        return 1 - (binom(n - sum(rew), k) / (binom(n, k)))
    pass_metric = np.mean(df['rewards'].apply(apply_fn))
    return pass_metric.item()

def main(reload=False, k_exponent=10):
    
    if reload:
        results_dir = '/u/rfechner/out/fixed_rollouts/Qwen/'
        model_filter = ['entropy0.01', 'entropy0.0', 'base']  # Load all model types
        size_filter = ['7b']  # None -> Load all sizes by default. Set to ['7b'] to only load 7B models
        
        dfs = load_data(results_dir, model_filter=model_filter, size_filter=size_filter, use_legacy_format=False)
        ks = 2**np.arange(k_exponent)
        pass_ats = {dataset : 
                        {variant : 
                            {str(k) : calculate_passat(df, k) for k in ks}
                        for variant, df in variants.items()} 
                    for dataset, variants in dfs.items()}
        with open('/u/rfechner/verl/workspace/plots/passat_values.json', 'w') as file:
            json.dump(pass_ats, file)

    else:
        path = '/u/rfechner/verl/workspace/plots/passat_values.json'
        with open(path, 'r') as file:
            pass_ats = json.load(file)
        
    n_datasets = len(pass_ats)

    fig, axes = plt.subplots(ncols=n_datasets, figsize=(5 * n_datasets, 4))
    axes = axes.flatten()
    counter = 0

    for dataset, variants in pass_ats.items():
        for variant, ks in variants.items():
            values = list(ks.values())
            axes[counter].scatter(range(len(values)), values)
            axes[counter].plot(values, label=variant)
            ticks = list(ks.keys())
    
        axes[counter].set_xticks(range(len(ticks)), ticks)
        axes[counter].set_title(f'Pass@k metric for dataset: {dataset}')
        axes[counter].legend()

        counter += 1
    
    fig.savefig(fname=f'passat.png', dpi=300)

if __name__ == "__main__":
    passrates = main(reload=False)
