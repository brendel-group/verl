import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

from tqdm import tqdm
from analysis import load_passrates_from_json, load_data_standard_format


def calc_difference_for_k(k : int, base_df, grpo_df, num_bootstrap_repetitions=100):
    """
        For each question, bootstrap the pass@k rate. Then, calculate the differences between base and grpo. The mean difference gets returned.
    """
    diffs = []
    for _ in tqdm(range(num_bootstrap_repetitions), desc='bootstrap_samples'):
        base_responses = base_df['rewards'].apply(lambda x: np.random.choice(x, size=k))
        grpo_responses = grpo_df['rewards'].apply(lambda x: np.random.choice(x, size=k))
        base_rates = base_responses.apply(np.nanmean)
        grpo_rates = grpo_responses.apply(np.nanmean)
        ds = np.array(grpo_rates - base_rates)
        ds = ds[~np.isnan(ds)]
        mean = np.mean(ds)
        if not np.isnan(mean):
            diffs.append(mean)

    return diffs

def main(reload=False):    
    """
        We want to plot the differences between passrates of the different benchmarks 
    """
    if reload:
        dfs = load_data_standard_format(results_dir='/u/rfechner/out/fixed_rollouts/Qwen',
                                        model_filter=['base', 'entropy0.0'],
                                        size_filter=['7b'],
                                        all_names=['math'])
        with open('hist_of_mean_diffs_per_k.pickle', 'wb') as file:
            pickle.dump(dfs, file)
    else:
        with open('hist_of_mean_diffs_per_k.pickle', 'rb') as file:
            dfs = pickle.load(file)

    diffs = [
        calc_difference_for_k(k, dfs['math']['base'], dfs['math']['entropy0.0']) for k in tqdm(2**np.arange(10), desc='k')
    ]

    # we need num_models (models besides the "base" model) * num_datasets histograms
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(5*5, 2*6))
    axes = axes.flatten()
    counter = 0
    for k, diff in zip(2**np.arange(10), diffs):
        axes[counter].hist(diff)
        axes[counter].set_title(f'k : {k}')
        counter += 1
    
    fig.tight_layout()
    plt.subplots_adjust(hspace = .5)
    fig.savefig(fname='hist_of_diffs_per_k.png', dpi=300)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    ax.hist([np.nanmean(diff) for diff in diffs])
    ax.set_title("Mean differences for all k's put into the same hist.")

    fig.tight_layout()
    plt.subplots_adjust(hspace = .5)
    fig.savefig(fname='hist_of_mean_diffs_per_k.png', dpi=300)

if __name__ == "__main__":
    main(reload=False)