import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import torch

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across various libraries and frameworks.

    Args:
        seed (int): The seed value to be used for random number generation.

    This function ensures that the random number generators in Python's `random` module,
    NumPy, and PyTorch produce the same results each time the code is run with the same seed.
    It also configures PyTorch to use deterministic algorithms for CUDA operations, ensuring
    reproducibility on GPU devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_learning_curve(experiments_data: Dict[str, List[Dict[str, any]]], 
                        bin_size: int = 500,
                        title: str = "Algorithm Performance Comparison",
                        output_filename: str = "algorithm_comparison.png") -> None:
    """
    Plots the learning curves of multiple experiments with mean and standard deviation shading.

    This function aggregates episode rewards from multiple experiments (possibly with different random seeds),
    bins the data by environment steps, and visualizes the mean and standard deviation of rewards for each bin.
    Each experiment's curve is plotted in a different color for easy comparison.

    Args:
        experiments_data (Dict[str, List[Dict[str, any]]]): A dictionary where each key is the experiment name
            and each value is a list of dictionaries containing training statistics (should include 'steps', 'reward', and 'seed').
        bin_size (int, optional): The width of each bin for grouping environment steps. Default is 500.
        title (str, optional): The title of the plot. Default is "Algorithm Performance Comparison".
        output_filename (str, optional): The filename for saving the plot. Default is "algorithm_comparison.png".

    Returns:
        None. The function saves the plot as a PNG file and displays it.

    Note:
        The bin_size parameter determines the interval of environment steps used to group and average
        the rewards. A larger bin_size results in smoother curves by averaging over more episodes,
        while a smaller bin_size provides more detailed but potentially noisier curves.
    """
    print("\n--- Generating multi-curve comparison plot ---")
    
    # Set plot style and canvas
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define a list of colors to distinguish different curves
    colors = ['dodgerblue', 'orangered', 'forestgreen', 'darkviolet', 'gold', 'cyan']

    # Iterate over the data of each experiment
    for i, (name, data) in enumerate(experiments_data.items()):
        if not data:
            print(f"Warning: Data for experiment '{name}' is empty, skipping.")
            continue
            
        df = pd.DataFrame(data)
        num_seeds = df['seed'].nunique()
        color = colors[i % len(colors)]

        max_steps = df['steps'].max()
        bins = np.arange(0, max_steps + bin_size, bin_size)
        df['step_bin'] = pd.cut(df['steps'], bins=bins, right=False, labels=bins[:-1])

        seed_bin_rewards = df.groupby(['seed', 'step_bin'], observed=True)['reward'].mean().reset_index()
        final_stats = seed_bin_rewards.groupby('step_bin', observed=True)['reward'].agg(['mean', 'std']).reset_index()

        # Fill missing values and drop any remaining NaN (usually at the beginning)
        final_stats = final_stats.ffill().dropna()

        # --- Plotting ---
        # Plot mean curve
        ax.plot(final_stats['step_bin'], final_stats['mean'], color=color, linewidth=2.5, 
                label=f"{name} ({num_seeds} Seeds)")
        # Fill standard deviation shading area
        ax.fill_between(final_stats['step_bin'],
                        final_stats['mean'] - final_stats['std'],
                        final_stats['mean'] + final_stats['std'],
                        color=color, alpha=0.2)

    # --- Set final style for the chart ---
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel('Total Environment Steps', fontsize=14)
    ax.set_ylabel('Episode Reward', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save and display the chart
    output_filename = output_filename.replace(" ", "_")
    plt.savefig(output_filename, dpi=300)
    plt.show()
    print(f"\n--- Comparison chart saved as {output_filename} ---")