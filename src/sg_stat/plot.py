import poly2graph as p2g
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict


def plot_properties_vs_A_separate(aggregated_stats_over_A, A_values, properties, output_folder, with_error_bars=True):
    """
    Plot each property against A separately, saving plots with or without error bars
    in corresponding subdirectories.

    Parameters:
        aggregated_stats_over_A (dict): Dictionary with A as keys and aggregated stats as values.
        A_values (array-like): Array of A values.
        properties (list): List of properties to plot.
        output_folder (str): Directory to save the plots.
        with_error_bars (bool): 
            - If True, plots include error bars and are saved in 'STD_MEAN'.
            - If False, plots show only mean values and are saved in 'ONLY_MEAN'.
    """
    # Determine subdirectory based on the plot type
    subfolder = 'STD_MEAN' if with_error_bars else 'ONLY_MEAN'
    subfolder_path = os.path.join(output_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    # Identify all possible p+q sums and p-q differences
    pq_sums = set()
    pq_diffs = set()
    for A in A_values:
        for group_type in ['p_plus_q', 'p_minus_q']:
            if group_type == 'p_plus_q':
                pq_sums.update(aggregated_stats_over_A[A][group_type].keys())
            elif group_type == 'p_minus_q':
                pq_diffs.update(aggregated_stats_over_A[A][group_type].keys())

    # Sort for consistent plotting
    pq_sums = sorted(pq_sums)
    pq_diffs = sorted(pq_diffs)

    # Define color maps
    color_map_sum = plt.get_cmap('tab10')
    color_map_diff = plt.get_cmap('tab20')

    for prop in properties:
        # Plot for p + q
        plt.figure(figsize=(12, 8))
        for idx, pq_sum in enumerate(pq_sums):
            means = []
            stds = []
            for A in A_values:
                group_props = aggregated_stats_over_A[A]['p_plus_q'].get(pq_sum, {}).get(prop, {})
                means.append(group_props.get('mean', np.nan))
                stds.append(group_props.get('std', np.nan))
            if with_error_bars:
                plt.errorbar(
                    A_values, 
                    means, 
                    yerr=stds, 
                    fmt='o-', 
                    color=color_map_sum(idx % color_map_sum.N),
                    label=f'p+q={pq_sum}', 
                    capsize=5
                )
            else:
                plt.plot(
                    A_values, 
                    means, 
                    'o-', 
                    color=color_map_sum(idx % color_map_sum.N),
                    label=f'p+q={pq_sum}'
                )

        plt.xlabel('Perturbation Amplitude A', fontsize=14)
        plt.ylabel(prop.replace('_', ' ').title(), fontsize=14)
        title_suffix = 'with Error Bars' if with_error_bars else 'Mean Only'
        plt.title(f'Graph Property: {prop.replace("_", " ").title()} vs Perturbation Amplitude A (Grouped by p+q) {title_suffix}', fontsize=16)
        plt.grid(True)
        plt.legend(title='p + q', fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot for p+q
        filename_sum = os.path.join(
            subfolder_path, 
            f"{'STD_MEAN' if with_error_bars else 'ONLY_MEAN'}_Property_vs_A_{prop.replace('_', '')}_grouped_by_p+q.png"
        )
        plt.savefig(filename_sum, dpi=300)
        plt.close()

        # Plot for p - q
        plt.figure(figsize=(12, 8))
        for idx, pq_diff in enumerate(pq_diffs):
            means = []
            stds = []
            for A in A_values:
                group_props = aggregated_stats_over_A[A]['p_minus_q'].get(pq_diff, {}).get(prop, {})
                means.append(group_props.get('mean', np.nan))
                stds.append(group_props.get('std', np.nan))
            if with_error_bars:
                plt.errorbar(
                    A_values, 
                    means, 
                    yerr=stds, 
                    fmt='s--', 
                    color=color_map_diff(idx % color_map_diff.N),
                    label=f'p-q={pq_diff}', 
                    capsize=5
                )
            else:
                plt.plot(
                    A_values, 
                    means, 
                    's--', 
                    color=color_map_diff(idx % color_map_diff.N),
                    label=f'p-q={pq_diff}'
                )

        plt.xlabel('Perturbation Amplitude A', fontsize=14)
        plt.ylabel(prop.replace('_', ' ').title(), fontsize=14)
        plt.title(f'Graph Property: {prop.replace("_", " ").title()} vs Perturbation Amplitude A (Grouped by p-q) {title_suffix}', fontsize=16)
        plt.grid(True)
        plt.legend(title='p - q', fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot for p-q
        filename_diff = os.path.join(
            subfolder_path, 
            f"{'STD_MEAN' if with_error_bars else 'ONLY_MEAN'}_Property_vs_A_{prop.replace('_', '')}_grouped_by_p-q.png"
        )
        plt.savefig(filename_diff, dpi=300)
        plt.close()

    print(f"{'STD_MEAN' if with_error_bars else 'ONLY_MEAN'} plots have been generated and saved in the '{subfolder}' subdirectory.")
 

def plot_correlations_vs_A_separate(
    aggregated_correlations, 
    A_values, 
    properties, 
    output_folder
):
    """
    Plot correlation coefficients between property pairs against perturbation amplitude A,
    grouped by 'p_plus_q' and 'p_minus_q', and save the plots in the specified output folder.

    Parameters:
        aggregated_correlations (dict): Nested dictionary with the following structure:
            {
                A_value: {
                    'p_plus_q': {
                        group_key_int: { (prop_x, prop_y): avg_corr, ... },
                        ...
                    },
                    'p_minus_q': {
                        group_key_int: { (prop_x, prop_y): avg_corr, ... },
                        ...
                    }
                },
                ...
            }
        A_values (list or array-like): Iterable of perturbation amplitude values.
        properties (list): List of property names to generate property pairs.
        output_folder (str): Directory path where the plots will be saved.

    Returns:
        None
    """
    # Define subdirectory for correlations
    subfolder = 'CORRELATIONS'
    subfolder_path = os.path.join(output_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    # Generate all unique property pairs
    property_pairs = list(combinations(properties, 2))

    # Collect and sort unique group_keys for 'p_plus_q' and 'p_minus_q'
    pq_groups = {
        'p_plus_q': sorted({
            pq 
            for A in A_values 
            for pq in aggregated_correlations.get(A, {}).get('p_plus_q', {})
        }),
        'p_minus_q': sorted({
            pq 
            for A in A_values 
            for pq in aggregated_correlations.get(A, {}).get('p_minus_q', {})
        })
    }

    # Define color maps for different group types
    color_maps = {
        'p_plus_q': plt.get_cmap('tab10'),
        'p_minus_q': plt.get_cmap('tab20')
    }

    # Helper function to plot and save correlations
    def plot_correlation(prop_pair, group_type):
        plt.figure(figsize=(12, 8))
        color_map = color_maps[group_type]
        
        for idx, group_key in enumerate(pq_groups[group_type]):
            # Extract correlation coefficients for the current property pair across all A_values
            correlations = []
            for A in A_values:
                group_data = aggregated_correlations.get(A, {}).get(group_type, {}).get(group_key, {})
                corr = group_data.get(prop_pair, np.nan)
                
                correlations.append(corr)
            
            label = f"{'p+q' if group_type == 'p_plus_q' else 'p-q'}={group_key}"
            fmt = 'o-' if group_type == 'p_plus_q' else 's--'
        
            plt.plot(
                A_values, 
                correlations, 
                fmt, 
                color=color_map(idx % color_map.N),
                label=label
            )

        plt.xlabel('Perturbation Amplitude A', fontsize=14)
        plt.ylabel('Correlation Coefficient', fontsize=14)
        prop_pair_str = ' & '.join([p.replace('_', ' ').title() for p in prop_pair])
        plt.title(
            f'Correlation: {prop_pair_str} vs Perturbation Amplitude A (Grouped by {"p+q" if group_type == "p_plus_q" else "p-q"})',
            fontsize=16
        )
        plt.grid(True)
        plt.legend(
            title='Group',
            fontsize=10, 
            title_fontsize=12, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )
        plt.tight_layout()

        # Create a safe filename by replacing spaces with underscores and removing special characters
        prop_pair_filename = '_'.join([p.replace(' ', '').replace('_', '') for p in prop_pair])
        filename = os.path.join(
            subfolder_path, 
            f"Correlation_vs_A_{prop_pair_filename}_grouped_by_{'p+q' if group_type == 'p_plus_q' else 'p-q'}.png"
        )
        plt.savefig(filename, dpi=300)
        plt.close()

    # Iterate over each property pair and plot correlations
    for prop_pair in property_pairs:
        # Plot for 'p_plus_q'
        if pq_groups['p_plus_q']:
            plot_correlation(prop_pair, 'p_plus_q')
        
        # Plot for 'p_minus_q'
        if pq_groups['p_minus_q']:
            plot_correlation(prop_pair, 'p_minus_q')

    print(f"Correlation plots have been generated and saved in the '{subfolder}' subdirectory.")


