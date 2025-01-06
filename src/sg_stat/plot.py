import poly2graph as p2g
print(p2g.__version__)
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
 


def plot_correlations_vs_A_separate(correlations_over_A, A_values, properties, output_folder):
    """
    Plot the Pearson correlation coefficients between pairs of properties against A,
    grouped by p+q and p-q, and save the plots in a 'Correlations' subdirectory.
    
    Parameters:
        correlations_over_A (dict): Dictionary with A as keys and correlation_dict as values.
                                    correlation_dict structure: { (p, q): {prop_x: {prop_y: corr, ...}, ...}, ...}
        A_values (array-like): Array of A values.
        properties (list): List of properties to consider for correlation.
        output_folder (str): Directory to save the correlation plots.
    """
    import itertools

    # Define the subdirectory for correlation plots
    correlation_subfolder = 'Correlations'
    correlation_subfolder_path = os.path.join(output_folder, correlation_subfolder)
    os.makedirs(correlation_subfolder_path, exist_ok=True)
    
    # Generate all unique pairs of properties
    property_pairs = list(combinations(properties, 2))
    
    # Define color maps
    color_map_sum = plt.get_cmap('tab10')
    color_map_diff = plt.get_cmap('tab20')
    
    # Precompute p+q and p-q for each (p, q) pair
    pq_groups = defaultdict(lambda: defaultdict(list))  # {group_type: {group_key: [(p, q), ...], ...}, ...}
    for A in A_values:
        correlation_dict = correlations_over_A.get(A, {})
        for (p, q) in correlation_dict.keys():
            p_plus_q = p + q
            p_minus_q = p - q
            pq_groups['p_plus_q'][p_plus_q].append((p, q))
            pq_groups['p_minus_q'][p_minus_q].append((p, q))
    
    # Iterate over each pair of properties to plot their correlation over A
    for pair in property_pairs:
        prop_x, prop_y = pair
        pair_label = f"{prop_x} vs {prop_y}"
        
        # --- Plot Correlations Grouped by p+q ---
        plt.figure(figsize=(12, 8))
        legend_handles_sum = []
        for idx, (pq_sum, pairs) in enumerate(sorted(pq_groups['p_plus_q'].items())):
            correlation_coeffs = []
            for A in A_values:
                corr_values = []
                for (p, q) in pairs:
                    corr_matrix = correlations_over_A.get(A, {}).get((p, q), {})
                    corr = corr_matrix.get(prop_x, {}).get(prop_y, np.nan)
                    if not np.isnan(corr):
                        corr_values.append(corr)
                # Compute average correlation for this group and A
                if corr_values:
                    avg_corr = np.nanmean(corr_values)
                else:
                    avg_corr = np.nan
                correlation_coeffs.append(avg_corr)
            
            # Plot the average correlation over A for this p+q group
            plt.plot(
                A_values,
                correlation_coeffs,
                'o-', 
                color=color_map_sum(idx % color_map_sum.N),
                label=f'p+q={pq_sum}'
            )
        
        plt.xlabel('Perturbation Amplitude A', fontsize=14)
        plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
        plt.title(f'Correlation between {prop_x} and {prop_y} vs Perturbation Amplitude A (Grouped by p+q)', fontsize=16)
        plt.grid(True)
        plt.legend(title='p + q', fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the correlation plot for p+q
        filename_corr_sum = os.path.join(
            correlation_subfolder_path, 
            f"Correlation_{prop_x}_vs_{prop_y}_grouped_by_p+q.png"
        )
        plt.savefig(filename_corr_sum, dpi=300)
        plt.close()
        
        # --- Plot Correlations Grouped by p-q ---
        plt.figure(figsize=(12, 8))
        legend_handles_diff = []
        for idx, (pq_diff, pairs) in enumerate(sorted(pq_groups['p_minus_q'].items())):
            correlation_coeffs = []
            for A in A_values:
                corr_values = []
                for (p, q) in pairs:
                    corr_matrix = correlations_over_A.get(A, {}).get((p, q), {})
                    corr = corr_matrix.get(prop_x, {}).get(prop_y, np.nan)
                    if not np.isnan(corr):
                        corr_values.append(corr)
                # Compute average correlation for this group and A
                if corr_values:
                    avg_corr = np.nanmean(corr_values)
                else:
                    avg_corr = np.nan
                correlation_coeffs.append(avg_corr)
            
            # Plot the average correlation over A for this p-q group
            plt.plot(
                A_values,
                correlation_coeffs,
                's--', 
                color=color_map_diff(idx % color_map_diff.N),
                label=f'p-q={pq_diff}'
            )
        
        plt.xlabel('Perturbation Amplitude A', fontsize=14)
        plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
        plt.title(f'Correlation between {prop_x} and {prop_y} vs Perturbation Amplitude A (Grouped by p-q)', fontsize=16)
        plt.grid(True)
        plt.legend(title='p - q', fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the correlation plot for p-q
        filename_corr_diff = os.path.join(
            correlation_subfolder_path, 
            f"Correlation_{prop_x}_vs_{prop_y}_grouped_by_p-q.png"
        )
        plt.savefig(filename_corr_diff, dpi=300)
        plt.close()