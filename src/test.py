
import os
import numpy as np
import poly2graph as p2g
from sg_stat import StatsAnalyzer, plot_properties_vs_A_separate, plot_correlations_vs_A_separate

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Example usage:
    print(f'\npoly2graph version: {p2g.__version__}\n')

    # Define parameters.
    p_values = range(5, 8)         # For example, p = 5, 6, 7
    q_values = range(5, 8)         # For example, q = 5, 6, 7
    A_min = 0.01                 # Minimum A value
    A_max = 1.5                  # Maximum A value
    num_A = 3                   # Number of A samples
    N = 10                       # Number of samples per (p,q) pair and A
    num_workers = None           # Use default parallelism
    properties = [
        "number_of_nodes",
        "number_of_edges",
        "average_degree",
        "degree_assortativity",
        # "average_shortest_path_length",
        # "diameter",
        # "radius",
        # "eccentricities",
        # "global_efficiency",
        # "local_efficiency",
        # "average_clustering",
        # "triad_census",
        "triangle_count"
    ]
    output_folder = "../plots_A_analysis"
    plot_nancases = False         # Whether to plot graphs with NaN properties
    save_graphs = True            # Set to True to save each generated graph as an image

    # Instantiate and run the analyzer.
    A_values = np.linspace(A_min, A_max, num_A)
    analyzer = StatsAnalyzer(p_values=p_values,
                             q_values=q_values,
                             A_values=A_values,
                             N=N,
                             properties=properties,
                             num_workers=num_workers,
                             distribution='gaussian',
                             mu=0,
                             output_folder=output_folder,
                             plot_nancases=plot_nancases)

    correlations_over_A, aggregated_stats_over_A, A_values = analyzer.run_workflow()

    print("Plotting properties with error bars (STD_MEAN)...")
    analyzer.plot_properties_vs_A(with_error_bars=True)
    print("Plotting properties with only mean values (ONLY_MEAN)...")
    analyzer.plot_properties_vs_A(with_error_bars=False)
    print("Plotting correlations between properties over A...")
    analyzer.plot_correlations_vs_A()
    print("All plots, including correlation plots, have been generated and saved.")