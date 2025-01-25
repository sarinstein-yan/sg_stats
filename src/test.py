
from sg_stat import main_workflow_mpc, plot_properties_vs_A_separate, plot_correlations_vs_A_separate

if __name__ == "__main__":

    # Define your parameters
    p_values = range(5,8)      # Example: p from 1 to 1 (adjust as needed)
    q_values = range(5,8)      # Example: q from 1 to 1 (adjust as needed)
    A_min = 0.01                # Minimum A value
    A_max = 1.5                  # Maximum A value
    num_A = 5             # Number of A samples
    N = int(5)                  # Number of samples per (p, q) pair and A
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
    output_folder = 'plots_A_analysis'
    plot_nancases = False  # Set to True if you want to plot NaN cases

    # Execute the workflow
    correlations_over_A,aggregated_stats_over_A,A_values,output_folder= main_workflow_mpc(
        p_values=p_values,
        q_values=q_values,
        A_min=A_min,
        A_max=A_max,
        num_A=num_A,
        N=N,
        properties=properties,
        output_folder=output_folder,
        plot_nancases=plot_nancases
    )

    # Step 4: Plot the results with error bars (STD_MEAN)
    print("Plotting properties with error bars (STD_MEAN)...")
    plot_properties_vs_A_separate(
        aggregated_stats_over_A=aggregated_stats_over_A,
        A_values=A_values,
        properties=properties,
        output_folder=output_folder,
        with_error_bars=True  # Generates plots with error bars in 'STD_MEAN' subdirectory
    )

    # Step 5: Plot the results with only mean values (ONLY_MEAN)
    print("Plotting properties with only mean values (ONLY_MEAN)...")
    plot_properties_vs_A_separate(
        aggregated_stats_over_A=aggregated_stats_over_A,
        A_values=A_values,
        properties=properties,
        output_folder=output_folder,
        with_error_bars=False  # Generates plots with only mean values in 'ONLY_MEAN' subdirectory
    )

    # Step 6: Plot correlations between properties over A
    print("Plotting correlations between properties over A...")
    plot_correlations_vs_A_separate(
            correlations_over_A,
            A_values=A_values,
            properties=properties,
            output_folder=output_folder
        )


    print("All plots, including correlation plots, have been generated and saved.")