import poly2graph as p2g
print(f'poly2graph version: {p2g.__version__}')
import math
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd
from itertools import combinations
import multiprocessing as mp



def generate_polynomial(q, p, random_perturbations=False, distribution='gaussian', mu=0, sigma=1, sample=100):
    """
    Generates a symmetric coefficient list for the polynomial H(z) = z^p + z^-q.
    If random_perturbations is True, it adds random coefficients for powers from
    (1-q) to (p-1).

    Distribution options:
    - 'gaussian': Gaussian distribution (use mu and sigma as mean and std)
    - 'uniform': Uniform distribution (mu and sigma define the lower and upper bounds)
    - Callable: A function or lambda that returns a random sample when called.

    If sample > 1, multiple samples of random perturbations are generated, resulting
    in multiple polynomial realizations. In this case, a 2D array is returned
    with shape (sample, size), where each row is one realization.

    Args:
    - q (int): Power of the term z^-q
    - p (int): Power of the term z^p
    - random_perturbations (bool): Whether to add random coefficients in between leading monomials
    - distribution (str or callable): Distribution type ('gaussian', 'uniform') or a callable returning random values.
    - mu (float): Parameter for the chosen distribution. For gaussian (mean), for uniform (lower bound).
    - sigma (float): Parameter for the chosen distribution. For gaussian (std), for uniform (upper bound).
    - sample (int): Number of random samples to generate if random_perturbations=True (default: 100)

    Returns:
    - np.array: 
        If sample=1 or random_perturbations=False, returns a 2D numpy array of shape (1, size).
        If sample>1 and random_perturbations=True, returns a 2D numpy array of shape (sample, size).
    """
    max_power = max(p, q)
    size = 2 * max_power + 1  # symmetric list length

    # Define a function to draw a random sample from the chosen distribution
    def draw_sample():
        if callable(distribution):
            return distribution()
        elif distribution == 'gaussian':
            return np.random.normal(mu, sigma)
        elif distribution == 'uniform':
            return np.random.uniform(mu, sigma)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}. Must be 'gaussian', 'uniform', or a Callable.")

    def generate_one():
        # Initialize the coefficient list with zeros
        coeffs = np.zeros(size, dtype=float)
        # Set the leading terms
        coeffs[max_power + p] = 1.0   # z^p
        coeffs[max_power - q] = 1.0   # z^-q

        #TODO: Parallelize the loop
        if random_perturbations:
            for r in range(1 - q, p):
                idx = r + max_power
                # Avoid overwriting the main terms
                if r != p and r != -q:
                    coeffs[idx] += draw_sample()
        return coeffs

    if random_perturbations and sample > 1:
        # Generate multiple polynomial realizations
        coeff_samples = [generate_one() for _ in range(sample)]
        return np.array(coeff_samples)
    else:
        # Generate a single polynomial and wrap it in a list to form a 2D array
        return np.array([generate_one()])


def get_irreducible_pairs(p_range, q_range):
    """Generate a list of irreducible (p, q) pairs where gcd(p, q) == 1."""
    return [(p, q) for p in p_range for q in q_range if math.gcd(p, q) == 1 and p != q]


def convert_to_simple_graph(sg):
    """Convert a MultiGraph or MultiDiGraph to a simple Graph."""
    if isinstance(sg, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph()
        G.add_nodes_from(sg.nodes(data=True))
        G.add_edges_from(sg.edges(data=True))
    else:
        G = sg
    return G



def compute_graph_properties(G, properties):
    """
    Compute specified graph properties.

    Parameters:
        G (networkx.Graph): The graph for which properties are computed.
        properties (list): List of properties to compute.

    Returns:
        dict: Dictionary of computed properties.
    """
    props = {}

    # Basic properties
    if "number_of_nodes" in properties:
        props["number_of_nodes"] = G.number_of_nodes() - 1  # Adjusting as per original code

    if "number_of_edges" in properties:
        props["number_of_edges"] = G.number_of_edges()

    if "average_degree" in properties:
        num_nodes = G.number_of_nodes() - 1
        num_edges = G.number_of_edges()
        props["average_degree"] = (2 * num_edges / num_nodes) if num_nodes > 0 else np.nan

    if "degree_assortativity" in properties:
        num_edges = G.number_of_edges()
        props["degree_assortativity"] = nx.degree_assortativity_coefficient(G) if num_edges > 0 else np.nan

    # Properties requiring connectedness
    connected_properties = ["average_shortest_path_length", "diameter", "radius", "eccentricities"]
    need_connected = any(prop in properties for prop in connected_properties)

    is_connected = G.number_of_nodes() > 1 and nx.is_connected(G) if need_connected else False

    if need_connected and is_connected:
        try:
            if "average_shortest_path_length" in properties:
                props["average_shortest_path_length"] = nx.average_shortest_path_length(G)

            if "diameter" in properties:
                props["diameter"] = nx.diameter(G)

            if "radius" in properties or "eccentricities" in properties:
                ecc = nx.eccentricity(G)
                if "eccentricities" in properties:
                    props["eccentricities"] = list(ecc.values())
                if "radius" in properties:
                    props["radius"] = min(ecc.values()) if ecc else np.nan

        except nx.NetworkXError:
            for prop in connected_properties:
                if prop in properties:
                    props[prop] = np.nan
    else:
        for prop in connected_properties:
            if prop in properties:
                props[prop] = np.nan

    # Efficiencies
    if "global_efficiency" in properties:
        try:
            props["global_efficiency"] = nx.global_efficiency(G)
        except:
            props["global_efficiency"] = np.nan

    if "local_efficiency" in properties:
        try:
            props["local_efficiency"] = nx.local_efficiency(G)
        except:
            props["local_efficiency"] = np.nan

    # Average Clustering
    if "average_clustering" in properties: # NA on multigraph
        try:
            num_nodes = G.number_of_nodes() - 1
            props["average_clustering"] = nx.average_clustering(G) if num_nodes > 1 else np.nan
        except:
            props["triangle_count"] = np.nan
    
    # Triad Census
    if "triad_census" in properties: # NA on multigraph
        DG = G.copy().to_directed()
        try:
            props["triad_census"] = nx.triadic_census(DG)
        except:
            props["triad_census"] = None

    # Triangle Count
    if "triangle_count" in properties:
        try:
            tri_per_node = nx.triangles(G)
            props["triangle_count"] = sum(tri_per_node.values()) / 3
        except:
            props["triangle_count"] = np.nan

    return props


def process_single_sample(args):
    """
    Worker function to handle a single polynomial sample.
    
    Parameters:
        args (tuple): Contains all arguments needed for computation.
            - i (int) : Index of the sample.
            - c (list or np.array) : Polynomial coefficients for this sample.
            - p (int) : p-value from the irreducible (p, q) pair.
            - q (int) : q-value from the irreducible (p, q) pair.
            - properties (list): List of properties to compute.
            - plot_nancases (bool): Whether or not to plot graphs with NaN properties.

    Returns:
        (i, dict): The index of the sample and the dictionary of computed properties.
    """
    i, c, p, q, properties, plot_nancases = args
    
    # Compute E_maxes and the spectral graph
    E_maxes = p2g.auto_Emaxes(c)
    sg = p2g.spectral_graph(
        c, E_max=E_maxes, 
        E_len=256, E_splits=4,
        s2g_kwargs={'add_pts': False}
    )
    # G = sg
    G = convert_to_simple_graph(sg)
    props = compute_graph_properties(G, properties)

    # Optionally handle NaN cases, but be aware that plotting from multiple processes
    # can sometimes cause race conditions or performance issues. Usually, you might
    # want to do plotting in the parent process if possible. 
    if plot_nancases:
        scalar_props = {k: v for k, v in props.items() if isinstance(v, float)}
        nan_props = [k for k, v in scalar_props.items() if np.isnan(v)]
        if nan_props:
            print(f"NaN encountered for (p,q)=({p},{q}) at sample {i}. NaN in: {nan_props}")
            plt.figure(figsize=(6, 4))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos=pos, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")
            plt.title(f"(p,q)=({p},{q}), Sample={i}, NaN in: {nan_props}")
            # plt.show()

    return i, props

def compute_graph_properties_for_pairs(
        irreducible_pairs, N, 
        perturb=True, 
        distribution='gaussian',
        mu=0, sigma=1, 
        properties=None, 
        num_workers=None,
        plot_nancases=False,
        return_correlation=False
    ):
    """
    Generate graph properties for each irreducible (p, q) pair.
    Now parallelized using multiprocessing.
    """
    if properties is None:
        properties = [
            "number_of_nodes",
            "number_of_edges",
            "average_degree",
            "degree_assortativity",
            "average_shortest_path_length",
            "diameter",
            "radius",
            "eccentricities",
            "global_efficiency",
            "local_efficiency",
            "average_clustering",
            "triad_census",
            "triangle_count"
        ]

    # Initialize the main dictionary
    graph_properties_dict = {pair: {prop: [] for prop in properties} for pair in irreducible_pairs}

    # Optionally initialize the correlation dictionary
    if return_correlation:
        correlation_dict = {pair: {} for pair in irreducible_pairs}

    # Iterate over irreducible pairs
    for (p, q) in irreducible_pairs:
        # Generate N samples (with or without perturbations)
        poly_samples = generate_polynomial(
            q, p,
            random_perturbations=perturb,
            distribution=distribution,
            mu=mu,
            sigma=sigma,
            sample=N
        )

        # Build argument list for multiprocessing
        # Each entry is a tuple containing all the arguments needed by 'process_single_sample'
        argument_list = [
            (i, c, p, q, properties, plot_nancases)
            for i, c in enumerate(poly_samples)
        ]

        # Use a pool of workers to process samples in parallel
        with mp.dummy.Pool(num_workers) as pool:
            results = pool.map(process_single_sample, argument_list)

        # 'results' is a list of (i, props) tuples, in the same order as argument_list
        # Store these in the main dictionary
        for i, props in results:
            for prop in properties:
                graph_properties_dict[(p, q)][prop].append(props.get(prop, np.nan))

    # Compute correlations if requested
    if return_correlation:
        for pair in irreducible_pairs:
            df = pd.DataFrame(graph_properties_dict[pair])
            # Drop columns with all NaNs to avoid errors
            df = df.dropna(axis=1, how='all')
            # Compute the correlation matrix
            corr_matrix = df.corr()
            # Convert the correlation matrix to a nested dictionary
            correlation_dict[pair] = corr_matrix.to_dict()

        return graph_properties_dict, correlation_dict

    else:
        return graph_properties_dict


def compute_statistics(graph_properties_dict, properties):
    """
    Compute mean and standard deviation for each property, ignoring NaNs.

    Parameters:
        graph_properties_dict (dict): Nested dictionary with properties.
        properties (list): List of properties to compute statistics for.

    Returns:
        dict: Nested dictionary with mean and std for each (p, q) pair and property.
    """
    mean_std_dict = {}
    for (p, q), props in graph_properties_dict.items():
        mean_std_dict[(p, q)] = {}
        for prop in properties:
            values = props[prop]
            if values:
                mean_std_dict[(p, q)][prop] = {
                    "mean": np.nanmean(values),
                    "std": np.nanstd(values)
                }
            else:
                mean_std_dict[(p, q)][prop] = {
                    "mean": np.nan,
                    "std": np.nan
                }
    return mean_std_dict



def group_statistics_by_pq(mean_std_dict, properties):
    """
    Aggregate statistics by grouping (p, q) pairs based on p+q and p-q.

    Parameters:
        mean_std_dict (dict): Nested dictionary with (p, q) as keys and property stats as values.
                              Example:
                              {
                                  (1, 2): {'property1': {'mean': 0.5, 'std': 0.1}, ...},
                                  (2, 3): {'property1': {'mean': 0.6, 'std': 0.2}, ...},
                                  ...
                              }
        properties (list): List of properties to group and aggregate.

    Returns:
        dict: Aggregated statistics grouped by 'p_plus_q' and 'p_minus_q'.
              Structure:
              {
                  'p_plus_q': {
                      3: {'property1': {'mean': aggregated_mean, 'std': aggregated_std}, ...},
                      5: {...},
                      ...
                  },
                  'p_minus_q': {
                      -1: {'property1': {'mean': aggregated_mean, 'std': aggregated_std}, ...},
                      1: {...},
                      ...
                  }
              }
    """
    grouped_stats = defaultdict(lambda: defaultdict(dict))
    for (p, q), props in mean_std_dict.items():
        p_plus_q = p + q
        p_minus_q = p - q
        for prop in properties:
            if prop in props:
                # Aggregate for p+q
                group_pq = grouped_stats['p_plus_q'][p_plus_q].setdefault(prop, {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0})
                group_pq['sum_mean'] += props[prop]['mean']
                group_pq['sum_std'] += props[prop]['std']
                group_pq['count'] += 1

                # Aggregate for p-q
                group_pq_diff = grouped_stats['p_minus_q'][p_minus_q].setdefault(prop, {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0})
                group_pq_diff['sum_mean'] += props[prop]['mean']
                group_pq_diff['sum_std'] += props[prop]['std']**2 # Using the average std of n indep random variables formula 
                # i.e sqrt(sigma_1^2+sigma_2^2+sigma_3^2+sigma_5^2+...sigma_n^2)=average_sigma
                group_pq_diff['count'] += 1
 
    # Compute overall mean and std for grouped statistics incrementally
    for group in ['p_plus_q', 'p_minus_q']:
        for key in grouped_stats[group]:
            for prop in properties:
                if prop in grouped_stats[group][key]:
                    sum_mean = grouped_stats[group][key][prop]['sum_mean']
                    sum_std = grouped_stats[group][key][prop]['sum_std']
                    count = grouped_stats[group][key][prop]['count']
                    grouped_stats[group][key][prop] = {
                        'mean': sum_mean / count if count > 0 else np.nan,
                        'std': np.sqrt(sum_std) / count if count > 0 else np.nan  #  sqrt(sigma_1^2+sigma_2^2+sigma_3^2+sigma_5^2+...sigma_n^2)=average_sigma
                    }
    
    return grouped_stats
 
 


def compute_avg_correlations_grouped(correlation_dict, properties):
    """
    Compute average Pearson correlation coefficients for each (p + q) and (p - q) group using a counting method.
    
    Parameters:
        correlation_dict (dict): Correlation data for a specific A.
                                 Structure: { (p, q): {prop_x: {prop_y: corr, ...}, ...}, ...}
        properties (list): List of properties to consider for correlation.
    
    Returns:
        dict: Average correlations grouped by 'p_plus_q' and 'p_minus_q'.
              Structure:
              {
                  'p_plus_q': {
                      group_key: { (prop_x, prop_y): avg_corr, ... },
                      ...
                  },
                  'p_minus_q': {
                      group_key: { (prop_x, prop_y): avg_corr, ... },
                      ...
                  }
              }
    """
    # Initialize dictionaries to accumulate sum and count for correlations
    sum_correlations = {
        'p_plus_q': defaultdict(lambda: defaultdict(float)),
        'p_minus_q': defaultdict(lambda: defaultdict(float))
    }
    count_correlations = {
        'p_plus_q': defaultdict(lambda: defaultdict(int)),
        'p_minus_q': defaultdict(lambda: defaultdict(int))
    }
    
    # Iterate over each (p, q) pair and its correlation matrix
    for (p, q), prop_corr in correlation_dict.items():
        p_plus_q = p + q
        p_minus_q = p - q

        # Iterate over each unique property pair to collect correlations
        for i in range(len(properties)):
            for j in range(i + 1, len(properties)):
                prop_x = properties[i]
                prop_y = properties[j]
                corr = prop_corr.get(prop_x, {}).get(prop_y, np.nan)
                if not np.isnan(corr):
                    pair = (prop_x, prop_y)
                    # Accumulate sum and count for p+q
                    sum_correlations['p_plus_q'][p_plus_q][pair] += corr
                    count_correlations['p_plus_q'][p_plus_q][pair] += 1
                    # Accumulate sum and count for p-q
                    sum_correlations['p_minus_q'][p_minus_q][pair] += corr
                    count_correlations['p_minus_q'][p_minus_q][pair] += 1
    
    # Compute the average correlation for each group and property pair
    final_avg_correlations = {
        'p_plus_q': defaultdict(dict),
        'p_minus_q': defaultdict(dict)
    }

    for group_type in ['p_plus_q', 'p_minus_q']:
        for group_key in sum_correlations[group_type]:
            for pair in sum_correlations[group_type][group_key]:
                total_corr = sum_correlations[group_type][group_key][pair]
                total_count = count_correlations[group_type][group_key][pair]
                avg_corr = total_corr / total_count if total_count > 0 else np.nan
                final_avg_correlations[group_type][group_key][pair] = avg_corr

    return final_avg_correlations
 

 
def main_workflow_mpc(p_values, q_values, A_min, A_max, num_A, N, properties, 
                  num_workers=1, output_folder='plots_A_analysis', plot_nancases=False):
    """
    Execute the workflow to generate graph properties, compute statistics, and plot results.

    Parameters:
        p_values (iterable): Range of p values.
        q_values (iterable): Range of q values.
        A_min (float): Minimum value of A (should include 0 for the base case).
        A_max (float): Maximum value of A.
        num_A (int): Number of A samples.
        N (int): Number of samples per (p, q) pair and A.
        properties (list): List of properties to compute and plot.
        num_workers (int): Number of parallel workers for multiprocessing.
        output_folder (str): Directory to save the plots.
        plot_nancases (bool): Whether to plot graphs with NaN properties.
    """
    # Step 1: Generate irreducible (p, q) pairs
    irreducible_pairs = get_irreducible_pairs(p_values, q_values)
    print(f"Total irreducible (p, q) pairs: {len(irreducible_pairs)}")

    # Step 2: Define range of A values (including A=0 for the base case)
    A_values = np.linspace(A_min, A_max, num_A)
    print(f"A values: {A_values}")

    # Initialize a dictionary to store aggregated stats over A
    aggregated_stats_over_A = {A: {'p_plus_q': defaultdict(dict), 'p_minus_q': defaultdict(dict)} for A in A_values}

    # Step 3: Iterate over A and compute properties
    correlations_over_A={}
    for A in A_values:
        print(f"Processing A = {A:.2f}")
        
        if A == 0:
            # Base case: No perturbations, single sample
            current_N = 1
            current_perturb = False
            current_sigma = 1  # Sigma is irrelevant when perturb=False
        else:
            # Perturbed cases
            current_N = N
            current_perturb = True
            current_sigma = A  # Sigma scales with A

   

        graph_properties_dict,correlations = compute_graph_properties_for_pairs(
            irreducible_pairs=irreducible_pairs,
            N=current_N,
            perturb=current_perturb,
            distribution='gaussian',
            mu=0,
            sigma=current_sigma,
            properties=properties,
            plot_nancases=plot_nancases, return_correlation=True
        )
         


        # Compute statistics (mean and std) for each (p, q)
        mean_std_dict = compute_statistics(graph_properties_dict, properties)

        # Group statistics by p + q and p - q
        aggregated_stats = group_statistics_by_pq(mean_std_dict, properties)

        # Store aggregated stats for this A
        aggregated_stats_over_A[A] = aggregated_stats

        # Group correlations by p + q and p - q
        aggregated_correlations = compute_avg_correlations_grouped(correlations, properties)
        # Store aggregated correlations for this A
        correlations_over_A[A] = aggregated_correlations
    return correlations_over_A,aggregated_stats_over_A,A_values,output_folder
     





if __name__ == "__main__":

    # Define your parameters
    p_values = range(5,8)      # Example: p from 1 to 1 (adjust as needed)
    q_values = range(5,8)      # Example: q from 1 to 1 (adjust as needed)
    A_min = 0.01                # Minimum A value
    A_max = 1.5                  # Maximum A value
    num_A = 15             # Number of A samples
    N = int(25)                  # Number of samples per (p, q) pair and A
    num_workers = None              # Number of parallel workers
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
        num_workers=num_workers,
        properties=properties,
        output_folder=output_folder,
        plot_nancases=plot_nancases
    )