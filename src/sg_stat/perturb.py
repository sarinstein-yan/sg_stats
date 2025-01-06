import poly2graph as p2g
print(p2g.__version__)
import math
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd
from itertools import combinations



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

    if "average_clustering" in properties:
        num_nodes = G.number_of_nodes() - 1
        props["average_clustering"] = nx.average_clustering(G) if num_nodes > 1 else np.nan

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

    # Triad Census
    if "triad_census" in properties:
        DG = G.to_directed()
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




def compute_graph_properties_for_pairs(irreducible_pairs, N, perturb=True, distribution='gaussian',
                                       mu=0, sigma=1, properties=None, plot_nancases=False,
                                       return_correlation=False):
    """
    Generate graph properties for each irreducible (p, q) pair.

    Parameters:
        irreducible_pairs (list of tuples): List of (p, q) pairs.
        N (int): Number of samples per pair.
        perturb (bool): Whether to apply perturbations.
        distribution (str or callable): Distribution type for perturbations.
        mu (float): Mean or lower bound for the distribution.
        sigma (float): Std or upper bound for the distribution.
        properties (list): List of properties to compute.
        plot_nancases (bool): Whether to plot graphs with NaN properties.
        return_correlation (bool): Whether to compute and return correlation matrices.

    Returns:
        dict or tuple:
            - If return_correlation is False:
                dict: Nested dictionary with properties for each (p, q) pair.
            - If return_correlation is True:
                tuple: (graph_properties_dict, correlation_dict)
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

    for (p, q) in irreducible_pairs:
        # Always generate N samples, with or without perturbations
        poly_samples = generate_polynomial(
            q, p,
            random_perturbations=perturb,
            distribution=distribution,
            mu=mu,
            sigma=sigma,
            sample=N
        )
        for i, c in enumerate(poly_samples):
            E_maxes = p2g.auto_Emaxes(c)
            sg = p2g.spectral_graph(c, E_max=E_maxes, s2g_kwargs={'add_pts': False})
            G = convert_to_simple_graph(sg)
            props = compute_graph_properties(G, properties)

            # Store scalar properties
            for prop in properties:
                graph_properties_dict[(p, q)][prop].append(props.get(prop, np.nan))

            # Handle NaN cases
            if plot_nancases:
                scalar_props = {k: v for k, v in props.items() if isinstance(v, float)}
                nan_props = [k for k, v in scalar_props.items() if np.isnan(v)]
                if nan_props:
                    print(f"NaN encountered for (p,q)=({p},{q}) at sample {i}. NaN in: {nan_props}")
                    plt.figure(figsize=(6, 4))
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw(G, pos=pos, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")
                    plt.title(f"(p,q)=({p},{q}), Sample={i}, NaN in: {nan_props}")
                    plt.show()

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
                grouped_stats['p_plus_q'][p_plus_q][prop] = grouped_stats['p_plus_q'][p_plus_q].get(prop, {'mean': [], 'std': []})
                grouped_stats['p_plus_q'][p_plus_q][prop]['mean'].append(props[prop]['mean'])
                grouped_stats['p_plus_q'][p_plus_q][prop]['std'].append(props[prop]['std'])

                # Aggregate for p-q
                grouped_stats['p_minus_q'][p_minus_q][prop] = grouped_stats['p_minus_q'][p_minus_q].get(prop, {'mean': [], 'std': []})
                grouped_stats['p_minus_q'][p_minus_q][prop]['mean'].append(props[prop]['mean'])
                grouped_stats['p_minus_q'][p_minus_q][prop]['std'].append(props[prop]['std'])

    # Compute overall mean and std for grouped statistics
    for group in ['p_plus_q', 'p_minus_q']:
        for key in grouped_stats[group]:
            for prop in properties:
                if prop in grouped_stats[group][key]:
                    means = grouped_stats[group][key][prop]['mean']
                    stds = grouped_stats[group][key][prop]['std']
                    grouped_stats[group][key][prop] = {
                        'mean': np.nanmean(means) if means else np.nan,
                        'std': np.nanstd(stds) if stds else np.nan
                    }

    return grouped_stats
 
 
def main_workflow(p_values, q_values, A_min, A_max, num_A, N, properties, output_folder='plots_A_analysis', plot_nancases=False):
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
        correlations_over_A[A] = correlations

        # Compute statistics (mean and std) for each (p, q)
        mean_std_dict = compute_statistics(graph_properties_dict, properties)

        # Group statistics by p + q and p - q
        aggregated_stats = group_statistics_by_pq(mean_std_dict, properties)

        # Store aggregated stats for this A
        aggregated_stats_over_A[A] = aggregated_stats
    return correlations_over_A,aggregated_stats_over_A,A_values,output_folder
     





if __name__ == "__main__":

    # Define your parameters
    p_values = range(5,8)      # Example: p from 1 to 1 (adjust as needed)
    q_values = range(5,8)      # Example: q from 1 to 1 (adjust as needed)
    A_min = 0.01                # Minimum A value
    A_max = 1.5                  # Maximum A value
    num_A = 15             # Number of A samples
    N = int(25)                  # Number of samples per (p, q) pair and A
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
    correlations_over_A,aggregated_stats_over_A,A_values,output_folder= main_workflow(
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