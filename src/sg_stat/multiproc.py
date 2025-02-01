import poly2graph as p2g
import math, os, pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from .plot import plot_properties_vs_A_separate, plot_correlations_vs_A_separate
import warnings



class StatsAnalyzer:
    def __init__(self, 
            p_values, q_values, A_values,
            N, properties, 
            num_workers=None,
            distribution='gaussian',
            mu=0,
            output_folder=None,
            plot_nancases=False,
        ):
        """
        Initialize the StatsAnalyzer.

        Parameters:
            p_values (iterable): p values to consider.
            q_values (iterable): q values to consider.
            A_values (iterable): A values to consider.
            N (int): Number of samples per (p,q, A).
            properties (list): List of graph properties to compute.
            num_workers (int): Number of parallel workers (default: None).
            distribution (str or callable): Random distribution for polynomial perturbations.
            mu (float): Mean or lower bound for the distribution.
            output_folder (str): Base folder to save graphs and noncases. If None, saving is disabled.
            plot_nancases (bool): If True, plot (and save) graphs with NaN properties.
        """
        self.p_values = p_values
        self.q_values = q_values
        self.A_values = A_values
        self.N = N
        self.properties = properties
        self.num_workers = num_workers
        self.distribution = distribution
        self.mu = mu
        self.output_folder = output_folder
        self.plot_nancases = plot_nancases
        self.correlations_over_A = None
        self.aggregated_stats_over_A = None

        # Create output directories.
        if self.output_folder is None:
            self.save_data = False
        elif isinstance(self.output_folder, str):
            self.save_data = True
        else:
            raise ValueError("output_folder must be a string or None")
        
        if self.save_data:
            self.graphs_folder = os.path.join(self.output_folder, "graphs")
            os.makedirs(self.graphs_folder, exist_ok=True)
            self.data_folder = os.path.join(self.output_folder, "data")
            os.makedirs(self.data_folder, exist_ok=True)
        if self.plot_nancases:
            self.noncase_folder = os.path.join(self.output_folder, "noncases")
            os.makedirs(self.noncase_folder, exist_ok=True)
            

    def generate_polynomial(self, q, p, random_perturbations, sigma, sample):
        """
        Generate a polynomial coefficient vector (or many samples) for h(z)=z^p+z^-q
        with optional random perturbations between the two main terms.
        """
        max_power = max(p, q)
        size = 2 * max_power + 1

        def draw_sample():
            if callable(self.distribution):
                return self.distribution()
            elif self.distribution == 'gaussian':
                return np.random.normal(self.mu, sigma)
            elif self.distribution == 'uniform':
                return np.random.uniform(self.mu, sigma)
            else:
                raise ValueError(f"Unsupported distribution type: {self.distribution}")

        def generate_one():
            coeffs = np.zeros(size, dtype=float)
            coeffs[max_power + p] = 1.0  # leading term z^p
            coeffs[max_power - q] = 1.0  # leading term z^-q
            if random_perturbations:
                for r in range(1 - q, p):
                    if r == p or r == -q:
                        continue
                    idx = r + max_power
                    coeffs[idx] += draw_sample()
            return coeffs

        if random_perturbations and sample > 1:
            coeff_samples = [generate_one() for _ in range(sample)]
            return np.array(coeff_samples)
        else:
            return np.array([generate_one()])

    def get_irreducible_pairs(self):
        """Return a list of (p, q) pairs such that gcd(p, q)==1 and p != q."""
        return [(p, q) for p in self.p_values
                for q in self.q_values if math.gcd(p, q) == 1 and p != q]

    def convert_to_simple_graph(self, sg):
        """Convert a (multi)graph into a simple graph if necessary."""
        if isinstance(sg, (nx.MultiGraph, nx.MultiDiGraph)):
            G = nx.Graph()
            G.add_nodes_from(sg.nodes(data=True))
            G.add_edges_from(sg.edges(data=True))
        else:
            G = sg
        return G

    def compute_graph_properties(self, G):
        """
        Compute the requested graph properties from a given networkx graph.
        
        Parameters:
            G (networkx.Graph or nx.MultiGraph): The graph for which to compute properties.
            
        Returns:
            dict: Computed properties.
        """
        props = {}

        if "number_of_nodes" in self.properties:
            props["number_of_nodes"] = G.number_of_nodes() # removed the -1 offset
        if "number_of_edges" in self.properties:
            props["number_of_edges"] = G.number_of_edges()
        if "average_degree" in self.properties:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            props["average_degree"] = (2 * num_edges / num_nodes) if num_nodes > 0 else np.nan
        if "degree_assortativity" in self.properties:
            num_edges = G.number_of_edges()
            props["degree_assortativity"] = (nx.degree_assortativity_coefficient(G)
                                             if num_edges > 0 else np.nan)

        # Properties that require a connected graph.
        connected_properties = ["average_shortest_path_length", "diameter", "radius", "eccentricities"]
        need_connected = any(prop in self.properties for prop in connected_properties)
        is_connected = (G.number_of_nodes() > 1 and nx.is_connected(G)) if need_connected else False

        if need_connected and is_connected:
            try:
                if "average_shortest_path_length" in self.properties:
                    props["average_shortest_path_length"] = nx.average_shortest_path_length(G)
                if "diameter" in self.properties:
                    props["diameter"] = nx.diameter(G)
                if "radius" in self.properties or "eccentricities" in self.properties:
                    ecc = nx.eccentricity(G)
                    if "eccentricities" in self.properties:
                        props["eccentricities"] = list(ecc.values())
                    if "radius" in self.properties:
                        props["radius"] = min(ecc.values()) if ecc else np.nan
            except nx.NetworkXError:
                for prop in connected_properties:
                    if prop in self.properties:
                        props[prop] = np.nan
        else:
            for prop in connected_properties:
                if prop in self.properties:
                    props[prop] = np.nan

        # Additional properties.
        if "global_efficiency" in self.properties:
            try:
                props["global_efficiency"] = nx.global_efficiency(G)
            except Exception:
                props["global_efficiency"] = np.nan

        if "local_efficiency" in self.properties:
            try:
                props["local_efficiency"] = nx.local_efficiency(G)
            except Exception:
                props["local_efficiency"] = np.nan

        if "average_clustering" in self.properties: # NA on multigraph
            try:
                num_nodes = G.number_of_nodes()
                props["average_clustering"] = nx.average_clustering(G) if num_nodes > 0 else np.nan
            except Exception:
                props["average_clustering"] = np.nan

        if "triad_census" in self.properties: # NA on multigraph
            DG = G.copy().to_directed()
            try:
                props["triad_census"] = nx.triadic_census(DG)
            except Exception:
                props["triad_census"] = None

        if "triangle_count" in self.properties:
            try:
                tri_per_node = nx.triangles(G)
                props["triangle_count"] = sum(tri_per_node.values()) / 3
            except Exception:
                props["triangle_count"] = np.nan

        return props

    def process_single_sample(self, args):
        """
        Worker function that processes one polynomial sample.
        
        Parameters:
            args (tuple): (i, coeff, p, q, A)
            
        Returns:
            tuple: (i, coeff, spectral graph object)
        """
        i, coeff, p, q, A = args
        E_maxes = p2g.auto_Emaxes(coeff)
        sg = p2g.spectral_graph(
            coeff,
            E_max=E_maxes,
            E_len=200,
            E_splits=4,
            s2g_kwargs={'add_pts': False} # do not save points on the edges
        )
        # sg = self.convert_to_simple_graph(sg)
        return i, coeff, sg

    def generate_graph(self, irreducible_pairs, N, perturb, sigma, A):
        """
        For each (p,q) pair, generate polynomial samples and compute the corresponding spectral graphs.
        
        Parameters:
            irreducible_pairs (list): List of (p,q) pairs.
            N (int): Number of samples per (p,q) pair.
            perturb (bool): Whether to add random perturbations.
            sigma (float): The sigma parameter for the perturbation distribution.
            A (float): The current A value (for record-keeping).
            
        Returns:
            dict: A dictionary mapping each (p,q) pair to a list of tuples (i, coeff, graph).
        """
        graphs_dict = {}
        # pool = mp.Pool(self.num_workers) # process-based pool
        pool = mp.dummy.Pool(self.num_workers) # thread-based pool
        for (p, q) in irreducible_pairs:
            poly_samples = self.generate_polynomial(q, p,
                                                    random_perturbations=perturb,
                                                    sigma=sigma,
                                                    sample=N)
            argument_list = [(i, coeff, p, q, A) for i, coeff in enumerate(poly_samples)]
            results = pool.map(self.process_single_sample, argument_list)
            graphs_dict[(p, q)] = results  # Each result is a tuple (i, coeff, graph)
        pool.close()
        pool.join()
        return graphs_dict

    def save_graph_partition(self, graphs_batch, filename):
        """
        Save a batch of graphs with their corresponding polynomial coefficient lists to disk.
        
        Parameters:
            graphs_batch (dict): Dictionary mapping each (p,q) pair to a list of tuples 
                                 (i, coeff, graph).
            filename (str): Full path (including filename) where the batch will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(graphs_batch, f)
        print(f"Saved graph partition to {filename}")
    
    def load_graph_partition(self, filename):
        """
        Load a batch of graphs with their corresponding polynomial coefficient lists from disk.
        
        Parameters:
            filename (str): Full path (including filename) of the saved partition.
            
        Returns:
            The loaded graphs batch (dict).
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded graph partition from {filename}")
        return data
    
    def load_graph_as_per_A(self, A_values=None):
        """
        Load graphs with their corresponding polynomial coefficient lists from disk.
        
        Parameters:
            A_values (iterable): The A values for which to load the graph partitions.
            
        Returns:
            dict: A dictionary mapping each A value to the loaded graphs.
        """
        loaded_graphs = {}
        if A_values is None:
            A_values = self.A_values
        for A in A_values:
            partition_filename = os.path.join(self.graphs_folder, f"graphs_A_{A:.3f}.pkl")
            if os.path.exists(partition_filename):
                with open(partition_filename, "rb") as f:
                    loaded_graphs[A] = pickle.load(f)
                print(f"Loaded graph partition from {partition_filename}")
            else:
                print(f"Partition file {partition_filename} does not exist.")
        return loaded_graphs

    def compute_statistics(self, graph_properties_dict):
        """
        Compute the mean and standard deviation for each property over all samples.
        
        Parameters:
            graph_properties_dict (dict): Dictionary mapping each (p,q) pair to property lists.
            
        Returns:
            dict: Nested dictionary with mean and std for each (p,q) pair.
        """
        mean_std_dict = {}
        for (p, q), props in graph_properties_dict.items():
            mean_std_dict[(p, q)] = {}
            for prop in self.properties:
                values = props[prop]
                if values:
                    mean_std_dict[(p, q)][prop] = {
                        "mean": np.nanmean(values),
                        "std": np.nanstd(values)
                    }
                else:
                    mean_std_dict[(p, q)][prop] = {"mean": np.nan, "std": np.nan}
        return mean_std_dict

    def group_statistics_by_pq(self, mean_std_dict):
        """
        Aggregate statistics by grouping (p,q) pairs according to p+q and p-q.
        Returns a plain dictionary that can be pickled.
        """
        grouped_stats = {'p_plus_q': {}, 'p_minus_q': {}}
        
        for (p, q), props in mean_std_dict.items():
            p_plus_q = p + q
            p_minus_q = p - q
            
            # Ensure the keys exist.
            if p_plus_q not in grouped_stats['p_plus_q']:
                grouped_stats['p_plus_q'][p_plus_q] = {}
            if p_minus_q not in grouped_stats['p_minus_q']:
                grouped_stats['p_minus_q'][p_minus_q] = {}
            
            for prop in self.properties:
                if prop in props:
                    # For p+q
                    if prop not in grouped_stats['p_plus_q'][p_plus_q]:
                        grouped_stats['p_plus_q'][p_plus_q][prop] = {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0}
                    grouped_stats['p_plus_q'][p_plus_q][prop]['sum_mean'] += props[prop]['mean']
                    grouped_stats['p_plus_q'][p_plus_q][prop]['sum_std'] += props[prop]['std']
                    grouped_stats['p_plus_q'][p_plus_q][prop]['count'] += 1

                    # For p-q
                    if prop not in grouped_stats['p_minus_q'][p_minus_q]:
                        grouped_stats['p_minus_q'][p_minus_q][prop] = {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0}
                    grouped_stats['p_minus_q'][p_minus_q][prop]['sum_mean'] += props[prop]['mean']
                    grouped_stats['p_minus_q'][p_minus_q][prop]['sum_std'] += props[prop]['std'] ** 2
                    grouped_stats['p_minus_q'][p_minus_q][prop]['count'] += 1

        # Convert the aggregated sums into mean and standard deviation.
        for group in ['p_plus_q', 'p_minus_q']:
            for key in grouped_stats[group]:
                for prop in grouped_stats[group][key]:
                    data = grouped_stats[group][key][prop]
                    count = data['count']
                    mean_val = data['sum_mean'] / count if count > 0 else np.nan
                    std_val = np.sqrt(data['sum_std']) / count if count > 0 else np.nan
                    grouped_stats[group][key][prop] = {'mean': mean_val, 'std': std_val}
        
        return grouped_stats

    def compute_avg_correlations_grouped(self, correlation_dict):
        """
        Compute average Pearson correlation coefficients for each (p+q) and (p-q) group.
        Returns a plain dictionary that can be pickled.
        """
        sum_correlations = {'p_plus_q': {}, 'p_minus_q': {}}
        count_correlations = {'p_plus_q': {}, 'p_minus_q': {}}
        
        for (p, q), prop_corr in correlation_dict.items():
            p_plus_q = p + q
            p_minus_q = p - q
            
            # Ensure keys exist for each group.
            if p_plus_q not in sum_correlations['p_plus_q']:
                sum_correlations['p_plus_q'][p_plus_q] = {}
                count_correlations['p_plus_q'][p_plus_q] = {}
            if p_minus_q not in sum_correlations['p_minus_q']:
                sum_correlations['p_minus_q'][p_minus_q] = {}
                count_correlations['p_minus_q'][p_minus_q] = {}
            
            for i in range(len(self.properties)):
                for j in range(i + 1, len(self.properties)):
                    prop_x = self.properties[i]
                    prop_y = self.properties[j]
                    corr = prop_corr.get(prop_x, {}).get(prop_y, np.nan)
                    if not np.isnan(corr):
                        pair_key = (prop_x, prop_y)
                        # For p+q
                        if pair_key not in sum_correlations['p_plus_q'][p_plus_q]:
                            sum_correlations['p_plus_q'][p_plus_q][pair_key] = 0.0
                            count_correlations['p_plus_q'][p_plus_q][pair_key] = 0
                        sum_correlations['p_plus_q'][p_plus_q][pair_key] += corr
                        count_correlations['p_plus_q'][p_plus_q][pair_key] += 1
                        
                        # For p-q
                        if pair_key not in sum_correlations['p_minus_q'][p_minus_q]:
                            sum_correlations['p_minus_q'][p_minus_q][pair_key] = 0.0
                            count_correlations['p_minus_q'][p_minus_q][pair_key] = 0
                        sum_correlations['p_minus_q'][p_minus_q][pair_key] += corr
                        count_correlations['p_minus_q'][p_minus_q][pair_key] += 1

        final_avg_correlations = {'p_plus_q': {}, 'p_minus_q': {}}
        for group_type in ['p_plus_q', 'p_minus_q']:
            for group_key in sum_correlations[group_type]:
                final_avg_correlations[group_type][group_key] = {}
                for pair_key in sum_correlations[group_type][group_key]:
                    total_corr = sum_correlations[group_type][group_key][pair_key]
                    total_count = count_correlations[group_type][group_key][pair_key]
                    avg_corr = total_corr / total_count if total_count > 0 else np.nan
                    final_avg_correlations[group_type][group_key][pair_key] = avg_corr

        return final_avg_correlations


    def run_workflow(self):
        """
        Main workflow:
         - Generate polynomial coefficients.
         - For each A in A_values:
             * Generate the batch of spectral graphs.
             * Save the batch (graphs and corresponding coefficients) to disk.
             * Compute graph properties for each generated graph.
             * Aggregate the statistics and correlations.
         - After processing all A values, save the aggregated statistics and correlations to disk.
         - Return correlations_over_A, aggregated_stats_over_A, A_values.
         
        Returns:
            tuple: (correlations_over_A, aggregated_stats_over_A, A_values)
        """
        irreducible_pairs = self.get_irreducible_pairs()
        print(f"Total irreducible (p,q) pairs: {len(irreducible_pairs)}")
        A_values = self.A_values
        print(f"A values: {A_values}")
        
        aggregated_stats_over_A = {}
        correlations_over_A = {}

        # For each A, generate graphs, save them, compute properties and correlations.
        for A in A_values:
            partition_filename = os.path.join(self.graphs_folder, f"graphs_A_{A:.3f}.pkl")
            if os.path.exists(partition_filename):
                print(f"Loading graph partition for A = {A:.3f}")
                graphs_batch = self.load_graph_partition(partition_filename)
            else:
                print(f"Processing A = {A:.3f}")
                if A == 0:
                    current_N = 1
                    current_perturb = False
                    current_sigma = 1  # Base case; sigma is not used.
                else:
                    current_N = self.N
                    current_perturb = True
                    current_sigma = A

                # Generate the batch of graphs for the current A.
                graphs_batch = self.generate_graph(irreducible_pairs, current_N, current_perturb, current_sigma, A)
                
                # If saving is enabled, save this batch.
                if self.save_data:
                    self.save_graph_partition(graphs_batch, partition_filename)
            
            # Compute graph properties for each (p,q) pair.
            graph_properties_dict = {pair: {prop: [] for prop in self.properties} for pair in irreducible_pairs}
            correlation_dict = {}
            for pair, samples in graphs_batch.items():
                props_list = []
                for i, coeff, G in samples:
                    props = self.compute_graph_properties(G)
                    props_list.append(props)
                    # Append each property value to the corresponding list.
                    for prop in self.properties:
                        graph_properties_dict[pair][prop].append(props.get(prop, np.nan))
                # Compute correlation for this (p,q) pair if there is more than one sample.
                if len(props_list) > 1:
                    df = pd.DataFrame(props_list)
                    df = df.dropna(axis=1, how='all')
                    corr_matrix = df.corr()
                    correlation_dict[pair] = corr_matrix.to_dict()
                else:
                    correlation_dict[pair] = {}
            
            # Compute statistics (mean and std) for the current A.
            mean_std_dict = self.compute_statistics(graph_properties_dict)
            aggregated_stats = self.group_statistics_by_pq(mean_std_dict)
            aggregated_stats_over_A[A] = aggregated_stats

            # Compute aggregated correlations.
            aggregated_correlations = self.compute_avg_correlations_grouped(correlation_dict)
            correlations_over_A[A] = aggregated_correlations

        # After processing all A values, save the aggregated data if enabled.
        if self.save_data:
            stats_filename = os.path.join(self.data_folder, "aggregated_stats_over_A.pkl")
            corr_filename = os.path.join(self.data_folder, "correlations_over_A.pkl")
            with open(stats_filename, "wb") as f:
                pickle.dump(aggregated_stats_over_A, f)
            print(f"Saved aggregated statistics to {stats_filename}")
            with open(corr_filename, "wb") as f:
                pickle.dump(correlations_over_A, f)
            print(f"Saved correlations data to {corr_filename}")

        self.correlations_over_A = correlations_over_A
        self.aggregated_stats_over_A = aggregated_stats_over_A
        return correlations_over_A, aggregated_stats_over_A, A_values

    def plot_properties_vs_A(self, with_error_bars=True):
        """
        Plot each property against A separately, saving plots with or without error bars
        in corresponding subdirectories.

        Parameters:
            with_error_bars (bool): 
                - If True, plots include error bars and are saved in 'STD_MEAN'.
                - If False, plots show only mean values and are saved in 'ONLY_MEAN'.
        """
        plot_properties_vs_A_separate(
            self.aggregated_stats_over_A,
            self.A_values,
            self.properties,
            os.path.join(self.output_folder, "plots"),
            with_error_bars
        )

    def plot_correlations_vs_A(self):
        """
        Plot correlation coefficients between property pairs against perturbation amplitude A,
        grouped by 'p_plus_q' and 'p_minus_q', and save the plots in the specified output folder.
        """
        plot_correlations_vs_A_separate(
            self.correlations_over_A,
            self.A_values,
            self.properties,
            os.path.join(self.output_folder, "plots")
        )


if __name__ == "__main__":
    # Example usage:
    import warnings
    warnings.filterwarnings("ignore")
    print(f'\npoly2graph version: {p2g.__version__}\n')

    # Define parameters.
    p_values = range(5, 8)         # For example, p = 5, 6, 7
    q_values = range(5, 8)         # For example, q = 5, 6, 7
    A_min = 0.01                 # Minimum A value
    A_max = 1.5                  # Maximum A value
    num_A = 4#60                   # Number of A samples
    N = 20#50                       # Number of samples per (p,q) pair and A
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
    output_folder = "plots_A_analysis"
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