import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mTE_graph_calculation import compute_MTE_embedded, compute_MTE, silvermans_rule
import csv
import re
import sys
from scipy.interpolate import interp1d
from itertools import combinations
from joblib import Parallel, delayed

# Function to interpolate matrices
def interpolate_matrices(agent_matrices, agent_time_stamps, all_time_points):
    """
    Interpolates a series of transition matrices to a common set of time points.

    Args:
        agent_matrices (list): List of transition matrices for an agent.
        agent_time_stamps (list): Corresponding timestamps for the matrices.
        all_time_points (list): Common set of time points to interpolate to.

    Returns:
        list: Interpolated transition matrices aligned to all_time_points.
    """
    interpolated_matrices = []
    # Flatten matrices for interpolation
    flattened_matrices = [matrix.flatten() for matrix in agent_matrices]
    flattened_matrices = np.array(flattened_matrices)

    # Interpolate each element in the flattened matrix
    interpolated_flat_matrices = []
    for i in range(flattened_matrices.shape[1]):
        element_series = flattened_matrices[:, i]
        f = interp1d(agent_time_stamps, element_series, kind='linear', fill_value='extrapolate')
        interpolated_element = f(all_time_points)
        interpolated_flat_matrices.append(interpolated_element)

    # Reconstruct interpolated matrices
    interpolated_flat_matrices = np.array(interpolated_flat_matrices).T  # Transpose back
    num_elements = agent_matrices[0].size
    matrix_shape = agent_matrices[0].shape

    for i in range(len(all_time_points)):
        interpolated_matrix = interpolated_flat_matrices[i].reshape(matrix_shape)
        interpolated_matrices.append(interpolated_matrix)

    return interpolated_matrices

def clean_character_name(name):
    # Strip unwanted characters and fix common formatting issues
    name = re.sub(r'[^a-zA-Z\s]', '', name)  # Remove any non-alphabetic and non-space characters
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with a single space
    return name.strip().lower()  # Trim spaces and convert to lower case

def load_character_aliases(file_path):
    character_aliases = []
    try:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Process each name in the row after splitting by commas or semicolons
                aliases = [clean_character_name(name) for name in row for name in re.split(r'[,;]\s*', name) if name]
                if aliases:
                    character_aliases.append(aliases)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return character_aliases

# Load character aliases from CSV
file_path = '../data_collection/characters_list.csv'
characters = load_character_aliases(file_path)
print("Characters with aliases:", characters)

# Extract the primary name from each list of aliases
agent_names = [aliases[0] for aliases in characters if aliases]  # Ensure there's at least one alias

def load_final_matrices(character):
    return np.load(f'transition_matrices_final_{character}.npy', allow_pickle=True)

def load_time_series_matrices(character):
    try:
        return np.load(f'transition_matrices_{character}.npy', allow_pickle=True)
    except Exception as e:
        print(f"Error loading matrix for {character}: {e}")
        return None

def load_time_stamps(character):
    try:
        return np.load(f'time_stamps_{character}.npy', allow_pickle=True)
    except Exception as e:
        print(f"Error loading timestamps for {character}: {e}")
        return None

# Load time series matrices and timestamps
time_series_agent_matrices = {agent: load_time_series_matrices(agent) for agent in agent_names}
time_stamps = {agent: load_time_stamps(agent) for agent in agent_names}

# Check if matrices were loaded successfully
if any(matrix is None for matrix in time_series_agent_matrices.values()):
    print("Error: Some matrices failed to load.")
    sys.exit(1)

if any(ts is None for ts in time_stamps.values()):
    print("Error: Some timestamps failed to load.")
    sys.exit(1)

print("agent_names:", agent_names)
print("time_series_agent_matrices keys:", time_series_agent_matrices.keys())

final_agent_matrices = {agent: load_final_matrices(agent) for agent in agent_names}

# Collect all time points
all_time_points = sorted(set().union(*[set(time_stamps[agent]) for agent in agent_names]))

# Interpolate matrices for each agent
interpolated_time_series_agent_matrices = {}
for agent in agent_names:
    agent_matrices = time_series_agent_matrices[agent]
    agent_time_stamps = time_stamps[agent]
    interpolated_matrices = interpolate_matrices(agent_matrices, agent_time_stamps, all_time_points)
    interpolated_time_series_agent_matrices[agent] = interpolated_matrices

print("Interpolated matrices for agents:", interpolated_time_series_agent_matrices.keys())

# Define these for the data
lag = 1
dimensions = 2
use_embeddings = True
significant_te_threshold = 0.002  # threshold for significant TE values

def generate_all_graphs():
    # Define the structure of categories and terms
    categories = ['high-level', 'task-specific', 'context-specific']
    terms = ['short-term', 'medium-term', 'long-term']

    all_graphs = []  # This will collect all graphs over time

    # Determine the number of timesteps
    num_timesteps = len(all_time_points)  # Use length of all_time_points

    for time in range(num_timesteps):
        G = nx.DiGraph()

        # Define node positions and customize spacing
        positions = {}
        x_offset, y_offset, group_offset = 2, 2, 10

        # Initialize nodes
        for i, category in enumerate(categories):
            for j, term in enumerate(terms):
                for k, agent in enumerate(agent_names):
                    node_id = f"{category}_{term}_{agent}"
                    positions[node_id] = (i * x_offset + k * group_offset, j * y_offset)
                    G.add_node(node_id, pos=positions[node_id], label=node_id, group=f"{category}_{term}")

        # Prepare tasks for parallel computation
        tasks = []
        for category in categories:
            for term in terms:
                agent_pairs = combinations(agent_names, 2)
                for agent_i, agent_j in agent_pairs:
                    # Prepare tasks for both directions
                    tasks.append((agent_i, agent_j, category, term, time))
                    tasks.append((agent_j, agent_i, category, term, time))

        # Define the function to compute TE for a pair of agents
        def compute_te_for_pair(args):
            source_agent, target_agent, category, term, time = args
            matrix_i = np.array([
                interpolated_time_series_agent_matrices[source_agent][t][category][term]
                for t in range(time + 1)
            ])
            matrix_j = np.array([
                interpolated_time_series_agent_matrices[target_agent][t][category][term]
                for t in range(time + 1)
            ])

            # Flatten matrices
            matrix_i_flat = matrix_i.flatten()
            matrix_j_flat = matrix_j.flatten()

            if matrix_i_flat.size >= lag * dimensions and matrix_j_flat.size >= lag * dimensions:
                try:
                    if use_embeddings:
                        te_value = compute_MTE_embedded(
                            X=matrix_i_flat, Y=matrix_j_flat, lag=lag, dimensions=dimensions
                        )
                    else:
                        te_value = compute_MTE(X_matrix=matrix_i_flat, Y_matrix=matrix_j_flat)

                    if np.isnan(te_value) or np.isinf(te_value):
                        print(f"TE value is NaN or Inf for agents {source_agent} and {target_agent} at time {time}")
                        return None

                    if abs(te_value) > significant_te_threshold:
                        return (source_agent, target_agent, category, term, te_value)
                except Exception as e:
                    print(f"Error calculating TE for {category}_{term} from {source_agent} to {target_agent}: {e}")
            return None

        # Compute TE values in parallel
        results = Parallel(n_jobs=-1)(delayed(compute_te_for_pair)(args) for args in tasks)

        # Add edges to the graph based on TE values
        for res in results:
            if res is not None:
                source_agent, target_agent, category, term, te_value = res
                G.add_edge(
                    f'{category}_{term}_{source_agent}',
                    f'{category}_{term}_{target_agent}',
                    weight=te_value
                )

        all_graphs.append(G)

    return all_graphs

# Call the function
all_graphs = generate_all_graphs()
print("Number of graphs:", len(all_graphs))

# Function to analyze and plot network evolution
def plot_network_evolution(graphs):
    metrics = {
        'Number of Nodes': [len(g.nodes()) for g in graphs],
        'Number of Edges': [len(g.edges()) for g in graphs],
        'Average Degree': [np.mean([d for n, d in g.degree()]) if len(g.nodes()) > 0 else 0 for g in graphs],
        'Clustering Coefficient': [nx.average_clustering(g) if len(g) > 0 else 0 for g in graphs]
    }

    plt.figure(figsize=(14, 10))
    for idx, (metric_name, values) in enumerate(metrics.items()):
        plt.subplot(2, 2, idx+1)
        plt.plot(values, marker='o', linestyle='-')
        plt.title(metric_name)
        plt.xlabel('Time')
        plt.ylabel(metric_name)
    
    plt.tight_layout()
    plt.show()

# Optionally, plot the network evolution
# plot_network_evolution(all_graphs)