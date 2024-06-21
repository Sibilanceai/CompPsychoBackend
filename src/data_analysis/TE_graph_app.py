import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mTE_graph_calculation import compute_MTE_embedded, compute_MTE, silvermans_rule
import csv
import re


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
    return np.load(f'transition_matrices_{character}.npy', allow_pickle=True)


final_agent_matrices = {agent: load_final_matrices(agent) for agent in agent_names}

# Example: Load time series matrices if needed
time_series_agent_matrices = {agent: load_time_series_matrices(agent) for agent in agent_names}

hierarchy = "high-level"
temporality = "short-term"
print(time_series_agent_matrices[agent_names[0]])
num_timesteps = len(time_series_agent_matrices[agent_names[0]][hierarchy][temporality])  # Assuming all characters have the same number of timesteps


print("num timesteps: ", num_timesteps)

all_graphs = []  # Store graphs for evolution analysis

# Process and create networks for each timestep
significant_te_threshold = 0.002  # threshold for significant TE values
# Print the first entry of each agent's matrix to see the structure


print("time_series_agent_matrices.keys()", time_series_agent_matrices.keys())
print("agent_names", agent_names)
# TODO vectorize and make this more efficient

def generate_all_graphs(): 
    # Define the structure of categories and terms
    categories = ['high-level', 'task-specific', 'context-specific']
    terms = ['short-term', 'medium-term', 'long-term']

    significant_te_threshold = 0.002  # Define a significant threshold for TE


    all_graphs = []  # This will collect all graphs over time

    # Create data structures to store time series for each category-term block
    time_series_data = {
        agent: {cat: {term: [] for term in terms} for cat in categories} for agent in agent_names
    }

    # Define these for the data
    lag = 1
    dimensions = 2
    use_embeddings = True


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


        # Aggregate data and compute TE
        for category in categories:
            for term in terms:
                for i, agent_i in enumerate(agent_names):
                    for j, agent_j in enumerate(agent_names):
                        if i != j:
                            matrix_i = np.concatenate([time_series_agent_matrices[agent_i][t][category][term] for t in range(time+1)], axis=0)
                            matrix_j = np.concatenate([time_series_agent_matrices[agent_j][t][category][term] for t in range(time+1)], axis=0)
                            # Check the shape before computation
                            # print(f"Shape before TE computation for {category}_{term} from {agent_i} to {agent_j}: matrix_i shape={matrix_i.shape}, matrix_j shape={matrix_j.shape}")

                            # Flatten the matrices before TE computation
                            matrix_i_flat = matrix_i.flatten()
                            matrix_j_flat = matrix_j.flatten()
                            # Check the shape before computation
                            # print(f"Shape after flattening and before TE computation for {category}_{term} from {agent_i} to {agent_j}: matrix_i shape={matrix_i_flat.shape}, matrix_j shape={matrix_j_flat.shape}")

                            # Check if there's enough data to perform embedding
                            if matrix_i_flat.size >= lag * dimensions and matrix_j_flat.size >= lag * dimensions:
                                try:
                                    if use_embeddings:
                                        if matrix_i_flat.size >= lag * dimensions and matrix_j_flat.size >= lag * dimensions:
                                            te_value = compute_MTE_embedded(X=matrix_i_flat, Y=matrix_j_flat, lag=1, dimensions=2)
                                    else:
                                        te_value = compute_MTE(X_matrix=matrix_i_flat, Y_matrix=matrix_j_flat)

                                    if abs(te_value) > significant_te_threshold:
                                        G.add_edge(f'{category}_{term}_{agent_i}', f'{category}_{term}_{agent_j}', weight=te_value)

                                except ValueError as e:
                                    print(f"Error calculating TE for {category}_{term} from {agent_i} to {agent_j}: {e}")


        all_graphs.append(G)
    # print(type(all_graphs))
    # print(all_graphs)
    # print("num graphs ", len(all_graphs))
    return all_graphs
    









# analyze_community_evolution(all_graphs)
def plot_network_evolution(graphs):
    metrics = {
        'Number of Nodes': [len(g.nodes()) for g in graphs],
        'Number of Edges': [len(g.edges()) for g in graphs],
        'Average Degree': [np.mean([d for n, d in g.degree()]) if len(g.nodes()) > 0 else 0 for g in graphs],
        'Clustering Coefficient': [nx.average_clustering(g) for g in graphs]
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

# Call this function with all_graphs
# plot_network_evolution(all_graphs)
all_graphs = generate_all_graphs()
print("num graphs: ", len(all_graphs))