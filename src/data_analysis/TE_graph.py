import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mTE_graph_calculation import compute_MTE_embedded, compute_MTE, silvermans_rule
import csv



# # Generate synthetic data
# num_agents = 2
# block_size = 4  # 4x4 blocks, can extend this with level of granularity in your schema
# num_timesteps = 5  # Number of timesteps
# agent_matrices = [np.random.rand(num_timesteps, 3, 3, block_size, block_size) for _ in range(num_agents)]

def read_characters_from_csv(file_path):
    """ Read character names from a CSV file and return them as a list. """
    characters = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Ensure the row is not empty
                characters.append(row[0])
    return characters

def load_final_matrices(character):
    return np.load(f'transition_matrices_final_{character}.npy', allow_pickle=True)

def load_time_series_matrices(character):
    return np.load(f'transition_matrices_{character}.npy', allow_pickle=True)

# Example: Load final matrices for graph analysis
characters_file_path = "../characters.csv"
agent_names = read_characters_from_csv(characters_file_path)  
final_agent_matrices = {agent: load_final_matrices(agent) for agent in agent_names}

# Example: Load time series matrices if needed
time_series_agent_matrices = {agent: load_time_series_matrices(agent) for agent in agent_names}


num_timesteps = len(time_series_agent_matrices[agent_names[0]])  # Assuming all characters have the same number of timesteps




all_graphs = []  # Store graphs for evolution analysis

# Process and create networks for each timestep
significant_te_threshold = 0.002  # threshold for significant TE values
# Print the first entry of each agent's matrix to see the structure
print(time_series_agent_matrices['Boy'][0])
print(time_series_agent_matrices['Eleanor'][0])

print("time_series_agent_matrices.keys()", time_series_agent_matrices.keys())
print("agent_names", agent_names)
# TODO vectorize and make this more efficient


# Define the structure of categories and terms
categories = ['high-level', 'task-specific', 'context-specific']
terms = ['short-term', 'medium-term', 'long-term']

# Map category and term combinations to a color index
label_to_index = {f'{cat}_{term}': i + j * len(categories)
                  for i, cat in enumerate(categories)
                  for j, term in enumerate(terms)}

significant_te_threshold = 0.002  # Define a significant threshold for TE

# Define a colormap with enough colors for each group
cmap = plt.cm.get_cmap('viridis', len(label_to_index))

def get_color_for_node(group):
    index = label_to_index.get(group, 0)  # Default index to 0 if group not found
    return cmap(index / len(label_to_index))  # Normalize index for colormap



# Example timestep number for visualization
num_timesteps = 9  # Adjust as per your setup

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
                        print(f"Shape before TE computation for {category}_{term} from {agent_i} to {agent_j}: matrix_i shape={matrix_i.shape}, matrix_j shape={matrix_j.shape}")

                        # Flatten the matrices before TE computation
                        matrix_i_flat = matrix_i.flatten()
                        matrix_j_flat = matrix_j.flatten()
                        # Check the shape before computation
                        print(f"Shape after flattening and before TE computation for {category}_{term} from {agent_i} to {agent_j}: matrix_i shape={matrix_i_flat.shape}, matrix_j shape={matrix_j_flat.shape}")

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
    # Draw the graph
    if G.number_of_edges() > 0:
        plt.figure(figsize=(12, 12))
        pos = nx.get_node_attributes(G, 'pos')
        # Check if all nodes have 'group' attribute before plotting
        # Print node details to confirm correct attribute assignment
        for node, data in G.nodes(data=True):
            print(f"Node: {node}, Data: {data}")

        missing_group = [n for n, attr in G.nodes(data=True) if 'group' not in attr]
        if missing_group:
            print("These nodes are missing 'group' attributes:", missing_group)
        else:
            print("All nodes have 'group' attributes.")
        node_colors = [get_color_for_node(data['group']) for n, data in G.nodes(data=True)]  # Use safe color mapping
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, edge_color='grey')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Network Graph")
        plt.show()
    else:
        print("No edges to display.")










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

