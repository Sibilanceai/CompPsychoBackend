import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mTE_graph_calculation import compute_MTE_embedded, compute_MTE, silvermans_rule
import csv


# def plot_network_evolution(snapshots):
#     """
#     Plots the evolution of various network metrics over time.
    
#     :param snapshots: A list of NetworkX graphs representing the network at different time points.
#     """
    

#     metrics = {
#         'Number of Nodes': [],
#         'Number of Edges': [],
#         'Average Degree': [],
#         'Clustering Coefficient': []
#     }
    
#     for G in snapshots:
#         metrics['Number of Nodes'].append(len(G.nodes()))
#         metrics['Number of Edges'].append(len(G.edges()))
#         degrees = [deg for node, deg in G.degree()]
#         metrics['Average Degree'].append(np.mean(degrees) if degrees else 0)
#         metrics['Clustering Coefficient'].append(nx.average_clustering(G))
    
#     # Plotting the evolution of network metrics
#     plt.figure(figsize=(14, 10))
#     for i, (metric, values) in enumerate(metrics.items(), start=1):
#         plt.subplot(2, 2, i)
#         plt.plot(values, marker='o', linestyle='-')
#         plt.title(metric)
#         plt.xlabel('Time Snapshot')
#         plt.ylabel(metric)
    
#     plt.tight_layout()
#     plt.show()



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

for time in range(num_timesteps):
    G = nx.DiGraph()

    # Define node positions
    positions = {}

    # Customize the spacing parameters
    x_offset = 2  # Horizontal offset between different categories
    y_offset = 2  # Vertical offset between different terms
    group_offset = 10  # Offset between groups of nodes (Boy vs Eleanor)


    for i, category in enumerate(categories):
        for j, term in enumerate(terms):
            for k, agent in enumerate(agent_names):
                node_id = f"{category}_{term}_{agent}"
                x_position = i * x_offset + k * group_offset
                y_position = j * y_offset
                positions[node_id] = (x_position, y_position)
                G.add_node(node_id, group=node_id,pos=(x_position, y_position), label=node_id)




    # Check if all nodes have 'group' attribute before plotting
    missing_group = [n for n, attr in G.nodes(data=True) if 'group' not in attr]
    if missing_group:
        print("These nodes are missing 'group' attributes:", missing_group)
    else:
        print("All nodes have 'group' attributes.")



    # Analyze the matrices and add edges based on TE
    for category in categories:
        for term in terms:
            matrix_boy = time_series_agent_matrices[agent_names[0]][time][category][term]
            matrix_eleanor = time_series_agent_matrices[agent_names[1]][time][category][term]
            matrix_dim = matrix_boy.shape[0]

            print(f"Time {time}, Category {category}, Term {term}: Boy matrix shape {matrix_boy.shape}, Eleanor matrix shape {matrix_eleanor.shape}")

            for i in range(matrix_dim):
                for j in range(matrix_dim):
                    Ax = matrix_boy[i][j].flatten()
                    Ay = matrix_eleanor[i][j].flatten()
                    print("matrix_boy[i][j], matrix_eleanor[i][j]: ", matrix_boy[i][j], matrix_eleanor[i][j])
                    # Check if data is too sparse for embeddings
                    min_data_points_required = 2 * 1 + 1  # This matches lag=1, dimensions=2, adjust as needed
                    if np.count_nonzero(Ax) < min_data_points_required or np.count_nonzero(Ay) < min_data_points_required:
                        # TODO check why this is the index so a specifical transition matrix value, we should also see why we are just giving a single matrix at each time step, shouldnt this be a running time series we add?
                        te_Ax_to_Ay = compute_MTE(matrix_boy, matrix_eleanor)
                        te_Ay_to_Ax = compute_MTE(matrix_eleanor, matrix_boy)
                    else:
                        # TODO this is wrong, should be running time series not just the 4x4 transition matrix from one block.
                        te_Ax_to_Ay = compute_MTE_embedded(X=Ax, Y=Ay, lag=1, dimensions=2)
                        te_Ay_to_Ax = compute_MTE_embedded(X=Ay, Y=Ax, lag=1, dimensions=2)

                    if abs(te_Ax_to_Ay) > significant_te_threshold:
                        G.add_edge(f'{category}_{term}_Boy', f'{category}_{term}_Eleanor', weight=te_Ax_to_Ay)
                    if abs(te_Ay_to_Ax) > significant_te_threshold:
                        G.add_edge(f'{category}_{term}_Eleanor', f'{category}_{term}_Boy', weight=te_Ay_to_Ax)


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

