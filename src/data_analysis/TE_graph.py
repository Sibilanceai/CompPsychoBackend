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
significant_te_threshold = 0.2  # threshold for significant TE values
# Print the first entry of each agent's matrix to see the structure
print(time_series_agent_matrices['Boy'][0])
print(time_series_agent_matrices['Eleanor'][0])

print("time_series_agent_matrices.keys()", time_series_agent_matrices.keys())
print("agent_names", agent_names)
# TODO vectorize and make this more efficient
# for time in range(num_timesteps):
#     G = nx.DiGraph()
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 for l in range(3):
#                     # TODO add logic to handle more than 2 characters
#                     Ax = time_series_agent_matrices[agent_names[0]][time][i][j].flatten()
#                     Ay = time_series_agent_matrices[agent_names[1]][time][k][l].flatten()
#                     te_Ax_to_Ay = compute_MTE_embedded(Ax, Ay)
#                     te_Ay_to_Ax = compute_MTE_embedded(Ay, Ax)

#                     if abs(te_Ax_to_Ay) > significant_te_threshold:
#                         G.add_edge(f'Agent1_Block_{i*3+j+1}', f'Agent2_Block_{k*3+l+1}', weight=te_Ax_to_Ay)
#                     if abs(te_Ay_to_Ax) > significant_te_threshold:
#                         G.add_edge(f'Agent2_Block_{k*3+l+1}', f'Agent1_Block_{i*3+j+1}', weight=te_Ay_to_Ax)

              

#     all_graphs.append(G)  # Store the graph

#     if G.number_of_edges() > 0:
#         plt.figure(figsize=(12, 12))
#         pos = nx.spring_layout(G)
#         edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
#         nx.draw(G, pos, node_color='lightblue', node_size=500, edgelist=edges, edge_color=weights, width=3, edge_cmap=plt.cm.Blues, with_labels=True)
#         plt.title(f"Network Graph at Time {time}")
#         plt.show()
#     else:
#         print(f"No edges to display at time {time}.")

# Assuming the matrices are 4x4 as per the data you provided
# Add debug statements to print the matrix details before accessing it


all_graphs = []  # This will collect all graphs over time
print("num_timesteps ", num_timesteps)
for time in range(num_timesteps):
    G = nx.DiGraph()
    # TODO we shouldnt need this
    category = 'high-level'
    term = 'long-term'
    # TODO fix this so it is dynamically creating the matrices and it should use all of the matrices not just high level long term
    matrix_boy = time_series_agent_matrices[agent_names[0]][time][category][term]
    matrix_eleanor = time_series_agent_matrices[agent_names[1]][time][category][term]
    matrix_dim = matrix_boy.shape[0]

    print(f"Time {time}: Boy matrix shape {matrix_boy.shape}, Eleanor matrix shape {matrix_eleanor.shape}")

    for i in range(matrix_dim):
        for j in range(matrix_dim):
            Ax = matrix_boy[i][j].flatten()
            Ay = matrix_eleanor[i][j].flatten()
            
            # Check if data is too sparse for embeddings
            # TODO make this dynamically calculated
            min_data_points_required = 2 * 1 + 1  # This matches lag=1, dimensions=2, adjust as needed
            if np.count_nonzero(Ax) < min_data_points_required or np.count_nonzero(Ay) < min_data_points_required:
                te_Ax_to_Ay = compute_MTE(matrix_boy[i][j], matrix_eleanor[i][j])
                print("te_Ax_to_Ay", te_Ax_to_Ay)
                te_Ay_to_Ax = compute_MTE(matrix_eleanor[i][j], matrix_boy[i][j])
                print("te_Ay_to_Ax", te_Ay_to_Ax)
            else:
                te_Ax_to_Ay = compute_MTE_embedded(X=Ax, Y=Ay, lag=1, dimensions=2)
                te_Ay_to_Ax = compute_MTE_embedded(X=Ay, Y=Ax, lag=1, dimensions=2)

            if abs(te_Ax_to_Ay) > significant_te_threshold:
                G.add_edge(f'Boy_{i*matrix_dim+j}', f'Eleanor_{i*matrix_dim+j}', weight=te_Ax_to_Ay)
            if abs(te_Ay_to_Ax) > significant_te_threshold:
                G.add_edge(f'Eleanor_{i*matrix_dim+j}', f'Boy_{i*matrix_dim+j}', weight=te_Ay_to_Ax)

    all_graphs.append(G)

    if G.number_of_edges() > 0:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G, pos, node_color='lightblue', node_size=500, edgelist=edges, edge_color=weights, width=3, edge_cmap=plt.cm.Blues, with_labels=True)
        plt.title(f"Network Graph at Time {time}")
        plt.show()
    else:
        print(f"No edges to display at time {time}.")


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

