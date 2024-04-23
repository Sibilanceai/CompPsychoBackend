import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mTE_graph_calculation import compute_MTE_embedded, silvermans_rule

# Generate synthetic data
num_agents = 2
block_size = 4  # 4x4 blocks
agent_matrices = [np.random.rand(3, 3, block_size, block_size) for _ in range(num_agents)]


def plot_network_evolution(snapshots):
    """
    Plots the evolution of various network metrics over time.
    
    :param snapshots: A list of NetworkX graphs representing the network at different time points.
    """
    

    metrics = {
        'Number of Nodes': [],
        'Number of Edges': [],
        'Average Degree': [],
        'Clustering Coefficient': []
    }
    
    for G in snapshots:
        metrics['Number of Nodes'].append(len(G.nodes()))
        metrics['Number of Edges'].append(len(G.edges()))
        degrees = [deg for node, deg in G.degree()]
        metrics['Average Degree'].append(np.mean(degrees) if degrees else 0)
        metrics['Clustering Coefficient'].append(nx.average_clustering(G))
    
    # Plotting the evolution of network metrics
    plt.figure(figsize=(14, 10))
    for i, (metric, values) in enumerate(metrics.items(), start=1):
        plt.subplot(2, 2, i)
        plt.plot(values, marker='o', linestyle='-')
        plt.title(metric)
        plt.xlabel('Time Snapshot')
        plt.ylabel(metric)
    
    plt.tight_layout()
    plt.show()

# from networkx.algorithms import community
# from community import community_louvain

# def analyze_community_evolution(snapshots):
#     """
#     Analyzes the evolution of community structure over time.
    
#     :param snapshots: A list of NetworkX graphs representing the network at different time points.
#     """
#     community_changes = []

#     for G in snapshots:
#         partition = community_louvain.best_partition(G)
#         num_communities = len(set(partition.values()))
#         community_changes.append(num_communities)
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(community_changes, marker='o', linestyle='-')
#     plt.title('Evolution of Community Structure')
#     plt.xlabel('Time Snapshot')
#     plt.ylabel('Number of Communities')
#     plt.grid(True)
#     plt.show()


# --------------------------------------------------- testing block end ----------------------------------------------------------





# Generate synthetic data
num_agents = 2
block_size = 4  # 4x4 blocks
num_timesteps = 5  # Number of timesteps
agent_matrices = [np.random.rand(num_timesteps, 3, 3, block_size, block_size) for _ in range(num_agents)]

all_graphs = []  # Store graphs for evolution analysis

# Process and create networks for each timestep
significant_te_threshold = 0.2  # threshold for significant TE values


for time in range(num_timesteps):
    G = nx.DiGraph()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Ax = agent_matrices[0][time][i][j].flatten()
                    Ay = agent_matrices[1][time][k][l].flatten()
                    te_Ax_to_Ay = compute_MTE_embedded(Ax, Ay)
                    te_Ay_to_Ax = compute_MTE_embedded(Ay, Ax)

                    if abs(te_Ax_to_Ay) > significant_te_threshold:
                        G.add_edge(f'Agent1_Block_{i*3+j+1}', f'Agent2_Block_{k*3+l+1}', weight=te_Ax_to_Ay)
                    if abs(te_Ay_to_Ax) > significant_te_threshold:
                        G.add_edge(f'Agent2_Block_{k*3+l+1}', f'Agent1_Block_{i*3+j+1}', weight=te_Ay_to_Ax)

              

    all_graphs.append(G)  # Store the graph

    if G.number_of_edges() > 0:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G, pos, node_color='lightblue', node_size=500, edgelist=edges, edge_color=weights, width=3, edge_cmap=plt.cm.Blues, with_labels=True)
        plt.title(f"Network Graph at Time {time}")
        plt.show()
    else:
        print(f"No edges to display at time {time}.")

# Optionally save all edges data to a DataFrame and CSV if needed for analysis
all_edges_data = [{
    'Time': time,
    'Source': source,
    'Target': target,
    'TE': data['weight']
} for time, graph in enumerate(all_graphs) for source, target, data in graph.edges(data=True)]
df = pd.DataFrame(all_edges_data)
df.to_csv('network_snapshots.csv', index=False)

plot_network_evolution(all_graphs)

# analyze_community_evolution(all_graphs)

