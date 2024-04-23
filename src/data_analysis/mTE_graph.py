import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# Generate synthetic data
num_agents = 2
block_size = 4  # 4x4 blocks
agent_matrices = [np.random.rand(3, 3, block_size, block_size) for _ in range(num_agents)]


def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * sigma**2))

def normalize_transition_matrix(T):
    n = T.shape[0]
    Gram = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Gram[i, j] = gaussian_kernel(T[i], T[j])
    diag_sum = np.sum(np.diag(Gram))
    if diag_sum != 0:
        Gram = Gram / diag_sum
    return Gram

def compute_entropy(A, alpha=1.01):
    eigenvalues = np.linalg.eigvalsh(A)
    eigenvalues = eigenvalues[eigenvalues > 0]
    entropy = (1 / (1 - alpha)) * np.log2(np.sum(eigenvalues ** alpha))
    return entropy



# TODO fix the calculation to work and use time lag embeddings and time series, placeholder for actual calculation of matrix based TE
def compute_transfer_entropy(Ax, Ay, alpha=1.01):
    joint_matrix = np.kron(Ax, Ay)
    Hxy = compute_entropy(joint_matrix, alpha)
    Hx = compute_entropy(Ax, alpha)
    Hy = compute_entropy(Ay, alpha)
    return Hxy - Hx - Hy

# from scipy.stats import entropy
# def compute_transfer_entropy(time_series_x, time_series_y, max_lags=3):
#     # Dummy example, you'll need an actual TE library like PyInform
#     # Assume `time_series_x` and `time_series_y` are arrays where rows are time points
#     # and columns are the values from different runs or simulations.
    
#     joint_entropy = entropy(np.hstack((time_series_x[:-1], time_series_y[1:])))
#     entropy_x = entropy(time_series_x[:-1])
#     entropy_y_given_x = entropy(time_series_y[1:] - time_series_x[:-1])  # Simplistic assumption
#     return entropy_y_given_x - joint_entropy + entropy_x


# testing functions from other scripts

# def plot_network_evolution(snapshots):
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
    
#     plt.figure(figsize=(14, 10))
#     for i, (metric, values) in enumerate(metrics.items(), start=1):
#         plt.subplot(2, 2, i)
#         plt.plot(values, marker='o', linestyle='-')
#         plt.title(metric)
#         plt.xlabel('Time Snapshot')
#         plt.ylabel(metric)
    
#     plt.tight_layout()
#     plt.show()

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
significant_te_threshold = 0.1e-15  # threshold for significant TE values


for time in range(num_timesteps):
    G = nx.DiGraph()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Ax = normalize_transition_matrix(agent_matrices[0][time][i][j].flatten())
                    Ay = normalize_transition_matrix(agent_matrices[1][time][k][l].flatten())
                    te = compute_transfer_entropy(Ax, Ay)
                    print("te: ", te)
                    if abs(te) > significant_te_threshold:  # Example threshold
                        G.add_edge(f'Agent1_Block_{i*3+j+1}', f'Agent2_Block_{k*3+l+1}', weight=te)

    all_graphs.append(G)  # Store the graph

    if G.number_of_edges() > 0:  # Check if there are any edges
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G, pos, node_color='lightblue', node_size=500, edgelist=edges, edge_color=weights, width=3, edge_cmap=plt.cm.Blues, with_labels=True)
        plt.title(f"Network Graph at Time {time}")
        plt.show()
    else:
        print(f"No edges to display at time {time}.")  # Handle the case where no edges were added

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
















# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# def vectorized_normalize_matrices(matrices):
#     """ Vectorized normalization of block matrices """
#     flattened_matrices = matrices.reshape(matrices.shape[0], -1)  # Flatten each matrix
#     norm = np.linalg.norm(flattened_matrices, axis=1, keepdims=True)
#     normalized_matrices = flattened_matrices / norm  # Normalize
#     return normalized_matrices

# def vectorized_transfer_entropy(Ax, Ay):
#     """ Vectorized computation of transfer entropy between sets of matrices """
#     # Assuming Ax and Ay are flattened and normalized
#     # For simplicity, assume TE calculation is directly possible; adjust as per actual TE calculation needs
#     joint_entropy = -np.sum(Ax * np.log(Ay), axis=1)
#     return joint_entropy

# # Example data generation
# num_timesteps = 5
# num_blocks = 3 * 3  # For 3x3 blocks per agent
# block_size = 4

# # Randomly generate matrices for two agents
# agent1_matrices = np.random.rand(num_timesteps, num_blocks, block_size, block_size)
# agent2_matrices = np.random.rand(num_timesteps, num_blocks, block_size, block_size)

# # Flatten and normalize
# agent1_matrices_flat = agent1_matrices.reshape(num_timesteps * num_blocks, block_size**2)
# agent2_matrices_flat = agent2_matrices.reshape(num_timesteps * num_blocks, block_size**2)
# agent1_norm = vectorized_normalize_matrices(agent1_matrices_flat)
# agent2_norm = vectorized_normalize_matrices(agent2_matrices_flat)

# # Calculate TE for all pairs (example simplified logic)
# tes = vectorized_transfer_entropy(agent1_norm, agent2_norm)

# # Reshape back to original timestep and block structure
# tes_reshaped = tes.reshape(num_timesteps, num_blocks, num_blocks)

# # Use results as needed, for example to populate a graph
# for time in range(num_timesteps):
#     G = nx.DiGraph()
#     for i in range(num_blocks):
#         for j in range(num_blocks):
#             if tes_reshaped[time, i, j] > 0.01:  # Threshold for significant TE
#                 G.add_edge(f'Agent1_Block_{i+1}', f'Agent2_Block_{j+1}', weight=tes_reshaped[time, i, j])

#     # Plotting logic can be here if needed
