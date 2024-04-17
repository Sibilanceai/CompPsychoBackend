import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# Generate synthetic data
num_agents = 2
block_size = 4  # 4x4 blocks
agent_matrices = [np.random.rand(3, 3, block_size, block_size) for _ in range(num_agents)]


# # Normalize matrices and compute TE between each pair of blocks across agents
# normalized_matrices = []
# for agent in agent_matrices:
#     normalized_agent = []
#     for i in range(3):
#         for j in range(3):
#             block = agent[i, j]
#             normalized_matrix = normalize_transition_matrix(block.flatten())
#             normalized_agent.append(normalized_matrix)
#     normalized_matrices.append(normalized_agent)

# # Assuming we want to compare the first block of the first agent with the first block of the second agent
# TE = compute_transfer_entropy(normalized_matrices[0][0], normalized_matrices[1][0])
# print(f"Transfer Entropy from Agent 1 Block 1 to Agent 2 Block 1: {TE}")




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

def compute_transfer_entropy(Ax, Ay, alpha=1.01):
    joint_matrix = np.kron(Ax, Ay)
    Hxy = compute_entropy(joint_matrix, alpha)
    Hx = compute_entropy(Ax, alpha)
    Hy = compute_entropy(Ay, alpha)
    return Hxy - Hx - Hy

# Assuming agent_matrices is correctly loaded with time as a dimension
num_agents = 2
num_timesteps = len(agent_matrices[0])
all_edges_data = []  # List to accumulate data across all timesteps

for time in range(num_timesteps):
    G = nx.DiGraph()
    
    for i in range(3):  # Assuming 3x3 blocks
        for j in range(3):
            Ax = normalize_transition_matrix(agent_matrices[0][time][i][j].flatten())
            Ay = normalize_transition_matrix(agent_matrices[1][time][i][j].flatten())
            te = compute_transfer_entropy(Ax, Ay)
            G.add_edge(f'Agent1_Block{i+1}', f'Agent2_Block{j+1}', weight=te)
            all_edges_data.append({
                'Time': time,
                'Source': f'Agent1_Block{i+1}',
                'Target': f'Agent2_Block{j+1}',
                'TE': te
            })

    # Optionally visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(f"TE Network at Time {time}")
    plt.show()

# Convert all edges data to DataFrame and save to a single CSV
df = pd.DataFrame(all_edges_data)
df.to_csv('network_snapshots.csv', index=False)

def plot_network_evolution(snapshots):
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
    
    plt.figure(figsize=(14, 10))
    for i, (metric, values) in enumerate(metrics.items(), start=1):
        plt.subplot(2, 2, i)
        plt.plot(values, marker='o', linestyle='-')
        plt.title(metric)
        plt.xlabel('Time Snapshot')
        plt.ylabel(metric)
    
    plt.tight_layout()
    plt.show()


all_graphs = []  # Store graphs for evolution analysis

for time in range(num_timesteps):
    G = nx.DiGraph()
    
    for i in range(3):  # Assuming 3x3 blocks
        for j in range(3):
            Ax = normalize_transition_matrix(agent_matrices[0][time][i][j].flatten())
            Ay = normalize_transition_matrix(agent_matrices[1][time][i][j].flatten())
            te = compute_transfer_entropy(Ax, Ay)
            G.add_edge(f'Agent1_Block{i+1}', f'Agent2_Block{j+1}', weight=te)

    all_graphs.append(G)  # Append the graph of this time step to the list

# Plot the evolution of the network using the collected graphs
plot_network_evolution(all_graphs)