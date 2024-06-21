import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform


# Step 2: Generate Synthetic Time Series Data
# np.random.seed(42)

# # Generate synthetic time series data
# lag = 1
# dimensions = 3
# t = np.linspace(0, 10, 100)
# X = np.sin(t) + 0.5 * np.random.normal(size=t.shape)
# Y = np.cos(t) + 0.5 * np.random.normal(size=t.shape)

def silvermans_rule(time_series_list):
    d = len(time_series_list)  # Number of time series
    N = len(time_series_list[0])  # Assuming all time series are of the same length
    variances = [np.var(ts) for ts in time_series_list]
    mean_variance = np.mean(variances)
    sigma = (4 / (d + 2) * N) ** (-1/(d + 4)) * np.sqrt(mean_variance)
    return sigma

# Example usage with multiple time series
# time_series_list = [X, Y]  # Assuming X and Y are numpy arrays
# sigma = silvermans_rule(time_series_list)

def gaussian_kernel_normalized(X, sigma):
    """Generate Gaussian Kernel Gram matrix with normalization."""
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-pairwise_sq_dists / (2 * sigma**2))


def matrix_entropy(G):
    trace_squared = np.trace(G @ G)
    print("Trace of G squared:", trace_squared)  # Debugging statement
    if trace_squared <= 0:
        raise ValueError("Trace of matrix squared is zero or negative, check the Gram matrix computation.")
    entropy = -np.log(trace_squared)
    print("Entropy:", entropy)  # More debugging
    return entropy


def compute_MTE(X_matrix, Y_matrix, sigma=0.4):
    """Compute the Matrix Transfer Entropy from time series Y to X."""
    # Reshape data for Gaussian kernel calculation
    X_matrix = X_matrix.reshape(-1, 1)
    Y_matrix = Y_matrix.reshape(-1, 1)
    

    # Computing Gram matrices using the normalized Gaussian kernel
    G_X = gaussian_kernel_normalized(X_matrix, sigma)
    G_Y = gaussian_kernel_normalized(Y_matrix, sigma)

    # Normalized matrices for the entropy calculation
    G_R = G_X  # Assuming G_R is based on X
    G_S = G_Y  # Assuming G_S is based on Y
    G_Q = G_X  # Assuming G_Q is also based on X

    # Compute Normalized Hadamard Products
    G_T = (G_R * G_S) / np.trace(G_R * G_S)
    G_QR = (G_Q * G_R) / np.trace(G_Q * G_R)
    G_QT = (G_Q * G_T) / np.trace(G_Q * G_T)

    # Compute entropies based on matrix traces
    entropy_R = -np.log(np.trace(G_R @ G_R))
    entropy_T = -np.log(np.trace(G_T @ G_T))

    # MTE computation
    MTE_Y_to_X = (-np.log(np.trace(G_QR)) +
                  entropy_R +
                  np.log(np.trace(G_QT)) -
                  entropy_T)
    return MTE_Y_to_X


def compute_normalized_matrix(H):
    """Normalize the matrix H by the trace of its Hadamard product with itself."""
    trace_H = np.trace(H)
    if trace_H == 0:
        raise ValueError("Trace of the matrix product is zero, normalization not possible.")
    return H / trace_H

def create_time_lagged_embeddings(X, lag, dimensions):
    """Create time-lagged embeddings for a given time series."""
    n = len(X)
    if n < lag * (dimensions - 1) + 1:
        raise ValueError("Not enough data points for the given lag and dimensions")
    embedded_data = np.zeros((n - lag * (dimensions - 1), dimensions))
    for i in range(dimensions):
        embedded_data[:, i] = X[(dimensions - i - 1) * lag : n - i * lag]
    return embedded_data

def compute_MTE_embedded(X, Y, lag=1, dimensions=3, sigma_X=0.4, sigma_Y=0.4):
    # print(f"Shape of X before operation: {X.shape}")
    # print(f"Shape of Y before operation: {Y.shape}")
    # Embedding the data with time lags
    # print("time lag start")
    X_embedded = create_time_lagged_embeddings(X, lag, dimensions)
    Y_embedded = create_time_lagged_embeddings(Y, lag, dimensions)
    # print("time lag end")

    # Computing Gram matrices using the normalized Gaussian kernel
    G_X = gaussian_kernel_normalized(X_embedded, sigma_X)
    G_Y = gaussian_kernel_normalized(Y_embedded, sigma_Y)
    # print("gaussian kernel end")
    # Normalized matrices for the entropy calculation
    G_R = G_X  # Assuming G_R is based on X
    G_S = G_Y  # Assuming G_S is based on Y
    G_Q = G_X  # Assuming G_Q is also based on X

    # Compute Normalized Hadamard Products
    G_T = (G_R * G_S) / np.trace(G_R * G_S)
    G_QR = (G_Q * G_R) / np.trace(G_Q * G_R)
    G_QT = (G_Q * G_T) / np.trace(G_Q * G_T)
    # print("compute normalized end")
    # Compute entropies based on matrix traces
    entropy_R = -np.log(np.trace(G_R @ G_R))
    entropy_T = -np.log(np.trace(G_T @ G_T))
    # print("entropy end")
    # MTE computation
    MTE_Y_to_X = (-np.log(np.trace(G_QR)) +
                  entropy_R +
                  np.log(np.trace(G_QT)) -
                  entropy_T)
    # print("MTE end")
    return MTE_Y_to_X


# MTE_value = compute_MTE(X, Y)
# MTE_value_embed = compute_MTE_embedded(X, Y, lag, dimensions, sigma_X=sigma, sigma_Y=sigma)
# When calling compute_MTE, add exception handling to catch potential errors
# try:
#     MTE_value_embed = compute_MTE_embedded(X, Y, lag, dimensions, sigma_X=sigma, sigma_Y=sigma)
#     print("MTE from Y to X:", MTE_value_embed)
# except Exception as e:
#     print("Error during MTE computation:", str(e))

def update_graph_with_MTE(time_series_data, sigma_values, graph, node_labels):
    """Update the directed graph with MTE calculations."""
    for i in range(len(time_series_data)):
        for j in range(len(time_series_data)):
            if i != j:
                MTE_value = compute_MTE_embedded(time_series_data[i], time_series_data[j], sigma_values[i], sigma_values[j])
                graph.add_edge(node_labels[i], node_labels[j], weight=MTE_value)


def update_graph(X=None, Y=None,dimensions=3, lag=1):
    time_series_list = [X, Y]
    sigma = silvermans_rule(time_series_list)

    # Plotting and computing MTE for each timestep
    for time_index in range(100 - (dimensions - 1) * lag):  # Adjust based on the number of embeddings
        G = nx.DiGraph()
        for i in range(len(time_series_list)):
            for j in range(len(time_series_list)):
                if i != j:
                    mte = compute_MTE_embedded(time_series_list[i][time_index:time_index + dimensions * lag],
                                    time_series_list[j][time_index:time_index + dimensions * lag], sigma, sigma)
                    if abs(mte) > 1e-5:  # Threshold for significant MTE
                        G.add_edge(f'Series {i}', f'Series {j}', weight=mte)

        # Drawing the graph for the current timestep
        if G.number_of_edges() > 0:
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(G)
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
            nx.draw(G, pos, node_color='skyblue', node_size=500, edgelist=edges,
                    edge_color=weights, width=3, edge_cmap=plt.cm.Blues, with_labels=True)
            plt.title(f'Network Graph at Timestep {time_index}')
            plt.show()
        else:
            print(f"No significant edges to display at timestep {time_index}.")


import unittest

class TestMTECalculations(unittest.TestCase):

    def test_gaussian_kernel_normalized(self):
        X = np.array([[1], [2], [3]])
        sigma = 1.0
        G = gaussian_kernel_normalized(X, sigma)
        self.assertTrue(G.shape == (3, 3), "Kernel matrix should be 3x3")
        # Ensure that the trace is not a string and convert it to a float if needed
        trace_G = float(np.trace(G))
        self.assertAlmostEqual(trace_G, 1.0, places=5, msg="Trace of normalized kernel matrix should be close to 1")

    def test_matrix_entropy(self):
        G = np.array([[1, 0], [0, 1]])
        entropy = matrix_entropy(G)
        self.assertAlmostEqual(entropy, 0.0, places=5, msg="Entropy of identity matrix should be 0")

    def test_compute_MTE(self):
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([1, 2, 3, 4, 5])
        sigma = 0.4
        mte = compute_MTE(X, Y, sigma)
        self.assertAlmostEqual(mte, 0.0, places=5, msg="MTE of identical time series should be close to 0")

    def test_create_time_lagged_embeddings(self):
        X = np.array([1, 2, 3, 4, 5])
        lag = 1
        dimensions = 3
        embedded = create_time_lagged_embeddings(X, lag, dimensions)
        expected_shape = (3, 3)  # (5 - 1 * (3 - 1), 3)
        self.assertTrue(embedded.shape == expected_shape, "Embedded data should have correct shape")

    def test_compute_MTE_embedded(self):
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([1, 2, 3, 4, 5])
        lag = 1
        dimensions = 3
        sigma = 0.4
        mte = compute_MTE_embedded(X, Y, lag, dimensions, sigma, sigma)
        self.assertAlmostEqual(mte, 0.0, places=5, msg="MTE of identical time series should be close to 0")

if __name__ == '__main__':
    unittest.main()