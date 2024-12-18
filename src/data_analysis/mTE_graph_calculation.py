# mTE_graph_calculation.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform


def silvermans_rule(time_series_list):
    """
    Calculate the bandwidth sigma using Silverman's rule.

    Args:
        time_series_list (list of np.ndarray): List of time series arrays.

    Returns:
        float: Calculated sigma value.
    """
    epsilon = 1e-5  # Small constant to prevent zero variance
    d = len(time_series_list)
    N = len(time_series_list[0])
    variances = [np.var(ts) for ts in time_series_list]
    mean_variance = np.mean(variances)
    mean_variance = max(mean_variance, epsilon)
    sigma = (4 / (d + 2) * N) ** (-1 / (d + 4)) * np.sqrt(mean_variance)
    return sigma

def gaussian_kernel_normalized(X, sigma):
    """
    Generate Gaussian Kernel Gram matrix with normalization.

    Args:
        X (np.ndarray): Input data array of shape (n_samples, n_features).
        sigma (float): Bandwidth parameter.

    Returns:
        np.ndarray: Normalized Gram matrix.
    """
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    K = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    K_normalized = K / np.trace(K)  # Normalize by trace
    return K_normalized

def create_time_lagged_embeddings(X, lag, dimensions):
    """
    Create time-lagged embeddings for a given multidimensional time series.

    Args:
        X (np.ndarray): Input time series array of shape (n_samples, n_features).
        lag (int): Lag between time steps.
        dimensions (int): Embedding dimension.

    Returns:
        np.ndarray: Embedded data array.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape
    if n_samples < lag * (dimensions - 1) + 1:
        raise ValueError("Not enough data points for the given lag and dimensions")
    n_embedded = n_samples - lag * (dimensions - 1)
    embedded_data = np.zeros((n_embedded, dimensions * n_features))
    for i in range(dimensions):
        start = i * lag
        end = start + n_embedded
        embedded_data[:, i * n_features : (i + 1) * n_features] = X[start:end, :]
    return embedded_data



def matrix_entropy(G):
    trace_squared = np.trace(G @ G)
    print("Trace of G squared:", trace_squared)  # Debugging statement
    if trace_squared <= 0:
        raise ValueError("Trace of matrix squared is zero or negative, check the Gram matrix computation.")
    entropy = -np.log(trace_squared)
    print("Entropy:", entropy)  # More debugging
    return entropy

def compute_MTE_embedded(X, Y, lag=1, dimensions=3, sigma_X=None, sigma_Y=None):
    """
    Compute the Matrix Transfer Entropy from time series Y to X with embeddings.

    Args:
        X (np.ndarray): Target time series data.
        Y (np.ndarray): Source time series data.
        lag (int): Time lag for embeddings.
        dimensions (int): Embedding dimension.
        sigma_X (float): Bandwidth for X. If None, it will be calculated.
        sigma_Y (float): Bandwidth for Y. If None, it will be calculated.

    Returns:
        float: Computed Matrix Transfer Entropy value.
    """
    # Embedding the data with time lags
    X_embedded = create_time_lagged_embeddings(X, lag, dimensions)
    Y_embedded = create_time_lagged_embeddings(Y, lag, dimensions)
    
    # Current values (x_i)
    x_i = X_embedded[:, -X.shape[1]:]
    
    # Previous values (x_i^{(k)} and y_i^{(k)})
    x_i_k = X_embedded[:, :-X.shape[1]]
    y_i_k = Y_embedded[:, :-Y.shape[1]]
    
    # Compute bandwidths if not provided
    if sigma_X is None:
        sigma_X = silvermans_rule([x_i, x_i_k])
    if sigma_Y is None:
        sigma_Y = silvermans_rule([y_i_k])
    print(f"sigma_X: {sigma_X}, sigma_Y: {sigma_Y}")

    # Check if sigma is zero
    if sigma_X == 0 or sigma_Y == 0:
        print("Sigma is zero, returning MTE value of 0")
        return 0.0

    # Compute Gram matrices
    Q = gaussian_kernel_normalized(x_i, sigma_X)
    R = gaussian_kernel_normalized(x_i_k, sigma_X)
    S = gaussian_kernel_normalized(y_i_k, sigma_Y)
    
    # Compute T
    R_S = R * S  # Element-wise multiplication
    T = R_S / np.trace(R_S)
    
    # Compute entropies
    # S2(Q|R) = S2(Q,R) - S2(R)
    Q_R = Q * R
    Q_R_normalized = Q_R / np.trace(Q_R)
    entropy_QR = -np.log(np.trace(Q_R_normalized @ Q_R_normalized))
    entropy_R = -np.log(np.trace(R @ R))
    S2_Q_given_R = entropy_QR - entropy_R
    
    # S2(Q|T) = S2(Q,T) - S2(T)
    Q_T = Q * T
    Q_T_normalized = Q_T / np.trace(Q_T)
    entropy_QT = -np.log(np.trace(Q_T_normalized @ Q_T_normalized))
    entropy_T = -np.log(np.trace(T @ T))
    S2_Q_given_T = entropy_QT - entropy_T
    
    # MTE computation
    MTE_Y_to_X = S2_Q_given_R - S2_Q_given_T
    return MTE_Y_to_X

def compute_MTE(X_matrix, Y_matrix, sigma=None):
    """
    Compute the Matrix Transfer Entropy from time series Y to X without embeddings.

    Args:
        X_matrix (np.ndarray): Target time series data.
        Y_matrix (np.ndarray): Source time series data.
        sigma (float): Bandwidth parameter. If None, it will be calculated.

    Returns:
        float: Computed Matrix Transfer Entropy value.
    """
    # Reshape data for Gaussian kernel calculation
    if X_matrix.ndim == 1:
        X_matrix = X_matrix.reshape(-1, 1)
    if Y_matrix.ndim == 1:
        Y_matrix = Y_matrix.reshape(-1, 1)
        
    # Compute bandwidth if not provided
    if sigma is None:
        sigma = silvermans_rule([X_matrix, Y_matrix])
    
    # Compute Gram matrices
    G_X = gaussian_kernel_normalized(X_matrix, sigma)
    G_Y = gaussian_kernel_normalized(Y_matrix, sigma)
    G_R = G_X
    G_S = G_Y
    G_Q = G_X

    # Compute T
    R_S = G_R * G_S  # Element-wise multiplication
    T = R_S / np.trace(R_S)

    # Compute entropies
    # S2(Q|R) = S2(Q,R) - S2(R)
    Q_R = G_Q * G_R
    Q_R_normalized = Q_R / np.trace(Q_R)
    entropy_QR = -np.log(np.trace(Q_R_normalized @ Q_R_normalized))
    entropy_R = -np.log(np.trace(G_R @ G_R))
    S2_Q_given_R = entropy_QR - entropy_R

    # S2(Q|T) = S2(Q,T) - S2(T)
    Q_T = G_Q * T
    Q_T_normalized = Q_T / np.trace(Q_T)
    entropy_QT = -np.log(np.trace(Q_T_normalized @ Q_T_normalized))
    entropy_T = -np.log(np.trace(T @ T))
    S2_Q_given_T = entropy_QT - entropy_T

    # MTE computation
    MTE_Y_to_X = S2_Q_given_R - S2_Q_given_T
    return MTE_Y_to_X

def compute_normalized_matrix(H):
    """
    Normalize the matrix H by the trace of its Hadamard product with itself.

    Args:
        H (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Normalized matrix.
    """
    trace_H = np.trace(H)
    if trace_H == 0:
        raise ValueError("Trace of the matrix product is zero, normalization not possible.")
    return H / trace_H

def update_graph_with_MTE(time_series_data, sigma_values, graph, node_labels):
    """
    Update the directed graph with MTE calculations.

    Args:
        time_series_data (list of np.ndarray): List of time series data for each node.
        sigma_values (list of float): Bandwidth values for each time series.
        graph (networkx.DiGraph): Graph to be updated.
        node_labels (list of str): Labels for the nodes.
    """
    num_series = len(time_series_data)
    for i in range(num_series):
        for j in range(num_series):
            if i != j:
                MTE_value = compute_MTE_embedded(
                    X=time_series_data[i], Y=time_series_data[j],
                    sigma_X=sigma_values[i], sigma_Y=sigma_values[j]
                )
                if not np.isnan(MTE_value) and not np.isinf(MTE_value):
                    graph.add_edge(node_labels[i], node_labels[j], weight=MTE_value)


def update_graph(time_series_list, node_labels, dimensions=3, lag=1, significant_threshold=1e-5):
    """
    Update and plot the graph over time based on MTE calculations.

    Args:
        time_series_list (list of np.ndarray): List of time series data.
        node_labels (list of str): Labels for the nodes.
        dimensions (int): Embedding dimension.
        lag (int): Time lag for embeddings.
        significant_threshold (float): Threshold for significant MTE values.
    """
    sigma = silvermans_rule(time_series_list)
    sigma_values = [sigma] * len(time_series_list)

    num_timesteps = min(len(ts) for ts in time_series_list) - (dimensions - 1) * lag
    for time_index in range(num_timesteps):
        G = nx.DiGraph()
        current_data = [ts[time_index:time_index + dimensions * lag] for ts in time_series_list]
        # Flatten the data
        current_data_flat = [data.reshape(len(data), -1) for data in current_data]
        update_graph_with_MTE(current_data_flat, sigma_values, G, node_labels)

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