import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist

# Define functions

def silvermans_rule(time_series_list):
    d = len(time_series_list)  # Number of time series
    N = len(time_series_list[0])  # Assuming all time series are of the same length
    variances = [np.var(ts) for ts in time_series_list]
    mean_variance = np.mean(variances)
    sigma = (4 / (d + 2) * N) ** (-1/(d + 4)) * np.sqrt(mean_variance)
    return sigma

def gaussian_kernel_normalized(X, sigma):
    """Generate Gaussian Kernel Gram matrix with normalization."""
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    K_normalized = K / np.trace(K)  # Normalize by trace
    return K_normalized

def matrix_entropy(G):
    trace_squared = np.trace(G @ G)
    print(f"Trace squared: {trace_squared}")  # Debugging print
    if trace_squared <= 0:
        raise ValueError("Trace of matrix squared is zero or negative, check the Gram matrix computation.")
    entropy = -np.log(trace_squared)
    print(f"Entropy: {entropy}")  # Debugging print
    return entropy

def compute_MTE(X_matrix, Y_matrix, sigma=0.4):
    """Compute the Matrix Transfer Entropy from time series Y to X."""
    X_matrix = X_matrix.reshape(-1, 1)
    Y_matrix = Y_matrix.reshape(-1, 1)
    G_X = gaussian_kernel_normalized(X_matrix, sigma)
    G_Y = gaussian_kernel_normalized(Y_matrix, sigma)
    G_R = G_X
    G_S = G_Y
    G_Q = G_X
    G_T = (G_R * G_S) / np.trace(G_R * G_S)
    G_QR = (G_Q * G_R) / np.trace(G_Q * G_R)
    G_QT = (G_Q * G_T) / np.trace(G_Q * G_T)
    entropy_R = -np.log(np.trace(G_R @ G_R))
    entropy_T = -np.log(np.trace(G_T @ G_T))
    print(f"G_R @ G_R trace: {np.trace(G_R @ G_R)}, entropy_R: {entropy_R}")  # Debugging print
    print(f"G_T @ G_T trace: {np.trace(G_T @ G_T)}, entropy_T: {entropy_T}")  # Debugging print
    MTE_Y_to_X = (-np.log(np.trace(G_QR)) + entropy_R + np.log(np.trace(G_QT)) - entropy_T)
    print(f"Final MTE: {MTE_Y_to_X}")  # Debugging print
    return MTE_Y_to_X

def compute_normalized_matrix(H):
    trace_H = np.trace(H)
    if trace_H == 0:
        raise ValueError("Trace of the matrix product is zero, normalization not possible.")
    return H / trace_H

def create_time_lagged_embeddings(X, lag, dimensions):
    n = len(X)
    if n < lag * (dimensions - 1) + 1:
        raise ValueError("Not enough data points for the given lag and dimensions")
    embedded_data = np.zeros((n - lag * (dimensions - 1), dimensions))
    for i in range(dimensions):
        embedded_data[:, i] = X[(dimensions - i - 1) * lag : n - i * lag]
    return embedded_data

def compute_MTE_embedded(X, Y, lag=1, dimensions=3, sigma_X=0.4, sigma_Y=0.4):
    X_embedded = create_time_lagged_embeddings(X, lag, dimensions)
    Y_embedded = create_time_lagged_embeddings(Y, lag, dimensions)
    G_X = gaussian_kernel_normalized(X_embedded, sigma_X)
    G_Y = gaussian_kernel_normalized(Y_embedded, sigma_Y)
    G_R = G_X
    G_S = G_Y
    G_Q = G_X
    G_T = (G_R * G_S) / np.trace(G_R * G_S)
    G_QR = (G_Q * G_R) / np.trace(G_Q * G_R)
    G_QT = (G_Q * G_T) / np.trace(G_Q * G_T)
    entropy_R = -np.log(np.trace(G_R @ G_R))
    entropy_T = -np.log(np.trace(G_T @ G_T))
    print(f"G_R @ G_R trace: {np.trace(G_R @ G_R)}, entropy_R: {entropy_R}")  # Debugging print
    print(f"G_T @ G_T trace: {np.trace(G_T @ G_T)}, entropy_T: {entropy_T}")  # Debugging print
    MTE_Y_to_X = (-np.log(np.trace(G_QR)) + entropy_R + np.log(np.trace(G_QT)) - entropy_T)
    print(f"Final MTE: {MTE_Y_to_X}")  # Debugging print
    return MTE_Y_to_X

# Unit testing
import unittest

class TestMTECalculations(unittest.TestCase):

    def test_gaussian_kernel_normalized(self):
        X = np.array([[1], [2], [3]])
        sigma = 1.0
        G = gaussian_kernel_normalized(X, sigma)
        self.assertTrue(G.shape == (3, 3), "Kernel matrix should be 3x3")
        trace_G = float(np.trace(G))
        self.assertAlmostEqual(trace_G, 1.0, places=4, msg="Trace of normalized kernel matrix should be close to 1")

    def test_matrix_entropy(self):
        G = np.array([[1, 0], [0, 1]])
        entropy = matrix_entropy(G)
        self.assertAlmostEqual(entropy, 0.0, places=4, msg="Entropy of identity matrix should be 0")

    def test_compute_MTE(self):
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([1, 2, 3, 4, 5])
        sigma = 0.4
        mte = compute_MTE(X, Y, sigma)
        self.assertAlmostEqual(mte, 0.0, places=4, msg="MTE of identical time series should be close to 0")

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
        self.assertAlmostEqual(mte, 0.0, places=4, msg="MTE of identical time series should be close to 0")

if __name__ == '__main__':
    unittest.main()