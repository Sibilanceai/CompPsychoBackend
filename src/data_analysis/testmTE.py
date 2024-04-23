import numpy as np
from scipy.linalg import eigh

def calculate_normalized_gram_matrix(data, kernel_width=1.0):
    """ Calculate the normalized Gram matrix for a given state-space data. """
    pairwise_distances = np.sum((data[:, None] - data[None, :]) ** 2, axis=2)
    K = np.exp(-pairwise_distances / (2 * kernel_width ** 2))
    return K / np.trace(K)

def matrix_renyi_entropy(K, alpha=1.01):
    """ Calculate the matrix-based Rényi entropy of a given Gram matrix. """
    eigenvalues = eigh(K, eigvals_only=True)
    return (1 / (1 - alpha)) * np.log2(np.sum(eigenvalues ** alpha))

def transfer_entropy(X, Y, alpha=1.01, kernel_width=1.0):
    """ Compute matrix-based transfer entropy from X to Y. """
    Ax = calculate_normalized_gram_matrix(X, kernel_width)
    Ay = calculate_normalized_gram_matrix(Y, kernel_width)
    Axy = calculate_normalized_gram_matrix(np.hstack((X, Y)), kernel_width)
    
    Hx = matrix_renyi_entropy(Ax, alpha)
    Hy = matrix_renyi_entropy(Ay, alpha)
    Hxy = matrix_renyi_entropy(Axy, alpha)
    
    return Hxy - Hx - Hy  # Adjust based on the definition used

# Example usage
X = np.random.randn(100, 4)  # 100 time points, 4-dimensional system
Y = np.random.randn(100, 4)
te = transfer_entropy(X, Y)
print("Transfer Entropy from X to Y:", te)
