import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform


# Step 2: Generate Synthetic Time Series Data
np.random.seed(42)


# Generate synthetic time series data
t = np.linspace(0, 10, 100)
X = np.sin(t) + 0.5 * np.random.normal(size=t.shape)
Y = np.cos(t) + 0.5 * np.random.normal(size=t.shape)





# Gaussian Kernel Calculations
def gaussian_kernel_normalized(X, sigma):
    # Compute pairwise squared Euclidean distances
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    # Calculate the Gaussian kernel with normalization
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-pairwise_sq_dists / (2 * sigma**2))


# making 2 dimensional
X_matrix = X.reshape(-1, 1)
Y_matrix = Y.reshape(-1, 1)

sigma_X = np.std(X) * np.power(4/(3*len(X)), 1/5)  # Silverman's rule of thumb
sigma_Y = np.std(Y) * np.power(4/(3*len(Y)), 1/5)  # Adjust as necessary

G_X = gaussian_kernel_normalized(X_matrix, sigma_X)
G_Y = gaussian_kernel_normalized(Y_matrix, sigma_Y)

# Assuming G_X, G_Y have been computed as described earlier
G_Q = G_X  # Simplification if Q represents the same as G_X
G_R = G_X  # Assuming R also represents the same; adjust as per actual definition
G_S = G_Y

# Step 4: Calculate Joint Gram Matrix
# We need a correct way to compute the joint Gram matrix. Here's a proper computation based on mutual dependencies:
# Calculate Hadamard products for joint matrices
G_T = (G_R * G_S) / np.trace(G_R * G_S)  # Adjust normalization as per actual data




# Step 3: Calculate Matrix Entropies

# Using the definition of matrix entropy based on the trace of the matrix raised to the power of 
# α
# α:

def matrix_entropy(G):
    return -np.log(np.trace(G**2))

# Compute entropies
entropy_Q = matrix_entropy(G_Q)
entropy_R = matrix_entropy(G_R)
entropy_T = matrix_entropy(G_T)


# Compute conditional entropies if not directly from joint entropies
# Assuming simplifications, directly calculate as per the described formula
MTE_Y_to_X = (-np.log(np.trace((G_Q * G_R / np.trace(G_Q * G_R))**2)) +
              np.log(np.trace(G_R**2)) +
              np.log(np.trace((G_Q * G_T / np.trace(G_Q * G_T))**2)) -
              np.log(np.trace(G_T**2)))


print(MTE_Y_to_X)