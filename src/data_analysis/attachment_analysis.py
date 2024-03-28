import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

# Load event sequences
event_sequences = pd.read_csv("event_sequences.csv")

# Example list of tracked individuals
tracked_individuals = ['person1', 'person2', 'person3']  # Update this list based on your actual tracked individuals

# Function to filter subjects and objects to those being tracked
def filter_tracked_individuals(subjects, objects, tracked_individuals):
    filtered_subjects = [subject for subject in subjects if subject in tracked_individuals]
    filtered_objects = [obj for obj in objects if obj in tracked_individuals]
    return np.union1d(filtered_subjects, filtered_objects)


# Initialize the graph
G = nx.Graph()

# Extract subjects and objects involved in events
subjects = event_sequences['subject'].unique()
objects = pd.unique(event_sequences['object'].str.split(';', expand=True).stack())  # Assuming objects are separated by semicolons

# Filter to only include tracked individuals
all_individuals = filter_tracked_individuals(subjects, objects, tracked_individuals)

G.add_nodes_from(all_individuals)


def calculate_differential_matrix(matrix_pre, matrix_post):
    """Calculate the differential transition matrix."""
    return matrix_post - matrix_pre

def update_graph_with_influence(G, interactions, transition_matrices_by_timestamp):
    """Update the graph with mutual information scores based on differential matrices."""
    for interaction in interactions:
        subject, object, timestamp = interaction  # Assuming interaction tuple format
        matrix_pre_subject = get_pre_interaction_matrix(subject, timestamp, transition_matrices_by_timestamp)
        matrix_post_subject = get_post_interaction_matrix(subject, timestamp, transition_matrices_by_timestamp)
        matrix_pre_object = get_pre_interaction_matrix(object, timestamp, transition_matrices_by_timestamp)
        matrix_post_object = get_post_interaction_matrix(object, timestamp, transition_matrices_by_timestamp)

        differential_matrix_subject = calculate_differential_matrix(matrix_pre_subject, matrix_post_subject)
        differential_matrix_object = calculate_differential_matrix(matrix_pre_object, matrix_post_object)

        # Flatten matrices to calculate mutual information
        mi_score = mutual_info_score(differential_matrix_subject.flatten(), differential_matrix_object.flatten())

        # Update the graph
        if mi_score > threshold:  # Define a suitable threshold based on your analysis
            if G.has_edge(subject, object):
                # Update existing weight (could average, sum, or replace)
                G[subject][object]['weight'] = (G[subject][object]['weight'] + mi_score) / 2
            else:
                G.add_edge(subject, object, weight=mi_score)

def get_pre_interaction_matrix(individual, timestamp, transition_matrices_by_timestamp):
    # Implement retrieval of the pre-interaction matrix for 'individual' just before 'timestamp'
    pass

def get_post_interaction_matrix(individual, timestamp, transition_matrices_by_timestamp):
    # Implement retrieval of the post-interaction matrix for 'individual' just after 'timestamp'
    pass

# Visualization
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)
plt.show()



"""
1. Node Degree Analysis
Degree Centrality: Measures the number of edges connected to each node. High degree centrality indicates influential individuals in the network.
In-Degree and Out-Degree (for directed graphs): Specifically looks at the number of incoming and outgoing connections, which can signify individuals who predominantly influence others versus those who are influenced.
2. Path Analysis
Shortest Paths: Identifies the shortest path(s) between pairs of nodes, which can reveal the most direct influences or connections between individuals.
Average Path Length: Gives a sense of the overall 'small-world' nature of the network, indicating how closely connected the individuals are on average.
3. Clustering and Community Detection
Clustering Coefficient: Measures the degree to which nodes in the network tend to cluster together. High clustering might indicate tight-knit groups or communities.
Community Detection Algorithms (e.g., Louvain method, Girvan-Newman algorithm): Identify clusters or communities within the network where individuals are more densely connected among themselves than with the rest of the network.
4. Centrality Measures
Betweenness Centrality: Quantifies the number of times a node acts as a bridge along the shortest path between two other nodes. High betweenness centrality points to individuals who serve as key connectors or influencers within the network.
Closeness Centrality: Measures how close a node is to all other nodes in the network, highlighting individuals who are well-positioned to influence the entire network quickly.
Eigenvector Centrality: Identifies nodes that are connected to other highly connected nodes, indicating not just individual influence but also the quality of connections in terms of network influence.
5. Assortativity and Homophily
Assortativity Coefficient: Measures the tendency of nodes to connect to other nodes that are similar or dissimilar to themselves in some attributes, such as behavioral states or hierarchical levels.
Homophily Analysis: Examines the extent to which similar individuals (e.g., with similar behavioral patterns) tend to cluster together.
6. Network Robustness and Resilience
Robustness Analysis: Evaluates how the network behaves under failures or targeted attacks, such as the removal of highly central nodes, which can simulate the loss of key individuals.
Connectivity and Components: Analyzes the network's connectivity, identifying which nodes or edges are critical for maintaining the network's connectedness.
7. Dynamics and Temporal Changes
Temporal Network Analysis: For networks that change over time, examining how the structure evolves can reveal dynamic patterns of influence and connectivity.
Influence Propagation Models: Applying models like the Independent Cascade or Linear Threshold model to simulate how influence or information spreads through the network.
8. Mutual Information and Influence
Mutual Information Network: Beyond pairwise mutual information, analyzing the network to identify motifs or structures that signify complex mutual influences among groups of individuals.
9. Spectral Analysis
Laplacian Spectrum: Investigating the eigenvalues of the graph Laplacian can reveal properties about the network's structure, such as the number of connected components or bipartite sections.
10. Graph Embeddings and Machine Learning
Node Embeddings: Using techniques like Node2Vec to generate vector representations of nodes, which can then be used for clustering, visualization, or as features in predictive modeling.


"""