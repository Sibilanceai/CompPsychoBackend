import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score

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


# Calculate mutual information based on interaction
def calculate_interaction_influence(interactions, transition_matrices):
    influence_scores = {}
    for subject, obj in interactions:
        pre_interaction_matrix_subject = transition_matrices[subject]
        pre_interaction_matrix_obj = transition_matrices[obj]
        # You need to implement logic to update matrices based on interactions and then calculate mutual information
        # This is a placeholder for demonstration
        post_interaction_matrix_subject = update_matrix_based_on_interaction(pre_interaction_matrix_subject, subject, obj)
        post_interaction_matrix_obj = update_matrix_based_on_interaction(pre_interaction_matrix_obj, obj, subject)
        mutual_info = calculate_mutual_information(post_interaction_matrix_subject, post_interaction_matrix_obj)
        influence_scores[(subject, obj)] = mutual_info
    return influence_scores

# Function to calculate mutual information for all pairs among tracked individuals
def calculate_all_mutual_informations(transition_matrices, tracked_individuals):
    mutual_informations = {}
    for person_a in tracked_individuals:
        for person_b in tracked_individuals:
            if person_a != person_b and person_a in transition_matrices and person_b in transition_matrices:
                matrix_a = transition_matrices[person_a].flatten()
                matrix_b = transition_matrices[person_b].flatten()
                mi_score = mutual_info_score(matrix_a, matrix_b)
                mutual_informations[(person_a, person_b)] = mi_score
    return mutual_informations

# Calculate mutual information scores among tracked individuals
mutual_informations = calculate_all_mutual_informations(transition_matrices, all_individuals)

# Define a threshold to determine significant influence
threshold = 0.1  # Adjust based on your data's scale and distribution

# Add edges based on mutual information (representing influence)
for pair, mi_score in mutual_informations.items():
    if mi_score > threshold:
        G.add_edge(pair[0], pair[1], weight=mi_score)

# Visualization
import matplotlib.pyplot as plt

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