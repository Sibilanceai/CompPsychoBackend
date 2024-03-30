import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from community import community_louvain
from node2vec import Node2Vec

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

# Define a threshold to determine significant influence
threshold = 0.1  # Adjust based on your data's scale and distribution

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

# analysis functions

def plot_degree_centrality(G):
    degree_centrality = nx.degree_centrality(G)
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(degree_centrality)), list(degree_centrality.values()), align='center')
    plt.xticks(range(len(degree_centrality)), list(degree_centrality.keys()), rotation='vertical')
    plt.xlabel('Node')
    plt.ylabel('Degree Centrality')
    plt.title('Node Degree Centrality')
    plt.show()

def plot_in_out_degree_centrality(G):
    if not G.is_directed():
        print("Graph is not directed. In-degree and out-degree analysis is not applicable.")
        return
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(in_degree_centrality)), list(in_degree_centrality.values()), align='center')
    plt.xticks(range(len(in_degree_centrality)), list(in_degree_centrality.keys()), rotation='vertical')
    plt.xlabel('Node')
    plt.ylabel('In-Degree Centrality')
    plt.title('Node In-Degree Centrality')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(out_degree_centrality)), list(out_degree_centrality.values()), align='center')
    plt.xticks(range(len(out_degree_centrality)), list(out_degree_centrality.keys()), rotation='vertical')
    plt.xlabel('Node')
    plt.ylabel('Out-Degree Centrality')
    plt.title('Node Out-Degree Centrality')

    plt.tight_layout()
    plt.show()

def analyze_shortest_paths(G):
    # Calculate shortest paths for all pairs of nodes
    # Note: This can be resource-intensive for large graphs
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Calculate average path length
    total_paths_length = sum([sum(lengths.values()) for source, lengths in path_lengths.items()])
    number_of_paths = sum([len(lengths) for source, lengths in path_lengths.items()])
    average_path_length = total_paths_length / number_of_paths
    
    print(f"Average Path Length: {average_path_length}")
    
    return path_lengths, average_path_length


def analyze_clustering_and_communities(G):
    # Clustering Coefficient
    clustering_coefficient = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {clustering_coefficient}")
    
    # Community Detection with Louvain Method
    partition = community_louvain.best_partition(G)
    
    # Number of communities
    num_communities = len(set(partition.values()))
    print(f"Number of communities detected: {num_communities}")
    
    return clustering_coefficient, partition

def analyze_centrality_measures(G):
    # Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)
    
    # Eigenvector Centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    return betweenness_centrality, closeness_centrality, eigenvector_centrality

def analyze_assortativity(G, attribute_name):
    # For numeric attributes
    assortativity_coefficient = nx.numeric_assortativity_coefficient(G, attribute_name)
    print(f"Assortativity Coefficient for {attribute_name}: {assortativity_coefficient}")
    
    return assortativity_coefficient

def homophily_analysis(G, attribute):
    """
    Perform a homophily analysis based on the distribution of node connections relative to an attribute.
    
    :param G: A NetworkX graph
    :param attribute: The node attribute to perform homophily analysis on (e.g., 'group')
    """
    groups = set(nx.get_node_attributes(G, attribute).values())
    connections_within_groups = {group: 0 for group in groups}
    total_connections = 0
    
    for node in G.nodes(data=True):
        node_group = node[1][attribute]
        for neighbor in G[node[0]]:
            total_connections += 1
            neighbor_group = G.nodes[neighbor].get(attribute, None)
            if node_group == neighbor_group:
                connections_within_groups[node_group] += 1
                
    print("Homophily Analysis:")
    for group, connections in connections_within_groups.items():
        print(f"Group {group}: {connections / total_connections * 100:.2f}% of connections are within this group")



def analyze_network_robustness(G, centrality_measure=nx.betweenness_centrality):
    # Calculate centrality
    centrality = centrality_measure(G)
    
    # Sort nodes by centrality
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    
    # Remove top 10% most central nodes
    G_reduced = G.copy()
    nodes_to_remove = sorted_nodes[:len(sorted_nodes) // 10]
    G_reduced.remove_nodes_from(nodes_to_remove)
    
    # Check if the network is still connected (for undirected graphs)
    is_connected = nx.is_connected(G_reduced)
    
    print(f"Is the network still connected after removing top 10% central nodes? {is_connected}")
    
    return is_connected

snapshots = [G_t1, G_t2, G_t3, ...]  # G_ti represents the network snapshot at time i


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
    
    # Plotting
    plt.figure(figsize=(14, 10))
    for i, (metric, values) in enumerate(metrics.items(), start=1):
        plt.subplot(2, 2, i)
        plt.plot(values, marker='o', linestyle='-')
        plt.title(metric)
        plt.xlabel('Time Snapshot')
        plt.ylabel(metric)
    
    plt.tight_layout()
    plt.show()



def analyze_community_evolution(snapshots):
    """
    Analyzes the evolution of community structure over time.
    
    :param snapshots: A list of NetworkX graphs representing the network at different time points.
    """
    community_changes = []

    for G in snapshots:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        community_changes.append(num_communities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(community_changes, marker='o', linestyle='-')
    plt.title('Evolution of Community Structure')
    plt.xlabel('Time Snapshot')
    plt.ylabel('Number of Communities')
    plt.grid(True)
    plt.show()

def prepare_for_temporal_motif_analysis(network_snapshots):
    # This method is a placeholder for preparing your network data
    # for temporal motif analysis, which might involve identifying 
    # recurring interaction patterns within and across snapshots.
    print("Prepare your network snapshots for temporal motif analysis.")
    # Actual implementation will depend on the motifs of interest
    # and the tool or algorithm you plan to use.

def longitudinal_network_analysis(network_snapshots):
    metrics = {'Average Clustering': [], 'Average Shortest Path Length': []}
    
    for G in network_snapshots:
        metrics['Average Clustering'].append(nx.average_clustering(G))
        if nx.is_connected(G):
            metrics['Average Shortest Path Length'].append(nx.average_shortest_path_length(G))
        else:
            metrics['Average Shortest Path Length'].append(float('inf'))
    
    # Plotting
    plt.figure(figsize=(10, 5))
    for metric, values in metrics.items():
        plt.plot(values, label=metric)
    plt.xlabel('Time (snapshot index)')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()


def generate_node_embeddings(G):
    """
    Generate node embeddings using Node2Vec.
    
    :param G: A NetworkX graph.
    :return: A model that contains the node embeddings.
    """
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1)
    return model

def node_embedding_evolution(network_snapshots):
    embeddings_over_time = []
    for G in network_snapshots:
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=2)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings_over_time.append(model.wv)
    return embeddings_over_time

def link_prediction_analysis(network_snapshots):
    for i in range(len(network_snapshots)-1):
        G1 = network_snapshots[i]
        G2 = network_snapshots[i+1]
        predicted_edges = nx.preferential_attachment(G1, G2.nodes())
        # Process and analyze predicted_edges as needed
        # This is a simplified placeholder. Actual implementation may vary based on
        # the prediction method and how you compare predictions to the subsequent snapshot.
        print(f"Snapshot {i} to {i+1}: Predicted new or strengthened connections.")


def laplacian_spectrum_analysis(G):
    """
    Analyze and plot the Laplacian spectrum of the network.
    
    :param G: A NetworkX graph.
    """
    laplacian = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues.sort()
    
    plt.figure()
    plt.plot(eigenvalues, marker='o')
    plt.title('Laplacian Spectrum')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.show()
    
    # The number of near-zero eigenvalues indicates the number of connected components
    # Small eigenvalues indicate potential community structures


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

1. Node Degree Analysis
Degree Centrality: Measures how connected or influential an individual is within the network. High degree centrality indicates individuals who have significant influence over or are influenced by many others, possibly acting as central figures in the group's social dynamics.

2. Path Analysis
Shortest Paths: Identifying the shortest paths between pairs of individuals can reveal the most direct lines of influence or pathways through which behaviors or information flow.
Average Path Length: Offers insight into the group's cohesion or separation. A shorter average path length suggests a tightly-knit group where influences can quickly spread, while a longer average implies more isolated subgroups or individuals.

3. Clustering and Community Detection
Clustering Coefficient: Indicates the presence of tightly-knit subgroups or cliques within the network, where individuals are more densely connected to each other. High clustering might suggest subgroups with strong internal bonds or shared behaviors.
Community Detection: Helps identify these subgroups or communities within the network explicitly. Communities might represent individuals with similar roles, behaviors, or positions within the group's social hierarchy.

4. Centrality Measures
Betweenness Centrality: Identifies individuals who serve as bridges or connectors between different parts of the network. These individuals might facilitate or control the flow of influence or information across the group.
Closeness Centrality: Highlights individuals who can quickly influence or be influenced by others in the network, indicating their strategic position in the social fabric.
Eigenvector Centrality: Points to individuals who are not just well-connected but are connected to other well-connected individuals, suggesting influential individuals in a network of influencers.

5. Assortativity and Homophily
Assortativity: Measures how similar individuals tend to connect more frequently than dissimilar ones, possibly in terms of behaviors, roles, or other attributes. This can indicate the presence of homophily - the tendency of individuals to associate with others who are similar to themselves.

6. Network Robustness and Resilience
Robustness Analysis: Evaluates how the network withstands disruptions, such as the loss of key individuals. This can reveal the network's vulnerability or resilience to changes, highlighting how dependent the group is on specific members for maintaining its social structure.

7. Dynamics and Temporal Changes
Temporal Network Analysis: Observing how the network structure evolves over time can provide insights into the dynamics of social relationships, the emergence of leaders or central figures, and how external events or internal group dynamics affect social bonds.

8. Mutual Information and Influence
Mutual Information Network: Analyzing the network based on mutual information can help understand the depth and complexity of influence between individuals, going beyond direct interactions to capture more subtle or indirect influences.

9. Spectral Analysis
Laplacian Spectrum: Offers a way to understand the network's fundamental properties, such as connectivity and the potential for partitioning the network into distinct communities or groups.

10. Graph Embeddings and Machine Learning
Node Embeddings: Utilizing machine learning to generate vector representations of individuals based on their network positions can facilitate clustering, visualization, and predictive modeling, revealing hidden patterns in social dynamics and potential predictors of behavior or role within the group.
"""



"""
To fully realize your project of quantifying the influence or mutual information based on differential matrices in the context of Computational Psychodynamics, focusing on interactions and their impact on transition matrices over time, the following components still need to be implemented:

### 1. **Pre- and Post-Interaction Matrix Retrieval**
   - **Implement `get_pre_interaction_matrix`**: Function to retrieve the transition matrix for an individual immediately before a specified interaction.
   - **Implement `get_post_interaction_matrix`**: Function to retrieve the transition matrix for an individual immediately after a specified interaction.

### 2. **Event and Interaction Identification**
   - **Identify Interactions**: A detailed method to parse event sequences and accurately identify interactions between individuals, including the participants and the timestamp of each interaction.

### 3. **Differential Matrix Calculation**
   - **Optimize `calculate_differential_matrix`**: Ensure this function efficiently computes the difference between pre- and post-interaction matrices, focusing on capturing meaningful changes in transition probabilities.

### 4. **Mutual Information Calculation**
   - **Refine Calculation**: Verify and potentially optimize the calculation of mutual information using differential matrices to ensure it accurately measures the influence between individuals.

### 5. **Graph Update Mechanism**
   - **Enhance `update_graph_with_influence`**: Ensure this function correctly updates the graph with mutual information scores, considering how to handle multiple interactions between the same pair of individuals (e.g., averaging, summing, or keeping the maximum score).

### 6. **Threshold for Influence**
   - **Define and Adjust Threshold**: Establish a suitable threshold for mutual information scores to decide when an influence is significant enough to be represented as an edge in the graph.

### 7. **Graph Visualization Enhancements**
   - **Improve Visualization**: Implement more sophisticated visualization techniques to effectively display the network, possibly including edge weights (influence scores) and highlighting key nodes based on centrality measures or clustering.

### 8. **Temporal Analysis**
   - **Longitudinal Tracking**: Develop methods for analyzing how influences evolve over time, potentially involving dynamic graph models or time-series analysis of mutual information scores.

### 9. **Validation and Sensitivity Analysis**
   - **Validate Approach**: Conduct tests to validate the approach, including checking the mutual information calculation against known benchmarks or simulated data.
   - **Sensitivity Analysis**: Explore how sensitive the results are to the choice of threshold for influence, the method of updating the graph with new information, and the handling of pre- and post-interaction matrices.

### 10. **Integration and Workflow Optimization**
   - **Workflow Integration**: Ensure all components are integrated into a coherent workflow, from data preprocessing to final analysis and visualization.
   - **Optimize Performance**: Optimize the performance for handling large datasets, including efficient data structures for transition matrices and scalable graph analysis techniques.

### 11. **Documentation and Testing**
   - **Comprehensive Documentation**: Write detailed documentation for each function and component, explaining inputs, outputs, and assumptions.
   - **Robust Testing**: Implement testing for each component to ensure reliability and accuracy, particularly for custom functions like mutual information calculation based on differential matrices.

By addressing these components, you'll build a robust system for analyzing the dynamic influence of interactions within a network, grounded in the principles of Computational Psychodynamics and informed by detailed behavioral data.
"""