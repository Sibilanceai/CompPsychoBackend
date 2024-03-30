# File: big5analysis.py

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import mutual_info_score
from node2vec import Node2Vec

# Function to load event sequences and any existing network data
def load_data(events_file, network_file=None, matrix_file=None):
    event_sequences = pd.read_csv(events_file)
    
    if network_file:
        network_snapshots = pd.read_pickle(network_file)
    else:
        network_snapshots = []
        
    if matrix_file:
        transition_matrices = pd.read_pickle(matrix_file)
    else:
        transition_matrices = {}
    
    return event_sequences, network_snapshots, transition_matrices

def assess_openness(network_snapshots, transition_matrices):
    """
    Assess Openness based on the diversity of connections and interaction patterns.
    """
    openness_scores = []
    for G in network_snapshots:
        diversity_scores = [len(list(G.neighbors(n))) for n in G.nodes()]
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        openness_scores.append(avg_diversity)
    # Additional analysis on transition_matrices can be done here
    return np.mean(openness_scores)

def assess_openness_static(network_snapshots):
    """
    Assess Openness based on diversity of connections in static network snapshots.
    Openness is indicated by a willingness to explore diverse social interactions, reflected in the variety of connections.
    """
    diversity_scores = [np.mean([len(list(G.neighbors(n))) for n in G.nodes()]) for G in network_snapshots if G.nodes()]
    return np.mean(diversity_scores)

def assess_openness_dynamic(transition_matrices):
    """
    Assess Openness through the evolution of social interactions over time, as reflected in transition matrices.
    Dynamic changes in behavior patterns may indicate openness to new experiences and intellectual curiosity.
    """
    interaction_variability = [np.mean([np.std(matrix) for matrix in matrices]) for matrices in transition_matrices.values()]
    return 1 / (1 + np.mean(interaction_variability))


def assess_conscientiousness(network_snapshots, transition_matrices):
    """
    Assess Conscientiousness based on the stability and consistency in interactions.
    """
    stability_scores = []
    for matrix in transition_matrices.values():
        # Assuming transition_matrices is a dictionary {node: [matrix_time1, matrix_time2, ...]}
        matrix_changes = [np.std(m) for m in matrix]
        stability_scores.extend(matrix_changes)
    return 1 / (1 + np.mean(stability_scores) if stability_scores else 1)  # Inverse of variability

def assess_conscientiousness_static(network_snapshots):
    """
    Assess Conscientiousness based on the stability of network centrality across snapshots.
    Consistency in centrality might indicate a stable social role, reflecting conscientious behavior.
    """
    if not network_snapshots:
        return 0

    centrality_changes = []
    previous_snapshot_centrality = nx.degree_centrality(network_snapshots[0])

    for snapshot in network_snapshots[1:]:
        current_snapshot_centrality = nx.degree_centrality(snapshot)
        centrality_diff = sum(abs(previous_snapshot_centrality[node] - current_snapshot_centrality[node]) 
                              for node in snapshot.nodes()) / len(snapshot.nodes())
        centrality_changes.append(centrality_diff)
        previous_snapshot_centrality = current_snapshot_centrality

    # Lower changes in centrality over time indicate higher conscientiousness
    return 1 - np.mean(centrality_changes)  # Normalize to a score between 0 and 1


def assess_conscientiousness_dynamic(transition_matrices):
    """
    Assess Conscientiousness from the consistency of behavior patterns in transition matrices.
    Stability in interactions and minimal erratic changes suggest higher conscientiousness.
    """
    stability_scores = [1 / (1 + np.std(matrix)) for matrix in transition_matrices.values()]  # Inverse variability as stability proxy
    return np.mean(stability_scores)

#def assess_extraversion_static(network_snapshots):
    """
    Assess Extraversion from network snapshots based on degree centrality.
    This method evaluates an individual's social connectivity at static points in time.
    """
    extraversion_scores = []
    for G in network_snapshots:
        centrality_scores = list(nx.degree_centrality(G).values())
        avg_centrality = sum(centrality_scores) / len(centrality_scores) if centrality_scores else 0
        extraversion_scores.append(avg_centrality)
    return np.mean(extraversion_scores)

def assess_extraversion_static(network_snapshots):
    """
    Assess Extraversion from degree centrality in static network snapshots.
    Extraversion is associated with broader social networks and active social engagement.
    """
    centrality_scores = [np.mean(list(nx.degree_centrality(G).values())) for G in network_snapshots]
    return np.mean(centrality_scores)

def assess_extraversion_dynamic(transition_matrices):
    """
    Assess Extraversion by examining the expansion in the range of interactions over time.
    Increasing engagement with a wider array of nodes indicates higher extraversion.
    """
    expansion_scores = []
    for matrices in transition_matrices.values():
        # Assuming matrices are ordered chronologically
        initial_interactions = np.count_nonzero(matrices[0])
        final_interactions = np.count_nonzero(matrices[-1])
        expansion = (final_interactions - initial_interactions) / initial_interactions if initial_interactions > 0 else 0
        expansion_scores.append(expansion)
    
    return np.mean(expansion_scores)



def assess_agreeableness(network_snapshots, transition_matrices):
    """
    Assess Agreeableness based on the formation of mutually beneficial connections.
    """
    clustering_scores = []
    for G in network_snapshots:
        clustering = nx.clustering(G).values()
        clustering_scores.extend(clustering)
    return np.mean(clustering_scores)

def assess_agreeableness_static(network_snapshots):
    """
    Assess Agreeableness based on community tightness and cooperation in static snapshots.
    Agreeableness may manifest as central roles within tightly-knit communities, indicating cooperative behavior.
    """
    clustering_scores = [np.mean(list(nx.clustering(G).values())) for G in network_snapshots]
    return np.mean(clustering_scores)

def assess_agreeableness_dynamic(transition_matrices):
    """
    Assess Agreeableness through the increase in reciprocal interactions over time.
    An increase in mutual engagements suggests cooperative behavior, reflecting higher agreeableness.
    """
    reciprocal_scores = []
    for matrices in transition_matrices.values():
        # Compare initial and final matrices for increased reciprocity
        initial_reciprocity = np.sum((matrices[0] > 0) & (matrices[0].T > 0))
        final_reciprocity = np.sum((matrices[-1] > 0) & (matrices[-1].T > 0))
        change_in_reciprocity = final_reciprocity - initial_reciprocity
        reciprocal_scores.append(change_in_reciprocity)
    
    return np.mean(reciprocal_scores) if reciprocal_scores else 0



def assess_neuroticism(network_snapshots, transition_matrices):
    """
    Assess Neuroticism based on volatility in network position and interactions.
    """
    volatility_scores = []
    for matrix in transition_matrices.values():
        for m in matrix:
            diffs = np.diff(m, axis=0)
            max_change = np.max(np.abs(diffs)) if diffs.size > 0 else 0
            volatility_scores.append(max_change)
    return np.mean(volatility_scores)

def assess_neuroticism_static(network_snapshots):
    """
    Assess Neuroticism by evaluating fluctuations in network connectivity patterns.
    Higher fluctuations might indicate emotional instability, reflecting higher neuroticism.
    """
    if not network_snapshots:
        return 0

    fluctuation_scores = []
    for G in network_snapshots:
        isolation_scores = [1 for n in G.nodes() if len(list(G.neighbors(n))) <= 1]  # Count nodes with 1 or fewer connections
        fluctuation_score = sum(isolation_scores) / len(G.nodes()) if G.nodes() else 0
        fluctuation_scores.append(fluctuation_score)
    
    return np.mean(fluctuation_scores)


def assess_neuroticism_dynamic(transition_matrices):
    """
    Assess Neuroticism through the volatility in behavior patterns, as indicated by transition matrices.
    High variability and erratic changes suggest emotional instability or higher neuroticism.
    """
    volatility_scores = [np.max(np.diff(matrix, axis=0)) for matrix in transition_matrices.values()]  # Max change between states as volatility indicator
    return np.mean(volatility_scores)


def main(events_file, network_file=None, matrix_file=None):
    event_sequences, network_snapshots, transition_matrices = load_data(events_file, network_file, matrix_file)
    
    # Code to update network_snapshots and transition_matrices based on event_sequences
    
    # Assess personality traits
    openness = assess_openness(network_snapshots, transition_matrices)
    conscientiousness = assess_conscientiousness(network_snapshots, transition_matrices)
    extraversion = assess_extraversion_dynamic(transition_matrices)
    agreeableness = assess_agreeableness(network_snapshots, transition_matrices)
    neuroticism = assess_neuroticism(network_snapshots, transition_matrices)
    
    # Combine and return or save the assessment results
    personality_assessments = {
        "Openness": openness,
        "Conscientiousness": conscientiousness,
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Neuroticism": neuroticism
    }
    
    return personality_assessments

if __name__ == "__main__":
    events_file = "path/to/your/events_data.csv"
    network_file = "path/to/your/network_snapshots.pkl"  # Optional
    matrix_file = "path/to/your/transition_matrices.pkl"  # Optional
    results = main(events_file, network_file, matrix_file)
    print(results)
