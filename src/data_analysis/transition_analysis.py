import numpy as np
import sys
import os

# Add the data_annotation directory to the path
sys.path.insert(0, os.path.abspath('../data_annotation'))

from eventlevelprotonet import ProtoNet, get_contextual_embedding, compute_prototypes, create_episode, train_proto_net, validate_proto_net, classify_new_events, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS
from sklearn.cluster import KMeans
import networkx as nx
from scipy.stats import ttest_ind
import csv

# Construct the path relative to transition_analysis.py
med_context_sentences_path = '../data_collection/med_context_sentences.csv'
med_context_tuples_path = '../data_collection/med_context_tuples.csv'

def read_csv_to_list(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return list(reader)
    

# Read the event sentences and tuples
med_context_sentences = read_csv_to_list(med_context_sentences_path)
med_context_tuples = read_csv_to_list(med_context_tuples_path)
# Flatten the list of lists if necessary
med_context_sentences = [item for sublist in med_context_sentences for item in sublist]
med_context_tuples = [item for sublist in med_context_tuples for item in sublist]


# Assuming the ProtoNet model is already trained or loaded
predicted_labels_BC_med_context_sent, predicted_labels_PS_med_context_sent = classify_new_events(med_context_sentences, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
predicted_labels_BC_med_context_tup, predicted_labels_PS_med_context_tup = classify_new_events(med_context_tuples, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)

# Define the size of the matrices
NUM_COG_VECTORS = 4  # This should be the number of cognitive vector categories you have
# Cognitive vectors mapping
cog_vectors = {'BP': 0, 'CP': 1, 'BS': 0, 'CS': 1}


# Initialize a dictionary to hold the transition matrices for each character
# Initialize the matrices for each category
transition_matrices = {
    'high-level': {
        'short-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
        'medium-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
        'long-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
    },
    'context-specific': {
        'short-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
        'medium-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
        'long-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
    },
    'task-specific': {
        'short-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
        'medium-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
        'long-term': np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS)),
    }
}

# Assume you have a way to get the subject of the event, placeholder here
def get_subject_from_event(event_tups):
    # Placeholder: replace with actual logic to extract subject
    return 'Bob'

def initialize_character_matrices(characters, hierarchy_levels, temporality_levels, num_vectors):
    matrices = {}
    for character in characters:
        matrices[character] = {}
        for hierarchy in hierarchy_levels:
            matrices[character][hierarchy] = {}
            for temporality in temporality_levels:
                matrices[character][hierarchy][temporality] = np.zeros((num_vectors, num_vectors))
    return matrices

def normalize_matrices(transition_matrices):
    for character_matrices in transition_matrices.values():
        for hierarchy_matrices in character_matrices.values():
            for matrix in hierarchy_matrices.values():
                row_sums = matrix.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1
                matrix[:] = matrix / row_sums  # Normalize in-place

def save_transition_matrices(matrices, filepath):
    # Saves the entire transition matrices dictionary as a NumPy binary file
    np.save(filepath, matrices)

def load_transition_matrices(filepath):
    # Loads the entire transition matrices dictionary from a NumPy binary file
    return np.load(filepath, allow_pickle=True).item()


characters = ['Bob', 'Jack', 'John', 'Sawyer']
hierarchy_levels = ['high-level', 'context-specific', 'task-specific']
temporality_levels = ['short-term', 'medium-term', 'long-term']


transition_matrices = initialize_character_matrices(characters, hierarchy_levels, temporality_levels, NUM_COG_VECTORS)

# After updating transition matrices with new event data
normalize_matrices(transition_matrices)

# Saving the matrices to a file for persistence
save_transition_matrices(transition_matrices, 'transition_matrices.npy')

# At a later point, to load the matrices
transition_matrices = load_transition_matrices('transition_matrices.npy')

# Mapping from BC and PS classification indices to overall states
bc_ps_to_state = {
    (0, 0): 'BP',  # (Blast, Play)
    (0, 1): 'BS',  # (Blast, Sleep)
    (1, 0): 'CP',  # (Consume, Play)
    (1, 1): 'CS',  # (Consume, Sleep)
}

# Initialize your transition matrices dictionary for each character as before

# Example function to combine BC and PS predictions into overall state
def combine_predictions(predicted_labels_BC, predicted_labels_PS):
    combined_states = [bc_ps_to_state[(bc, ps)] for bc, ps in zip(predicted_labels_BC_med_context_sent, predicted_labels_PS_med_context_sent)]
    return combined_states

# Classify and update the matrices with combined states
for i, sentence in enumerate(med_context_sentences[:-1]):
    subject = get_subject_from_event(sentence)  # Get the subject from the event sentence
    
    # Use the combine_predictions function to get the overall state for each pair of consecutive events
    # do it for all of the levels and temporal categories
    combined_states = combine_predictions(predicted_labels_BC_med_context_sent, predicted_labels_PS_med_context_sent)
    
    if i < len(combined_states) - 1:  # Ensure we have the next state to form a transition
        prev_state = combined_states[i]
        next_state = combined_states[i + 1]

        # Get the indices for the cognitive vector states
        prev_vector = cog_vectors[prev_state]
        next_vector = cog_vectors[next_state]

        # Assuming you have a 'medium-term' and 'context-specific' category for every character
        # Update the transition matrix for the character in context-specific medium term
        transition_matrices[subject]['context-specific']['medium-term'][prev_vector, next_vector] += 1

# After updating the matrices, normalize them and save for persistence
normalize_matrices(transition_matrices)  # Normalize your matrices as implemented previously
save_transition_matrices(transition_matrices, 'transition_matrices.npy')  # Save your matrices as implemented previously

# Print out the matrices to check them
for char in characters:
    print(f"Transition matrices for {char}:")
    print(transition_matrices[char]['context-specific']['medium-term']['BC'])
    print(transition_matrices[char]['context-specific']['medium-term']['PS'])

# Save the transition matrices if needed
np.save('bob_bc_matrix.npy', transition_matrices['Bob']['context-specific']['medium-term']['BC'])
np.save('bob_ps_matrix.npy', transition_matrices['Bob']['context-specific']['medium-term']['PS'])

# ... Additional code for loading and using the matrices


# Example function to map events to hierarchical/temporal indices and cognitive vectors
def map_event_to_indices(event):
    # Implement your logic here
    return None
    return temporal_index, hierarchical_index, prev_vector, next_vector

def categorize_event_with_langchain(event_description):
    # Interaction with LangChain to categorize the event
    # This is pseudo-code; you'll need to replace it with your actual LangChain interaction code
    category = "BS"  # Placeholder for the actual category determined by LangChain
    return category


# Update transition matrices based on an event
def update_transition_matrix(event, prev_event, transition_matrices, cog_vectors):
    if prev_event is None:
        return  # Skip if there's no previous event to form a transition
    
    # Example logic to determine matrix indices based on event categories
    prev_vector = cog_vectors.get(prev_event)
    next_vector = cog_vectors.get(event)
    
    if prev_vector is not None and next_vector is not None:
        # Update the matrix; details depend on your specific logic for mapping events to matrices
        pass


def find_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1)]
    stationary_distribution = stationary / stationary.sum()
    return stationary_distribution.real.flatten()

# Iterate over matrices to calculate stationary distributions
# for hierarchy_level, temporal_matrices in transition_matrices.items():
#     for temporal_level, matrix in temporal_matrices.items():
#         dist = find_stationary_distribution(matrix)
#         # Store or process the stationary distribution




def flatten_matrices(matrices):
    return [matrix.flatten() for matrix in matrices]

# Assuming you've collected all matrices into a list for clustering
# all_matrices = flatten_matrices([...])  # Flatten all matrices for clustering
# kmeans = KMeans(n_clusters=3).fit(all_matrices)
# labels = kmeans.labels_


# Pseudo-code, as implementation depends on the specific patterns you're looking for
# for sequence in sequences:
#     analyze_sequence(sequence)
#     # Implement pattern recognition or sequence mining based on your criteria


def simulate_behavior(start_state, transition_matrix, num_steps):
    current_state = start_state
    states = [current_state]
    for _ in range(num_steps):
        current_state = np.random.choice(a=range(len(transition_matrix)), p=transition_matrix[current_state])
        states.append(current_state)
    return states


# from scipy.stats import ttest_ind

# # Assuming `group1_features` and `group2_features` are arrays of features derived from transition matrices
# t_stat, p_val = ttest_ind(group1_features, group2_features)


# 6. Absorption Probabilities
# Calculate the probability of reaching an absorbing state from other states.

# This is complex and requires setting up a modified transition matrix that identifies absorbing states.
# Implement a method to modify the transition matrix and calculate absorption probabilities.

# # Pseudo-code
# for matrix in transition_matrices:
#     identify_transient_and_recurrent_states(matrix)
#     # Analyze matrix structure to classify states

# def entropy_rate(transition_matrix):
#     # Calculate entropy rate from the transition probabilities
#     stationary_dist = find_stationary_distribution(transition_matrix)
#     entropy_rate = -np.sum(stationary_dist * np.sum(transition_matrix * np.log(transition_matrix), axis=1))
#     return entropy_rate


# def mean_first_passage_time(transition_matrix, state_a, state_b):
#     # Calculate MFPT from state_a to state_b
#     pass  # Implement MFPT calculation


# # testing


# # Simulate behavior example
# start_state = 0  # Assuming starting from 'Sleep' state
# num_steps = 10  # Example number of steps
# simulated_states = simulate_behavior(start_state, transition_matrices[0, 0], num_steps)  # Use a specific matrix for simulation

# # Conduct t-test example (Assuming you have defined group1_features and group2_features arrays)
# t_stat, p_val = ttest_ind(group1_features, group2_features)
# print("T-test results: T-statistic = {}, P-value = {}".format(t_stat, p_val))

