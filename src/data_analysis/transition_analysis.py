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

def get_file_path(hierarchy, temporality):
    return f'../data_collection/events_{hierarchy.replace(" ", "_")}_{temporality.replace(" ", "_")}.csv'

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

def load_events(hierarchy_levels, temporality_levels):
    events = {}
    for hierarchy in hierarchy_levels:
        for temporality in temporality_levels:
            file_path = get_file_path(hierarchy, temporality)
            unprocessedeventlist = read_csv_to_list(file_path)
            event_list = [item for sublist in unprocessedeventlist for item in sublist]
            events[(hierarchy, temporality)] = event_list
    return events


# Assuming the ProtoNet model is already trained or loaded
predicted_labels_BC_med_context_sent, predicted_labels_PS_med_context_sent = classify_new_events(med_context_sentences, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
predicted_labels_BC_med_context_tup, predicted_labels_PS_med_context_tup = classify_new_events(med_context_tuples, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)


# TODO need to make functions to make these 
# Define the size of the matrices
NUM_COG_VECTORS = 4  # This should be the number of cognitive vector categories you have
# Cognitive vectors mapping
cog_vectors = {'BP': 0, 'BS': 1, 'CP': 2, 'CS': 3}

characters = ['Kate', 'Jack', 'John', 'Sawyer']
hierarchy_levels = ['high-level', 'context-specific', 'task-specific']
temporality_levels = ['short-term', 'medium-term', 'long-term']

# Mapping from BC and PS classification indices to overall states
bc_ps_to_state = {
    (0, 0): 'BP',  # (Blast, Play)
    (0, 1): 'BS',  # (Blast, Sleep)
    (1, 0): 'CP',  # (Consume, Play)
    (1, 1): 'CS',  # (Consume, Sleep)
}

# Initialize your transition matrices dictionary for each character as before
"""
Predicted Labels Key:
For BC:
0: Blast (B)
1: Consume (C)

For PS:
0: Play (P)
1: Sleep (S)
"""
# Initialization and normalize Transition matrices methods

# Assume you have a way to get the subject of the event, placeholder here
def get_subject_from_event(event_tups):
    # Placeholder: replace with actual logic to extract subject
    return 'Kate'

def initialize_transition_matrices(characters, hierarchy_levels, temporality_levels, num_vectors):
    matrices = {}
    for character in characters:
        character_matrices = {}
        for hierarchy in hierarchy_levels:
            hierarchy_matrices = {}
            for temporality in temporality_levels:
                # Initialize a zero matrix for each combination of hierarchy and temporality
                hierarchy_matrices[temporality] = np.zeros((num_vectors, num_vectors))
            character_matrices[hierarchy] = hierarchy_matrices
        matrices[character] = character_matrices
    return matrices



def normalize_matrices(matrices):
    for character_matrices in matrices.values():
        for hierarchy_matrices in character_matrices.values():
            for temporality_matrices in hierarchy_matrices.values():
                row_sums = temporality_matrices.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                temporality_matrices[:] = temporality_matrices / row_sums


def save_transition_matrices(matrices, filepath):
    with open(filepath, 'wb') as f:
        np.save(f, matrices, allow_pickle=True)

def load_transition_matrices(filepath):
    with open(filepath, 'rb') as f:
        return np.load(f, allow_pickle=True).item()
# hierarchical/temporal indices and cognitive vectors
def map_event_to_indices(event):
    # Might not need
    return None
    return temporal_index, hierarchical_index, prev_vector, next_vector

def categorize_event_with_langchain(event_description):
    # Interaction with LangChain to categorize the event
    # This is pseudo-code; need to replace it with actual LangChain interaction code
    category = "BS"  # Placeholder for the actual category determined by LangChain
    return category


# Update transition matrices based on an event
def update_transition_matrix(subject, hierarchy, temporality, prev_state, next_state, transition_matrices, cog_vectors):
    # Convert states to matrix indices
    prev_vector = cog_vectors.get(prev_state)
    next_vector = cog_vectors.get(next_state)

    # Validate indices
    if prev_vector is not None and next_vector is not None:
        # Update the specified matrix
        transition_matrices[subject][hierarchy][temporality][prev_vector, next_vector] += 1

def process_events_for_all_matrices(characters, hierarchy_levels, temporality_levels, events, transition_matrices):
    for hierarchy, temporality in events:
        event_list = events[(hierarchy, temporality)]
        for event in event_list:
            subject = get_subject_from_event(event)
            if subject in characters:
                prev_state, next_state = extract_states_from_event(event)  # Implement this
                update_transition_matrix(subject, hierarchy, temporality, prev_state, next_state, transition_matrices, cog_vectors)



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



# set up the transition matrices

transition_matrices = initialize_transition_matrices(characters, hierarchy_levels, temporality_levels, NUM_COG_VECTORS)
# After updating transition matrices with new event data
normalize_matrices(transition_matrices)

# Saving the matrices to a file for persistence
save_transition_matrices(transition_matrices, 'transition_matrices.npy')

# At a later point, to load the matrices
transition_matrices = load_transition_matrices('transition_matrices.npy')


# Function to combine BC and PS predictions into overall state
def combine_predictions(predicted_labels_BC, predicted_labels_PS):
    combined_states = []
    for bc, ps in zip(predicted_labels_BC, predicted_labels_PS):
        if bc == 0 and ps == 0:
            combined_states.append('BP')
        elif bc == 0 and ps == 1:
            combined_states.append('BS')
        elif bc == 1 and ps == 0:
            combined_states.append('CP')
        else:  # bc == 1 and ps == 1
            combined_states.append('CS')
    return combined_states



# Generate combined states for all sentences, with actual data
# combined_states_all_sentences = combine_predictions(predicted_labels_BC_med_context_sent, predicted_labels_PS_med_context_sent)

# synthetic data for testing
import random

def generate_combined_states(num_events, states):
    """ Generate a random sequence of states for a given number of events. """
    return [random.choice(states) for _ in range(num_events)]

# Example states and number of events
states = ['BP', 'BS', 'CP', 'CS']
num_events = 100  # Number of transitions to simulate

combined_states_all_sentences = generate_combined_states(num_events, states)


print("combined_states_all_sentences: ", combined_states_all_sentences)
# Iterate over sentences, except the last since we're looking at transitions
for i, sentence in enumerate(med_context_sentences[:-1]):
    subject = get_subject_from_event(sentence)  # Extract subject from the sentence
    
    # Ensure we have the next state to form a transition
    if i < len(combined_states_all_sentences) - 1:
        prev_state = combined_states_all_sentences[i]
        next_state = combined_states_all_sentences[i + 1]

        # Convert states to matrix indices
        prev_vector = cog_vectors[prev_state]
        next_vector = cog_vectors[next_state]

        # Update matrices for all hierarchical levels and temporal categories
        for hierarchy in hierarchy_levels:
            for temporality in temporality_levels:
                transition_matrices[subject][hierarchy][temporality][prev_vector, next_vector] += 1

# After updating the matrices, normalize them and save for persistence
normalize_matrices(transition_matrices)  # Normalize your matrices as implemented previously
save_transition_matrices(transition_matrices, 'transition_matrices.npy')  # Save your matrices as implemented previously

bp_index = cog_vectors['BP']  # cog_vectors['BP'] should give you the index for 'BP'
# Print out the matrices to check them
for char in characters:
    print(f"Transition matrices for {char} in 'context-specific' 'medium-term':")
    print(transition_matrices[char]['context-specific']['medium-term'])
    print(f"Transition from 'BP' for {char} in 'context-specific' 'medium-term':")
    bp_transitions = transition_matrices[char]['context-specific']['medium-term'][bp_index]
    print(bp_transitions)

# Save the transition matrices if needed
np.save('bob_matrix.npy', transition_matrices['Bob']['context-specific']['medium-term'])

# ... Additional code for loading and using the matrices
# Example usage for a specific character, hierarchical level, and temporal category
char = 'Bob'
hierarchy = 'context-specific'
temporality = 'medium-term'

# Ensure the matrix is normalized
normalize_matrices(transition_matrices)  # Assuming this normalizes all matrices

# Get the transition matrix
transition_matrix = transition_matrices[char][hierarchy][temporality]

# Find the stationary distribution
stationary_distribution = find_stationary_distribution(transition_matrix)
print(f"Stationary distribution for {char} in {hierarchy} {temporality}: {stationary_distribution}")



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

