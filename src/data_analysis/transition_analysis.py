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
import re
import copy

# Construct the path relative to transition_analysis.py
def get_file_path(hierarchy, temporality):
    return f'../data_collection/events_{hierarchy.replace(" ", "_")}_{temporality.replace(" ", "_")}.csv'

def read_csv_to_list(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return list(reader)

def read_characters_from_csv(file_path):
    """ Read character names from a CSV file and return them as a list. """
    characters = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Ensure the row is not empty
                characters.append(row[0])
    return characters

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

def load_events_from_csv(sentences_path, tuples_path):
    """Loads events from specified CSV files for sentences and tuples."""
    sentences = read_csv_to_list(sentences_path)
    tuples = read_csv_to_list(tuples_path)
    return sentences, tuples

def load_and_classify_events():
    events = {}
    # Define paths for each hierarchical and temporal category
    paths = {
        ('high-level', 'long-term'): ('../data_collection/long_high_sentences.csv', '../data_collection/long_high_tuples.csv'),
        ('high-level', 'medium-term'): ('../data_collection/medium_high_sentences.csv', '../data_collection/medium_high_tuples.csv'),
        ('high-level', 'short-term'): ('../data_collection/short_high_sentences.csv', '../data_collection/short_high_tuples.csv'),
        ('context-specific', 'long-term'): ('../data_collection/long_context_sentences.csv', '../data_collection/long_context_tuples.csv'),
        ('context-specific', 'medium-term'): ('../data_collection/medium_context_sentences.csv', '../data_collection/medium_context_tuples.csv'),
        ('context-specific', 'short-term'): ('../data_collection/short_context_sentences.csv', '../data_collection/short_context_tuples.csv'),
        ('task-specific', 'long-term'): ('../data_collection/long_task_sentences.csv', '../data_collection/long_task_tuples.csv'),
        ('task-specific', 'medium-term'): ('../data_collection/medium_task_sentences.csv', '../data_collection/medium_task_tuples.csv'),
        ('task-specific', 'short-term'): ('../data_collection/short_task_sentences.csv', '../data_collection/short_task_tuples.csv'),
    }
    
    # Load events for each combination of hierarchy and temporality
    for (hierarchy, temporality), (sentences_path, tuples_path) in paths.items():

        sentences, tuples = load_events_from_csv(sentences_path, tuples_path)
        # print("loading code sentences ", sentences)
        # print("loading code tuples ", tuples)

        temporal_hierarchy_sentences = [' '.join(map(str, sublist)) for sublist in sentences]
        tuples_str = [' '.join(map(str, sublist)) for sublist in tuples]
        temporal_hierarchy_tuples = [tuple(sublist) for sublist in tuples]
        # print("loading code temporal_hierarchy_sentences ", sentences)
        # print("loading code temporal_hierarchy_tuples ", tuples)
        # temporal_hierarchy_sentences = sentences
        # temporal_hierarchy_tuples = tuples
        predicted_labels_BC_sent, predicted_labels_PS_sent = classify_new_events(temporal_hierarchy_sentences, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
        predicted_labels_BC_tup, predicted_labels_PS_tup = classify_new_events(tuples_str, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
        combined_states_all_sentences = combine_predictions(predicted_labels_BC_sent, predicted_labels_PS_sent)
        combined_states_all_tuples = combine_predictions(predicted_labels_BC_tup, predicted_labels_PS_tup)
        # print("loading code tuples: ", temporal_hierarchy_tuples)
        
        # Store the classified events
        events[(hierarchy, temporality)] = {
            'sentences': temporal_hierarchy_sentences,
            'tuples': temporal_hierarchy_tuples,
            'combined_predicted_labels_sentences': combined_states_all_sentences,
            'combined_predicted_labels_tuples': combined_states_all_tuples
        }
    return events, combined_states_all_sentences, combined_states_all_tuples



def clean_character_name(name):
    # Strip unwanted characters and fix common formatting issues
    name = re.sub(r'[^a-zA-Z\s]', '', name)  # Remove any non-alphabetic and non-space characters
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with a single space
    return name.strip().lower()  # Trim spaces and convert to lower case

def load_character_aliases(file_path):
    character_aliases = []
    try:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # Process each name in the row after splitting by commas or semicolons
                aliases = [clean_character_name(name) for name in row for name in re.split(r'[,;]\s*', name) if name]
                if aliases:
                    character_aliases.append(aliases)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return character_aliases



# TODO need to make functions to make these 
# Define the size of the matrices
NUM_COG_VECTORS = 4  # This should be the number of cognitive vector categories behavioral states you have
# Cognitive vectors mapping
cog_vectors = {'BP': 0, 'BS': 1, 'CP': 2, 'CS': 3}

# Load characters from CSV
characters = load_character_aliases('../data_collection/characters_list.csv')
print(f"Loaded characters: {characters}")

# Extract the first element (primary name) from each sublist using list comprehension
primary_names = [aliases[0] for aliases in characters if aliases]
print("Primary names list:", primary_names)

# EXAMPLE: characters = ['Kate', 'Jack', 'John', 'Sawyer']
hierarchy_levels = ['high-level', 'context-specific', 'task-specific']
temporality_levels = ['short-term', 'medium-term', 'long-term']

# Mapping from BC and PS classification indices to overall states
bc_ps_to_state = {
    (0, 0): 'BP',  # (Blast, Play)
    (0, 1): 'BS',  # (Blast, Sleep)
    (1, 0): 'CP',  # (Consume, Play)
    (1, 1): 'CS',  # (Consume, Sleep)
}

def combine_predictions(predicted_labels_BC, predicted_labels_PS):
    combined_states = []
    # Handle the case for single values directly
    if np.isscalar(predicted_labels_BC) and np.isscalar(predicted_labels_PS):
        return [bc_ps_to_state.get((predicted_labels_BC, predicted_labels_PS), 'Unknown')]

    # Assuming labels are arrays where each position corresponds to a label for an event
    for bc_label, ps_label in zip(predicted_labels_BC, predicted_labels_PS):
        combined_state = bc_ps_to_state.get((bc_label, ps_label), 'Unknown')  # Handle unknown combinations
        combined_states.append(combined_state)
    return combined_states




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

def get_subject_from_event(event_tup, characters):
    """
    Check if the subject of the event tuple contains any of the character names in the list and return the character's primary name if found,
    ensuring the returned name matches the primary name despite variations in the event tuple.
    
    Args:
    event_tup (tuple): The event tuple where the first item is the subject.
    characters (list of lists): A list of lists where each sublist contains names that the same character could go by, case-sensitive.
    
    Returns:
    str or None: The primary name of the character if found in the characters list; otherwise, returns None.
    """

    print("Character aliases", characters)
    print("Event tuple", event_tup)
    
    # Extract the subject from the first index of the tuple
    subject = event_tup[0]

    # Normalize the subject for robust checking (e.g., lowercasing, removing punctuation)
    normalized_subject = re.sub(r'[^\w\s]', '', subject.lower())  # Remove punctuation and convert to lower case

    # Iterate over each character's alias list
    for aliases in characters:
        # Check each alias in the sublist
        for alias in aliases:
            # Normalize alias similarly for matching
            normalized_alias = re.sub(r'[^\w\s]', '', alias.lower())
            
            # Check if the normalized alias is in the normalized subject text
            if normalized_alias in normalized_subject:
                return aliases[0]  # Return the primary name from the alias list if found

    print("No match found between: subject:", normalized_subject, "and character aliases", characters)
    return None  # Return None if no match is found

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


def find_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1)]
    stationary_distribution = stationary / stationary.sum()
    return stationary_distribution.real.flatten()


def flatten_matrices(matrices):
    return [matrix.flatten() for matrix in matrices]


def simulate_behavior(start_state, transition_matrix, num_steps):
    current_state = start_state
    states = [current_state]
    for _ in range(num_steps):
        current_state = np.random.choice(a=range(len(transition_matrix)), p=transition_matrix[current_state])
        states.append(current_state)
    return states


def process_and_update_matrices(events_dict, transition_matrices):
    time_series_matrices = {char: [] for char in primary_names}  # Prepare a dict to hold all time-series data
    iter_time_dict_list = []
    for (hierarchy, temporality), data in events_dict.items():
        iter_time = {name: 0 for name in primary_names}
        sentences, tuples = data['sentences'], data['tuples']
        combined_states_all_sentences = data.get('combined_predicted_labels_sentences', [])
        combined_states_all_tuples = data.get('combined_predicted_labels_tuples', [])

        # print("tuples: ", tuples)
        # print("sentences: ", sentences)
        print("compare tuple and sentence list lengths: ", "tuples: ", len(tuples), "sentences: ", len(sentences))
        
        for i in range(len(tuples) - 1):
            print("tup range ", len(tuples) -1)
            subject = get_subject_from_event(tuples[i], characters=characters)  # Ensure subject extraction logic matches data structure
            if subject is not None: 
                prev_state = combined_states_all_sentences[i]
                next_state = combined_states_all_sentences[i + 1]
                update_transition_matrix(subject, hierarchy, temporality, prev_state, next_state, transition_matrices, cog_vectors)
                iter_time[subject] += 1
            # Store the current state of the matrices for this timestep
            for char in primary_names:
                time_series_matrices[char].append(copy.deepcopy(transition_matrices[char]))

        print("num time step ", iter_time)        
        iter_time_dict_list.append(iter_time)

            
        

    return time_series_matrices, transition_matrices, iter_time_dict_list, len(time_series_matrices[primary_names[0]]), len(time_series_matrices[primary_names[1]])


def find_min_value(dicts):
    try:
        # Extract all values from all dictionaries into a single list
        all_values = [value for d in dicts for value in d.values() if isinstance(value, (int, float))]
        
        # Check if there are any values to process
        if not all_values:
            return None
        
        # Find the minimum value from all extracted values
        min_value = min(all_values)
        return min_value
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load all events and their predictions
event_dict, pred_labels_sent, pred_labels_tups = load_and_classify_events()

# Initialize your transition matrices
transition_matrices = initialize_transition_matrices(primary_names, hierarchy_levels, temporality_levels, NUM_COG_VECTORS)
print(transition_matrices)

# Process events and update matrices
# At the end of your data processing script, save the time series matrices
time_series_matrices, final_matrices, iter_time_list, num1, num2 = process_and_update_matrices(event_dict, transition_matrices)

print("length of time series matrices for agent 1: ", num1)
print("length of time series matrices for agent 2: ", num2)
min_time_steps = find_min_value(iter_time_list)
print("max time we can allot: ", min_time_steps)

# Path to the file where you want to save the minimum time steps
file_path = 'min_time_steps.txt'

# Write the minimum time steps to a file
with open(file_path, 'w') as file:
    file.write(str(min_time_steps))

for character, matrices in time_series_matrices.items():
    np.save(f'transition_matrices_{character}.npy', matrices)

# Save final matrices for each character
for character, matrix in final_matrices.items():
    np.save(f'transition_matrices_final_{character}.npy', matrix)

# Normalize and save matrices
normalize_matrices(transition_matrices)
print(transition_matrices)
save_transition_matrices(transition_matrices, 'transition_matrices.npy')

# Find the stationary distribution
for char in primary_names:
    for hierarchy in hierarchy_levels: 
        for temporality in temporality_levels:
            transition_matrix = transition_matrices[char][hierarchy][temporality]
            stationary_distribution = find_stationary_distribution(transition_matrix)
            print(f"Stationary distribution for {char} in {hierarchy} {temporality}: {stationary_distribution}")


#  TODO need to add other transition analysis methods
# print("combined_states_all_sentences: ", combined_states_all_sentences)
# # Iterate over sentences, except the last since we're looking at transitions
# for i, sentence in enumerate(med_context_sentences[:-1]):
#     subject = get_subject_from_event(sentence)  # Extract subject from the sentence
    
#     # Ensure we have the next state to form a transition
#     if i < len(combined_states_all_sentences) - 1:
#         prev_state = combined_states_all_sentences[i]
#         next_state = combined_states_all_sentences[i + 1]

#         # Convert states to matrix indices
#         prev_vector = cog_vectors[prev_state]
#         next_vector = cog_vectors[next_state]

#         # Update matrices for all hierarchical levels and temporal categories
#         for hierarchy in hierarchy_levels:
#             for temporality in temporality_levels:
#                 transition_matrices[subject][hierarchy][temporality][prev_vector, next_vector] += 1

# # After updating the matrices, normalize them and save for persistence
# normalize_matrices(transition_matrices)  # Normalize your matrices as implemented previously
# save_transition_matrices(transition_matrices, 'transition_matrices.npy')  # Save your matrices as implemented previously

# bp_index = cog_vectors['BP']  # cog_vectors['BP'] should give you the index for 'BP'
# # Print out the matrices to check them
# for char in characters:
#     print(f"Transition matrices for {char} in 'context-specific' 'medium-term':")
#     print(transition_matrices[char]['context-specific']['medium-term'])
#     print(f"Transition from 'BP' for {char} in 'context-specific' 'medium-term':")
#     bp_transitions = transition_matrices[char]['context-specific']['medium-term'][bp_index]
#     print(bp_transitions)

# # Save the transition matrices if needed
# np.save('bob_matrix.npy', transition_matrices['Bob']['context-specific']['medium-term'])

# # ... Additional code for loading and using the matrices
# # Example usage for a specific character, hierarchical level, and temporal category
# char = 'Bob'
# hierarchy = 'context-specific'
# temporality = 'medium-term'

# # Ensure the matrix is normalized
# normalize_matrices(transition_matrices)  # Assuming this normalizes all matrices

# # Get the transition matrix
# transition_matrix = transition_matrices[char][hierarchy][temporality]

# # Find the stationary distribution
# stationary_distribution = find_stationary_distribution(transition_matrix)
# print(f"Stationary distribution for {char} in {hierarchy} {temporality}: {stationary_distribution}")



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

