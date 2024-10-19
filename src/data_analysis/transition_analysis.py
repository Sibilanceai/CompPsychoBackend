import numpy as np
import sys
import os

# Add the data_annotation directory to the path
sys.path.insert(0, os.path.abspath('../data_annotation'))

from eventlevelprotonet import ProtoNet, get_contextual_embedding, compute_prototypes, create_episode, train_proto_net, validate_proto_net, classify_new_events, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS
from sklearn.cluster import KMeans
import networkx as nx
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.linalg import eig  # For spectral analysis
import csv
import re
import copy
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated character list with aliases
characters = [
    ['sherlock', 'holmes', 'sherlock holmes', 'protagonist'],
    ['watson', 'dr watson', 'doctor watson', 'narrator'],
    ['irene adler', 'adler', 'woman', 'female character'],
    ['king of bohemia', 'king', 'king of bohemia'],
    ['jabez wilson', 'mr wilson', 'wilson'],
    ['john clay', 'clay', 'vincent spaulding', 'spaulding'],
    ['duncan ross', 'mr duncan ross'],
    ['godfrey norton', 'mr godfrey norton', 'male character'],
    ['jones', 'mr jones', 'detective jones'],
    ['merryweather', 'mr merryweather'],
    ['count von kramm'],
    # Add more characters as needed
]

# Define the list of states corresponding to your transition matrix indices
states = ['BP', 'BS', 'CP', 'CS']  


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
        temporal_hierarchy_sentences = [' '.join(map(str, sublist)) for sublist in sentences]
        tuples_str = [' '.join(map(str, sublist)) for sublist in tuples]
        temporal_hierarchy_tuples = [tuple(sublist) for sublist in tuples]

        predicted_labels_BC_sent, predicted_labels_PS_sent = classify_new_events(temporal_hierarchy_sentences, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
        predicted_labels_BC_tup, predicted_labels_PS_tup = classify_new_events(tuples_str, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS)
        combined_states_all_sentences = combine_predictions(predicted_labels_BC_sent, predicted_labels_PS_sent)
        combined_states_all_tuples = combine_predictions(predicted_labels_BC_tup, predicted_labels_PS_tup)
        
        # Store the classified events
        events[(hierarchy, temporality)] = {
            'sentences': temporal_hierarchy_sentences,
            'tuples': temporal_hierarchy_tuples,
            'combined_predicted_labels_sentences': combined_states_all_sentences,
            'combined_predicted_labels_tuples': combined_states_all_tuples
        }
    return events

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

# Define the size of the matrices
NUM_COG_VECTORS = 4  # Number of cognitive vector categories
# Cognitive vectors mapping
cog_vectors = {'BP': 0, 'BS': 1, 'CP': 2, 'CS': 3}

# Load characters from CSV
characters = load_character_aliases('../data_collection/characters_list.csv')
print(f"Loaded characters: {characters}")

# Extract the primary names
primary_names = [aliases[0] for aliases in characters if aliases]
print("Primary names list:", primary_names)

# Hierarchy and Temporality Levels
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
    if np.isscalar(predicted_labels_BC) and np.isscalar(predicted_labels_PS):
        return [bc_ps_to_state.get((predicted_labels_BC, predicted_labels_PS), 'Unknown')]

    for bc_label, ps_label in zip(predicted_labels_BC, predicted_labels_PS):
        combined_state = bc_ps_to_state.get((bc_label, ps_label), 'Unknown')
        combined_states.append(combined_state)
    return combined_states

def get_subject_from_event(event_tup, characters):
    subject = event_tup[0]
    # Remove 'Subject:' prefix if present, and strip whitespace
    subject = re.sub(r'^Subject:\s*', '', subject).strip()
    # Remove 'The', 'A', 'An' from the beginning
    subject = re.sub(r'^(The|A|An)\s+', '', subject, flags=re.I)
    normalized_subject = re.sub(r'[^\w\s]', '', subject.lower()).strip()

    for aliases in characters:
        for alias in aliases:
            normalized_alias = re.sub(r'[^\w\s]', '', alias.lower()).strip()
            if normalized_alias in normalized_subject or normalized_subject in normalized_alias:
                return aliases[0]
    logger.warning(f"No match found for subject '{subject}' in event: {event_tup}")
    return None

def initialize_transition_matrices(characters, hierarchy_levels, temporality_levels, num_vectors):
    matrices = {}
    for character in primary_names:
        character_matrices = {}
        for hierarchy in hierarchy_levels:
            hierarchy_matrices = {}
            for temporality in temporality_levels:
                hierarchy_matrices[temporality] = np.zeros((num_vectors, num_vectors))
            character_matrices[hierarchy] = hierarchy_matrices
        matrices[character] = character_matrices
    return matrices

def normalize_matrices(matrices):
    for character_matrices in matrices.values():
        for hierarchy_matrices in character_matrices.values():
            for temporality_matrix in hierarchy_matrices.values():
                row_sums = temporality_matrix.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1
                temporality_matrix[:] = temporality_matrix / row_sums
                # Ensure no negative values
                temporality_matrix[temporality_matrix < 0] = 0

def save_transition_matrices(matrices, filepath):
    with open(filepath, 'wb') as f:
        np.save(f, matrices, allow_pickle=True)

def load_transition_matrices(filepath):
    with open(filepath, 'rb') as f:
        return np.load(f, allow_pickle=True).item()

def update_transition_matrix(subject, hierarchy, temporality, prev_state, next_state, transition_matrices, cog_vectors):
    prev_vector = cog_vectors.get(prev_state)
    next_vector = cog_vectors.get(next_state)

    if prev_vector is not None and next_vector is not None:
        transition_matrices[subject][hierarchy][temporality][prev_vector, next_vector] += 1

def find_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.real(eigenvectors[:, idx]).flatten()
    stationary = stationary / np.sum(stationary)
    # Ensure non-negative
    stationary[stationary < 0] = 0
    stationary = stationary / np.sum(stationary)
    return stationary

def process_and_update_matrices(events_dict, transition_matrices):
    time_series_matrices = {char: [] for char in primary_names}
    time_stamps = {char: [] for char in primary_names}
    iter_time = {name: 0 for name in primary_names}

    all_events = []
    for (hierarchy, temporality), data in events_dict.items():
        sentences, tuples = data['sentences'], data['tuples']
        combined_states = data.get('combined_predicted_labels_sentences', [])
        for i in range(len(tuples)):
            all_events.append({
                'sentence': sentences[i],
                'tuple': tuples[i],
                'combined_state': combined_states[i],
                'hierarchy': hierarchy,
                'temporality': temporality,
                'index': i
            })

    # Optionally sort all_events if you have timestamps

    for i in range(len(all_events) - 1):
        current_event = all_events[i]
        next_event = all_events[i + 1]

        subject = get_subject_from_event(current_event['tuple'], characters=characters)
        if subject is not None:
            prev_state = current_event['combined_state']
            next_state = next_event['combined_state']
            hierarchy = current_event['hierarchy']
            temporality = current_event['temporality']
            update_transition_matrix(
                subject, hierarchy, temporality, prev_state, next_state, transition_matrices, cog_vectors
            )
            iter_time[subject] += 1
            time_stamps[subject].append(iter_time[subject])
            time_series_matrices[subject].append(copy.deepcopy(transition_matrices[subject]))
        else:
            pass

    return time_series_matrices, transition_matrices, time_stamps

# Load all events and their predictions
event_dict = load_and_classify_events()

# Initialize your transition matrices
transition_matrices = initialize_transition_matrices(primary_names, hierarchy_levels, temporality_levels, NUM_COG_VECTORS)

# Process events and update matrices
time_series_matrices, final_matrices, time_stamps = process_and_update_matrices(event_dict, transition_matrices)

# Save time series matrices and timestamps for each character
for character in primary_names:
    np.save(f'transition_matrices_{character}.npy', time_series_matrices[character])
    np.save(f'time_stamps_{character}.npy', time_stamps[character])  # Save timestamps

# Normalize and save matrices
normalize_matrices(transition_matrices)
save_transition_matrices(transition_matrices, 'transition_matrices.npy')

# Save final matrices for each character
for character in primary_names:
    final_matrix = transition_matrices[character]
    np.save(f'transition_matrices_final_{character}.npy', final_matrix)

# Find the stationary distribution
for char in primary_names:
    for hierarchy in hierarchy_levels: 
        for temporality in temporality_levels:
            transition_matrix = transition_matrices[char][hierarchy][temporality]
            stationary_distribution = find_stationary_distribution(transition_matrix)
            print(f"Stationary distribution for {char} in {hierarchy} {temporality}: {stationary_distribution}")

# Additional Analysis Methods

import scipy.linalg

def entropy_rate(transition_matrix):
    """
    Calculate the entropy rate of a Markov chain given its transition matrix.
    """
    stationary_dist = find_stationary_distribution(transition_matrix)
    entropy = 0.0
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            if transition_matrix[i][j] > 0 and stationary_dist[i] > 0:
                entropy -= stationary_dist[i] * transition_matrix[i][j] * np.log2(transition_matrix[i][j])
    return entropy

def mean_first_passage_time(P):
    """
    Compute the mean first passage times for an ergodic Markov chain.
    """
    n = P.shape[0]
    stationary = find_stationary_distribution(P)
    I = np.eye(n)
    Z = np.linalg.inv(I - P + np.outer(np.ones(n), stationary))
    mfpt = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if stationary[j] > 0:
                    mfpt[i, j] = (Z[j, j] - Z[i, j]) / stationary[j]
                else:
                    mfpt[i, j] = np.inf  # Set to infinity or handle as needed
    return mfpt

def absorption_probabilities(transition_matrix, absorbing_states):
    n = transition_matrix.shape[0]
    transient_states = [i for i in range(n) if i not in absorbing_states]

    if not transient_states:
        logger.warning("No transient states in the transition matrix.")
        return None

    Q = transition_matrix[np.ix_(transient_states, transient_states)]
    R = transition_matrix[np.ix_(transient_states, absorbing_states)]
    try:
        N = np.linalg.inv(np.eye(len(transient_states)) - Q)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix encountered while computing fundamental matrix N.")
        return None
    B = N @ R
    # Create full absorption probability matrix
    absorption_probs = np.zeros((n, len(absorbing_states)))
    absorption_probs[np.ix_(transient_states, range(len(absorbing_states)))] = B
    # Absorption probabilities for absorbing states are 1 for themselves
    for idx, state in enumerate(absorbing_states):
        absorption_probs[state, idx] = 1.0
    return absorption_probs

def simulate_behavior(start_state, transition_matrix, num_steps):
    """
    Simulate a sequence of states starting from start_state.
    """
    current_state = start_state
    states = [current_state]
    for _ in range(num_steps):
        current_state = np.random.choice(
            range(len(transition_matrix)),
            p=transition_matrix[current_state]
        )
        states.append(current_state)
    return states

# Example usage of additional methods

# Choose a character, hierarchy, and temporality for analysis
char = primary_names[0]  # e.g., 'alice'
hierarchy = 'context-specific'
temporality = 'medium-term'

# Get the transition matrix for the chosen character and levels
transition_matrix = transition_matrices[char][hierarchy][temporality]

# Ensure the matrix is normalized
normalize_matrices({char: {hierarchy: {temporality: transition_matrix}}})

# Calculate entropy rate
entropy = entropy_rate(transition_matrix)
print(f"Entropy rate for {char} in {hierarchy} {temporality}: {entropy}")

# Calculate mean first passage times
mfpt_matrix = mean_first_passage_time(transition_matrix)
print(f"Mean First Passage Times for {char} in {hierarchy} {temporality}:\n{mfpt_matrix}")

# Calculate absorption probabilities (assuming state 3 is absorbing)
absorbing_states = [3]  # Assuming 'CS' is an absorbing state
abs_prob = absorption_probabilities(transition_matrix, absorbing_states)
print(f"Absorption probabilities to state {absorbing_states} for {char}:\n{abs_prob}")

# Simulate behavior
start_state = 0  # Starting from 'BP'
num_steps = 10
simulated_states = simulate_behavior(start_state, transition_matrix, num_steps)
state_names = ['BP', 'BS', 'CP', 'CS']
simulated_state_names = [state_names[state] for state in simulated_states]
print(f"Simulated states for {char}: {simulated_state_names}")

# After processing events
for char in primary_names:
    for hierarchy in hierarchy_levels:
        for temporality in temporality_levels:
            tm = transition_matrices[char][hierarchy][temporality]
            if not np.any(tm):
                logger.warning(f"Transition matrix for {char} in {hierarchy} {temporality} is empty.")

# Updated analysis section
# Your analysis code
for char in primary_names:
    for hierarchy in hierarchy_levels:
        for temporality in temporality_levels:
            tm = transition_matrices[char][hierarchy][temporality]
            if np.any(tm):
                normalize_matrices({char: {hierarchy: {temporality: tm}}})
                stationary = find_stationary_distribution(tm)
                print(f"Stationary distribution for {char} in {hierarchy} {temporality}: {stationary}")
                entropy = entropy_rate(tm)
                print(f"Entropy rate for {char} in {hierarchy} {temporality}: {entropy}")
                mfpt = mean_first_passage_time(tm)
                print(f"Mean First Passage Times for {char} in {hierarchy} {temporality}:\n{mfpt}")

                # Additional Analysis Methods

                # 1. Communicating Classes and Classification of States
                G = nx.DiGraph(tm)
                components = list(nx.strongly_connected_components(G))
                print(f"Communicating classes for {char} in {hierarchy} {temporality}: {components}")

                # Classify states as recurrent or transient
                recurrent_states = []
                transient_states = []
                for component in components:
                    is_recurrent = True
                    for state in component:
                        if not np.isclose(tm[state, state], 1.0):
                            is_recurrent = False
                            break
                    if is_recurrent:
                        recurrent_states.extend(component)
                    else:
                        transient_states.extend(component)
                print(f"Recurrent states: {recurrent_states}")
                print(f"Transient states: {transient_states}")

                # 2. Spectral Analysis
                eigenvalues, eigenvectors = eig(tm.T)
                # Sort eigenvalues and eigenvectors
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                print(f"Eigenvalues for {char} in {hierarchy} {temporality}: {eigenvalues}")

                # Spectral Gap
                spectral_gap = abs(eigenvalues[0]) - abs(eigenvalues[1])
                print(f"Spectral gap for {char} in {hierarchy} {temporality}: {spectral_gap}")

                # 3. Mixing Time Estimation
                # Mixing time is the time until the distribution is close to stationary distribution
                # This is complex; here, we provide an approximate mixing time
                # For ergodic chains, mixing time is related to the second-largest eigenvalue
                if abs(eigenvalues[1]) < 1:
                    mixing_time = int(np.ceil(np.log(0.01) / np.log(abs(eigenvalues[1]))))
                    print(f"Approximate mixing time for {char} in {hierarchy} {temporality}: {mixing_time}")
                else:
                    print(f"Cannot compute mixing time for {char} in {hierarchy} {temporality}")

                # 4. Expected Number of Visits
                # For ergodic chains, the expected number of visits is infinite
                # Instead, compute expected visits in first n steps
                n_steps = 10
                expected_visits = np.zeros((len(states), len(states)))
                P_power = np.identity(len(states))
                for step in range(1, n_steps + 1):
                    P_power = P_power @ tm
                    expected_visits += P_power
                print(f"Expected number of visits in first {n_steps} steps for {char} in {hierarchy} {temporality}:\n{expected_visits}")

                # 5. First Return Times
                # For recurrent states, mean recurrence time is 1 / stationary_probability
                recurrence_times = {}
                for i, state in enumerate(states):
                    if stationary[i] > 0:
                        recurrence_times[state] = 1.0 / stationary[i]
                    else:
                        recurrence_times[state] = np.inf
                print(f"Mean recurrence times for {char} in {hierarchy} {temporality}:\n{recurrence_times}")

                # 6. Hitting Times
                # Compute expected hitting times using fundamental matrix for ergodic chains
                # For simplicity, use mean first passage times computed earlier
                print(f"Hitting times (mean first passage times) for {char} in {hierarchy} {temporality}:\n{mfpt}")

                # 7. Commute Times
                # Commute time between states i and j is mfpt[i, j] + mfpt[j, i]
                commute_times = mfpt + mfpt.T
                print(f"Commute times between states for {char} in {hierarchy} {temporality}:\n{commute_times}")

            else:
                print(f"No data for {char} in {hierarchy} {temporality}.")
# Updated t-test or Mann-Whitney U test
# Handle statistical tests appropriately
group1_features = []
group2_features = []

for idx, char in enumerate(primary_names):
    tm = transition_matrices[char][hierarchy][temporality]
    if np.any(tm):
        normalize_matrices({char: {hierarchy: {temporality: tm}}})
        entropy = entropy_rate(tm)
        if idx % 2 == 0:
            group1_features.append(entropy)
        else:
            group2_features.append(entropy)

if len(group1_features) > 1 and len(group2_features) > 1:
    t_stat, p_val = ttest_ind(group1_features, group2_features, equal_var=False)
    print(f"T-test results: T-statistic = {t_stat}, P-value = {p_val}")
elif len(group1_features) > 0 and len(group2_features) > 0:
    u_stat, p_val = mannwhitneyu(group1_features, group2_features, alternative='two-sided')
    print(f"Mann-Whitney U test results: U-statistic = {u_stat}, P-value = {p_val}")
else:
    print("Not enough samples to perform statistical test.")