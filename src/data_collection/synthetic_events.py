import pandas as pd
import numpy as np
import random
import networkx as nx

# Define characters and possible states
characters = ['Jack', 'Kate', 'John', 'Sawyer']
states = ['BP', 'BS', 'CP', 'CS']

# Initialize transition matrices for each character
transition_matrices = {
    char: np.zeros((len(states), len(states))) for char in characters
}

# Initialize a list to store event tuples
events = []


def simulate_event(i, characters, states, transition_matrices, events):
    # Randomly select subject and object characters
    subject = random.choice(characters)
    object = random.choice([char for char in characters if char != subject])  # Ensure different characters
    
    # Randomly select an action and environment
    action = random.choice(['laughs with', 'reminds', 'finds', 'realizes'])
    environment = random.choice(['in the room', 'in "Whispers of the Past"', 'in the city'])
    
    # Simulate a transition for the subject based on the interaction
    prev_state = random.choice(states)
    next_state = random.choice(states)
    transition_matrices[subject][states.index(prev_state)][states.index(next_state)] += 1
    
    # Create event tuple and append to events list
    event = (i, subject, action, object, environment)
    events.append(event)
    
    return transition_matrices, events

# Simulate a series of events
num_events = 100
for i in range(num_events):
    transition_matrices, events = simulate_event(i, characters, states, transition_matrices, events)



# Convert events to DataFrame and save to CSV
events_df = pd.DataFrame(events, columns=['Timestamp', 'Subject', 'Action', 'Object', 'Environment'])
events_df.to_csv('simulated_events.csv', index=False)

# Save transition matrices (example for one character)
for char in characters:
    np.savetxt(f"{char}_transition_matrix.csv", transition_matrices[char], delimiter=",")
