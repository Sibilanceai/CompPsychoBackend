# import re
# import ast

# text = """```python
# [
#     {'subject': 'group of wolves', 'action': 'surrounding', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'crowding around', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'following', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'attacking', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'bison', 'action': 'running', 'objects': ['wolves'], 'environment': 'snowy field'},
#     {'subject': 'bison', 'action': 'running', 'objects': {'other bison', 'wolves'}, 'environment': 'snowy field'},
#     {'subject': 'bison', 'action': 'charging', 'objects': ['wolves'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'chasing', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'pursuing', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'following', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'hunting', 'objects': ['bison'], 'environment': 'snowy field'},
#     {'subject': 'bison', 'action': 'escaping', 'objects': ['wolves'], 'environment': 'snowy field'},
#     {'subject': 'bison', 'action': 'stumbling', 'objects': ['wolves'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'attacking', 'objects': ['fallen bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'taking down', 'objects': ['fallen bison'], 'environment': 'snowy field'},
#     {'subject': 'wolves', 'action': 'biting', 'objects': ['fallen bison'], 'environment': 'snow"""

# # Regular expression pattern to match dictionary-like structures
# #pattern = r"\{[^{}]*\}"
# # Updated regular expression pattern to match more specific dictionary-like structures
# # Regular expression pattern to match more specific dictionary-like structures
# pattern = r"\{'[^']+'[^{}]+\}"

# # Find all matches
# matches = re.findall(pattern, text)

# # Safely evaluate each match as a Python dictionary
# dict_list = [ast.literal_eval(match) for match in matches]

# print(dict_list)

import re

# def parse_response(single_chunk_text):
#     # Split the chunk by new lines and filter out empty lines
#     lines = [line.strip() for line in single_chunk_text.split('\n') if line.strip()]

#     event_sentences = []
#     event_tuples = []

#     # Regular expression to match the tuple format
#     tuple_regex = r"\(([^)]+)\)"
    
#     for line in lines:
#         # Check if the line contains an event tuple and description
#         match = re.search(tuple_regex, line)
#         if match:
#             # Extract the tuple
#             event_tuple = tuple(match.group(1).split(', '))
#             event_tuples.append(event_tuple)

#             # Extract the description, which follows the tuple
#             description = re.sub(tuple_regex, '', line).strip()
#             event_sentences.append(description)

#     return event_sentences, event_tuples

# def parse_response(single_chunk_text):
#     lines = [line.strip() for line in single_chunk_text.split('\n') if line.strip()]
#     event_sentences = []
#     event_tuples = []

#     # Adjusted regular expression to match the tuple format and ignore leading numbers
#     tuple_regex = r"\(([^)]+)\)"
#     prefix_regex = r"^\d+\.\s*(.*)"

#     for line in lines:
#         # Check if the line contains an event tuple and description
#         match = re.search(tuple_regex, line)
#         if match:
#             # Extract the tuple
#             event_tuple = tuple(match.group(1).split(', '))
#             event_tuples.append(event_tuple)

#             # Remove the tuple part to better isolate the description
#             description_only = re.sub(tuple_regex, '', line).strip()
#             # Now remove any leading numerical prefix from the description
#             description_match = re.match(prefix_regex, description_only)
#             if description_match:
#                 # This removes the "1. " like prefixes from descriptions
#                 clean_description = description_match.group(1)
#                 event_sentences.append(clean_description)

#     return event_sentences, event_tuples

# def parse_response(single_chunk_text):
#     # Split the chunk into lines, filtering out empty lines
#     lines = [line.strip() for line in single_chunk_text.split('\n') if line.strip()]
#     event_sentences = []
#     event_tuples = []

#     # Loop through each line in the provided text
#     for line in lines:
#         # First, separate the descriptive part from the tuple using regex
#         match = re.match(r"^\d+\.\s*\((.*?)\)\s*(.*)", line)
#         if match:
#             # Extract and clean the tuple part
#             tuple_str = match.group(1)
#             event_tuple = tuple(map(str.strip, tuple_str.split(',')))
#             event_tuples.append(event_tuple)
            
#             # Extract and store the description part
#             description = match.group(2).strip()
#             event_sentences.append(description)

#     return event_sentences, event_tuples

# def parse_response(single_chunk_text):
#     lines = [line.strip() for line in single_chunk_text.split('\n') if line.strip()]
#     event_sentences = []
#     event_tuples = []

#     # Regular expression to match and remove numerical prefixes from event descriptions
#     prefix_regex = re.compile(r'^\d+\.\s*')
    
#     for line in lines:
#         # Identify the tuple within parentheses and separate it from the rest of the text
#         match = re.search(r"\(([^)]+)\)", line)
#         if match:
#             event_tuple = tuple(match.group(1).split(', '))
#             event_tuples.append(event_tuple)

#             # Remove the tuple to isolate the description, then strip the numerical prefix
#             description_with_prefix = re.sub(r"\(([^)]+)\)", "", line).strip()
#             clean_description = prefix_regex.sub("", description_with_prefix)
#             event_sentences.append(clean_description)

#     return event_sentences, event_tuples


def parse_response(single_chunk_text):
    # Initialize lists for storing event sentences and tuples
    event_sentences = []
    event_tuples = []

    # Debug: Print the received input to ensure it's as expected
    print("Received text:", single_chunk_text[:100])  # Print first 100 chars for a quick check

    # Split the input text into lines
    lines = single_chunk_text.strip().split('\n')

    # Pattern to identify and remove the numerical prefix from sentences
    numerical_prefix_pattern = re.compile(r'^Event\s+\d+:\s+')

    for line in lines:
        # Debug: Print each line to see if splitting works as expected
        print("Processing line:", line)

        # Attempt to find the tuple within parentheses
        tuple_match = re.search(r'\((.*?)\)', line)
        if tuple_match:
            # Extract and process the tuple string
            tuple_str = tuple_match.group(1)
            event_tuple = tuple(tuple_str.split('; '))
            event_tuples.append(event_tuple)

            # Remove the tuple from the line to isolate the description
            description_with_numerical_prefix = re.sub(r'\(.*?\)', '', line).strip()

            # Remove the numerical prefix from the description
            description = numerical_prefix_pattern.sub('', description_with_numerical_prefix).strip()
            event_sentences.append(description)

    return event_sentences, event_tuples
# Example usage with one of the provided chunks
single_chunk_text = '\n\nEvent 1: Eleanor\'s laughter filled the room, creating a warm and magical atmosphere that captivated the boy. (Subject: Eleanor; Action: Filled; Object: Room; Environment: Magical and warm atmosphere)\n\nEvent 2: The stories they shared danced in the air, conveying a sense of wonder and enchantment. (Subject: Eleanor and the boy; Action: Shared; Object: Stories; Environment: Air)\n\nEvent 3: The boy found solace and comfort in the haven of "Whispers of the Past", a reminder that magic can be found in even the smallest of places. (Subject: The boy; Action: Found; Object: Solace and comfort; Environment: "Whispers of the Past")\n\nEvent 4: The boy\'s newfound understanding of magic and its presence in everyday life aligned with his overarching goal of seeking wonder and enchantment. (Subject: The boy; Action: Aligned; Object: Understanding of magic; Environment: Everyday life)'

event_sentences, event_tuples = parse_response(single_chunk_text)

# Printing the results for verification
print("Event Sentences:")
for sentence in event_sentences:
    print("-", sentence)
print("\nEvent Tuples:")
for event_tuple in event_tuples:
    print("-", event_tuple)
