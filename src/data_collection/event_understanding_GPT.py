from IPython.display import display, Image, Audio

import cv2
import base64
import time
import openai
from openai import OpenAI
import os
import requests
import ast
import re

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
openai.api_key = api_key
client = OpenAI()

# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
def extract_frames(video_path, interval=1):
    """
    Extract frames from the video at the specified interval.
    """
    video = cv2.VideoCapture(video_path)
    print("attempted extracted video")
    frames = []
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        frame_id = video.get(1)  # Current frame number
        success, frame = video.read()
        if not success:
            break
        if frame_id % (frame_rate * interval) == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return frames

# using this for the direct LLM inference testing for each event without the need for prototype classes
def analyze_frames_with_gpt4_direct_classification(frames, client):
    """
    Send frames to GPT-4 for direct classification of behavioral vectors
    at each hierarchical level and temporal category.
    """
    all_frame_classifications = []

    for frame in frames:
        # Creating a detailed prompt asking for classifications across all levels and categories
        prompt_message = """
        Analyze this scene and classify the observed behaviors according to their hierarchical levels 
        (High, Medium, Low) and temporal categories (1, 2, 3). Provide the classification in the format of 
        a 3x3 matrix, where each cell contains the behavioral vector appropriate for its hierarchical level 
        and temporal category. Use the following structure for your response:

        High: 
          - Temporal Category 1: [Vector]
          - Temporal Category 2: [Vector]
          - Temporal Category 3: [Vector]
        Medium: 
          - Temporal Category 1: [Vector]
          - Temporal Category 2: [Vector]
          - Temporal Category 3: [Vector]
        Low: 
          - Temporal Category 1: [Vector]
          - Temporal Category 2: [Vector]
          - Temporal Category 3: [Vector]

        Please fill in [Vector] with the appropriate behavioral vector classification.
        """

        # Note: Adjust the prompt as necessary to fit your specific vector categories and descriptors
        
        # Construct the prompt for GPT-4, including the frame image
        messages = [
            {"role": "user", "content": prompt_message, "image": frame, "resize": 768}
        ]

        params = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": 1024,  # Adjust as needed based on expected response length
        }

        # Make the API call for direct classification
        result = client.chat.completions.create(**params)
        classification = result.choices[0].message.content
        
        # Store the classification for this frame
        all_frame_classifications.append(classification)
    
    return all_frame_classifications


def analyze_frames_with_gpt4(frames, client):
    """
    Send frames to GPT-4 for analysis and return the descriptions.
    """

    # "Describe these video frames in terms of subject, action, and objects involved and format it like so (Subject: [subject], Action [action], Object [objects])."
    # "add option to add hierachical levels and/or temporal dynamics and add that to a richer event representation"
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "Please describe these video frames in terms of subject, action, objects involved, and environment. Format each description as a dictionary in a Python list. Here is an example of the format I am looking for:\n\n"
                "[\n"
                "    {'subject': 'cat', 'action': 'sitting', 'objects': ['mat'], 'environment': 'living room'},\n"
                "    {'subject': 'dog', 'action': 'barking', 'objects': ['mailman'], 'environment': 'front yard'}\n"
                "]\n\n"
                "Now, please format the descriptions of the video frames in the same way:"
            ] + list(map(lambda x: {"image": x, "resize": 768}, frames))
        }
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }

    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


# analyze frames with GPT using hierarchical and temporal categories

def analyze_frames_with_gpt4_leveled(frames, client, hierarchical_levels, temporal_categories):
    """
    Send frames to GPT-4 for analysis across hierarchical levels and temporal categories.
    """
    all_frame_descriptions = []

    for frame in frames:
        frame_description = {}
        for level in hierarchical_levels:
            for category in temporal_categories:
                prompt_message = f"Given this scene, identify {level} level behaviors observable within temporal category {category}."
                
                # Construct the prompt for GPT-4
                messages = [
                    {"role": "user", "content": prompt_message, "image": frame["image"], "resize": 768}
                ]

                params = {
                    "model": "gpt-4-vision-preview",
                    "messages": messages,
                    "max_tokens": 500,
                }

                # Make the API call for each hierarchical level and temporal category
                result = client.chat.completions.create(**params)
                description = result.choices[0].message.content
                
                # Append the description to the frame description under the specific level and category
                frame_description[f"{level}_{category}"] = description
        
        all_frame_descriptions.append(frame_description)
    
    return all_frame_descriptions


def organize_event_representations(frame_descriptions, hierarchical_levels, temporal_categories):
    """
    Organize the descriptions into a structured 3x3 matrix format.
    """
    organized_descriptions = []

    for frame_description in frame_descriptions:
        matrix = []
        for level in hierarchical_levels:
            level_descriptions = []
            for category in temporal_categories:
                key = f"{level}_{category}"
                level_descriptions.append(frame_description.get(key, ""))
            matrix.append(level_descriptions)
        organized_descriptions.append(matrix)
    
    return organized_descriptions


def gpt_parse_events_batch(descriptions, batch_size):
    """
    Use GPT in a chat-style interaction to parse descriptions into structured data.
    Processes the descriptions in batches to adhere to API limits.
    """
    all_structured_events = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        structured_events = gpt_parse_events(batch)
        all_structured_events.extend(structured_events)

    return all_structured_events



def gpt_parse_events(descriptions):
    """
    Use GPT to parse a batch of descriptions into structured (subject, action, object(s)) tuples.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please reformat each description into structured data with subject, action, and objects."}
    ]
    
    for desc in descriptions:
        messages.append({"role": "user", "content": desc})
    
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0  # Adjust if needed
    )

    # Extract the response content
    response_message = response.choices[0].message.content if response.choices else ""
    return process_gpt_responses(response_message)



def extract_list_from_string(description):
    """
    Extracts all complete dictionaries from a string that resembles a Python list of dictionaries.
    Handles incomplete final dictionary entries.
    """
    try:
        # Adjusted regex pattern to match complete dictionary entries
        pattern = r"\{'[^']+'[^{}]+\}"
        
        # Find all complete dictionaries using regex
        matches = re.findall(pattern, description, re.DOTALL)
        if matches:
            # Construct the list string with the matched dictionaries
            dict_list = [ast.literal_eval(match) for match in matches]
            return dict_list
        else:
            print("No complete dictionaries found in the string.")
    except (ValueError, SyntaxError) as e:
        print(f"Error extracting or evaluating the list: {e}")

    return []

def process_gpt_responses(response_text):
    """
    Process the GPT responses to extract (subject, action, object(s)) tuples.
    """
    events = []
    print("expected response", response_text)
    # Implement parsing logic based on the expected response format
    # Placeholder for parsing logic
    return events

def process_video_frames(video_frames):
    """
    Process the structured data from video frames into a list of events.
    """
    events = []
    video_frames = extract_list_from_string(video_frames)
    print("video frames after extraction", video_frames)
    for frame in video_frames:
        # Directly extract data from the dictionary
        subject = frame.get("subject", "")
        action = frame.get("action", "")
        objects = frame.get("objects", [])
        environment = frame.get("environment", "")

        event = {
            "subject": subject,
            "action": action,
            "objects": objects,
            "environment": environment
        }
        events.append(event)

    return events


def append_timestamps(events, interval):
    """
    Append timestamps to each event.
    """
    timestamped_events = []
    for i, event in enumerate(events):
        timestamp = i * interval  # Assuming interval is in seconds
        timestamped_events.append((timestamp, event))
    return timestamped_events

def main():
    
    video_path = '../../data/bison.mp4'
    print("file exists?", os.path.exists(video_path))
    interval = 1  # Interval in seconds for frame extraction

    frames = extract_frames(video_path, interval)
    print("got frames")
    descriptions = analyze_frames_with_gpt4(frames, client)
    print("descriptions", descriptions)
    print("description type", type(descriptions))
    processed_events = process_video_frames(descriptions)
    print("processed_events", processed_events)
    timestamped_events = append_timestamps(processed_events, interval)
    print("timestamped_events", timestamped_events)
   

if __name__ == "__main__":
    main()
