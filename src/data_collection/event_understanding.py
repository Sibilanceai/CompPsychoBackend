import langchain
from typing import Sequence
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import transformers
import openai
import os
import csv
import json
import re
import numpy as np
print(transformers.__version__)

def initialize_openai_model():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # openai_api_key = ""
    return OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)


class EventRepresentation:
    def __init__(self, subject, action, obj, environment, temporal_category, hierarchical_level):
        self.subject = subject
        self.action = action
        self.obj = obj
        self.environment = environment
        self.temporal_category = temporal_category
        self.hierarchical_level = hierarchical_level

    def to_dict(self):
        return {
            "subject": self.subject,
            "action": self.action,
            "object": self.obj,
            "environment": self.environment,
            "temporal_category": self.temporal_category,
            "hierarchical_level": self.hierarchical_level,
        }


def extract_event_representation(text_chunk, temporal_category, hiearchical_level):
    '''
    prompt = f"""
    Given the narrative text: "{text_chunk}"
    Return a JSON string that identifies the main event, including Subject, Action, Object, and Environment. 
    Also classify the Action into its temporal category (Short-term, Medium-term, Long-term) 
    and hierarchical level (High-level, Context-specific, Task-specific).
    """
    '''

    model = initialize_openai_model()
    temporal_category = str(temporal_category).lower()
    hiearchical_level = str(hiearchical_level).lower()

    # NOTE: human message prompts are the same for each temporal category
    if temporal_category == "medium" and hiearchical_level == "high":
        # Define the narrative text as a human message
        # Medium Term, High Level
        human_message_med_High = HumanMessagePromptTemplate.from_template(f"Within the narrative text: \"{text_chunk}\", identify and describe the sequence of events, focusing on actions that align with the Extended Dynamic Cognitive Vector Theory (EDCVT). EDCVT posits that actions and behaviors can be classified into hierarchical levels: High-level (overarching themes or goals), Context-specific (influenced by situational factors), and Task-specific (directed toward immediate tasks). Additionally, actions are influenced by temporal dynamics: Short-term (immediate decisions), Medium-term (days to months), and Long-term (lifelong trends). For this medium-term (days to months) temporal category and High-level (overarching themes or goals) focused analysis, identify the Subject, Action, Object, and Environment for key actions and that can be classified as High-level, indicative of overarching themes driving the narrative.")
        
        # Define the system's task in response to the human message
        system_message_med_High = SystemMessagePromptTemplate.from_template(f"Using the narrative text: \"{text_chunk}\", extract and construct a sequence of event descriptions based on the Extended Dynamic Cognitive Vector Theory (EDCVT), which categorizes cognitive behaviors into hierarchical levels and temporal dynamics. In this task, focus on High-level vectors within a Medium-term temporal category. High-level vectors represent overarching themes or goals that guide behavior over days to months. Format each event as a meaningful sentence, include by each sentence event description a tuple of the (Subject, Action, Objects, Environment) of the event description, emphasizing the influence of overarching goals or themes on the actions within the medium-term scope of the narrative.")

        # Combine into a chat prompt template if needed
        chat_prompt_med_High = ChatPromptTemplate(messages=[human_message_med_High, system_message_med_High])
        llmchain = LLMChain(llm=model, prompt=chat_prompt_med_High)
        response = llmchain.invoke({}, Temperature=1)
        events, eventtups = parse_response(response["text"])
        return response, events, eventtups
    
    elif temporal_category == "medium" and hiearchical_level == "context":
        # Define the narrative text as a human message
        # Medium Term, High Level
        human_message_med_Context = HumanMessagePromptTemplate.from_template(f"Within the narrative text: \"{text_chunk}\", identify and describe the sequence of events, focusing on actions that align with the Extended Dynamic Cognitive Vector Theory (EDCVT). EDCVT posits that actions and behaviors can be classified into hierarchical levels: High-level (overarching themes or goals), Context-specific (influenced by situational factors), and Task-specific (directed toward immediate tasks). Additionally, actions are influenced by temporal dynamics: Short-term (immediate decisions), Medium-term (days to months), and Long-term (lifelong trends). For this medium-term (days to months) and Context-specific (influenced by situational factors) focused analysis, identify the Subject, Action, Object, and Environment for key actions and that can be classified as High-level, indicative of overarching themes driving the narrative.")

        # Define the system's task in response to the human message
        system_message_med_Context = SystemMessagePromptTemplate.from_template(f"Analyze \"{text_chunk}\" to identify and sequence events according to the Extended Dynamic Cognitive Vector Theory (EDCVT), focusing specifically on Context-specific vectors within the Medium-term temporal category. Context-specific vectors account for behaviors influenced by situational factors over days to months. Format each event as a meaningful sentence, include by each sentence event description a tuple of the (Subject, Action, Objects, Environment) of the event description, highlighting how situational factors shape the narrative's progression. Ensure each event captures the essence of Context-specific actions within the medium-term framework of the story.")

        # Combine into a chat prompt template if needed
        chat_prompt_med_Context = ChatPromptTemplate(messages=[human_message_med_Context, system_message_med_Context])
        llmchain = LLMChain(llm=model, prompt=chat_prompt_med_Context)
        response = llmchain.invoke({}, Temperature=1)
        events, eventtups = parse_response(response["text"])
        return response, events, eventtups
    
    elif temporal_category == "medium" and hiearchical_level == "task":
        # Define the narrative text as a human message
        # Medium Term, Task Level
        human_message_med_Task = HumanMessagePromptTemplate.from_template(f"Within the narrative text: \"{text_chunk}\", identify and describe the sequence of events, focusing on actions that align with the Extended Dynamic Cognitive Vector Theory (EDCVT). EDCVT posits that actions and behaviors can be classified into hierarchical levels: High-level (overarching themes or goals), Context-specific (influenced by situational factors), and Task-specific (directed toward immediate tasks). Additionally, actions are influenced by temporal dynamics: Short-term (immediate decisions), Medium-term (days to months), and Long-term (lifelong trends). For this medium-term (days to months) and Task-specific (directed toward immediate tasks) focused analysis, identify the Subject, Action, Object, and Environment for key actions and that can be classified as High-level, indicative of overarching themes driving the narrative.")

        # Define the system's task in response to the human message
        system_message_med_Task = SystemMessagePromptTemplate.from_template(f"Examine the narrative text: \"{text_chunk}\", and generate a sequence of event descriptions following the Extended Dynamic Cognitive Vector Theory (EDCVT), concentrating on Task-specific vectors within a Medium-term temporal framework. Task-specific vectors detail behaviors aimed at immediate tasks or challenges occurring over days to months. Format each event as a meaningful sentence, include by each sentence event description a tuple of the (Subject, Action, Objects, Environment) of the event description, showcasing how specific tasks or challenges are addressed and influenced by the narrative’s medium-term dynamics. The focus should be on the immediate objectives guiding the characters’ actions.")

        # Combine into a chat prompt template if needed
        chat_prompt_med_Context = ChatPromptTemplate(messages=[human_message_med_Task, system_message_med_Task])
        llmchain = LLMChain(llm=model, prompt=chat_prompt_med_Context)
        response = llmchain.invoke({}, Temperature=1)
        events, eventtups = parse_response(response["text"])
        return response, events, eventtups
    
    # TODO do the other temporal categories and hiearchical levels

    print("ERROR: Invalid temporal category and or hiearchical level")
    return None



# extracts events with numerical relative positional value within a chunk that may be useful
def parse_response(single_chunk_text):
    # Split the chunk by new lines and filter out empty lines
    lines = [line.strip() for line in single_chunk_text.split('\n') if line.strip()]

    event_sentences = []
    event_tuples = []

    # Regular expression to match the tuple format
    tuple_regex = r"\(([^)]+)\)"
    
    for line in lines:
        # Check if the line contains an event tuple and description
        match = re.search(tuple_regex, line)
        if match:
            # Extract the tuple
            event_tuple = tuple(match.group(1).split(', '))
            event_tuples.append(event_tuple)

            # Extract the description, which follows the tuple
            description = re.sub(tuple_regex, '', line).strip()
            event_sentences.append(description)

    return event_sentences, event_tuples

# extracts events without numerical prefix that may just add noise
#def parse_response(single_chunk_text):
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

def text_into_chunks(story): 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(type(embeddings))
    # Create a Semantic Chunker.
    text_splitter = SemanticChunker(embeddings)
    print(type(text_splitter))
    # Split the text into chunks.
    chunks = text_splitter.split_text(story) 
    print("chunk length ", len(chunks))
    return chunks

def refine_chunks(chunks):
    refined_chunks = []
    for chunk in chunks:
        # Example criterion: split further based on sentence count if too long
        sentences = chunk.split('.')
        if len(sentences) > 10:  # Arbitrary threshold for demonstration
            midpoint = len(sentences) // 2
            refined_chunks.append('.'.join(sentences[:midpoint]))
            refined_chunks.append('.'.join(sentences[midpoint:]))
        else:
            refined_chunks.append(chunk)
    return refined_chunks



"""
Step 1: Enhance Narrative Chunk Embedding Generation
First, ensure each narrative chunk is not only split but also embedded using diverse embedding strategies (semantic, sentiment, thematic). 
You've started with HuggingFaceEmbeddings for semantic chunking; next, generate embeddings that capture different narrative aspects.
"""

# Assuming additional embedding functions are defined elsewhere
def generate_additional_embeddings(chunk):
    # Placeholder for generating sentiment, thematic, etc., embeddings
    sentiment_embedding = generate_sentiment_embedding(chunk)
    thematic_embedding = generate_thematic_embedding(chunk)
    return sentiment_embedding, thematic_embedding

# Assuming an external function or system for sentiment and thematic embedding generation
def generate_sentiment_embedding(chunk):
    # Placeholder for generating sentiment embeddings
    return np.random.rand(768)  # Example: Random vector as a placeholder

def generate_thematic_embedding(chunk):
    # Placeholder for generating thematic embeddings
    return np.random.rand(768)  # Example: Random vector as a placeholder

"""
Step 2: Store Embeddings in a Vector Store
Before storing, initialize a vector database or choose an appropriate vector store (e.g., FAISS, Weaviate). 
Then, for each chunk and its embeddings, store these in the database with associated metadata.
"""


# Assuming an external system or database for storing and retrieving embeddings
# This could be implemented with FAISS, Elasticsearch, or similar
def store_embeddings_in_vector_store(chunk_id, embeddings):
    # Store embeddings in a vector store with a unique identifier for the chunk
    pass



"""
Step 3: Implement Context Retrieval Mechanism
Develop a retrieval mechanism that fetches relevant context embeddings from the vector store based on the current narrative chunk. 
This involves querying the vector store to find embeddings that are similar or contextually relevant.
"""

# Placeholder for context retrieval function
def retrieve_contextual_embeddings(chunk_id):
    # Retrieve relevant embeddings based on the current chunk's id or content
    # This might involve querying the vector store for similar embeddings
    return {"semantic": np.random.rand(768), "sentiment": np.random.rand(768), "thematic": np.random.rand(768)}

def extract_event_representation_with_context(text_chunk, context):
    # Modified function to include context in the prompt for the LLM
    # The context can be summarized or transformed as needed to fit the LLM's input format
    prompt = f"""
    With the given context: {context},
    and the narrative text: \"{text_chunk}\",
    Identify the main event, including Subject, Action, Object, and Environment...
    """
    # Call to LLM with the modified prompt
    # TODO finish, use without context implementation and add retrieved context
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200, temperature=0.5)
    return parse_response(response.choices[0].text)

"""
Step 4: Feed Context to LLM for Event Extraction
Modify extract_event_representation to incorporate the context retrieved from the vector store, 
possibly by summarizing it or directly appending it to the LLM's prompt.
"""


"""
Step 5: Proxy Model for Event Sequence Prediction
Implement a proxy model (e.g., a fine-tuned transformer) that predicts future event sequences based on past sequences, 
used to evaluate the effectiveness of the retrieval mechanism.
"""

def train_proxy_model(event_sequences):
    # Placeholder for training logic of the proxy model
    pass

def evaluate_proxy_model(performance_metric):
    # Use the proxy model's performance as feedback to adjust the retrieval mechanism
    pass


"""
Step 6: Optimization and Feedback Loop
Based on the proxy model's performance (e.g., perplexity in predicting event sequences), 
adjust the retrieval mechanism to optimize context provision for event extraction.
"""
def optimize_retrieval_mechanism(feedback):
    # Adjust retrieval criteria based on feedback from the proxy model's performance
    pass

"""
Integration and Continuous Improvement
Integrate these components into a cohesive system that iterates over narrative texts, 
continuously refining the retrieval mechanism based on feedback from the proxy model's performance. 
Ensure modularity in your design to facilitate independent updates and improvements to each component.
"""


def main_loop(narrative_texts):
    for text in narrative_texts:
        chunks = text_into_chunks(text)
        for chunk in chunks:
            # Example: Generate and store multiple types of embeddings for each chunk
            semantic_embedding = generate_additional_embeddings(chunk)  # Existing function
            sentiment_embedding = generate_sentiment_embedding(chunk)
            thematic_embedding = generate_thematic_embedding(chunk)
            embeddings = {
                "semantic": semantic_embedding,
                "sentiment": sentiment_embedding,
                "thematic": thematic_embedding
            }
            # Placeholder: Store embeddings with a unique identifier for the chunk
            chunk_id = hash(chunk)  # Simplistic approach for generating a unique ID
            store_embeddings_in_vector_store(chunk_id, embeddings)
            
            # Retrieve context based on current chunk
            context = retrieve_contextual_embeddings(chunk_id)
            
            # Extract event representations with the provided context
            event_representation = extract_event_representation_with_context(chunk, context)
            print(event_representation.to_dict())  # Assuming the event_representation is processed accordingly

            # Additional steps would involve training the proxy model, evaluating it, and optimizing the retrieval mechanism based on feedback
            # These steps require further implementation details and are omitted for brevity

# Placeholder function for text_into_chunks is assumed to be defined as per the initial code snippet provided

# This code outline integrates the initial processing steps with enhanced context retrieval and usage in event representation extraction. 
# Further development is required for the optimization loop and proxy model training, which depend on specific project requirements and data.


# testing

story = """In the heart of a sprawling metropolis, beneath the shadow of skyscrapers that pierced the sky like needles, there existed a quaint little bookstore named "Whispers of the Past." Its wooden door, etched with the tales of a thousand years, opened not just to shelves laden with books but to a portal between worlds. The owner, Eleanor, a woman with eyes as deep as the stories she cherished, believed each book was a living soul, waiting to share its secrets with those who dared to listen.

One rainy evening, as the city sighed under a blanket of silver droplets, a young boy, lost and drenched, stumbled upon the bookstore. His eyes, wide with wonder, reflected the myriad of worlds hidden within the pages of the books he beheld. Eleanor watched him from behind the counter, a knowing smile curling at the edges of her lips. She handed him a peculiar book, its cover worn, its pages yellowed with time. "This," she whispered, "will show you the way."

The boy, driven by an inexplicable pull, opened the book to find not words, but a whirlwind of colors that lifted him off his feet, spiriting him away to an adventure beyond his wildest dreams. When he finally returned, the storm had ceased, and the first light of dawn was painting the sky in hues of gold and pink. He found himself back in the bookstore, the book clutched tightly in his hands, a newfound gleam in his eyes. Eleanor's laughter filled the room, as magical and warm as the stories that danced in the air between them. In "Whispers of the Past," the boy had found a haven, a reminder that even in the largest of cities, magic could be found in the smallest of places."""

chunks = text_into_chunks(story=story)
print(chunks)

def save_events_to_csv(event_sentences, event_tuples, filename_prefix):
    # Save event sentences
    with open(f'{filename_prefix}_sentences.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for sentence in event_sentences:
            writer.writerow([sentence])

    # Save event tuples
    with open(f'{filename_prefix}_tuples.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for event_tuple in event_tuples:
            writer.writerow(event_tuple)

eventsentlist_med_context = []
eventsentlist_med_high = []
eventenclist_med_context = []
eventenclist_med_high = []
for chunk in chunks:
    eventsbundle, events, eventtups = extract_event_representation(chunk, "medium", "context")
    eventsentlist_med_context.extend(events)
    eventenclist_med_context.extend(eventtups)
    print(len(events), events)
    print(eventtups)

# Save medium-context events and encodings to CSV
save_events_to_csv(eventsentlist_med_context, eventenclist_med_context, "med_context")
    
for chunk in chunks:
    eventsbundle, events, eventtups = extract_event_representation(chunk, "medium", "high")
    eventsentlist_med_high.extend(events)
    eventenclist_med_high.extend(eventtups)
    print(len(events), events)
    print(eventtups)

# Save medium-high events and encodings to CSV
save_events_to_csv(eventsentlist_med_high, eventenclist_med_high, "med_high")