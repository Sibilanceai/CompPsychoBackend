import langchain
from typing import Sequence
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import transformers
print(transformers.__version__)

def text_into_chunks(story): 
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Create a Semantic Chunker.
    text_splitter = SemanticChunker(embeddings)

    # Split the text into chunks.
    chunks = text_splitter.split_text(story) 