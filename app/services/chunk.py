import re
import json
from typing import List, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list:
    """
    Chunk text into parts of a given word count with overlap.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The number of words in each chunk.
        overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks


def semantic_chunk_text(text: str, model, min_chunk_size: int = 300, max_chunk_size: int = 500) -> List[str]:
    """
    Chunks text using semantic similarity and sentence boundaries.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Get embeddings for sentences
    embeddings = model.encode(sentences)
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        linkage='ward'
    )
    clusters = clustering.fit_predict(embeddings)
    
    # Group sentences by clusters and merge into chunks
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence, cluster in zip(sentences, clusters):
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
            
        if current_size >= min_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks
