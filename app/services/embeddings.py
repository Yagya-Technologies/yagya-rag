import numpy as np
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.services.chunk import semantic_chunk_text, chunk_text_with_overlap
from app.services.vectorizer_manager import VectorizerManager
def create_sentence_embeddings(texts: list, model):
    """
    Get embeddings using SentenceTransformer
    """
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print("Embeddings generated.")
    return embeddings

def create_tfidf_embeddings(texts: list, tfidf_vectorizer: TfidfVectorizer, collection_name: str):
    """
    Create TF-IDF embeddings for a list of texts.
    
    Args:
        texts (list): List of text strings
        
    Returns:
        scipy.sparse.csr_matrix: TF-IDF embeddings matrix
    """
    vectorizer_manager = VectorizerManager()
    tfidf_matrix = vectorizer_manager.create_vectorizer(collection_name, texts)
    return tfidf_matrix

def create_hybrid_embeddings(chunks, model, tfidf_vectorizer, collection_name):
    """
    Create hybrid embeddings combining SentenceTransformer and TF-IDF
    """
    sentence_embeddings = create_sentence_embeddings(chunks, model)
    print(f"Created sentence embeddings with shape: {sentence_embeddings.shape}")
    
    tfidf_embeddings = create_tfidf_embeddings(chunks, tfidf_vectorizer, collection_name)
    print(f"Created TF-IDF embeddings with shape: {tfidf_embeddings.shape}")
    
    return sentence_embeddings, tfidf_embeddings, chunks

def store_embeddings_in_db(chunks, collection_name, client, model, tfidf_vectorizer):
    """
    Store document embeddings (BERT + TF-IDF) in Qdrant and verify the storage.
    """
 
    # Generate embeddings
    bert_embeddings, tfidf_embeddings, processed_texts = create_hybrid_embeddings(chunks, model, tfidf_vectorizer, collection_name)
    
    print(f"Created BERT embeddings with shape: {bert_embeddings.shape}")
    print(f"Created TF-IDF embeddings with shape: {tfidf_embeddings.shape}")

    # Check if collection exists and create it if necessary
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=bert_embeddings.shape[1], 
                distance=Distance.COSINE
            )
        )
    except Exception as e:
        # Collection already exists, ignore error
        print(f"Collection already exists: {str(e)}")

    # Prepare points for Qdrant (store both BERT and TF-IDF embeddings)
    points = [
        PointStruct(
            id=random.randint(1, 1000000),
            vector=bert_embedding.tolist(),
            payload={
                'content': doc,
                'tfidf_vector': tfidf_embedding.toarray().tolist()[0]  # Store TF-IDF embedding
            }
        )
        for doc, bert_embedding, tfidf_embedding in 
        zip(processed_texts, bert_embeddings, tfidf_embeddings)
    ]
    print(f"Prepared {len(points)} points for upserting.")
    # Upsert points into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Embeddings stored in Qdrant.")

    # # Verify the storage by querying one point (for example, the first one)
    # query_result = client.search(
    #     collection_name=collection_name,
    #     query_vector=bert_embeddings[0].tolist(),  # Search with the BERT vector of the first document
    #     limit=1
    # )

    # Print the result to verify the point was stored correctly
    # print(f"Verification Query Result: {query_result}")
    
    # Optionally, you can return a success message if needed
    return {"message": "Embeddings stored and verified successfully!"}
