

from app.services.vectorizer_manager import VectorizerManager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.services.embeddings import create_sentence_embeddings
from app.services.grokLlm import expand_query
from sentence_transformers import CrossEncoder

def hybrid_retriever(
    client,
    collection_name: str,
    query: str,
    model,
    tfidf_vectorizer,
    top_k: int = 5,
    schemantic_weight: float = 0.7
):
    """
    Hybrid document retrieval using SentenceTransformer and TF-IDF
    """
    query = expand_query(query)
    print('query after expansion:', query)
    
    query_embedding = create_sentence_embeddings([query], model)

    vectorizer_manager = VectorizerManager()
    vectorizer = vectorizer_manager.get_vectorizer(collection_name)
    query_tfidf = vectorizer.transform([query])
    
    qdrant_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
        limit=top_k * 2,
        with_payload=True
    )
    
    results = []
    for result in qdrant_results:
        payload = result.payload
        
        try:
            tfidf_vector = np.array(payload.get('tfidf_vector', [])).reshape(1, -1)
            tfidf_similarity = cosine_similarity(query_tfidf, tfidf_vector)[0][0]
        except Exception as e:
            print(f"TF-IDF computation error: {e}")
            tfidf_similarity = 0
        
        schemantic_score = result.score
        combined_score = (schemantic_weight * schemantic_score + 
                         (1 - schemantic_weight) * tfidf_similarity)
        
        results.append({
            'document': payload.get('content', 'No content'),
            'schemantic_similarity': schemantic_score,
            'tfidf_similarity': tfidf_similarity,
            'combined_similarity': combined_score
        })
    
    return sorted(results, key=lambda x: x['combined_similarity'], reverse=True)[:top_k]

def hybrid_retriever_reranked(
    client,
    collection_name: str,
    query: str,
    model,
    tfidf_vectorizer,
    top_k: int = 5,
    schemantic_weight: float = 0.7,
    rerank_weight: float = 0.4
):
    """
    Enhanced hybrid retrieval with cross-encoder reranking
    """
    # Initialize cross-encoder for reranking
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Initial retrieval
    query = expand_query(query)
    print('query after expansion:', query)
    query_embedding = create_sentence_embeddings([query], model)
    vectorizer_manager = VectorizerManager()
    vectorizer = vectorizer_manager.get_vectorizer(collection_name)
    query_tfidf = vectorizer.transform([query])
    
    # Get more candidates for reranking
    initial_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
        limit=top_k * 3,
        with_payload=True
    )
    
    results = []
    pairs_to_rerank = []
    
    for result in initial_results:
        payload = result.payload
        document = payload.get('content', 'No content')
        
        # Prepare pairs for batch reranking
        pairs_to_rerank.append([query, document])
        
        try:
            tfidf_vector = np.array(payload.get('tfidf_vector', [])).reshape(1, -1)
            tfidf_similarity = cosine_similarity(query_tfidf, tfidf_vector)[0][0]
        except Exception as e:
            print(f"TF-IDF computation error: {e}")
            tfidf_similarity = 0
        
        schemantic_score = result.score
        initial_score = (schemantic_weight * schemantic_score + 
                        (1 - schemantic_weight) * tfidf_similarity)
        
        results.append({
            'document': document,
            'schemantic_similarity': schemantic_score,
            'tfidf_similarity': tfidf_similarity,
            'initial_score': initial_score
        })
    
    # Batch reranking
    rerank_scores = reranker.predict(pairs_to_rerank)
    
    # Combine scores
    for idx, result in enumerate(results):
        result['rerank_score'] = rerank_scores[idx]
        result['combined_similarity'] = (
            (1 - rerank_weight) * result['initial_score'] + 
            rerank_weight * result['rerank_score']
        )
    
    return sorted(results, key=lambda x: x['combined_similarity'], reverse=True)[:top_k - 2]