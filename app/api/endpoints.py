from fastapi import APIRouter, HTTPException, Path, UploadFile, File, Query
from typing import List, Optional
from pydantic import BaseModel
from app.services.embeddings import store_embeddings_in_db
# from app.services.retrival import get_hybrid_similar_chunks
from app.services.retrival import hybrid_retriever, hybrid_retriever_reranked
from app.services.qdrant_client_init import get_qdrant_client
import fitz
from models.models_init import init_models
from app.services.grokLlm import get_answer
from datetime import datetime
from pydantic import Field, BaseModel
import os
import uuid
from app.services.chunk import chunk_text_with_overlap
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Response Models
class DocumentMetadata(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for the document/collection")
    filename: str = Field(..., description="Original filename")
    upload_timestamp: datetime = Field(..., description="When the document was uploaded")
    chunk_count: int = Field(..., description="Number of chunks the document was split into")
    
class ChunkInfo(BaseModel):
    content: str = Field(..., description="Content of the document chunk")
    schemantic_similarity: float = Field(..., description="Schemantic similarity score")
    tfidf_similarity: float = Field(..., description="TF-IDF similarity score")
    combined_similarity: float = Field(..., description="Combined similarity score")
    doc_id: str = Field(..., description="ID of the source document")
   

class QueryResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer")
    doc_id: str = Field(..., description="Document ID that was queried")
    referenced_chunks: List[ChunkInfo] = Field(..., description="Source chunks used for the answer")
    confidence_score: float = Field(..., description="Overall confidence score for the answer")

class UploadResponse(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for the document")
    message: str = Field(..., description="Status message")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

router = APIRouter(prefix="/rag", tags=["RAG Operations"])
model,  tfidf_vectorizer = init_models()
client = get_qdrant_client()

def generate_doc_id() -> str:
    """Generate a unique document ID that will also serve as collection name"""
    return f"doc_{str(uuid.uuid4())}"

class SmResponse(BaseModel):
    answer: str
    referenced_documents: List[str]

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its content, and store it in the vector database.
    """
    try:
        print('file uploaded')
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        doc_id = generate_doc_id()
        # Synchronously read the file content
        pdf_content = ""
        pdf_bytes = file.file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf_document:
            pdf_content += page.get_text()
        
        # Create chunks
        chunks = chunk_text_with_overlap(pdf_content, chunk_size=300, overlap=50)
        print('pdf content read and storing in db')
        # Store embeddings in the vector database
        store_embeddings_in_db(
           
            chunks=chunks,
            collection_name=doc_id,
            client=client,
            model=model,
            tfidf_vectorizer=tfidf_vectorizer
        )
        print('pdf content stored in db')
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=file.filename,
            upload_timestamp=datetime.utcnow(),
            chunk_count=len(chunks)
        )
        return UploadResponse(
            doc_id=doc_id,
            message="PDF content processed and stored successfully.",
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


    


@router.post("/query/{doc_id}", response_model=QueryResponse)
async def query_documents(    
    doc_id: str,
    question: str = Query(..., description="Question to ask about the document")):
    """
    Query the stored  documents and retrieve relevant chunks.
    """
    try:
        # try:
        #     client.delete_collection(collection_name="sms")
        # except Exception as e:
        #     print(f"Collection deletion error (might not exist): {e}")
        print('querying documents with question:', question)
        # Use hybrid retriever to get similar chunks
        # results = hybrid_retriever(
        #     client=client,
        #     collection_name="sm",
        #     query=question,
        #     model=model,
        #     tfidf_vectorizer=tfidf_vectorizer,
        #     top_k=5,
        #     schemantic_weight=0.7
        # )
        results = hybrid_retriever_reranked(
            client=client,
            collection_name=doc_id,
            query=question,
            model=model,
            tfidf_vectorizer=tfidf_vectorizer,
            top_k=5,
            schemantic_weight=0.7,
            rerank_weight=0.4
        )
        if not results:
            return QueryResponse(
                answer="No relevant information found in this document.",
                doc_id=doc_id,
                referenced_chunks=[],
                confidence_score=0.0
            )
        chunks = [
            ChunkInfo(
                content=result['document'],
                schemantic_similarity=result['schemantic_similarity'],
                tfidf_similarity=result['tfidf_similarity'],
                combined_similarity=result['combined_similarity'],
                doc_id=doc_id,
               
            )
            for result in results
        ]
        
        # Generate answer
        documents = [chunk.content for chunk in chunks]
        llm_answer = get_answer(question, documents)

        confidence_score = sum(r['combined_similarity'] for r in results) / len(results)

        return QueryResponse(
            answer=llm_answer,
            doc_id=doc_id,
            referenced_chunks=chunks,
            confidence_score=confidence_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str = Path(..., description="Document ID to delete")
):
    """
    Delete a document and its collection.
    """
    try:
       
        # Delete the collection
        client.delete_collection(doc_id)  # Use doc_id as collection name
        # delete file from vectorizer dir
        
        # Delete vectorizer file
        vectorizer_path = os.path.join("vectorizers", f"{doc_id}_vectorizer.pkl")
        try:
            if os.path.exists(vectorizer_path):
                os.remove(vectorizer_path)
                logger.info(f"Successfully deleted vectorizer file for document {doc_id}")
            else:
                logger.warning(f"Vectorizer file not found for document {doc_id}")
        except OSError as e:
            logger.error(f"Error deleting vectorizer file: {e}")
            # Don't fail the whole operation if just the file deletion fails
            # but include it in the response
            return {
                "message": f"Document {doc_id} deleted but failed to remove vectorizer file",
                "doc_id": doc_id,
                "warning": f"Failed to delete vectorizer file: {str(e)}"
            }

        return {
            "message": f"Document {doc_id} successfully deleted",
            "doc_id": doc_id,
            "details": "Removed both collection and vectorizer file"
        }

    except Exception as e:
        logger.error(f"Unexpected error in delete_document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )