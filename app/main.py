from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from qdrant_client import QdrantClient
from app.services.qdrant_client_init import get_qdrant_client


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="RAG Search API",
        description="RAG-powered  search",
        version="0.1.0"
    )

    # Add CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the API endpoints
    app.include_router(api_router, prefix="/api/v1")

    return app


# Initialize the Qdrant client and app
client = get_qdrant_client()

app = create_app()
