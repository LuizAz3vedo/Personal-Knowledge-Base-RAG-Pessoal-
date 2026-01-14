"""Core - Logica central do RAG."""

from src.core.config import settings
from src.core.embeddings import EmbeddingCache, EmbeddingService, get_embedding_service
from src.core.hybrid_search import (
    BM25,
    HybridSearcher,
    build_bm25_index_from_vectorstore,
    get_bm25_index,
)
from src.core.llm import LLMService, get_llm_service
from src.core.reranker import Reranker, get_reranker
from src.core.retriever import Retriever, RetrievalResult, get_retriever
from src.core.vectorstore import VectorStoreManager, vector_store

__all__ = [
    "settings",
    "EmbeddingCache",
    "EmbeddingService",
    "get_embedding_service",
    "BM25",
    "HybridSearcher",
    "get_bm25_index",
    "build_bm25_index_from_vectorstore",
    "LLMService",
    "get_llm_service",
    "Reranker",
    "get_reranker",
    "Retriever",
    "RetrievalResult",
    "get_retriever",
    "VectorStoreManager",
    "vector_store",
]
