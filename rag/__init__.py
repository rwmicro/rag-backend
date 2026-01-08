"""
RAG Pipeline - Core Modules
Modern Python implementation with advanced retrieval techniques
"""

__version__ = "2.0.0"

from .ingest import DocumentIngestor, parse_pdf, parse_markdown
from .chunking import SemanticChunker, RecursiveChunker, create_chunker
from .embeddings import EmbeddingModel, create_embedding_model, HybridEmbeddingModel, AdaptiveEmbeddingFusion
from .vectordb import VectorStore, create_vector_store
from .retrieval import Retriever, HybridRetriever, MultiQueryRetriever, Reranker
from .compression import ContextCompressor
from .generation import LLMGenerator
from .cache import EmbeddingCache, get_cache, get_embedding_cache, get_query_cache
from .semantic_cache import SemanticCache, get_semantic_cache
from .graph_rag import GraphRAG, Entity, Relation
from .hyde import HyDE, AdaptiveHyDE

__all__ = [
    "DocumentIngestor",
    "parse_pdf",
    "parse_markdown",
    "SemanticChunker",
    "RecursiveChunker",
    "create_chunker",
    "EmbeddingModel",
    "create_embedding_model",
    "HybridEmbeddingModel",
    "AdaptiveEmbeddingFusion",
    "VectorStore",
    "create_vector_store",
    "Retriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "Reranker",
    "ContextCompressor",
    "LLMGenerator",
    "EmbeddingCache",
    "get_cache",
    "get_embedding_cache",
    "get_query_cache",
    "SemanticCache",
    "get_semantic_cache",
    "GraphRAG",
    "Entity",
    "Relation",
    "HyDE",
    "AdaptiveHyDE",
]
