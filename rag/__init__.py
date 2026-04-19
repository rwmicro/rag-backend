"""RAG Pipeline - Core Modules.

Public symbols are exposed lazily via PEP-562 module-level ``__getattr__`` so
that importing a lightweight submodule (e.g. ``from rag.schemas import X``)
doesn't force the heavy ML dependencies (torch, sentence-transformers,
pymupdf, chromadb…) to load just because they happen to be used elsewhere in
the package. Callers can still ``from rag import DocumentIngestor`` — that
form triggers the import only when the attribute is first accessed.
"""

import importlib
from typing import TYPE_CHECKING

__version__ = "2.0.0"


_LAZY_EXPORTS = {
    "DocumentIngestor": "rag.ingest",
    "parse_pdf": "rag.ingest",
    "parse_markdown": "rag.ingest",
    "SemanticChunker": "rag.chunking",
    "RecursiveChunker": "rag.chunking",
    "create_chunker": "rag.chunking",
    "Chunk": "rag.chunking",
    "EmbeddingModel": "rag.embeddings",
    "create_embedding_model": "rag.embeddings",
    "HybridEmbeddingModel": "rag.embeddings",
    "AdaptiveEmbeddingFusion": "rag.embeddings",
    "VectorStore": "rag.vectordb",
    "create_vector_store": "rag.vectordb",
    "Retriever": "rag.retrieval",
    "HybridRetriever": "rag.retrieval",
    "MultiQueryRetriever": "rag.retrieval",
    "Reranker": "rag.retrieval",
    "ContextCompressor": "rag.compression",
    "LLMGenerator": "rag.generation",
    "EmbeddingCache": "rag.cache",
    "get_cache": "rag.cache",
    "get_embedding_cache": "rag.cache",
    "get_query_cache": "rag.cache",
    "SemanticCache": "rag.semantic_cache",
    "get_semantic_cache": "rag.semantic_cache",
    "GraphRAG": "rag.graph_rag",
    "Entity": "rag.graph_rag",
    "Relation": "rag.graph_rag",
    "HyDE": "rag.hyde",
    "AdaptiveHyDE": "rag.hyde",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'rag' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent accesses
    return value


if TYPE_CHECKING:
    # Static type checkers don't execute __getattr__; point them at the real
    # modules so IDE autocomplete still works.
    from rag.ingest import DocumentIngestor, parse_pdf, parse_markdown
    from rag.chunking import SemanticChunker, RecursiveChunker, create_chunker, Chunk
    from rag.embeddings import (
        EmbeddingModel,
        create_embedding_model,
        HybridEmbeddingModel,
        AdaptiveEmbeddingFusion,
    )
    from rag.vectordb import VectorStore, create_vector_store
    from rag.retrieval import Retriever, HybridRetriever, MultiQueryRetriever, Reranker
    from rag.compression import ContextCompressor
    from rag.generation import LLMGenerator
    from rag.cache import EmbeddingCache, get_cache, get_embedding_cache, get_query_cache
    from rag.semantic_cache import SemanticCache, get_semantic_cache
    from rag.graph_rag import GraphRAG, Entity, Relation
    from rag.hyde import HyDE, AdaptiveHyDE
