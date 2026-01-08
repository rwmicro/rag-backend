"""
FastAPI Server for RAG Pipeline
Modern Python backend with streaming support
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import uvicorn
from loguru import logger
import sys
import json
import os
import gc
import torch
import re
import pickle
import numpy as np

from config.settings import settings, ensure_directories
from rag import (
    DocumentIngestor,
    create_chunker,
    create_embedding_model,
    create_vector_store,
    Retriever,
    HybridRetriever,
    MultiQueryRetriever,
    Reranker,
    ContextCompressor,
    LLMGenerator,
    get_embedding_cache,
    get_query_cache,
    get_semantic_cache,
    GraphRAG,
    HyDE,
    AdaptiveHyDE,
)
from rag.query_classifier import QueryClassifier, QueryType
from rag.confidence_evaluator import ConfidenceEvaluator, ConfidenceLevel
from rag.contrastive_retrieval import ContrastiveRetriever, NegationDetector
from rag.multi_hop_retrieval import MultiHopRetriever, QueryDecomposer
from rag.answer_verification import AnswerVerifier
from rag.feedback_logger import FeedbackLogger, FeedbackType
from rag.domain_ner import DomainSpecificNER, HybridNER, Domain
from rag.metadata_store import MetadataStore
from rag.observability import (
    get_query_logger,
    create_query_log,
    measure_time,
)
from rag.query_router import create_query_router
from rag.document_summary import get_document_summarizer
from rag.chunking import apply_contextual_embeddings
from rag.evaluation import RAGEvaluator, EvaluationSample
from rag.validation import (
    validate_file_upload,
    validate_query_text,
    validate_top_k,
    validate_chunk_params,
    validate_temperature,
    validate_max_tokens,
    validate_collection_id,
    sanitize_filename,
    ValidationError,
)

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Advanced RAG system with hybrid search, reranking, and compression",
    version="2.0.0",
)

# Add CORS middleware
# SECURITY: Configure CORS_ORIGINS in .env with specific domains for production
# Example: CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
cors_origins = (
    settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # IMPORTANT: Set CORS_ORIGINS in .env for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


# ============================================================================
# Pydantic Models
# ============================================================================


class ConversationMessage(BaseModel):
    """Message in conversation history"""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""

    query: str = Field(..., description="Search query")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list, description="Previous conversation for context"
    )
    collection_id: Optional[str] = Field(
        None, description="Collection ID to query (uses collection's LLM model)"
    )
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    auto_route: bool = Field(
        False, description="Automatically select optimal retrieval strategy"
    )
    use_hybrid_search: bool = Field(
        True, description="Use hybrid search (vector + BM25)"
    )
    use_multi_query: bool = Field(
        False, description="Use multi-query retrieval (generate query variations)"
    )
    num_query_variations: int = Field(
        2, description="Number of query variations for multi-query", ge=1, le=5
    )
    use_reranking: bool = Field(True, description="Apply reranking")
    use_compression: bool = Field(False, description="Apply context compression")
    use_graph_rag: bool = Field(
        False, description="Use Graph RAG for enhanced retrieval"
    )
    graph_expansion_depth: int = Field(
        1, description="Graph expansion depth for Graph RAG", ge=1, le=3
    )
    graph_alpha: float = Field(
        0.7, description="Weight for vector vs graph scores (0-1)", ge=0.0, le=1.0
    )
    use_hyde: bool = Field(
        False, description="Use HyDE (Hypothetical Document Embeddings)"
    )
    hyde_fusion: str = Field(
        "rrf", description="HyDE fusion method: 'average', 'max', or 'rrf'"
    )
    num_hypothetical_docs: int = Field(
        3, description="Number of hypothetical docs for HyDE", ge=1, le=5
    )
    use_adaptive_fusion: bool = Field(
        False, description="Use adaptive embedding fusion"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filters (e.g., {'file_type': 'pdf', 'date': {'$gte': '2024-01-01'}})",
    )
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    stream: bool = Field(True, description="Stream the response")

    # NEW: Advanced RAG features
    enable_query_classification: bool = Field(
        False, description="Enable query classification and routing"
    )
    enable_adaptive_alpha: bool = Field(
        False, description="Enable adaptive hybrid search alpha"
    )
    enable_mmr: bool = Field(False, description="Enable MMR diversity enforcement")
    mmr_lambda: Optional[float] = Field(
        None, description="MMR lambda parameter (0-1), uses settings default if None"
    )
    enable_contrastive: bool = Field(
        False, description="Enable contrastive retrieval for negation handling"
    )
    enable_multi_hop: bool = Field(
        False, description="Enable multi-hop retrieval for complex queries"
    )
    max_hops: int = Field(
        3, description="Maximum reasoning hops for multi-hop retrieval", ge=1, le=5
    )
    enable_answer_verification: bool = Field(
        False, description="Enable answer verification before presenting"
    )
    verification_threshold: Optional[float] = Field(
        None, description="Minimum verification score (0-1)"
    )
    enable_confidence_evaluation: bool = Field(
        False, description="Enable confidence evaluation with fallback strategies"
    )
    enable_feedback_logging: bool = Field(
        True, description="Enable feedback and performance logging"
    )

    # NEW: Multilingual features
    enable_multilingual: bool = Field(
        False, description="Enable full multilingual pipeline"
    )
    query_language: Optional[str] = Field(
        None,
        description="Query language code (auto-detected if None, e.g., 'en', 'fr', 'es')",
    )
    use_multilingual_embeddings: bool = Field(
        False,
        description="Use multilingual-e5-large embeddings for cross-lingual retrieval",
    )
    use_multilingual_bm25: bool = Field(
        False, description="Use language-specific BM25 tokenization"
    )
    use_multilingual_hyde: bool = Field(
        False, description="Generate hypothetical documents in query's language"
    )
    use_multilingual_classifier: bool = Field(
        False, description="Use multilingual query classification patterns"
    )
    detect_language: bool = Field(
        True, description="Automatically detect query language"
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint"""

    answer: str
    sources: List[Dict[str, Any]]
    query: str
    metadata: Dict[str, Any] = {}
    llm_model: Optional[str] = None  # Model used for generation
    collection_id: Optional[str] = None  # Collection queried


class IngestRequest(BaseModel):
    """Request model for /ingest endpoint"""

    recursive: bool = Field(True, description="Process subdirectories recursively")
    chunk_size: int = Field(1000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    chunking_strategy: str = Field("semantic", description="Chunking strategy")


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint"""

    success: bool
    message: str
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """Response model for /stats endpoint"""

    total_chunks: int
    total_files: int
    embedding_model: str
    llm_model: Optional[str] = None
    vector_store_type: str
    cache_stats: Dict[str, Any]


class EvaluationRequest(BaseModel):
    """Request model for /evaluate endpoint"""

    test_dataset: List[Dict[str, Any]] = Field(
        ..., description="List of evaluation samples"
    )
    collection_id: Optional[str] = Field(None, description="Collection to evaluate on")
    k_values: List[int] = Field([1, 3, 5, 10], description="K values for @k metrics")
    evaluate_generation: bool = Field(
        False, description="Also evaluate generation quality"
    )


class EvaluationResponse(BaseModel):
    """Response model for /evaluate endpoint"""

    retrieval_metrics: Dict[str, Any]
    generation_metrics: Optional[Dict[str, Any]] = None
    sample_count: int
    timestamp: str


# ============================================================================
# Helper Functions
# ============================================================================

COLLECTIONS_FILE = os.path.join(settings.DATA_DIR, "collections.json")


def _load_collections() -> Dict[str, Any]:
    """Load all collections metadata"""
    if os.path.exists(COLLECTIONS_FILE):
        with open(COLLECTIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_collections(collections: Dict[str, Any]):
    """Save collections metadata"""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(COLLECTIONS_FILE, "w") as f:
        json.dump(collections, f, indent=2)


def _get_or_create_collection(
    collection_id: str,
    title: str,
    llm_model: str,
    embedding_model: str,
    embedding_dimension: Optional[int] = None,
    reranker_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Get or create a collection"""
    collections = _load_collections()

    if collection_id not in collections:
        from datetime import datetime
        from rag.model_registry import get_model_dimension

        # Get embedding dimension from registry if not provided
        if embedding_dimension is None:
            embedding_dimension = get_model_dimension(embedding_model, "embedding")

        collections[collection_id] = {
            "id": collection_id,
            "title": title,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "reranker_model": reranker_model or settings.RERANKER_MODEL,
            "file_count": 0,
            "chunk_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "files": [],
            "file_metadata": {},  # NEW: stores per-file metadata {filename: {size, chunks, date}}
        }
        _save_collections(collections)

    return collections[collection_id]


def _update_collection_stats(
    collection_id: str, filename: str, num_chunks: int, file_size: int = 0
):
    """Update collection statistics after adding a file"""
    from datetime import datetime

    collections = _load_collections()

    if collection_id in collections:
        collection = collections[collection_id]

        # Ensure file_metadata exists (for legacy collections)
        if "file_metadata" not in collection:
            collection["file_metadata"] = {}

        # Add/update file metadata
        collection["file_metadata"][filename] = {
            "size": file_size,
            "chunks": num_chunks,
            "uploaded_at": datetime.utcnow().isoformat(),
        }

        # Update legacy files list for backward compatibility
        if filename not in collection["files"]:
            collection["files"].append(filename)
            collection["file_count"] = len(collection["files"])

        # Update or increment chunk count
        collection["chunk_count"] = collection.get("chunk_count", 0) + num_chunks

        _save_collections(collections)


def _delete_collection(collection_id: str):
    """Delete a collection"""
    collections = _load_collections()
    if collection_id in collections:
        del collections[collection_id]
        _save_collections(collections)


def _get_collection_llm_model(collection_id: Optional[str] = None) -> str:
    """Get LLM model for a specific collection, or use global default"""
    if collection_id:
        collections = _load_collections()
        if collection_id in collections:
            return collections[collection_id].get("llm_model", settings.LLM_MODEL)

    # Fallback: try to load from legacy config, then settings
    rag_config = _load_rag_config()
    return rag_config.get("llm_model", settings.LLM_MODEL)


def _get_collection_embedding_model(collection_id: Optional[str] = None) -> str:
    """Get embedding model for a specific collection, or use global default"""
    if collection_id:
        collections = _load_collections()
        if collection_id in collections:
            return collections[collection_id].get(
                "embedding_model", settings.EMBEDDING_MODEL
            )

    # Fallback to global settings
    return settings.EMBEDDING_MODEL


def _get_collection_reranker_model(collection_id: Optional[str] = None) -> str:
    """Get reranker model for a specific collection, or use global default"""
    if collection_id:
        collections = _load_collections()
        if collection_id in collections:
            return collections[collection_id].get(
                "reranker_model", settings.RERANKER_MODEL
            )

    # Fallback to global settings
    return settings.RERANKER_MODEL


def _validate_embedding_compatibility(
    collection_id: str, new_embedding_model: str
) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Check if new embedding model is compatible with existing collection

    Args:
        collection_id: Collection ID to check
        new_embedding_model: New embedding model to validate

    Returns:
        (is_compatible, error_message, existing_dimension)
    """
    collections = _load_collections()

    if collection_id not in collections:
        # New collection - any model is fine
        return True, None, None

    collection = collections[collection_id]
    existing_dimension = collection.get("embedding_dimension")
    existing_model = collection.get("embedding_model")

    # If no dimension stored (legacy collection), can't validate
    if existing_dimension is None:
        logger.warning(
            f"Collection {collection_id} has no dimension metadata, skipping validation"
        )
        return True, None, None

    # Get dimension of new model
    from rag.model_registry import get_model_dimension, list_models_by_dimension

    new_dimension = get_model_dimension(new_embedding_model, "embedding")

    if new_dimension is None:
        # Unknown model - warn but allow
        logger.warning(
            f"Unknown model '{new_embedding_model}', cannot validate dimension"
        )
        return True, None, existing_dimension

    if new_dimension != existing_dimension:
        # Dimension mismatch!
        compatible_models = list_models_by_dimension(existing_dimension, "embedding")

        error_msg = (
            f"Embedding dimension mismatch!\n\n"
            f"Collection '{collection_id}' was created with '{existing_model}' ({existing_dimension}D embeddings).\n"
            f"You're trying to use '{new_embedding_model}' ({new_dimension}D embeddings).\n\n"
            f"Cannot mix different embedding dimensions in the same collection.\n\n"
            f"Compatible {existing_dimension}D models: {', '.join(compatible_models) if compatible_models else 'None in registry'}\n\n"
            f"Options:\n"
            f"1. Use a compatible {existing_dimension}D model\n"
            f"2. Create a new collection for {new_dimension}D embeddings"
        )

        return False, error_msg, existing_dimension

    return True, None, existing_dimension


def _update_collection_model(collection_id: str, llm_model: str) -> bool:
    """Update the LLM model for a collection"""
    collections = _load_collections()
    if collection_id not in collections:
        return False

    collections[collection_id]["llm_model"] = llm_model
    _save_collections(collections)
    logger.info(f"Updated collection {collection_id} LLM model to {llm_model}")
    return True


# Legacy support for single collection
RAG_CONFIG_FILE = os.path.join(settings.DATA_DIR, "rag_config.json")


def _save_rag_config(llm_model: str):
    """Save RAG configuration (LLM model preference) - legacy"""
    config = {"llm_model": llm_model}
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(RAG_CONFIG_FILE, "w") as f:
        json.dump(config, f)


def _load_rag_config() -> Dict[str, Any]:
    """Load RAG configuration - legacy"""
    if os.path.exists(RAG_CONFIG_FILE):
        with open(RAG_CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


# ============================================================================
# Global instances (initialized on startup)
# ============================================================================

embedding_model = None
vector_stores: Dict[str, Any] = {}  # Per-collection vector stores
retriever = None
reranker = None
compressor = None
llm_generator = None
ingestor = None
graph_rag = None
hyde = None
query_router = None

# NEW: Advanced RAG components
query_classifier = None
confidence_evaluator = None
answer_verifier = None
feedback_logger = None
metadata_store = None

# Reranker cache: stores reranker instances by model name
_reranker_cache: Dict[str, Any] = {}

# Embedding model cache: stores embedding model instances by cache key
_embedding_model_cache: Dict[str, Any] = {}


def get_or_create_embedding_model(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    use_ollama: bool = False,
    use_hybrid: bool = False,
    use_adaptive: bool = False,
    structural_weight: float = 0.3,
):
    """
    Get or create an embedding model instance with caching
    Prevents memory leaks by reusing models across requests

    Args:
        model_name: Model name
        provider: Provider (ollama, huggingface)
        use_ollama: Use Ollama for embeddings
        use_hybrid: Use hybrid embeddings
        use_adaptive: Use adaptive fusion
        structural_weight: Structural weight for hybrid

    Returns:
        Cached or new embedding model instance
    """
    global _embedding_model_cache

    # Use default model if not specified
    model_name = model_name or settings.EMBEDDING_MODEL

    # Create cache key from parameters
    cache_key = f"{model_name}|{provider}|{use_ollama}|{use_hybrid}|{use_adaptive}|{structural_weight}"

    # Check cache first
    if cache_key in _embedding_model_cache:
        cached_model = _embedding_model_cache[cache_key]
        logger.info(
            f"♻️  CACHE HIT: Using cached embedding model: {model_name} on {cached_model.device} (cache size: {len(_embedding_model_cache)})"
        )
        return cached_model

    # Create new embedding model
    logger.info(
        f"🔄 CACHE MISS: Loading embedding model on-demand: {model_name} (will cache for reuse)"
    )
    try:
        model = create_embedding_model(
            model_name=model_name,
            provider=provider,
            use_ollama=use_ollama,
            use_hybrid=use_hybrid,
            use_adaptive=use_adaptive,
            structural_weight=structural_weight if use_hybrid else None,
        )

        # Cache the model
        _embedding_model_cache[cache_key] = model
        logger.info(
            f"✅ Loaded and cached embedding model: {model_name} on {model.device} (cache size: {len(_embedding_model_cache)})"
        )

        return model

    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def _get_or_create_vector_store(
    collection_id: str, embedding_dimension: Optional[int] = None
):
    """
    Get or create a vector store for a specific collection

    Args:
        collection_id: Collection ID
        embedding_dimension: Embedding dimension (optional, will use settings default if not provided)

    Returns:
        VectorStore instance for the collection
    """
    global vector_stores

    # Check if store already exists for this collection
    if collection_id in vector_stores:
        logger.debug(f"Using cached vector store for collection: {collection_id}")
        return vector_stores[collection_id]

    # Create new vector store for this collection
    logger.info(f"Creating vector store for collection: {collection_id}")

    dimension = embedding_dimension or settings.EMBEDDING_DIMENSION

    vector_store = create_vector_store(collection_id=collection_id, dimension=dimension)

    # Cache the store
    vector_stores[collection_id] = vector_store
    logger.info(f"✓ Created and cached vector store for collection: {collection_id}")

    return vector_store


def get_or_create_reranker(model_name: str):
    """
    Get or create a reranker instance for the given model
    Implements caching to avoid reloading the same model multiple times

    Args:
        model_name: Reranker model name

    Returns:
        Reranker instance
    """
    global _reranker_cache

    # Check cache first
    if model_name in _reranker_cache:
        logger.debug(f"Using cached reranker: {model_name}")
        return _reranker_cache[model_name]

    # Create new reranker instance
    logger.info(f"Loading reranker on-demand: {model_name}")
    try:
        from rag.retrieval import Reranker

        reranker_instance = Reranker(model_name=model_name)

        # Cache the instance
        _reranker_cache[model_name] = reranker_instance
        logger.info(f"✓ Loaded and cached reranker: {model_name}")

        return reranker_instance
    except Exception as e:
        logger.error(f"Failed to load reranker '{model_name}': {e}")
        # Fall back to default reranker if available
        if (
            settings.RERANKER_MODEL != model_name
            and settings.RERANKER_MODEL in _reranker_cache
        ):
            logger.warning(
                f"Falling back to default reranker: {settings.RERANKER_MODEL}"
            )
            return _reranker_cache[settings.RERANKER_MODEL]
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global \
        embedding_model, \
        vector_stores, \
        retriever, \
        reranker, \
        compressor, \
        llm_generator, \
        ingestor, \
        graph_rag, \
        hyde, \
        query_router
    global \
        query_classifier, \
        confidence_evaluator, \
        answer_verifier, \
        feedback_logger, \
        metadata_store

    logger.info("🚀 Starting RAG Pipeline API v2.0 (Advanced Features)")

    # Ensure directories exist
    ensure_directories()

    # Embedding model will be loaded lazily per collection (saves memory & startup time)
    logger.info("Embedding models will be loaded on-demand per collection")
    embedding_model = None
    retriever = None

    # Vector stores will be created lazily per collection
    logger.info("Vector stores will be created on-demand per collection")
    vector_stores = {}

    # Initialize default reranker (optional) and cache it
    if settings.USE_RERANKING:
        logger.info("Loading default reranker...")
        try:
            reranker = get_or_create_reranker(settings.RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"Failed to load default reranker: {e}")
            reranker = None

    # Initialize compressor (optional)
    if settings.USE_COMPRESSION:
        compressor = ContextCompressor(max_tokens=settings.MAX_CONTEXT_TOKENS)

    # Initialize LLM generator
    logger.info("Initializing LLM generator...")
    # Use configured LLM model if available, otherwise use default from settings
    rag_config = _load_rag_config()
    llm_model = rag_config.get("llm_model") or settings.LLM_MODEL
    logger.info(f"Using LLM model: {llm_model}")

    llm_generator = LLMGenerator(
        base_url=settings.LLM_BASE_URL,
        model=llm_model,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )

    # Graph RAG and HyDE will be initialized lazily on first use
    graph_rag = None
    hyde = None

    # Initialize document ingestor
    ingestor = DocumentIngestor()

    # Initialize query router (if enabled)
    if settings.ENABLE_AUTO_ROUTING:
        logger.info("Initializing query router...")
        query_router = create_query_router(llm_generator=llm_generator)
        logger.info(f"✓ Query router initialized (mode: {settings.ROUTER_MODE})")

    # Preload models if enabled
    if settings.PRELOAD_MODELS_ON_STARTUP:
        logger.info("Preloading models on startup...")
        from rag.model_validation import preload_models

        preload_results = preload_models()
        successful = sum(1 for v in preload_results.values() if v)
        logger.info(f"✓ Preloaded {successful}/{len(preload_results)} models")

    # NEW: Initialize advanced RAG components
    logger.info("Initializing advanced RAG components...")

    # Query Classifier
    if settings.ENABLE_QUERY_CLASSIFICATION:
        logger.info("Initializing query classifier...")
        query_classifier = QueryClassifier(
            use_llm=settings.QUERY_CLASSIFIER_USE_LLM,
            llm_client=llm_generator if settings.QUERY_CLASSIFIER_USE_LLM else None,
        )
        logger.info("✓ Query classifier initialized")

    # Confidence Evaluator
    if settings.ENABLE_FALLBACK_STRATEGIES:
        logger.info("Initializing confidence evaluator...")
        confidence_evaluator = ConfidenceEvaluator(
            high_threshold=settings.RETRIEVAL_CONFIDENCE_HIGH,
            medium_threshold=settings.RETRIEVAL_CONFIDENCE_MEDIUM,
            minimum_threshold=settings.RETRIEVAL_CONFIDENCE_MINIMUM,
            enable_fallback=settings.ENABLE_FALLBACK_STRATEGIES,
        )
        logger.info("✓ Confidence evaluator initialized")

    # Answer Verifier
    logger.info("Initializing answer verifier...")
    answer_verifier = AnswerVerifier(
        llm_generator=llm_generator,
        embedding_model=None,  # Will use collection-specific embedding model
        grounding_threshold=0.7,
        verification_threshold=0.6,
    )
    logger.info("✓ Answer verifier initialized")

    # Feedback Logger
    logger.info("Initializing feedback logger...")
    feedback_logger = FeedbackLogger(
        db_path=settings.FEEDBACK_DB_PATH,
        enable_logging=True,
        log_queries=True,
        log_retrievals=True,
        log_generations=True,
        log_feedback=True,
    )
    logger.info("✓ Feedback logger initialized")

    # Metadata Store
    logger.info("Initializing metadata store...")
    metadata_store = MetadataStore(db_path=settings.METADATA_DB_PATH)
    logger.info("✓ Metadata store initialized")

    logger.info("✅ RAG Pipeline API v2.0 ready with advanced features enabled!")


# ============================================================================
# Shared RAG Pipeline Logic
# ============================================================================


async def _execute_rag_pipeline(
    request: QueryRequest,
    query_log: Any,
) -> tuple:
    """
    Execute RAG pipeline - shared logic between streaming and non-streaming endpoints

    This function handles the complete retrieval pipeline:
    1. Query contextualization (with conversation history)
    2. Automatic query routing (if enabled)
    3. Embedding model selection (collection-specific)
    4. Retrieval strategy (HyDE / multi-query / standard / hybrid)
    5. Graph RAG enhancement (optional)
    6. Reranking with deduplication (optional)
    7. Context compression (optional)
    8. LLM model selection (collection-specific)

    Args:
        request: Query request with all parameters
        query_log: Query log object for observability

    Returns:
        tuple: (
            chunks_with_scores: List[Tuple[Chunk, float]],
            current_llm_generator: LLMGenerator,
            collection_llm_model: str,
            contextualized_query: str,
            retrieval_start_time: float
        )
    """
    import time

    # Step 0: Get or create vector store for this collection
    if request.collection_id:
        collection = _load_collections().get(request.collection_id)
        embedding_dimension = (
            collection.get("embedding_dimension") if collection else None
        )
        vector_store = _get_or_create_vector_store(
            request.collection_id, embedding_dimension
        )
        logger.info(f"Using vector store for collection: {request.collection_id}")
    else:
        # Fallback to default collection
        vector_store = _get_or_create_vector_store(
            "default", settings.EMBEDDING_DIMENSION
        )
        logger.info("No collection specified, using default collection")

    # Step 1: Contextualize query if history exists
    contextualized_query = request.query
    if request.conversation_history:
        # Convert Pydantic models to dicts if needed
        history_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]
        contextualized_query = llm_generator.contextualize_query(
            request.query, history_dicts
        )
        logger.info(f"Using contextualized query for retrieval: {contextualized_query}")

    # Store contextualized query in log
    query_log.contextualized_query = contextualized_query

    # Step 1.5: Check semantic cache before retrieval
    if settings.USE_CACHE:
        try:
            # Get collection-specific embedding model for cache lookup
            collection_embedding_model_name = _get_collection_embedding_model(
                request.collection_id
            )
            current_embedding_model = get_or_create_embedding_model(
                model_name=collection_embedding_model_name
            )

            semantic_cache = get_semantic_cache(current_embedding_model)

            # Check cache
            cached_results = semantic_cache.get(
                query=contextualized_query,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )

            if cached_results:
                logger.info(
                    f"Semantic cache HIT for query: '{contextualized_query[:50]}...'"
                )
                query_log.metadata["semantic_cache_hit"] = True
                query_log.retrieval.final_candidates = len(cached_results)

                # Get collection-specific LLM model for the cached results
                collection_llm_model = _get_collection_llm_model(request.collection_id)

                # Create collection-specific LLM generator
                current_llm_generator = llm_generator
                if collection_llm_model != llm_generator.model:
                    current_llm_generator = LLMGenerator(
                        base_url=settings.LLM_BASE_URL,
                        model=collection_llm_model,
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.LLM_MAX_TOKENS,
                        timeout=settings.LLM_TIMEOUT,
                    )

                return (
                    cached_results,
                    current_llm_generator,
                    collection_llm_model,
                    contextualized_query,
                    retrieval_start,
                )

        except Exception as e:
            logger.warning(
                f"Semantic cache lookup failed: {e}, continuing with normal retrieval"
            )

    # Step 2: Apply query routing if enabled
    if request.auto_route and settings.ENABLE_AUTO_ROUTING and query_router:
        logger.info("Using automatic query routing...")
        query_type, auto_strategy = query_router.route(contextualized_query)

        # Override request parameters with router's strategy
        request.use_hybrid_search = auto_strategy.use_hybrid_search
        request.use_multi_query = auto_strategy.use_multi_query
        request.use_hyde = auto_strategy.use_hyde
        request.use_graph_rag = auto_strategy.use_graph_rag
        request.use_reranking = auto_strategy.use_reranking
        request.num_query_variations = auto_strategy.num_query_variations
        request.num_hypothetical_docs = auto_strategy.num_hypothetical_docs
        request.hyde_fusion = auto_strategy.hyde_fusion
        request.graph_expansion_depth = auto_strategy.graph_expansion_depth
        request.graph_alpha = auto_strategy.graph_alpha

        # Log routing decision
        query_log.routing_decision = {
            "query_type": query_type.value,
            "strategy": auto_strategy.to_dict(),
            "mode": settings.ROUTER_MODE,
        }

        logger.info(
            f"Router selected strategy for {query_type.value}: "
            f"hybrid={auto_strategy.use_hybrid_search}, "
            f"multi_query={auto_strategy.use_multi_query}, "
            f"hyde={auto_strategy.use_hyde}, "
            f"graph_rag={auto_strategy.use_graph_rag}, "
            f"reranking={auto_strategy.use_reranking}"
        )

    # Step 3: Check if multilingual pipeline should be used
    if request.enable_multilingual:
        logger.info("Using multilingual retrieval pipeline...")
        retrieval_start = time.time()

        # Import multilingual pipeline
        from rag.multilingual_pipeline import create_multilingual_pipeline

        # Create multilingual pipeline with collection-specific settings
        ml_pipeline = create_multilingual_pipeline(
            vector_store=vector_store,
            llm_generator=llm_generator,
            enable_all_features=False,  # Use individual flags
        )

        # Override pipeline settings based on request
        ml_pipeline.use_multilingual_embeddings = request.use_multilingual_embeddings
        ml_pipeline.use_multilingual_bm25 = request.use_multilingual_bm25
        ml_pipeline.use_multilingual_hyde = request.use_multilingual_hyde
        ml_pipeline.use_multilingual_classifier = request.use_multilingual_classifier

        # Detect language if requested
        detected_language = None
        language_confidence = None
        if request.detect_language:
            detected_language, language_confidence = ml_pipeline.detect_language(
                contextualized_query
            )
            logger.info(
                f"Detected language: {detected_language} (confidence: {language_confidence:.2f})"
            )
            query_log.metadata["detected_language"] = detected_language
            query_log.metadata["language_confidence"] = float(language_confidence)

        # Use provided language or detected language
        query_language = request.query_language or detected_language

        # Classify query with multilingual patterns
        if request.use_multilingual_classifier:
            query_analysis = ml_pipeline.classify_query(
                contextualized_query, language=query_language
            )
            logger.info(
                f"Query type: {query_analysis['query_type']} (language: {query_analysis.get('language', 'unknown')})"
            )
            query_log.metadata["query_classification"] = query_analysis

        # Determine retrieval parameters
        initial_top_k = request.top_k
        if request.use_reranking:
            initial_top_k = settings.INITIAL_RETRIEVAL_K
        elif request.use_graph_rag:
            initial_top_k = request.top_k * 3

        # Retrieve using multilingual pipeline
        chunks_with_scores = ml_pipeline.retrieve(
            query=contextualized_query,
            top_k=initial_top_k,
            use_hyde=request.use_multilingual_hyde,
            use_bm25=request.use_multilingual_bm25,
            fusion_method="rrf",
            metadata_filter=request.metadata_filter,
            language=query_language,
        )

        logger.info(f"Multilingual pipeline retrieved {len(chunks_with_scores)} chunks")
        query_log.retrieval.initial_candidates = len(chunks_with_scores)

        # Get collection-specific embedding model for downstream processing
        collection_embedding_model_name = _get_collection_embedding_model(
            request.collection_id
        )
        current_embedding_model = get_or_create_embedding_model(
            model_name=collection_embedding_model_name
        )

        # Skip to post-retrieval processing (reranking, compression, etc.)
        # Note: We'll continue with the existing pipeline below

    else:
        # Standard (non-multilingual) retrieval pipeline
        logger.info("Using standard retrieval pipeline...")
        retrieval_start = time.time()

        # Get collection-specific embedding model (lazy loading with cache)
        collection_embedding_model_name = _get_collection_embedding_model(
            request.collection_id
        )
        current_embedding_model = get_or_create_embedding_model(
            model_name=collection_embedding_model_name
        )

        # Create retriever with the collection-specific embedding model
        if request.use_hybrid_search or settings.USE_HYBRID_SEARCH:
            # Determine alpha for hybrid search
            hybrid_alpha = settings.HYBRID_ALPHA

            # Apply adaptive alpha if enabled
            if request.enable_adaptive_alpha and settings.ENABLE_ADAPTIVE_ALPHA:
                logger.info(
                    "Using adaptive alpha for hybrid search based on query type..."
                )
                try:
                    # Analyze query characteristics to determine optimal alpha
                    query_lower = contextualized_query.lower()

                    # Keyword-heavy queries (IDs, codes, exact terms) → higher alpha (favor vector search)
                    if any(
                        pattern in query_lower
                        for pattern in ["id:", "code:", "exact:", "specific:", "#"]
                    ):
                        hybrid_alpha = 0.9
                        logger.info(
                            f"Detected keyword/exact-match query → alpha={hybrid_alpha}"
                        )

                    # Semantic/conceptual queries → lower alpha (favor BM25)
                    elif any(
                        pattern in query_lower
                        for pattern in [
                            "explain",
                            "describe",
                            "what is",
                            "how does",
                            "why",
                        ]
                    ):
                        hybrid_alpha = 0.6
                        logger.info(f"Detected semantic query → alpha={hybrid_alpha}")

                    # Comparative queries → balanced alpha
                    elif any(
                        pattern in query_lower
                        for pattern in ["compare", "difference", "versus", "vs"]
                    ):
                        hybrid_alpha = 0.7
                        logger.info(
                            f"Detected comparative query → alpha={hybrid_alpha}"
                        )

                    # Default: use settings alpha
                    else:
                        logger.info(f"Using default alpha={hybrid_alpha}")

                    query_log.metadata["adaptive_alpha"] = hybrid_alpha
                except Exception as e:
                    logger.warning(
                        f"Adaptive alpha failed: {e}, using default alpha={settings.HYBRID_ALPHA}"
                    )
                    hybrid_alpha = settings.HYBRID_ALPHA

            current_retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=current_embedding_model,
                alpha=hybrid_alpha,
            )
            logger.info(f"Created HybridRetriever with alpha={hybrid_alpha}")
        else:
            current_retriever = Retriever(
                vector_store=vector_store,
                embedding_model=current_embedding_model,
            )
        logger.info(f"Created retriever with {collection_embedding_model_name}")

        # Step 4: Determine retrieval strategy
        # Use INITIAL_RETRIEVAL_K for two-stage retrieval if reranking is enabled
        if request.use_reranking:
            initial_top_k = settings.INITIAL_RETRIEVAL_K
        elif request.use_graph_rag:
            initial_top_k = request.top_k * 3
        else:
            initial_top_k = request.top_k

        # Step 5: Execute retrieval strategy
        # Priority order: Multi-Hop > Contrastive > HyDE > Multi-Query > Standard

        # Use Multi-Hop if requested (complex reasoning queries)
        if request.enable_multi_hop and settings.ENABLE_MULTI_HOP:
            logger.info(f"Using Multi-Hop Retrieval with max_hops={request.max_hops}")
            if request.metadata_filter:
                logger.info(f"Applying metadata filters: {request.metadata_filter}")
            try:
                from rag.multi_hop_retrieval import MultiHopRetriever, QueryDecomposer

                decomposer = QueryDecomposer(llm_generator=llm_generator)
                multi_hop_retriever = MultiHopRetriever(
                    retriever=current_retriever,
                    llm_generator=llm_generator,
                    decomposer=decomposer,
                    graph_rag=None,  # Graph RAG applied later as post-processing
                )

                chunks_with_scores = multi_hop_retriever.retrieve(
                    query=contextualized_query,
                    top_k=initial_top_k,
                    max_hops=request.max_hops,
                    metadata_filter=request.metadata_filter,
                )
                logger.info(
                    f"Multi-hop retrieval returned {len(chunks_with_scores)} chunks"
                )
            except Exception as e:
                logger.warning(
                    f"Multi-hop retrieval failed: {e}, falling back to standard retrieval"
                )
                chunks_with_scores = current_retriever.retrieve(
                    query=contextualized_query,
                    top_k=initial_top_k,
                    metadata_filter=request.metadata_filter,
                )

        # Use Contrastive retrieval if requested (negation queries)
        elif request.enable_contrastive:
            logger.info("Using Contrastive Retrieval for negation handling...")
            if request.metadata_filter:
                logger.info(f"Applying metadata filters: {request.metadata_filter}")
            try:
                from rag.contrastive_retrieval import (
                    ContrastiveRetriever,
                    NegationDetector,
                )

                negation_detector = NegationDetector()
                contrastive_retriever = ContrastiveRetriever(
                    retriever=current_retriever,
                    embedding_model=current_embedding_model,
                    negation_detector=negation_detector,
                )

                chunks_with_scores = contrastive_retriever.retrieve(
                    query=contextualized_query,
                    top_k=initial_top_k,
                    metadata_filter=request.metadata_filter,
                )
                logger.info(
                    f"Contrastive retrieval returned {len(chunks_with_scores)} chunks"
                )
            except Exception as e:
                logger.warning(
                    f"Contrastive retrieval failed: {e}, falling back to standard retrieval"
                )
                chunks_with_scores = current_retriever.retrieve(
                    query=contextualized_query,
                    top_k=initial_top_k,
                    metadata_filter=request.metadata_filter,
                )

        # Use HyDE if requested
        elif request.use_hyde:
            logger.info(
                f"Using HyDE with {request.num_hypothetical_docs} hypothetical docs and {request.hyde_fusion} fusion"
            )
            if request.metadata_filter:
                logger.info(f"Applying metadata filters: {request.metadata_filter}")
            # Create HyDE on-demand with collection-specific embedding model
            from rag.hyde import AdaptiveHyDE

            collection_hyde = AdaptiveHyDE(
                embedding_model=current_embedding_model,
                llm_generator=llm_generator,
                num_hypothetical_docs=request.num_hypothetical_docs,
            )
            chunks_with_scores = collection_hyde.retrieve(
                query=contextualized_query,
                vector_store=vector_store,
                top_k=initial_top_k,
                fusion_method=request.hyde_fusion,
                metadata_filter=request.metadata_filter,
            )
        # Use multi-query retrieval if requested
        elif request.use_multi_query:
            logger.info(
                f"Using multi-query retrieval with {request.num_query_variations} variations"
            )
            if request.metadata_filter:
                logger.info(f"Applying metadata filters: {request.metadata_filter}")
            multi_query_retriever = MultiQueryRetriever(
                retriever=current_retriever,
                llm_generator=llm_generator,
            )
            chunks_with_scores = multi_query_retriever.retrieve(
                query=contextualized_query,
                top_k=initial_top_k,
                num_variations=request.num_query_variations,
                metadata_filter=request.metadata_filter,
            )
        # Standard retrieval
        else:
            if request.metadata_filter:
                logger.info(f"Applying metadata filters: {request.metadata_filter}")
            chunks_with_scores = current_retriever.retrieve(
                query=contextualized_query,
                top_k=initial_top_k,
                metadata_filter=request.metadata_filter,
            )

        # Record retrieval metrics
        query_log.retrieval.initial_candidates = len(chunks_with_scores)

    if not chunks_with_scores:
        query_log.retrieval.final_candidates = 0
        return chunks_with_scores, None, None, contextualized_query, retrieval_start

    logger.info(f"Retrieved {len(chunks_with_scores)} chunks")

    # Normalize RRF scores to 0-1 range if using RRF
    # RRF scores are in range [0, 1/(RRF_K+1)] so we need to normalize them
    if request.use_hybrid_search and settings.SCORE_NORMALIZATION_METHOD == "rrf":
        if chunks_with_scores:
            # Find the maximum RRF score in the results
            max_rrf_score = max(score for _, score in chunks_with_scores)
            if max_rrf_score > 0:
                # Normalize to 0-1 range
                chunks_with_scores = [
                    (chunk, score / max_rrf_score)
                    for chunk, score in chunks_with_scores
                ]
                logger.info(
                    f"Normalized RRF scores to 0-1 range (max was {max_rrf_score:.4f})"
                )

    # Filter chunks by minimum similarity score (remove low-quality results)
    initial_count = len(chunks_with_scores)

    # Debug: Log score distribution before filtering
    if chunks_with_scores and settings.LOG_LEVEL == "DEBUG":
        scores = [score for _, score in chunks_with_scores]
        logger.debug(
            f"Score distribution - Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {sum(scores) / len(scores):.4f}"
        )
        logger.debug(f"Top 5 scores: {sorted(scores, reverse=True)[:5]}")

    chunks_with_scores = [
        (chunk, score)
        for chunk, score in chunks_with_scores
        if score >= settings.MIN_SIMILARITY_SCORE
    ]
    filtered_count = initial_count - len(chunks_with_scores)

    if filtered_count > 0:
        logger.info(
            f"Filtered out {filtered_count} chunks below {settings.MIN_SIMILARITY_SCORE:.0%} relevance threshold"
        )
        logger.info(f"Remaining chunks: {len(chunks_with_scores)}")

    # If all chunks filtered out, return early
    if not chunks_with_scores:
        logger.warning(
            f"All retrieved chunks below {settings.MIN_SIMILARITY_SCORE:.0%} relevance threshold"
        )
        query_log.retrieval.final_candidates = 0
        query_log.retrieval.filtered_by_min_score = filtered_count
        return chunks_with_scores, None, None, contextualized_query, retrieval_start

    # Step 6: Apply Graph RAG enhancement (optional, post-processing)
    if request.use_graph_rag:
        logger.info(
            f"Applying Graph RAG with expansion depth={request.graph_expansion_depth}, alpha={request.graph_alpha}"
        )

        # Create Graph RAG on-demand with collection-specific embedding model
        try:
            from rag.graph_rag import GraphRAG

            collection_graph_rag = GraphRAG(
                embedding_model=current_embedding_model,
                min_entity_mentions=2,
                use_pagerank=True,
                cache_dir="data/graph_cache",
            )

            # Build graph (lazy initialization with caching)
            logger.info("Building knowledge graph from indexed chunks...")
            all_chunks = [chunk for chunk, _ in chunks_with_scores]
            collection_graph_rag.build_graph(
                all_chunks, cache_name="global", force_rebuild=False
            )

            # Enhance retrieval with graph
            chunks_with_scores = collection_graph_rag.retrieve_with_graph(
                query=contextualized_query,
                chunks_with_scores=chunks_with_scores,
                top_k=request.top_k * 2 if request.use_reranking else request.top_k,
                expansion_depth=request.graph_expansion_depth,
                alpha=request.graph_alpha,
            )
            logger.info(f"Graph RAG returned {len(chunks_with_scores)} enhanced chunks")
        except Exception as e:
            logger.warning(
                f"Graph RAG failed: {e}, continuing without graph enhancement"
            )

    # Step 7: Rerank with deduplication (optional)
    if request.use_reranking:
        logger.info("Reranking results with deduplication...")
        rerank_start = time.time()

        # Get collection-specific reranker model
        collection_reranker_model = _get_collection_reranker_model(
            request.collection_id
        )
        logger.info(f"Loading reranker on-demand: {collection_reranker_model}")

        try:
            initial_count = len(chunks_with_scores)
            logger.info("Reranking results...")

            # Choose reranker based on settings
            if settings.RERANKER_TYPE == "llm":
                from rag.retrieval import LLMReranker

                reranker = LLMReranker(
                    llm_generator=current_llm_generator, domain=settings.RERANKER_DOMAIN
                )
            else:
                # Default Cross-Encoder
                reranker = get_or_create_reranker(
                    collection_reranker_model
                )  # Re-using existing factory for cross-encoder

            chunks_with_scores = reranker.rerank(
                query=contextualized_query,  # Use contextualized_query for reranking
                chunks_with_scores=chunks_with_scores,  # Use chunks_with_scores as input
                top_k=request.top_k,
                apply_deduplication=True,
                embedding_model=current_embedding_model,
            )

            query_log.timing.reranking_ms = (time.time() - rerank_start) * 1000
            query_log.retrieval.deduplication_removed = initial_count - len(
                chunks_with_scores
            )
            query_log.retrieval.rerank_scores = [
                float(score) for _, score in chunks_with_scores
            ]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, continuing without reranking")
            # Continue without reranking on error

    # Step 8: Apply MMR for diversity enforcement (optional)
    if request.enable_mmr and settings.ENABLE_MMR and chunks_with_scores:
        logger.info(
            f"Applying MMR diversity enforcement with lambda={request.mmr_lambda or settings.MMR_LAMBDA}"
        )
        try:
            from rag.retrieval import apply_mmr

            mmr_lambda = (
                request.mmr_lambda
                if request.mmr_lambda is not None
                else settings.MMR_LAMBDA
            )
            initial_count = len(chunks_with_scores)

            chunks_with_scores = apply_mmr(
                query_embedding=current_embedding_model.encode(contextualized_query),
                chunks_with_scores=chunks_with_scores,
                top_k=request.top_k,
                lambda_param=mmr_lambda,
                diversity_threshold=settings.MMR_DIVERSITY_THRESHOLD,
            )
            logger.info(
                f"MMR reduced results from {initial_count} to {len(chunks_with_scores)} diverse chunks"
            )
            query_log.retrieval.strategy_used.append("mmr")
            query_log.retrieval.mmr_lambda = mmr_lambda
        except Exception as e:
            logger.warning(
                f"MMR diversity enforcement failed: {e}, continuing without MMR"
            )

    # Step 9: Compress context (optional)
    if request.use_compression and compressor:
        logger.info("Compressing context...")
        chunks_with_scores = compressor.compress(
            chunks_with_scores=chunks_with_scores,
            query=contextualized_query,
        )

    # Step 10: Get collection-specific LLM model
    collection_llm_model = _get_collection_llm_model(request.collection_id)

    # Create collection-specific LLM generator if different from global
    current_llm_generator = llm_generator
    if collection_llm_model != llm_generator.model:
        logger.info(f"Using collection-specific LLM model: {collection_llm_model}")
        current_llm_generator = LLMGenerator(
            base_url=settings.LLM_BASE_URL,
            model=collection_llm_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
        )

    # Step 10: Record final retrieval metrics
    query_log.timing.retrieval_ms = (time.time() - retrieval_start) * 1000
    query_log.retrieval.final_candidates = len(chunks_with_scores)
    query_log.retrieval.chunk_ids = [chunk.chunk_id for chunk, _ in chunks_with_scores]
    query_log.retrieval.final_scores = [float(score) for _, score in chunks_with_scores]
    query_log.retrieval.strategy_used = []
    if request.use_hybrid_search:
        query_log.retrieval.strategy_used.append("hybrid_search")
        query_log.retrieval.normalization_method = settings.SCORE_NORMALIZATION_METHOD
    if request.use_multi_query:
        query_log.retrieval.strategy_used.append("multi_query")
    if request.use_hyde:
        query_log.retrieval.strategy_used.append("hyde")
    if request.use_graph_rag:
        query_log.retrieval.strategy_used.append("graph_rag")
    if request.use_reranking:
        query_log.retrieval.strategy_used.append("reranking")

    # Step 11: Store in semantic cache for future queries
    if settings.USE_CACHE and chunks_with_scores:
        try:
            semantic_cache = get_semantic_cache(current_embedding_model)
            semantic_cache.set(
                query=contextualized_query,
                chunks_with_scores=chunks_with_scores,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
            logger.debug(
                f"Stored results in semantic cache for query: '{contextualized_query[:50]}...'"
            )
        except Exception as e:
            logger.warning(f"Failed to cache results in semantic cache: {e}")

    return (
        chunks_with_scores,
        current_llm_generator,
        collection_llm_model,
        contextualized_query,
        retrieval_start,
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "RAG Pipeline API",
        "version": "2.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "service": "rag-backend",
        "version": "2.0.0",
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query endpoint - retrieves relevant documents and generates response (non-streaming)

    This endpoint uses the shared RAG pipeline and returns a complete response.
    For streaming responses, use /query/stream instead.
    """
    import time

    # Create query log
    query_log = create_query_log(request.query, request.collection_id)
    start_time = time.time()

    try:
        # Validate query text
        try:
            request.query = validate_query_text(request.query)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate top_k
        try:
            request.top_k = validate_top_k(request.top_k)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate collection_id if provided
        try:
            request.collection_id = validate_collection_id(request.collection_id)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(f"Query received: {request.query[:100]}...")

        # Check cache first
        query_cache = get_query_cache()
        cached_result = (
            query_cache.get(request.query, request.top_k)
            if settings.USE_CACHE
            else None
        )

        if cached_result:
            logger.info("Returning cached result")
            query_log.metadata["cached"] = True
            query_log.timing.total_ms = (time.time() - start_time) * 1000
            get_query_logger().log_query(query_log)
            return cached_result

        # Execute RAG pipeline (shared logic)
        (
            chunks_with_scores,
            current_llm_generator,
            collection_llm_model,
            contextualized_query,
            retrieval_start,
        ) = await _execute_rag_pipeline(request, query_log)

        # Handle empty results
        if not chunks_with_scores:
            query_log.timing.retrieval_ms = (time.time() - retrieval_start) * 1000
            query_log.timing.total_ms = (time.time() - start_time) * 1000
            get_query_logger().log_query(query_log)

            return QueryResponse(
                answer=f"No relevant documents found for your query. All retrieved documents had relevance scores below {settings.MIN_SIMILARITY_SCORE:.0%}. Try rephrasing your question or checking if the information exists in your indexed documents.",
                sources=[],
                query=request.query,
                metadata={
                    "num_chunks_retrieved": 0,
                    "min_similarity_threshold": settings.MIN_SIMILARITY_SCORE,
                },
            )

        # Generate response (non-streaming)
        if not request.stream:
            logger.info(f"Generating response with model: {collection_llm_model}...")
            generation_start = time.time()

            answer = current_llm_generator.generate_rag_response(
                query=request.query,
                chunks_with_scores=chunks_with_scores,
                system_prompt=request.system_prompt,
            )

            query_log.timing.generation_ms = (time.time() - generation_start) * 1000
            query_log.generation.model = collection_llm_model

            # Format sources
            sources = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:500] + "..."
                    if len(chunk.content) > 500
                    else chunk.content,
                    "metadata": chunk.metadata,
                    "score": float(score),
                }
                for chunk, score in chunks_with_scores
            ]

            response = QueryResponse(
                answer=answer,
                sources=sources,
                query=request.query,
                llm_model=collection_llm_model,
                collection_id=request.collection_id,
                metadata={
                    "num_chunks_retrieved": len(chunks_with_scores),
                    "multi_query": request.use_multi_query,
                    "num_query_variations": request.num_query_variations
                    if request.use_multi_query
                    else 0,
                    "hyde": request.use_hyde,
                    "hyde_fusion": request.hyde_fusion if request.use_hyde else None,
                    "num_hypothetical_docs": request.num_hypothetical_docs
                    if request.use_hyde
                    else 0,
                    "graph_rag": request.use_graph_rag,
                    "graph_expansion_depth": request.graph_expansion_depth
                    if request.use_graph_rag
                    else 0,
                    "graph_alpha": request.graph_alpha if request.use_graph_rag else 0,
                    "adaptive_fusion": request.use_adaptive_fusion,
                    "reranked": request.use_reranking,
                    "compressed": request.use_compression,
                },
            )

            # Cache result
            if settings.USE_CACHE:
                query_cache.set(request.query, request.top_k, response)

            # Log query completion
            query_log.timing.total_ms = (time.time() - start_time) * 1000
            get_query_logger().log_query(query_log)

            return response

        # Streaming response (handled separately)
        else:
            # For streaming, we return a StreamingResponse
            # Note: This is a simplified example
            raise HTTPException(
                status_code=400,
                detail="Streaming not supported in this endpoint. Use /query/stream",
            )

    except Exception as e:
        logger.error(f"Query error: {e}")

        # Log error
        query_log.error = str(e)
        query_log.timing.total_ms = (time.time() - start_time) * 1000
        get_query_logger().log_query(query_log)

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Query endpoint with streaming response with status updates

    This endpoint uses the shared RAG pipeline and returns a streaming response.
    It emits status updates as JSON followed by text response.
    For non-streaming responses, use /query instead.
    """
    import time
    import json

    try:
        logger.info(f"Streaming query received: {request.query[:100]}...")

        # Create query log for observability
        query_log = create_query_log(request.query, request.collection_id)
        start_time = time.time()

        # Stream response with status updates
        async def stream_generator():
            # Emit status: starting
            yield f"data: {json.dumps({'type': 'status', 'status': 'started', 'message': 'Starting RAG pipeline'})}\n\n"

            # Emit status: contextualizing (if conversation history exists)
            if request.conversation_history:
                yield f"data: {json.dumps({'type': 'status', 'status': 'contextualizing', 'message': 'Contextualizing query with conversation history'})}\n\n"

            # Emit status: retrieving
            strategy = []
            if request.use_hyde:
                strategy.append("HyDE")
            if request.use_multi_query:
                strategy.append(f"Multi-query({request.num_query_variations})")
            if request.use_hybrid_search:
                strategy.append("Hybrid")
            if request.use_graph_rag:
                strategy.append("GraphRAG")

            strategy_str = " + ".join(strategy) if strategy else "Standard"
            yield f"data: {json.dumps({'type': 'status', 'status': 'retrieving', 'message': f'Retrieving documents ({strategy_str})', 'strategy': strategy_str})}\n\n"

            # Execute RAG pipeline
            (
                chunks_with_scores,
                current_llm_generator,
                collection_llm_model,
                contextualized_query,
                retrieval_start,
            ) = await _execute_rag_pipeline(request, query_log)

            # Handle empty results
            if not chunks_with_scores:
                query_log.timing.total_ms = (time.time() - start_time) * 1000
                get_query_logger().log_query(query_log)
                yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'No relevant documents found (threshold: {settings.MIN_SIMILARITY_SCORE:.0%})'})}\n\n"
                yield f"No relevant documents found for your query. All retrieved documents had relevance scores below {settings.MIN_SIMILARITY_SCORE:.0%}. Try rephrasing your question or checking if the information exists in your indexed documents."
                return

            # Emit status: reranking (if enabled)
            if request.use_reranking:
                yield f"data: {json.dumps({'type': 'status', 'status': 'reranking', 'message': f'Reranking {len(chunks_with_scores)} candidates', 'candidates': len(chunks_with_scores)})}\n\n"

            # Emit status: generating
            yield f"data: {json.dumps({'type': 'status', 'status': 'generating', 'message': 'Generating response', 'model': collection_llm_model, 'chunks': len(chunks_with_scores)})}\n\n"

            # Prepare conversation history for LLM
            conversation = (
                [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.conversation_history
                ]
                if request.conversation_history
                else []
            )

            # Stream the LLM response with conversation context
            async for chunk in current_llm_generator.generate_rag_response_stream(
                query=request.query,
                chunks_with_scores=chunks_with_scores,
                system_prompt=request.system_prompt,
                conversation_history=conversation,
            ):
                # Format as SSE with JSON to preserve newlines properly
                # The frontend will parse this JSON and extract the token
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            # After streaming is complete, send RAG sources as JSON event
            if chunks_with_scores:
                # Prepare sources in frontend-expected format
                sources_data = [
                    {
                        "filename": chunk.metadata.get("filename", "Unknown"),
                        "chunkIndex": chunk.metadata.get("chunk_index", 0),
                        "pageNumber": chunk.metadata.get("page_number"),
                        "content": chunk.content[:500],  # First 500 chars
                        "score": float(score),
                    }
                    for chunk, score in chunks_with_scores[:5]  # Top 5 sources
                ]

                # Emit sources as JSON event
                yield f"\n\ndata: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

            # Emit final status: completed
            yield f"data: {json.dumps({'type': 'status', 'status': 'completed', 'message': 'Response generated successfully'})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _is_big_document(document: Any) -> bool:
    """
    Check if document is 'big' enough to warrant immediate memory cleanup
    Criteria: > 20 pages (PDF) or > 50,000 characters
    """
    # Get metadata safely
    metadata = getattr(document, "metadata", {})
    content = getattr(document, "content", "")

    page_count = metadata.get("page_count", 0)
    char_count = len(content)

    is_big = page_count > 20 or char_count > 50000
    if is_big:
        logger.info(
            f"⚠️ Identified big document: {metadata.get('filename', 'unknown')} "
            f"(pages: {page_count}, chars: {char_count})"
        )
    return is_big


def _cleanup_gpu_memory():
    """Force garbage collection and empty CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("🧹 Freed GPU memory cache")


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection_title: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    chunking_strategy: str = Form("semantic"),
    embedding_model_name: Optional[str] = Form(None),
    embedding_provider: Optional[str] = Form(
        None
    ),  # "ollama", "huggingface", or "auto"
    use_ollama_embedding: bool = Form(False),
    use_hybrid_embedding: bool = Form(False),
    use_adaptive_fusion: bool = Form(False),
    structural_weight: float = Form(0.3),
    reranker_model: Optional[str] = Form(None),  # NEW: per-collection reranker
):
    """
    Ingest a single file (PDF or Markdown)
    Supports custom embedding model selection
    """
    try:
        # Validate file upload
        try:
            validate_file_upload(file)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        logger.info(f"Ingesting file: {safe_filename}")

        # Validate chunk parameters
        try:
            chunk_size, chunk_overlap = validate_chunk_params(chunk_size, chunk_overlap)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate collection_id if provided
        try:
            collection_title = validate_collection_id(collection_title)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(
            f"Parameters: collection_title={collection_title}, llm_model={llm_model}, embedding_model={embedding_model_name}"
        )

        # Validate embedding compatibility if adding to existing collection
        effective_embedding_model = embedding_model_name or settings.EMBEDDING_MODEL
        is_compatible, error_msg, _ = _validate_embedding_compatibility(
            collection_id=collection_title,
            new_embedding_model=effective_embedding_model,
        )

        if not is_compatible:
            logger.error(f"Embedding compatibility check failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Read file content
        content = await file.read()

        # Determine file type
        doc_type = "pdf" if file.filename.endswith(".pdf") else "markdown"

        # Ingest document
        document = ingestor.ingest_from_buffer(
            content=content,
            filename=file.filename,
            doc_type=doc_type,
        )

        logger.info(f"Document ingested: {len(document.content)} characters")
        if len(document.content) < 50:
            logger.warning(
                f"Document content preview (first 50 chars): {document.content[:50]}"
            )
        else:
            logger.info(f"Document content preview: {document.content[:200]}...")

        # Implement Smart Chunking Strategy
        if chunking_strategy == "smart":
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in [".md", ".markdown"]:
                # Check for headers to decide between Markdown and Semantic
                has_headers = bool(re.search(r"^#+\s", document.content, re.MULTILINE))
                if has_headers:
                    chunking_strategy = "markdown"
                    logger.info(
                        f"Smart chunking: Selected 'markdown' strategy for {file.filename} (headers found)"
                    )
                else:
                    chunking_strategy = "semantic"
                    logger.info(
                        f"Smart chunking: Selected 'semantic' strategy for {file.filename} (no headers found)"
                    )
            elif ext in [
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".c",
                ".h",
            ]:
                chunking_strategy = "recursive"
                logger.info(
                    f"Smart chunking: Selected 'recursive' strategy for code file {file.filename}"
                )
            else:
                chunking_strategy = "semantic"
                logger.info(
                    f"Smart chunking: Selected 'semantic' strategy for {file.filename}"
                )

        # Create chunker
        chunker = create_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
        )

        # Chunk document
        chunks = chunker.chunk_document(
            content=document.content,
            metadata=document.metadata,
        )

        logger.info(f"Created {len(chunks)} chunks")

        # Check if any chunks were created
        if not chunks or len(chunks) == 0:
            error_msg = f"No chunks created from document '{file.filename}'. "
            if doc_type == "pdf":
                error_msg += "This may be because the PDF is image-based/scanned and OCR is not available. "
                error_msg += "Install Tesseract OCR: 'sudo apt-get install tesseract-ocr tesseract-ocr-eng'"
            else:
                error_msg += "The document may be empty or contain no extractable text."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Apply contextual embeddings if enabled
        if settings.USE_CONTEXTUAL_EMBEDDINGS:
            logger.info("Generating document summary for contextual embeddings...")

            # Get document summarizer
            summarizer = get_document_summarizer(llm_generator=llm_generator)

            # Generate summary
            document_title = document.metadata.get("filename", file.filename)
            doc_summary = summarizer.summarize(
                content=document.content,
                title=document_title,
            )

            # Apply contextual embeddings
            chunks = apply_contextual_embeddings(
                chunks=chunks,
                document_summary=doc_summary,
                document_title=document_title,
            )

            logger.info(
                f"Applied contextual embeddings with summary: {doc_summary[:100]}..."
            )

        # Generate embeddings with specified model
        model_desc = f"{embedding_model_name or 'default model'}"
        if use_adaptive_fusion:
            model_desc += " (adaptive fusion)"
        elif use_hybrid_embedding:
            model_desc += f" (hybrid, {structural_weight:.0%} structural)"
        if settings.USE_CONTEXTUAL_EMBEDDINGS:
            model_desc += " (contextual)"
        logger.info(f"Generating embeddings with {model_desc}...")

        # Get or create embedding model (cached to prevent memory leaks)
        logger.info(f"📊 Getting embedding model for {len(chunks)} chunks...")
        embedding_model_to_use = get_or_create_embedding_model(
            model_name=embedding_model_name,
            provider=embedding_provider,
            use_ollama=use_ollama_embedding,
            use_hybrid=use_hybrid_embedding,
            use_adaptive=use_adaptive_fusion,
            structural_weight=structural_weight,
        )
        logger.info(
            f"🚀 Starting embedding encoding on device: {embedding_model_to_use.device}"
        )

        # Use content_for_embedding if available (contextual embeddings)
        texts = [chunk.content_for_embedding or chunk.content for chunk in chunks]
        embeddings = embedding_model_to_use.encode(texts, show_progress=True)

        logger.info(f"✅ Finished encoding {len(chunks)} chunks into embeddings")

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        # Get actual embedding dimension from the embeddings we just created
        actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else None

        # Free GPU memory after embedding encoding
        del embeddings
        del texts

        if _is_big_document(document):
            logger.info(
                f"🧹 Triggering memory cleanup for big document: {document.metadata.get('filename')}"
            )
            _cleanup_gpu_memory()

        # Get or create collection first (to get dimension)
        _get_or_create_collection(
            collection_id=collection_title,
            title=collection_title,
            llm_model=llm_model or settings.LLM_MODEL,
            embedding_model=embedding_model_name or settings.EMBEDDING_MODEL,
            embedding_dimension=actual_dimension,
        )

        # Get or create vector store for this collection
        logger.info(f"Getting vector store for collection: {collection_title}")
        vector_store = _get_or_create_vector_store(collection_title, actual_dimension)

        # Index chunks in vector store
        logger.info("Indexing chunks in vector store...")
        vector_store.add_chunks(chunks)
        _update_collection_stats(
            collection_title, file.filename, len(chunks), file_size=len(content)
        )

        # Update collection LLM model if provided
        if llm_model:
            _update_collection_model(collection_title, llm_model)

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {file.filename}",
            stats={
                "filename": file.filename,
                "num_chunks": len(chunks),
                "doc_type": doc_type,
                "embedding_model": embedding_model_name or settings.EMBEDDING_MODEL,
            },
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(
    url: str = Form(...),
    collection_title: str = Form(...),
    llm_model: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    chunking_strategy: str = Form("semantic"),
    embedding_model_name: Optional[str] = Form(None),
    embedding_provider: Optional[str] = Form(None),
    use_ollama_embedding: bool = Form(False),
    use_hybrid_embedding: bool = Form(False),
    use_adaptive_fusion: bool = Form(False),
    structural_weight: float = Form(0.3),
    reranker_model: Optional[str] = Form(None),
):
    """
    Ingest content from a URL
    """
    try:
        logger.info(f"Ingesting URL: {url}")

        # Validate embedding compatibility if adding to existing collection
        effective_embedding_model = embedding_model_name or settings.EMBEDDING_MODEL
        is_compatible, error_msg, _ = _validate_embedding_compatibility(
            collection_id=collection_title,
            new_embedding_model=effective_embedding_model,
        )

        if not is_compatible:
            logger.error(f"Embedding compatibility check failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Ingest URL
        document = ingestor.ingest_url(url)

        # Create chunker
        chunker = create_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
        )

        # Chunk document
        chunks = chunker.chunk_document(
            content=document.content,
            metadata=document.metadata,
        )

        logger.info(f"Created {len(chunks)} chunks")

        # Check if any chunks were created
        if not chunks or len(chunks) == 0:
            error_msg = f"No chunks created from URL '{url}'. The page may be empty or contain no extractable text."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Apply contextual embeddings if enabled
        if settings.USE_CONTEXTUAL_EMBEDDINGS:
            logger.info("Generating document summary for contextual embeddings...")

            # Get document summarizer
            summarizer = get_document_summarizer(llm_generator=llm_generator)

            # Generate summary
            document_title = document.metadata.get("title", url)
            doc_summary = summarizer.summarize(
                content=document.content,
                title=document_title,
            )

            # Apply contextual embeddings
            chunks = apply_contextual_embeddings(
                chunks=chunks,
                document_summary=doc_summary,
                document_title=document_title,
            )

            logger.info(
                f"Applied contextual embeddings with summary: {doc_summary[:100]}..."
            )

        # Get or create embedding model (cached to prevent memory leaks)
        logger.info(f"📊 Getting embedding model for {len(chunks)} chunks...")
        embedding_model_to_use = get_or_create_embedding_model(
            model_name=embedding_model_name,
            provider=embedding_provider,
            use_ollama=use_ollama_embedding,
            use_hybrid=use_hybrid_embedding,
            use_adaptive=use_adaptive_fusion,
            structural_weight=structural_weight,
        )
        logger.info(
            f"🚀 Starting embedding encoding on device: {embedding_model_to_use.device}"
        )

        # Use content_for_embedding if available (contextual embeddings)
        texts = [chunk.content_for_embedding or chunk.content for chunk in chunks]
        embeddings = embedding_model_to_use.encode(texts, show_progress=True)

        logger.info(f"✅ Finished encoding {len(chunks)} chunks into embeddings")

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        # Get actual embedding dimension from the embeddings we just created
        actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else None

        # Free GPU memory after embedding encoding
        del embeddings
        del texts

        # Get or create collection first
        _get_or_create_collection(
            collection_id=collection_title,
            title=collection_title,
            llm_model=llm_model or settings.LLM_MODEL,
            embedding_model=embedding_model_name or settings.EMBEDDING_MODEL,
            embedding_dimension=actual_dimension,
            reranker_model=reranker_model,
        )

        # Get or create vector store for this collection
        logger.info(f"Getting vector store for collection: {collection_title}")
        vector_store = _get_or_create_vector_store(collection_title, actual_dimension)

        # Index in vector store
        vector_store.add_chunks(chunks)
        _update_collection_stats(
            collection_title,
            url,
            len(chunks),
            file_size=len(document.content.encode("utf-8")),
        )

        if llm_model:
            _update_collection_model(collection_title, llm_model)

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {url}",
            stats={
                "filename": url,
                "num_chunks": len(chunks),
                "doc_type": "url",
                "embedding_model": embedding_model_name or settings.EMBEDDING_MODEL,
            },
        )

    except Exception as e:
        logger.error(f"URL ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest all files from corpus directory (background task)
    """
    try:

        def ingest_task():
            logger.info(f"Ingesting directory: {settings.CORPUS_DIR}")

            # Ingest all documents
            documents = ingestor.ingest_directory(
                directory=settings.CORPUS_DIR,
                recursive=request.recursive,
            )

            # Create chunker
            chunker = create_chunker(
                strategy=request.chunking_strategy,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                min_chunk_size=settings.MIN_CHUNK_SIZE,
            )

            # Get or create embedding model (cached)
            embedding_model_to_use = get_or_create_embedding_model()

            # Process each document
            total_chunks = 0
            for doc in documents:
                # Chunk
                chunks = chunker.chunk_document(doc.content, doc.metadata)

                # Embed
                texts = [chunk.content for chunk in chunks]
                embeddings = embedding_model_to_use.encode(texts)

                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding.tolist()

                # Free GPU memory only if document is big
                del embeddings
                del texts

                # Add to vector store (use "default" collection for directory ingestion)
                collection_id = "default"
                vector_store = _get_or_create_vector_store(
                    collection_id, settings.EMBEDDING_DIMENSION
                )
                vector_store.add_chunks(chunks)
                total_chunks += len(chunks)

            logger.info(
                f"✅ Ingested {len(documents)} documents, {total_chunks} chunks"
            )

            # Always perform final cleanup at the end of batch
            _cleanup_gpu_memory()

        # Run in background
        background_tasks.add_task(ingest_task)

        return IngestResponse(
            success=True,
            message=f"Ingestion started for {settings.CORPUS_DIR}",
            stats={"status": "processing"},
        )

    except Exception as e:
        logger.error(f"Directory ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        # Aggregate vector store stats from all collections
        total_chunks = 0
        total_files = 0

        for collection_id, store in vector_stores.items():
            try:
                stats = store.get_stats()
                total_chunks += stats.get("total_chunks", 0)
                total_files += stats.get("total_files", 0)
            except Exception as e:
                logger.warning(
                    f"Failed to get stats for collection {collection_id}: {e}"
                )

        # Get cache stats
        cache = get_query_cache().cache
        cache_stats = {}
        if hasattr(cache, "stats"):
            cache_stats = cache.stats()

        # Get RAG config (LLM model)
        rag_config = _load_rag_config()

        return StatsResponse(
            total_chunks=total_chunks,
            total_files=total_files,
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=rag_config.get("llm_model"),
            vector_store_type=settings.VECTOR_STORE_TYPE,
            cache_stats=cache_stats,
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear all caches"""
    try:
        from rag.cache import get_cache

        cache = get_cache()
        cache.clear()

        return {"success": True, "message": "Cache cleared"}

    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/graph")
async def clear_graph_cache(cache_name: Optional[str] = None):
    """Clear graph cache"""
    try:
        if graph_rag:
            graph_rag.clear_cache(cache_name)
            return {
                "success": True,
                "message": f"Graph cache {'(' + cache_name + ') ' if cache_name else ''}cleared",
            }
        else:
            return {"success": False, "message": "Graph RAG not initialized"}

    except Exception as e:
        logger.error(f"Graph cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """List all RAG collections"""
    try:
        collections = _load_collections()

        # Migration: If no collections but we have data in vector store, create a default collection
        # Disable this after initial migration to allow users to delete all collections
        ENABLE_AUTO_MIGRATION = (
            False  # Set to True only if you need to migrate old data
        )

        if len(collections) == 0 and ENABLE_AUTO_MIGRATION and len(vector_stores) > 0:
            # Try to get stats from "default" collection store if it exists
            default_store = vector_stores.get("default")
            if default_store:
                vs_stats = default_store.get_stats()
                total_chunks = vs_stats.get("total_chunks", 0)
                total_files = vs_stats.get("total_files", 0)

                if total_chunks > 0:
                    # Load legacy config
                    rag_config = _load_rag_config()
                    llm_model = rag_config.get("llm_model", settings.LLM_MODEL)

                    # Create default collection from existing data
                    from datetime import datetime

                    default_collection = {
                        "id": "default",
                        "title": "My Documents",
                        "llm_model": llm_model,
                        "embedding_model": settings.EMBEDDING_MODEL,
                        "file_count": total_files,
                        "chunk_count": total_chunks,
                        "created_at": datetime.utcnow().isoformat(),
                        "files": [],
                    }
                    collections["default"] = default_collection
                _save_collections(collections)
                logger.info(
                    f"Migrated existing data to default collection: {total_files} files, {total_chunks} chunks"
                )

        collection_list = list(collections.values())

        return {"collections": collection_list, "total": len(collection_list)}

    except Exception as e:
        logger.error(f"List collections error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
    """Get a specific RAG collection"""
    try:
        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        return collections[collection_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/collections/{collection_id}")
async def update_collection(
    collection_id: str,
    title: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
):
    """
    Update a RAG collection's configuration

    Args:
        collection_id: Collection ID
        title: New collection title/name (optional)
        llm_model: LLM model for generation (optional)
        embedding_model: Embedding model (optional, metadata only)
        reranker_model: Reranker model (optional)

    Returns:
        Updated collection data
    """
    try:
        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        collection = collections[collection_id]
        updates = []

        # Update title if provided
        if title:
            collection["name"] = title
            updates.append(f"title to '{title}'")
            logger.info(f"Updated collection {collection_id} title to {title}")

        # Update LLM model if provided
        if llm_model:
            collection["llm_model"] = llm_model
            updates.append(f"LLM model to '{llm_model}'")
            logger.info(f"Updated collection {collection_id} LLM model to {llm_model}")

        # Update embedding model if provided (metadata only, existing chunks unchanged)
        if embedding_model:
            collection["embedding_model"] = embedding_model
            updates.append(f"embedding model to '{embedding_model}'")
            logger.info(
                f"Updated collection {collection_id} embedding model to {embedding_model}"
            )

        # Update reranker model if provided
        if reranker_model:
            collection["reranker_model"] = reranker_model
            updates.append(f"reranker model to '{reranker_model}'")
            logger.info(
                f"Updated collection {collection_id} reranker model to {reranker_model}"
            )

        # Save changes
        _save_collections(collections)

        update_message = ", ".join(updates) if updates else "no changes"

        return {
            "success": True,
            "message": f"Collection {collection_id} updated: {update_message}",
            "collection": collection,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_id}/documents/{filename}")
async def delete_document_from_collection(collection_id: str, filename: str):
    """
    Delete a specific document from a collection

    Args:
        collection_id: Collection ID
        filename: Document filename to delete

    Returns:
        Success message with deletion details
    """
    try:
        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        collection = collections[collection_id]
        files = collection.get("files", [])

        # Check if file exists in collection
        if filename not in files:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found in collection '{collection_id}'",
            )

        # Delete chunks from vector store for this collection
        try:
            vector_store = _get_or_create_vector_store(collection_id)
            vector_store.delete_by_filename(filename)
            logger.info(
                f"Deleted chunks for file: {filename} from collection {collection_id}"
            )
        except Exception as e:
            logger.error(f"Failed to delete chunks for file {filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document from vector store: {str(e)}",
            )

        # Remove file from collection metadata
        collection["files"].remove(filename)

        # Update document count
        if "document_count" in collection:
            collection["document_count"] = max(0, collection["document_count"] - 1)

        # Save updated collection
        _save_collections(collections)

        logger.info(f"Deleted document {filename} from collection {collection_id}")

        return {
            "success": True,
            "message": f"Document '{filename}' deleted from collection '{collection_id}'",
            "collection": collection,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_id}/urls")
async def list_collection_urls(collection_id: str):
    """
    List all URLs ingested in a collection

    Args:
        collection_id: Collection ID

    Returns:
        List of URLs with metadata
    """
    try:
        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        collection = collections[collection_id]

        # Get URLs from collection metadata
        urls = collection.get("urls", [])

        # Return URLs with additional info
        return {
            "success": True,
            "collection_id": collection_id,
            "collection_name": collection.get("name", collection_id),
            "urls": urls,
            "url_count": len(urls),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List collection URLs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a RAG collection and its associated vector store data"""
    try:
        from pathlib import Path
        import shutil

        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        collection = collections[collection_id]
        deleted_files = []

        # Delete vector store for this collection
        if collection_id in vector_stores:
            try:
                vector_store = vector_stores[collection_id]

                # Close the vector store connection (for SQLite)
                if hasattr(vector_store, "close"):
                    vector_store.close()

                # Remove from cache
                del vector_stores[collection_id]

                logger.info(f"Deleted vector store for collection: {collection_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to delete vector store for collection {collection_id}: {e}"
                )
        else:
            logger.info(f"No vector store found for collection {collection_id}")

        # Delete vector store files based on type
        data_dir = Path(settings.DATA_DIR)

        # 1. Delete FAISS files
        if settings.VECTOR_STORE_TYPE == "faiss":
            faiss_dir = data_dir / "faiss"
            faiss_index = faiss_dir / f"{collection_id}.faiss"
            faiss_metadata = faiss_dir / f"{collection_id}.metadata.pkl"

            for file_path in [faiss_index, faiss_metadata]:
                if file_path.exists():
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    logger.info(f"Deleted FAISS file: {file_path}")

        # 2. Delete LanceDB files
        elif settings.VECTOR_STORE_TYPE == "lancedb":
            lancedb_dir = data_dir / "lancedb"
            collection_table = lancedb_dir / f"{collection_id}.lance"

            if collection_table.exists():
                shutil.rmtree(collection_table)
                deleted_files.append(str(collection_table))
                logger.info(f"Deleted LanceDB table: {collection_table}")

        # 3. Delete SQLite files
        elif settings.VECTOR_STORE_TYPE == "sqlite":
            base_dir = Path(
                getattr(settings, "SQLITE_VECTOR_DB_PATH", "./data/rag.db")
            ).parent
            db_path = base_dir / f"rag-{collection_id}.db"

            if db_path.exists():
                db_path.unlink()
                deleted_files.append(str(db_path))
                logger.info(f"Deleted SQLite database: {db_path}")

        # 4. Delete cache files related to this collection
        cache_dir = data_dir / "cache"

        # Delete semantic cache entries for this collection
        semantic_cache_file = cache_dir / "semantic_cache.json"
        if semantic_cache_file.exists():
            try:
                import json

                with open(semantic_cache_file, "r") as f:
                    cache_data = json.load(f)

                # Filter out entries for this collection
                original_count = len(cache_data)
                cache_data = [
                    entry
                    for entry in cache_data
                    if not any(
                        chunk.get("chunk", {}).get("collection_id") == collection_id
                        for chunk in entry.get("chunks_with_scores", [])
                    )
                ]

                if len(cache_data) < original_count:
                    with open(semantic_cache_file, "w") as f:
                        json.dump(cache_data, f)
                    logger.info(
                        f"Cleaned {original_count - len(cache_data)} entries from semantic cache"
                    )
            except Exception as e:
                logger.warning(f"Failed to clean semantic cache: {e}")

        # Delete collection-specific cache database
        cache_db = cache_dir / f"cache-{collection_id}.db"
        if cache_db.exists():
            cache_db.unlink()
            deleted_files.append(str(cache_db))
            logger.info(f"Deleted cache database: {cache_db}")

        # Delete summary cache files for this collection
        summaries_dir = cache_dir / "summaries"
        if summaries_dir.exists():
            for summary_file in summaries_dir.glob(f"*{collection_id}*"):
                summary_file.unlink()
                deleted_files.append(str(summary_file))
                logger.info(f"Deleted summary cache: {summary_file}")

        # Delete collection metadata
        _delete_collection(collection_id)
        logger.info(f"Deleted collection metadata: {collection_id}")

        # Delete corpus files associated with this collection
        corpus_dir = Path(settings.CORPUS_DIR)
        if corpus_dir.exists():
            for corpus_file in corpus_dir.glob(f"*{collection_id}*"):
                if corpus_file.is_file():
                    corpus_file.unlink()
                    deleted_files.append(str(corpus_file))
                    logger.info(f"Deleted corpus file: {corpus_file}")

        # Delete any temporary files related to this collection
        temp_dir = Path("/tmp")
        if temp_dir.exists():
            for temp_file in temp_dir.glob(f"**/*{collection_id}*"):
                if temp_file.is_file():
                    try:
                        temp_file.unlink()
                        deleted_files.append(str(temp_file))
                        logger.info(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete temporary file {temp_file}: {e}"
                        )

        # Delete index files if they exist
        index_dir = Path(settings.INDEX_DIR)
        if index_dir.exists():
            for index_file in index_dir.glob(f"*{collection_id}*"):
                if index_file.is_file():
                    index_file.unlink()
                    deleted_files.append(str(index_file))
                    logger.info(f"Deleted index file: {index_file}")

        return {
            "success": True,
            "message": f"Collection {collection_id} deleted successfully",
            "deleted_files": deleted_files,
            "files_count": len(deleted_files),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_all_chunks_from_store(vector_store, collection_id: str) -> List:
    """
    Helper function to get all chunks from any vector store type
    """
    from .vectordb import LanceDBStore, FAISSStore, ChromaDBStore, SQLiteVectorStore
    from .chunking import Chunk
    import json

    chunks = []

    try:
        if isinstance(vector_store, LanceDBStore):
            # LanceDB: use to_pandas()
            if vector_store.table is not None:
                df = vector_store.table.to_pandas()
                for _, row in df.iterrows():
                    try:
                        metadata = (
                            json.loads(row["metadata"])
                            if isinstance(row["metadata"], str)
                            else row["metadata"]
                        )
                        chunk = Chunk(
                            chunk_id=row["chunk_id"],
                            content=row["content"],
                            metadata=metadata,
                            embedding=np.array(row["vector"])
                            if "vector" in row
                            else None,
                        )
                        chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Error parsing chunk: {e}")
                        continue

        elif isinstance(vector_store, FAISSStore):
            # FAISS: retrieve embeddings and metadata
            if (
                hasattr(vector_store, "chunks_metadata")
                and vector_store.index.ntotal > 0
            ):
                # First, try to use stored embeddings (saved separately for visualization)
                embeddings_available = False
                all_embeddings = None

                if (
                    hasattr(vector_store, "stored_embeddings")
                    and vector_store.stored_embeddings is not None
                ):
                    all_embeddings = vector_store.stored_embeddings
                    embeddings_available = True
                    logger.info(
                        f"Using {len(all_embeddings)} stored embeddings from FAISS"
                    )
                else:
                    # Try to reconstruct from index (works for some index types)
                    num_vectors = vector_store.index.ntotal
                    dimension = vector_store.dimension

                    try:
                        # Check if index supports reconstruction
                        if hasattr(vector_store.index, "reconstruct"):
                            all_embeddings = np.zeros(
                                (num_vectors, dimension), dtype=np.float32
                            )
                            for i in range(num_vectors):
                                try:
                                    all_embeddings[i] = vector_store.index.reconstruct(
                                        i
                                    )
                                    embeddings_available = True
                                except:
                                    embeddings_available = False
                                    break

                        # Alternative: if index is IndexFlatL2 or IndexFlatIP, access xb directly
                        if not embeddings_available and hasattr(
                            vector_store.index, "xb"
                        ):
                            import faiss

                            all_embeddings = faiss.vector_to_array(
                                vector_store.index.xb
                            ).reshape(num_vectors, dimension)
                            embeddings_available = True
                    except Exception as e:
                        logger.warning(
                            f"Could not reconstruct embeddings from FAISS index: {e}"
                        )
                        embeddings_available = False

                # Create chunks with metadata
                for idx, metadata_entry in enumerate(vector_store.chunks_metadata):
                    embedding = (
                        all_embeddings[idx]
                        if embeddings_available
                        and all_embeddings is not None
                        and idx < len(all_embeddings)
                        else None
                    )
                    chunk = Chunk(
                        chunk_id=metadata_entry["chunk_id"],
                        content=metadata_entry["content"],
                        metadata=metadata_entry["metadata"],
                        embedding=embedding,
                    )
                    chunks.append(chunk)

                if not embeddings_available:
                    logger.warning(
                        "FAISS embeddings not available. To enable UMAP visualization, re-index your documents or switch to LanceDB vector store."
                    )

            # Fallback: try chunks attribute
            elif hasattr(vector_store, "chunks"):
                chunks = list(vector_store.chunks.values())

        elif isinstance(vector_store, ChromaDBStore):
            # ChromaDB: get all from collection
            if vector_store.collection is not None:
                results = vector_store.collection.get(
                    include=["embeddings", "metadatas", "documents"]
                )
                for i, chunk_id in enumerate(results["ids"]):
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        content=results["documents"][i],
                        metadata=results["metadatas"][i],
                        embedding=np.array(results["embeddings"][i])
                        if results.get("embeddings")
                        else None,
                    )
                    chunks.append(chunk)

        elif isinstance(vector_store, SQLiteVectorStore):
            # SQLite: query all from database
            if hasattr(vector_store, "conn"):
                cursor = vector_store.conn.cursor()
                cursor.execute(
                    "SELECT chunk_id, content, metadata, embedding FROM chunks"
                )
                rows = cursor.fetchall()
                for row in rows:
                    metadata = json.loads(row[2]) if row[2] else {}
                    embedding = pickle.loads(row[3]) if row[3] else None
                    chunk = Chunk(
                        chunk_id=row[0],
                        content=row[1],
                        metadata=metadata,
                        embedding=embedding,
                    )
                    chunks.append(chunk)

    except Exception as e:
        logger.error(f"Error getting chunks from vector store: {e}")

    return chunks


@app.get("/collections/{collection_id}/graph")
async def get_collection_graph(collection_id: str):
    """
    Get Graph RAG visualization data for a collection

    Returns:
        nodes: List of nodes (entities and chunks)
        edges: List of edges (relations)
        stats: Graph statistics
    """
    try:
        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        collection = collections[collection_id]

        # Load or build graph for this collection
        vector_store = _get_or_create_vector_store(collection_id, collection)
        embedding_model = create_embedding_model(
            model_name=collection.get("embedding_model", settings.EMBEDDING_MODEL),
            use_ollama=collection.get("use_ollama_embedding", False),
        )

        # Initialize Graph RAG
        graph_rag = GraphRAG(
            embedding_model=embedding_model,
            min_entity_mentions=2,
            max_entities_per_chunk=10,
            use_pagerank=True,
            cache_dir=os.path.join(settings.DATA_DIR, "graph_cache"),
        )

        # Get chunks from vector store
        chunks = _get_all_chunks_from_store(vector_store, collection_id)

        if not chunks:
            return {
                "nodes": [],
                "edges": [],
                "stats": {"num_entities": 0, "num_chunks": 0, "num_edges": 0},
            }

        # Build or load graph
        graph_rag.build_graph(chunks, cache_name=collection_id)

        # Extract nodes (entities + chunks)
        nodes = []

        # Add entity nodes
        for entity_name, entity in graph_rag.entities.items():
            importance = graph_rag.graph.nodes[entity_name].get("importance", 0.0)
            nodes.append(
                {
                    "id": entity_name,
                    "type": "entity",
                    "entity_type": entity.entity_type,
                    "label": entity_name,
                    "mentions": len(entity.chunk_ids),
                    "importance": float(importance),
                    "embedding": entity.embedding.tolist()
                    if entity.embedding is not None
                    else None,
                }
            )

        # Add chunk nodes (sample to avoid overload)
        max_chunks_to_show = 100
        chunk_nodes = [
            {
                "id": chunk.chunk_id,
                "type": "chunk",
                "label": chunk.content[:50] + "...",
                "content": chunk.content[:200],
                "embedding": chunk.embedding[:128].tolist()
                if chunk.embedding is not None
                else None,  # Truncate for size
            }
            for chunk in chunks[:max_chunks_to_show]
        ]
        nodes.extend(chunk_nodes)

        # Extract edges
        edges = []
        for source, target, data in graph_rag.graph.edges(data=True):
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "type": data.get("type", "related"),
                    "weight": float(data.get("weight", 1.0)),
                }
            )

        # Get stats
        stats = graph_rag.get_stats()

        return {"nodes": nodes, "edges": edges, "stats": stats}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get collection graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_id}/umap")
async def get_collection_umap(
    collection_id: str,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
):
    """
    Get UMAP projection of collection embeddings for visualization

    Args:
        collection_id: Collection ID
        n_components: Number of dimensions (2 or 3)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric (cosine, euclidean, etc.)

    Returns:
        points: List of projected points with metadata
        labels: Chunk labels/content
        stats: UMAP projection statistics
    """
    try:
        # Try importing UMAP
        try:
            import umap
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="UMAP not installed. Install with: pip install umap-learn",
            )

        collections = _load_collections()

        if collection_id not in collections:
            raise HTTPException(status_code=404, detail="Collection not found")

        collection = collections[collection_id]

        # Load vector store
        vector_store = _get_or_create_vector_store(collection_id, collection)

        # Get all chunks
        chunks = _get_all_chunks_from_store(vector_store, collection_id)

        if not chunks:
            return {
                "points": [],
                "labels": [],
                "stats": {"num_points": 0, "n_components": n_components},
            }

        # Extract embeddings
        embeddings = []
        labels = []
        metadata = []

        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                labels.append(chunk.content[:100])
                metadata.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.metadata.get("source", "unknown"),
                        "file_name": chunk.metadata.get("file_name", "unknown"),
                    }
                )

        if not embeddings:
            return {
                "points": [],
                "labels": [],
                "stats": {"num_points": 0, "n_components": n_components},
            }

        # Convert to numpy array
        import numpy as np

        embeddings_array = np.array(embeddings)

        logger.info(f"Running UMAP projection on {len(embeddings)} embeddings...")

        # Run UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, len(embeddings) - 1),
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )

        projection = reducer.fit_transform(embeddings_array)

        # Prepare response
        points = [
            {"coordinates": proj.tolist(), "label": label, "metadata": meta}
            for proj, label, meta in zip(projection, labels, metadata)
        ]

        return {
            "points": points,
            "labels": labels,
            "stats": {
                "num_points": len(points),
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get collection UMAP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_available_models(
    check_availability: bool = False,
    model_type: Optional[str] = None,
):
    """
    Get available models with metadata

    Args:
        check_availability: Check actual model availability (slower, queries Ollama/HF)
        model_type: Filter by type: "embedding", "reranker", "all" (default: "all")

    Returns:
        Dictionary with embedding and reranker models and their metadata
    """
    try:
        from rag.model_registry import (
            EMBEDDING_MODELS,
            RERANKER_MODELS,
            OLLAMA_EMBEDDING_MODELS,
            to_dict,
        )
        from rag.model_validation import get_model_validator

        response = {}

        # Get embedding models if requested
        if model_type in [None, "all", "embedding"]:
            embedding_models = []

            # HuggingFace embedding models
            for shortcut, info in EMBEDDING_MODELS.items():
                model_data = to_dict(info)
                model_data["shortcut"] = shortcut
                model_data["available"] = None

                # Check availability if requested
                if check_availability:
                    validator = get_model_validator()
                    is_valid, error_msg = validator.validate_embedding_model(
                        info.name, check_availability=True
                    )
                    model_data["available"] = is_valid
                    model_data["error"] = error_msg if not is_valid else None

                embedding_models.append(model_data)

            # Ollama embedding models - query actual installed models
            try:
                import requests

                ollama_response = requests.get(
                    f"{settings.LLM_BASE_URL}/api/tags", timeout=5
                )

                if ollama_response.status_code == 200:
                    ollama_data = ollama_response.json()
                    ollama_model_names = [
                        model["name"] for model in ollama_data.get("models", [])
                    ]

                    # Filter for embedding models (contain "embed" in name)
                    ollama_embedding_names = [
                        name for name in ollama_model_names if "embed" in name.lower()
                    ]

                    # Add each Ollama embedding model
                    for model_name in ollama_embedding_names:
                        # Check if it's in our registry first
                        info = None
                        for shortcut, registry_info in OLLAMA_EMBEDDING_MODELS.items():
                            if (
                                registry_info.name == model_name
                                or model_name.startswith(registry_info.name)
                            ):
                                info = registry_info
                                break

                        if info:
                            # Use registry info if available
                            model_data = to_dict(info)
                            model_data["shortcut"] = (
                                None  # No shortcut for Ollama models
                            )
                        else:
                            # Create model data for unknown Ollama embedding model
                            model_data = {
                                "name": model_name,
                                "dimension": 768,  # Default, may vary
                                "max_seq_length": 2048,  # Default for Ollama
                                "model_type": "ollama",
                                "description": f"Ollama embedding model - {model_name}",
                                "size_mb": None,
                                "shortcut": None,
                            }

                        model_data["available"] = True  # If listed, it's available
                        embedding_models.append(model_data)
                else:
                    logger.warning(
                        f"Failed to fetch Ollama models: {ollama_response.status_code}"
                    )
                    # Fallback to static registry if Ollama query fails
                    for shortcut, info in OLLAMA_EMBEDDING_MODELS.items():
                        model_data = to_dict(info)
                        model_data["shortcut"] = shortcut
                        model_data["available"] = False
                        embedding_models.append(model_data)

            except Exception as e:
                logger.warning(f"Could not query Ollama for models: {e}")
                # Fallback to static registry
                for shortcut, info in OLLAMA_EMBEDDING_MODELS.items():
                    model_data = to_dict(info)
                    model_data["shortcut"] = shortcut
                    model_data["available"] = None
                    embedding_models.append(model_data)

            response["embedding_models"] = embedding_models

        # Get reranker models if requested
        if model_type in [None, "all", "reranker"]:
            reranker_models = []

            for shortcut, info in RERANKER_MODELS.items():
                model_data = to_dict(info)
                model_data["shortcut"] = shortcut
                model_data["available"] = None

                # Check availability if requested
                if check_availability:
                    validator = get_model_validator()
                    is_valid, error_msg = validator.validate_reranker_model(info.name)
                    model_data["available"] = is_valid
                    model_data["error"] = error_msg if not is_valid else None

                reranker_models.append(model_data)

            response["reranker_models"] = reranker_models

        # Add current configuration
        response["current_config"] = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
        }

        # Add metadata
        response["metadata"] = {
            "total_embedding_models": len(response.get("embedding_models", [])),
            "total_reranker_models": len(response.get("reranker_models", [])),
            "availability_checked": check_availability,
        }

        return response

    except Exception as e:
        logger.error(f"Get models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/models")
async def health_models():
    """
    Get health status of models and model infrastructure

    Returns:
        Status of Ollama connection, cached models, and model readiness
    """
    try:
        from rag.model_validation import get_model_validator

        health_info = {
            "status": "ok",
            "ollama": {},
            "cached_models": {},
            "reranker_cache": {},
        }

        # Check Ollama connection
        validator = get_model_validator()
        try:
            response = requests.get(
                f"{settings.LLM_BASE_URL}/api/tags",
                timeout=settings.OLLAMA_HEALTH_CHECK_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                ollama_models = data.get("models", [])

                health_info["ollama"] = {
                    "status": "connected",
                    "url": settings.LLM_BASE_URL,
                    "models_available": len(ollama_models),
                    "models": [
                        {
                            "name": model.get("name"),
                            "size": model.get("size"),
                            "modified_at": model.get("modified_at"),
                        }
                        for model in ollama_models
                    ],
                }
            else:
                health_info["ollama"] = {
                    "status": "error",
                    "url": settings.LLM_BASE_URL,
                    "error": f"HTTP {response.status_code}",
                }
                health_info["status"] = "degraded"

        except Exception as e:
            health_info["ollama"] = {
                "status": "disconnected",
                "url": settings.LLM_BASE_URL,
                "error": str(e),
            }
            health_info["status"] = "degraded"

        # Check reranker cache
        global _reranker_cache
        if _reranker_cache:
            health_info["reranker_cache"] = {
                "cached_models": list(_reranker_cache.keys()),
                "count": len(_reranker_cache),
            }
        else:
            health_info["reranker_cache"] = {"cached_models": [], "count": 0}

        # Check if HuggingFace cache directory exists and list cached models
        try:
            import os
            from pathlib import Path

            # Default HuggingFace cache location
            cache_home = os.getenv("HF_HOME") or os.path.join(
                Path.home(), ".cache", "huggingface"
            )
            hub_cache = os.path.join(cache_home, "hub")

            if os.path.exists(hub_cache):
                # List cached models (model directories start with "models--")
                cached_dirs = [
                    d for d in os.listdir(hub_cache) if d.startswith("models--")
                ]
                cached_model_names = [
                    d.replace("models--", "").replace("--", "/") for d in cached_dirs
                ]

                health_info["cached_models"] = {
                    "cache_path": hub_cache,
                    "models": cached_model_names,
                    "count": len(cached_model_names),
                }
            else:
                health_info["cached_models"] = {
                    "cache_path": hub_cache,
                    "models": [],
                    "count": 0,
                    "note": "Cache directory does not exist yet",
                }

        except Exception as e:
            health_info["cached_models"] = {"error": str(e)}

        # Add current configuration
        health_info["config"] = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
            "preload_on_startup": settings.PRELOAD_MODELS_ON_STARTUP,
            "preload_models": settings.PRELOAD_MODELS,
        }

        return health_info

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/pull")
async def pull_model(
    model_name: str,
    model_type: str = "embedding",
    auto_download: bool = True,
):
    """
    Pull/download a model

    Args:
        model_name: Model name to pull
        model_type: "embedding", "reranker", or "llm"
        auto_download: Whether to actually download (vs just check)

    Returns:
        Status of the download operation
    """
    try:
        from rag.model_validation import ensure_model_available

        logger.info(f"Pull request for {model_type} model: {model_name}")

        # Validate model_type
        if model_type not in ["embedding", "reranker", "llm"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type '{model_type}'. Must be 'embedding', 'reranker', or 'llm'",
            )

        # Attempt to ensure model is available
        is_available, error_msg = ensure_model_available(
            model_name=model_name, model_type=model_type, auto_download=auto_download
        )

        if is_available:
            return {
                "success": True,
                "message": f"Model '{model_name}' is available",
                "model_name": model_name,
                "model_type": model_type,
                "status": "ready",
            }
        else:
            # Model not available and download failed/was not attempted
            if not auto_download:
                return {
                    "success": False,
                    "message": f"Model '{model_name}' is not available",
                    "model_name": model_name,
                    "model_type": model_type,
                    "status": "not_available",
                    "error": error_msg,
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download model '{model_name}': {error_msg}",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pull model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/queries")
async def get_query_stats(
    limit: int = 100,
    collection_id: Optional[str] = None,
):
    """
    Get recent query logs and statistics

    Args:
        limit: Maximum number of logs to return (default: 100)
        collection_id: Optional collection ID to filter logs

    Returns:
        Query logs and aggregate statistics
    """
    try:
        query_logger = get_query_logger()

        # Get logs
        if collection_id:
            logs = query_logger.get_logs_by_collection(collection_id, limit)
        else:
            logs = query_logger.get_recent_logs(limit)

        # Get statistics
        stats = query_logger.get_stats()

        return {
            "logs": logs,
            "stats": stats,
            "total_logs": len(logs),
        }

    except Exception as e:
        logger.error(f"Query stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    Evaluate RAG pipeline performance

    Args:
        request: Evaluation request with test dataset

    Returns:
        Evaluation metrics
    """
    try:
        logger.info(
            f"Evaluation request received with {len(request.test_dataset)} samples"
        )

        # Parse evaluation samples
        samples = []
        for item in request.test_dataset:
            samples.append(
                EvaluationSample(
                    query=item["query"],
                    relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                    expected_answer=item.get("expected_answer"),
                    metadata=item.get("metadata", {}),
                )
            )

        # Get or create vector store for the collection
        if request.collection_id:
            collection = _load_collections().get(request.collection_id)
            embedding_dimension = (
                collection.get("embedding_dimension") if collection else None
            )
            vector_store = _get_or_create_vector_store(
                request.collection_id, embedding_dimension
            )
            logger.info(f"Using vector store for collection: {request.collection_id}")
        else:
            vector_store = _get_or_create_vector_store(
                "default", settings.EMBEDDING_DIMENSION
            )
            logger.info("No collection specified, using default collection")

        # Get collection-specific embedding model (cached)
        collection_embedding_model_name = _get_collection_embedding_model(
            request.collection_id
        )
        current_embedding_model = get_or_create_embedding_model(
            model_name=collection_embedding_model_name
        )

        # Create retriever
        if settings.USE_HYBRID_SEARCH:
            current_retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=current_embedding_model,
                alpha=settings.HYBRID_ALPHA,
            )
        else:
            current_retriever = Retriever(
                vector_store=vector_store,
                embedding_model=current_embedding_model,
            )

        # Create evaluator
        evaluator = RAGEvaluator(
            retriever=current_retriever,
            llm_generator=llm_generator if request.evaluate_generation else None,
        )

        # Evaluate retrieval
        retrieval_metrics = evaluator.evaluate_retrieval(
            samples=samples,
            k_values=request.k_values,
        )

        # Optionally evaluate generation
        generation_metrics = None
        if request.evaluate_generation and llm_generator:
            logger.info("Evaluating generation quality...")

            # Generate answers for evaluation
            generated_answers = []
            contexts = []

            for sample in samples:
                # Retrieve context
                chunks_with_scores = current_retriever.retrieve(
                    sample.query, top_k=settings.FINAL_TOP_K
                )

                # Generate answer
                answer = llm_generator.generate_rag_response(
                    query=sample.query,
                    chunks_with_scores=chunks_with_scores,
                )

                generated_answers.append(answer)
                contexts.append([chunk for chunk, _ in chunks_with_scores])

            # Evaluate
            generation_metrics = evaluator.evaluate_generation(
                samples=samples,
                generated_answers=generated_answers,
                contexts=contexts,
            )

        # Return results
        from dataclasses import asdict

        return EvaluationResponse(
            retrieval_metrics=asdict(retrieval_metrics),
            generation_metrics=asdict(generation_metrics)
            if generation_metrics
            else None,
            sample_count=len(samples),
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Configuration Management
# ============================================================================


@app.get("/config/env")
async def get_env_config():
    """
    Get current runtime configuration (settings + .env overrides)

    Returns:
        Dictionary of effective settings grouped by category
    """
    try:
        # Get actual runtime settings
        config = settings.model_dump()

        # Structure it to match the frontend's expected categories
        # This maps the flat settings structure to the categorized structure
        structured_config = {
            "server": {
                "HOST": config.get("HOST"),
                "PORT": config.get("PORT"),
                "LOG_LEVEL": config.get("LOG_LEVEL"),
            },
            "paths": {
                "DATA_DIR": config.get("DATA_DIR"),
                "CORPUS_DIR": config.get("CORPUS_DIR"),
                "INDEX_DIR": config.get("INDEX_DIR"),
            },
            "vector_store": {
                "VECTOR_STORE_TYPE": config.get("VECTOR_STORE_TYPE"),
                "LANCEDB_URI": config.get("LANCEDB_URI"),
                "CHROMA_HOST": config.get("CHROMA_HOST"),
                "CHROMA_PORT": config.get("CHROMA_PORT"),
            },
            "embedding": {
                "EMBEDDING_MODEL": config.get("EMBEDDING_MODEL"),
                "EMBEDDING_DIMENSION": config.get("EMBEDDING_DIMENSION"),
                "EMBEDDING_DEVICE": config.get("EMBEDDING_DEVICE"),
                "EMBEDDING_BATCH_SIZE": config.get("EMBEDDING_BATCH_SIZE"),
                "NORMALIZE_EMBEDDINGS": config.get("NORMALIZE_EMBEDDINGS"),
            },
            "llm": {
                "LLM_BASE_URL": config.get("LLM_BASE_URL"),
                "LLM_MODEL": config.get("LLM_MODEL"),
                "LLM_TEMPERATURE": config.get("LLM_TEMPERATURE"),
                "LLM_MAX_TOKENS": config.get("LLM_MAX_TOKENS"),
            },
            "chunking": {
                "CHUNK_SIZE": config.get("CHUNK_SIZE"),
                "CHUNK_OVERLAP": config.get("CHUNK_OVERLAP"),
                "CHUNKING_STRATEGY": config.get("CHUNKING_STRATEGY"),
            },
            "retrieval": {
                "TOP_K": config.get("TOP_K"),
                "MIN_SIMILARITY_SCORE": config.get("MIN_SIMILARITY_SCORE"),
                "USE_HYBRID_SEARCH": config.get("USE_HYBRID_SEARCH"),
                "HYBRID_ALPHA": config.get("HYBRID_ALPHA"),
            },
            "reranking": {
                "USE_RERANKING": config.get("USE_RERANKING"),
                "RERANKER_MODEL": config.get("RERANKER_MODEL"),
                "RERANKER_TOP_K": config.get("RERANKER_TOP_K"),
            },
            "compression": {
                "USE_COMPRESSION": config.get("USE_COMPRESSION"),
                "MAX_CONTEXT_TOKENS": config.get("MAX_CONTEXT_TOKENS"),
                "COMPRESSION_RATIO": config.get("COMPRESSION_RATIO"),
            },
            "other": {
                # Fallback for any other keys not mapped
                k: v
                for k, v in config.items()
                if not any(
                    k in sub
                    for sub in [
                        "HOST",
                        "PORT",
                        "LOG_LEVEL",
                        "DATA_DIR",
                        "CORPUS_DIR",
                        "INDEX_DIR",
                        "VECTOR_STORE_TYPE",
                        "LANCEDB_URI",
                        "CHROMA_HOST",
                        "CHROMA_PORT",
                        "EMBEDDING_MODEL",
                        "EMBEDDING_DIMENSION",
                        "EMBEDDING_DEVICE",
                        "EMBEDDING_BATCH_SIZE",
                        "NORMALIZE_EMBEDDINGS",
                        "LLM_BASE_URL",
                        "LLM_MODEL",
                        "LLM_TEMPERATURE",
                        "LLM_MAX_TOKENS",
                        "CHUNK_SIZE",
                        "CHUNK_OVERLAP",
                        "CHUNKING_STRATEGY",
                        "TOP_K",
                        "MIN_SIMILARITY_SCORE",
                        "USE_HYBRID_SEARCH",
                        "HYBRID_ALPHA",
                        "USE_RERANKING",
                        "RERANKER_MODEL",
                        "RERANKER_TOP_K",
                        "USE_COMPRESSION",
                        "MAX_CONTEXT_TOKENS",
                        "COMPRESSION_RATIO",
                    ]
                )
            },
        }

        # Determine current .env path for information purposes
        env_path = Path(__file__).parent.parent / ".env"

        return {
            "config": structured_config,
            "flat_config": config,  # Also return flat config for easier frontend mapping
            "file_path": str(env_path) if env_path.exists() else "Using defaults",
        }

    except Exception as e:
        logger.error(f"Failed to read config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/env")
async def update_env_config(updates: Dict[str, Dict[str, str]]):
    """
    Update .env configuration

    NOTE: This endpoint is suitable for local/internal use only.
    For public deployment, add authentication and authorization.

    Args:
        updates: Dictionary of category -> {key: value} updates

    Returns:
        Success message
    """
    try:
        env_path = Path(__file__).parent.parent / ".env"

        if not env_path.exists():
            raise HTTPException(status_code=404, detail=".env file not found")

        # Read current file
        with open(env_path, "r") as f:
            lines = f.readlines()

        # Build flat update dict
        flat_updates = {}
        for category_updates in updates.values():
            flat_updates.update(category_updates)

        # Update lines
        new_lines = []
        for line in lines:
            stripped = line.strip()

            # Keep comments and empty lines
            if stripped.startswith("#") or not stripped:
                new_lines.append(line)
            elif "=" in stripped:
                key = stripped.split("=", 1)[0].strip()

                # Update if key is in updates
                if key in flat_updates:
                    # Preserve any inline comment
                    if "#" in stripped:
                        comment = stripped.split("#", 1)[1]
                        new_lines.append(f"{key}={flat_updates[key]}  # {comment}\n")
                    else:
                        new_lines.append(f"{key}={flat_updates[key]}\n")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Write back to file
        with open(env_path, "w") as f:
            f.writelines(new_lines)

        logger.info(f"Updated .env configuration with {len(flat_updates)} changes")

        return {
            "success": True,
            "message": f"Updated {len(flat_updates)} configuration values",
            "note": "Restart the backend for changes to take effect",
        }

    except Exception as e:
        logger.error(f"Failed to update .env config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
