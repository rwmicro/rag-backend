"""
FastAPI Server for RAG Pipeline
Modern Python backend with streaming support
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, Tuple
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timezone
import uvicorn
from loguru import logger
import sys
import json
import os
import time
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
from rag.collections_db import CollectionsDB
from rag.job_store import JobStore, JobStatus
from rag.doc_registry import DocRegistry

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
)

# FastAPI app + CORS are constructed further down, after the lifespan handler
# is defined (it depends on module-level helpers declared below).


# ============================================================================
# Pydantic Models (moved to rag/schemas.py)
# ============================================================================

from rag.schemas import (
    ConversationMessage,
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    StatsResponse,
    AsyncIngestResponse,
    IngestFolderRequest,
)


# ============================================================================
# Helper Functions
# ============================================================================

# Store singletons live in rag/app_state.py — re-export with the legacy
# underscore names so existing callers in this module keep working.
from rag.app_state import (
    COLLECTIONS_FILE,
    get_collections_db as _get_collections_db,
    get_job_store as _get_job_store,
    get_doc_registry as _get_doc_registry,
)


def _load_collections() -> Dict[str, Any]:
    """Load all collections metadata"""
    return _get_collections_db().load_all()


def _save_collections(collections: Dict[str, Any]):
    """Save collections metadata - now writes each collection individually"""
    db = _get_collections_db()
    for cid, data in collections.items():
        db.upsert(cid, data)


def _get_or_create_collection(
    collection_id: str,
    title: str,
    llm_model: str,
    embedding_model: str,
    embedding_dimension: Optional[int] = None,
    reranker_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Get or create a collection"""
    db = _get_collections_db()
    existing = db.get(collection_id)
    if existing:
        return existing

    from rag.model_registry import get_model_dimension

    if embedding_dimension is None:
        embedding_dimension = get_model_dimension(embedding_model, "embedding")

    collection = {
        "id": collection_id,
        "title": title,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "reranker_model": reranker_model or settings.RERANKER_MODEL,
        "file_count": 0,
        "chunk_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "file_metadata": {},
    }
    db.upsert(collection_id, collection)
    return collection


def _update_collection_stats(
    collection_id: str, filename: str, num_chunks: int, file_size: int = 0
):
    """Update collection statistics after adding a file"""
    db = _get_collections_db()
    collection = db.get(collection_id)
    if not collection:
        return

    if "file_metadata" not in collection:
        collection["file_metadata"] = {}

    collection["file_metadata"][filename] = {
        "size": file_size,
        "chunks": num_chunks,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    if filename not in collection["files"]:
        collection["files"].append(filename)
        collection["file_count"] = len(collection["files"])

    collection["chunk_count"] = collection.get("chunk_count", 0) + num_chunks
    db.upsert(collection_id, collection)


def _delete_collection(collection_id: str):
    """Delete a collection"""
    _get_collections_db().delete(collection_id)


def _get_collection_llm_model(collection_id: Optional[str] = None) -> str:
    """Get LLM model for a specific collection, or use global default"""
    if collection_id:
        collection = _get_collections_db().get(collection_id)
        if collection:
            return collection.get("llm_model", settings.LLM_MODEL)
    return settings.LLM_MODEL


def _get_collection_embedding_model(collection_id: Optional[str] = None) -> str:
    """Get embedding model for a specific collection, or use global default"""
    if collection_id:
        collection = _get_collections_db().get(collection_id)
        if collection:
            return collection.get("embedding_model", settings.EMBEDDING_MODEL)
    return settings.EMBEDDING_MODEL


def _get_collection_reranker_model(collection_id: Optional[str] = None) -> str:
    """Get reranker model for a specific collection, or use global default"""
    if collection_id:
        collection = _get_collections_db().get(collection_id)
        if collection:
            return collection.get("reranker_model", settings.RERANKER_MODEL)
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
    collection = _get_collections_db().get(collection_id)

    if collection is None:
        return True, None, None

    existing_dimension = collection.get("embedding_dimension")
    existing_model = collection.get("embedding_model")

    if existing_dimension is None:
        logger.warning(f"Collection {collection_id} has no dimension metadata, skipping validation")
        return True, None, None

    from rag.model_registry import get_model_dimension, list_models_by_dimension

    new_dimension = get_model_dimension(new_embedding_model, "embedding")

    if new_dimension is None:
        logger.warning(f"Unknown model '{new_embedding_model}', cannot validate dimension")
        return True, None, existing_dimension

    if new_dimension != existing_dimension:
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
    db = _get_collections_db()
    collection = db.get(collection_id)
    if not collection:
        return False
    collection["llm_model"] = llm_model
    db.upsert(collection_id, collection)
    logger.info(f"Updated collection {collection_id} LLM model to {llm_model}")
    return True


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

# LRU-bounded model caches — prevents VRAM exhaustion when many models are tried.
# Kept intentionally small since each model (especially rerankers) can be 1-4GB on GPU.
MAX_EMBEDDING_CACHE = 3
MAX_RERANKER_CACHE = 2

_reranker_cache: "OrderedDict[str, Any]" = OrderedDict()
_embedding_model_cache: "OrderedDict[str, Any]" = OrderedDict()


def _evict_lru(cache: "OrderedDict[str, Any]", max_size: int, cache_name: str) -> None:
    """Evict least-recently-used entries until cache fits within max_size.

    Drops references and frees GPU memory so unloaded models actually release VRAM.
    """
    while len(cache) > max_size:
        key, evicted = cache.popitem(last=False)
        logger.info(f"♻️  LRU evict {cache_name}: {key}")
        try:
            del evicted
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
        _embedding_model_cache.move_to_end(cache_key)
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
        _evict_lru(_embedding_model_cache, MAX_EMBEDDING_CACHE, "embedding")
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
        _reranker_cache.move_to_end(model_name)
        logger.debug(f"Using cached reranker: {model_name}")
        return _reranker_cache[model_name]

    # Create new reranker instance
    logger.info(f"Loading reranker on-demand: {model_name}")
    try:
        from rag.retrieval import Reranker

        reranker_instance = Reranker(model_name=model_name)

        # Cache the instance
        _reranker_cache[model_name] = reranker_instance
        _evict_lru(_reranker_cache, MAX_RERANKER_CACHE, "reranker")
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup + shutdown hooks (replaces deprecated @app.on_event)."""
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

    ensure_directories()

    logger.info("Embedding models will be loaded on-demand per collection")
    embedding_model = None
    retriever = None

    logger.info("Vector stores will be created on-demand per collection")
    vector_stores = {}

    if settings.USE_RERANKING:
        logger.info("Loading default reranker...")
        try:
            reranker = get_or_create_reranker(settings.RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"Failed to load default reranker: {e}")
            reranker = None

    if settings.USE_COMPRESSION:
        compressor = ContextCompressor(max_tokens=settings.MAX_CONTEXT_TOKENS)

    logger.info("Initializing LLM generator...")
    llm_model = settings.LLM_MODEL
    logger.info(f"Using LLM model: {llm_model}")
    llm_generator = LLMGenerator(
        base_url=settings.LLM_BASE_URL,
        model=llm_model,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )

    graph_rag = None
    hyde = None
    ingestor = DocumentIngestor()

    if settings.ENABLE_AUTO_ROUTING:
        logger.info("Initializing query router...")
        query_router = create_query_router(llm_generator=llm_generator)
        logger.info(f"✓ Query router initialized (mode: {settings.ROUTER_MODE})")

    if settings.PRELOAD_MODELS_ON_STARTUP:
        logger.info("Preloading models on startup...")
        from rag.model_validation import preload_models

        preload_results = preload_models()
        successful = sum(1 for v in preload_results.values() if v)
        logger.info(f"✓ Preloaded {successful}/{len(preload_results)} models")

    logger.info("Preloading default embedding model...")
    try:
        embedding_model = get_or_create_embedding_model(model_name=settings.EMBEDDING_MODEL)
        logger.info(f"✓ Preloaded embedding model: {settings.EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to preload embedding model: {e}")
        embedding_model = None

    _get_collections_db()
    job_store = _get_job_store()
    _get_doc_registry()
    logger.info("✓ Database singletons initialized")

    # Sweep orphaned jobs from a previous run (server crashed mid-ingest)
    orphaned = job_store.mark_orphans_failed()
    if orphaned:
        logger.warning(f"Marked {orphaned} orphaned ingest job(s) as failed")

    logger.info("Initializing advanced RAG components...")

    if settings.ENABLE_QUERY_CLASSIFICATION:
        query_classifier = QueryClassifier(
            use_llm=settings.QUERY_CLASSIFIER_USE_LLM,
            llm_client=llm_generator if settings.QUERY_CLASSIFIER_USE_LLM else None,
        )
        logger.info("✓ Query classifier initialized")

    if settings.ENABLE_FALLBACK_STRATEGIES or settings.ENABLE_CORRECTIVE_RAG:
        confidence_evaluator = ConfidenceEvaluator(
            high_threshold=settings.RETRIEVAL_CONFIDENCE_HIGH,
            medium_threshold=settings.RETRIEVAL_CONFIDENCE_MEDIUM,
            minimum_threshold=settings.RETRIEVAL_CONFIDENCE_MINIMUM,
            enable_fallback=settings.ENABLE_FALLBACK_STRATEGIES,
        )
        logger.info("✓ Confidence evaluator initialized")

    answer_verifier = AnswerVerifier(
        llm_generator=llm_generator,
        embedding_model=None,
        grounding_threshold=0.7,
        verification_threshold=0.6,
    )
    logger.info("✓ Answer verifier initialized")

    feedback_logger = FeedbackLogger(
        db_path=settings.FEEDBACK_DB_PATH,
        enable_logging=True,
        log_queries=True,
        log_retrievals=True,
        log_generations=True,
        log_feedback=True,
    )
    logger.info("✓ Feedback logger initialized")

    metadata_store = MetadataStore(db_path=settings.METADATA_DB_PATH)
    logger.info("✓ Metadata store initialized")

    logger.info("✅ RAG Pipeline API v2.0 ready")

    yield

    logger.info("🛑 Shutting down RAG Pipeline API")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Advanced RAG system with hybrid search, reranking, and compression",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — configure CORS_ORIGINS in .env for production
cors_origins = (
    settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Domain routers (the rest of the endpoints remain inline below until they
# are extracted alongside the pipeline logic).
from rag.routers import (
    health_router,
    jobs_router,
    cache_router,
    config_router,
    stats_router,
    models_router,
    evaluate_router,
)

app.include_router(health_router)
app.include_router(jobs_router)
app.include_router(cache_router)
app.include_router(config_router)
app.include_router(stats_router)
app.include_router(models_router)
app.include_router(evaluate_router)


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
        collection = _get_collections_db().get(request.collection_id)
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
                effective_model = request.llm_model_override or collection_llm_model
                effective_timeout = request.llm_timeout or settings.LLM_TIMEOUT
                if effective_model != llm_generator.model or effective_timeout != settings.LLM_TIMEOUT:
                    current_llm_generator = LLMGenerator(
                        base_url=request.llm_base_url_override or settings.LLM_BASE_URL,
                        model=effective_model,
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.LLM_MAX_TOKENS,
                        timeout=effective_timeout,
                    )

                return (
                    cached_results,
                    current_llm_generator,
                    collection_llm_model,
                    contextualized_query,
                    retrieval_start,
                )

        except Exception:
            logger.exception("Semantic cache lookup failed, continuing with normal retrieval")
            query_log.metadata["semantic_cache_error"] = True

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
                except Exception:
                    logger.exception(f"Adaptive alpha failed, using default alpha={settings.HYBRID_ALPHA}")
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
            except Exception:
                logger.exception("Multi-hop retrieval failed, falling back to standard retrieval")
                query_log.metadata["multi_hop_error"] = True
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
            except Exception:
                logger.exception("Contrastive retrieval failed, falling back to standard retrieval")
                query_log.metadata["contrastive_error"] = True
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

        # Corrective RAG: grade the result set and retry with a rewritten query
        # if confidence is low. Only runs in the standard branch where
        # `current_retriever` is well-defined; multilingual branch keeps its
        # own pipeline.
        if (
            request.enable_confidence_evaluation
            and settings.ENABLE_CORRECTIVE_RAG
            and confidence_evaluator is not None
        ):
            try:
                from rag.corrective import CorrectiveLoop

                _metadata_filter = request.metadata_filter
                _retriever = current_retriever

                def _retry_retrieve(rewritten: str):
                    return _retriever.retrieve(
                        query=rewritten,
                        top_k=initial_top_k,
                        metadata_filter=_metadata_filter,
                    )

                loop = CorrectiveLoop(
                    evaluator=confidence_evaluator,
                    llm_generator=llm_generator,
                    max_attempts=settings.CORRECTIVE_MAX_ATTEMPTS,
                    trigger_level=settings.CORRECTIVE_TRIGGER_LEVEL,
                    merge_method=settings.CORRECTIVE_MERGE_METHOD,
                )
                chunks_with_scores, corrective_trace = loop.run(
                    query=contextualized_query,
                    initial_results=chunks_with_scores,
                    retry_retrieve=_retry_retrieve,
                    top_k=initial_top_k,
                )
                query_log.metadata["corrective_rag"] = corrective_trace.to_dict()
                if corrective_trace.triggered:
                    logger.info(
                        f"Corrective loop: strategy={corrective_trace.final_strategy}, "
                        f"attempts={len(corrective_trace.attempts)}"
                    )
                # Re-record candidate count after a possible merge
                query_log.retrieval.initial_candidates = len(chunks_with_scores)
            except Exception:
                logger.exception("Corrective RAG loop failed, continuing with initial results")
                query_log.metadata["corrective_rag_error"] = True

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

            graph_cache_name = request.collection_id or "default"
            collection_graph_rag = GraphRAG(
                embedding_model=current_embedding_model,
                min_entity_mentions=2,
                use_pagerank=True,
                cache_dir=settings.GRAPH_CACHE_DIR,
            )

            # Try to load persisted graph first
            cache_loaded = collection_graph_rag.load_cache(graph_cache_name)
            if not cache_loaded:
                # Fall back to building from current chunks
                logger.info("No persisted graph found, building from retrieved chunks...")
                all_chunks = [chunk for chunk, _ in chunks_with_scores]
                collection_graph_rag.build_graph(
                    all_chunks, cache_name=graph_cache_name, force_rebuild=False
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

            # Capture traversal trace so downstream endpoints can surface it
            if collection_graph_rag.last_traversal is not None:
                query_log.metadata["graph_traversal"] = collection_graph_rag.last_traversal
        except Exception:
            logger.exception("Graph RAG failed, continuing without graph enhancement")
            query_log.metadata["graph_rag_error"] = True

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
        except Exception:
            logger.exception("Reranking failed, continuing without reranking")
            query_log.metadata["reranking_error"] = True

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
        except Exception:
            logger.exception("MMR diversity enforcement failed, continuing without MMR")
            query_log.metadata["mmr_error"] = True

    # Step 9: Compress context (optional)
    if request.use_compression and compressor:
        logger.info("Compressing context...")
        chunks_with_scores = compressor.compress(
            chunks_with_scores=chunks_with_scores,
            query=contextualized_query,
        )

    # Step 10: Get collection-specific LLM model (with per-request overrides)
    collection_llm_model = _get_collection_llm_model(request.collection_id)

    # Per-request overrides take precedence
    effective_model = request.llm_model_override or collection_llm_model
    effective_base_url = request.llm_base_url_override or settings.LLM_BASE_URL
    effective_timeout = request.llm_timeout or settings.LLM_TIMEOUT

    # Reuse global generator if nothing changed
    current_llm_generator = llm_generator
    if (effective_model != llm_generator.model or
            effective_base_url != llm_generator.base_url or
            effective_timeout != settings.LLM_TIMEOUT or
            request.llm_provider is not None):
        logger.info(f"Using per-request LLM: model={effective_model}, url={effective_base_url}")
        from rag.llm_provider import create_llm_provider
        provider = create_llm_provider(
            base_url=effective_base_url,
            model=effective_model,
            timeout=effective_timeout,
            provider_hint=request.llm_provider,
        )
        current_llm_generator = LLMGenerator(
            base_url=effective_base_url,
            model=effective_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=effective_timeout,
        )
        current_llm_generator.provider = provider
    collection_llm_model = effective_model

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
        except Exception:
            logger.exception("Failed to cache results in semantic cache")

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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query endpoint - retrieves relevant documents and generates response (non-streaming)

    This endpoint uses the shared RAG pipeline and returns a complete response.
    For streaming responses, use /query/stream instead, or pass stream=true.
    """
    # Delegate to the streaming endpoint when the client asks for streaming
    if request.stream:
        return await query_stream(request)

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

        # Generate response (non-streaming; stream=true was delegated at entry)
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

            # Emit graph traversal metadata (only populated when Graph RAG ran)
            graph_traversal = query_log.metadata.get("graph_traversal")
            if graph_traversal:
                yield f"data: {json.dumps({'type': 'graph_traversal', 'traversal': graph_traversal})}\n\n"

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

        # Document deduplication check
        effective_collection = collection_title or "default"
        file_hash = DocRegistry.compute_hash(content)
        registry = _get_doc_registry()

        if settings.SKIP_DUPLICATE_INGESTION and registry.is_duplicate(effective_collection, safe_filename, file_hash):
            logger.info(f"Skipping duplicate document: {safe_filename} (hash match)")
            return IngestResponse(
                success=True,
                message=f"Document '{safe_filename}' unchanged (identical content), skipping re-ingestion.",
                stats={"filename": safe_filename, "skipped": True, "num_chunks": 0},
            )

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

        # Incremental diff: if this file was previously ingested, only embed
        # chunks whose content hash changed, and drop chunks that no longer
        # appear. Falls back to full re-ingest for legacy records.
        chunks, ingest_diff, final_chunk_ids, final_chunk_hashes = registry.plan_ingest(
            effective_collection, safe_filename, chunks
        )
        if ingest_diff:
            logger.info(
                f"Incremental ingest for {safe_filename}: "
                f"{len(ingest_diff.unchanged)} unchanged, "
                f"{len(ingest_diff.added)} new, "
                f"{len(ingest_diff.removed)} removed"
            )

        # Short-circuit: only removals (no new chunks to embed).
        if ingest_diff and not chunks and ingest_diff.removed:
            vector_store = _get_or_create_vector_store(collection_title, None)
            vector_store.delete_by_chunk_ids(ingest_diff.removed)
            registry.upsert(
                effective_collection,
                safe_filename,
                file_hash,
                final_chunk_ids,
                final_chunk_hashes,
            )
            _update_collection_stats(
                collection_title, file.filename, -len(ingest_diff.removed), file_size=len(content)
            )
            if llm_model:
                _update_collection_model(collection_title, llm_model)
            return IngestResponse(
                success=True,
                message=f"Updated {file.filename}: {len(ingest_diff.removed)} chunks removed, nothing new to embed",
                stats={
                    "filename": file.filename,
                    "num_chunks": len(final_chunk_ids),
                    "removed": len(ingest_diff.removed),
                    "doc_type": doc_type,
                },
            )

        # Short-circuit: diff was fully unchanged — refresh file_hash only.
        if ingest_diff and not chunks and not ingest_diff.removed:
            registry.upsert(
                effective_collection,
                safe_filename,
                file_hash,
                final_chunk_ids,
                final_chunk_hashes,
            )
            return IngestResponse(
                success=True,
                message=f"Document '{safe_filename}' chunks unchanged; refreshed file hash.",
                stats={"filename": safe_filename, "num_chunks": len(final_chunk_ids), "skipped": True},
            )

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

        # Drop stale chunks before adding the new ones so chunk_id re-use is safe.
        if ingest_diff and ingest_diff.removed:
            vector_store.delete_by_chunk_ids(ingest_diff.removed)

        # Index chunks in vector store
        logger.info("Indexing chunks in vector store...")
        vector_store.add_chunks(chunks)
        _update_collection_stats(
            collection_title, file.filename, len(chunks), file_size=len(content)
        )

        # Register document hash for future deduplication
        registry.upsert(
            effective_collection,
            safe_filename,
            file_hash,
            final_chunk_ids,
            final_chunk_hashes,
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


@app.post("/ingest/file/async", response_model=AsyncIngestResponse)
async def ingest_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_title: Optional[str] = Form(None),
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
    Async file ingestion - returns immediately with a job_id.
    Poll GET /ingest/jobs/{job_id} to check status.
    """
    try:
        validate_file_upload(file)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Read file bytes immediately (cannot read after returning)
    content = await file.read()
    filename = file.filename
    job_id = _get_job_store().create_job()

    def _run_ingest_job():
        try:
            _get_job_store().update(job_id, JobStatus.RUNNING, progress=0.05)
            safe_fname = sanitize_filename(filename)
            effective_col = collection_title or "default"
            file_hash = DocRegistry.compute_hash(content)

            # Dedup check
            if settings.SKIP_DUPLICATE_INGESTION and _get_doc_registry().is_duplicate(effective_col, safe_fname, file_hash):
                _get_job_store().update(
                    job_id, JobStatus.COMPLETED, progress=1.0,
                    result={"filename": safe_fname, "skipped": True, "num_chunks": 0}
                )
                return

            doc_type = "pdf" if filename.endswith(".pdf") else "markdown"
            document = ingestor.ingest_from_buffer(content=content, filename=filename, doc_type=doc_type)
            _get_job_store().update(job_id, JobStatus.RUNNING, progress=0.2)

            chunker = create_chunker(
                strategy=chunking_strategy, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap, min_chunk_size=settings.MIN_CHUNK_SIZE,
            )
            chunks = chunker.chunk_document(content=document.content, metadata=document.metadata)
            if not chunks:
                raise ValueError(f"No chunks created from '{filename}'")
            _get_job_store().update(job_id, JobStatus.RUNNING, progress=0.4)

            registry = _get_doc_registry()
            chunks, ingest_diff, final_chunk_ids, final_chunk_hashes = registry.plan_ingest(
                effective_col, safe_fname, chunks
            )
            if ingest_diff:
                logger.info(
                    f"Incremental async ingest for {safe_fname}: "
                    f"{len(ingest_diff.unchanged)} unchanged, "
                    f"{len(ingest_diff.added)} new, "
                    f"{len(ingest_diff.removed)} removed"
                )

            actual_dimension = None
            if chunks:
                embedding_model_to_use = get_or_create_embedding_model(
                    model_name=embedding_model_name,
                    provider=embedding_provider,
                    use_ollama=use_ollama_embedding,
                    use_hybrid=use_hybrid_embedding,
                    use_adaptive=use_adaptive_fusion,
                    structural_weight=structural_weight,
                )
                texts = [chunk.content_for_embedding or chunk.content for chunk in chunks]
                embeddings = embedding_model_to_use.encode(texts, show_progress=False)
                actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else None
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding.tolist()
                del embeddings, texts
            _get_job_store().update(job_id, JobStatus.RUNNING, progress=0.8)

            _get_or_create_collection(
                collection_id=effective_col, title=effective_col,
                llm_model=llm_model or settings.LLM_MODEL,
                embedding_model=embedding_model_name or settings.EMBEDDING_MODEL,
                embedding_dimension=actual_dimension,
                reranker_model=reranker_model,
            )
            vector_store = _get_or_create_vector_store(effective_col, actual_dimension)
            if ingest_diff and ingest_diff.removed:
                vector_store.delete_by_chunk_ids(ingest_diff.removed)
            if chunks:
                vector_store.add_chunks(chunks)
                _update_collection_stats(effective_col, safe_fname, len(chunks), file_size=len(content))
            elif ingest_diff and ingest_diff.removed:
                _update_collection_stats(effective_col, safe_fname, -len(ingest_diff.removed), file_size=len(content))
            if llm_model:
                _update_collection_model(effective_col, llm_model)

            registry.upsert(effective_col, safe_fname, file_hash, final_chunk_ids, final_chunk_hashes)

            # Persist graph if enabled (skip on delete-only re-ingest — no new chunks)
            if chunks and settings.USE_GRAPH_RAG_PERSISTENCE:
                try:
                    from rag.graph_rag import GraphRAG
                    g = GraphRAG(embedding_model=embedding_model_to_use, cache_dir=settings.GRAPH_CACHE_DIR)
                    g.load_cache(effective_col)
                    g.build_graph(chunks, cache_name=effective_col, force_rebuild=False)
                    g.save_cache(effective_col)
                except Exception as graph_err:
                    logger.warning(f"Graph persistence failed (non-fatal): {graph_err}")

            _get_job_store().update(
                job_id, JobStatus.COMPLETED, progress=1.0,
                result={"filename": safe_fname, "num_chunks": len(chunks),
                        "doc_type": doc_type, "embedding_model": embedding_model_name or settings.EMBEDDING_MODEL}
            )
        except Exception as e:
            logger.error(f"Async ingest job {job_id} failed: {e}")
            _get_job_store().update(job_id, JobStatus.FAILED, error=str(e))

    background_tasks.add_task(_run_ingest_job)

    return AsyncIngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion queued for '{file.filename}'. Poll /ingest/jobs/{job_id} for status.",
    )


@app.post("/ingest/folder", response_model=AsyncIngestResponse)
async def ingest_folder(
    request: IngestFolderRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest all PDF and Markdown files from a local folder on the server.
    Runs as a background job. Returns job_id immediately.
    """
    import os
    folder = Path(request.folder_path)

    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.folder_path}")

    job_id = _get_job_store().create_job()

    def _run_folder_job():
        try:
            _get_job_store().update(job_id, JobStatus.RUNNING, progress=0.0)
            pattern = "**/*" if request.recursive else "*"
            supported_exts = {".pdf", ".md", ".markdown", ".txt"}
            files = [f for f in folder.glob(pattern) if f.is_file() and f.suffix.lower() in supported_exts]

            if not files:
                _get_job_store().update(
                    job_id, JobStatus.COMPLETED, progress=1.0,
                    result={"folder": str(folder), "files_found": 0, "total_chunks": 0}
                )
                return

            embedding_model_to_use = get_or_create_embedding_model(
                model_name=request.embedding_model_name
            )
            effective_col = request.collection_title

            _get_or_create_collection(
                collection_id=effective_col, title=effective_col,
                llm_model=request.llm_model or settings.LLM_MODEL,
                embedding_model=request.embedding_model_name or settings.EMBEDDING_MODEL,
            )

            total_chunks = 0
            processed = 0

            for file_path in files:
                try:
                    content = file_path.read_bytes()
                    safe_fname = sanitize_filename(file_path.name)
                    file_hash = DocRegistry.compute_hash(content)

                    if settings.SKIP_DUPLICATE_INGESTION and _get_doc_registry().is_duplicate(effective_col, safe_fname, file_hash):
                        logger.info(f"Skipping duplicate: {file_path.name}")
                        processed += 1
                        _get_job_store().update(job_id, JobStatus.RUNNING, progress=processed / len(files))
                        continue

                    doc_type = "pdf" if file_path.suffix.lower() == ".pdf" else "markdown"
                    document = ingestor.ingest_from_buffer(content=content, filename=file_path.name, doc_type=doc_type)

                    chunker = create_chunker(
                        strategy=request.chunking_strategy,
                        chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap,
                        min_chunk_size=settings.MIN_CHUNK_SIZE,
                    )
                    chunks = chunker.chunk_document(content=document.content, metadata=document.metadata)
                    if not chunks:
                        continue

                    registry = _get_doc_registry()
                    chunks, ingest_diff, final_chunk_ids, final_chunk_hashes = registry.plan_ingest(
                        effective_col, safe_fname, chunks
                    )

                    actual_dimension = None
                    if chunks:
                        texts = [chunk.content_for_embedding or chunk.content for chunk in chunks]
                        embeddings = embedding_model_to_use.encode(texts, show_progress=False)
                        actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else None
                        for chunk, emb in zip(chunks, embeddings):
                            chunk.embedding = emb.tolist()
                        del embeddings, texts

                    vector_store = _get_or_create_vector_store(effective_col, actual_dimension)
                    if ingest_diff and ingest_diff.removed:
                        vector_store.delete_by_chunk_ids(ingest_diff.removed)
                    if chunks:
                        vector_store.add_chunks(chunks)
                        _update_collection_stats(effective_col, safe_fname, len(chunks), file_size=len(content))

                    registry.upsert(effective_col, safe_fname, file_hash, final_chunk_ids, final_chunk_hashes)
                    total_chunks += len(chunks)

                except Exception as e:
                    logger.warning(f"Failed to ingest {file_path.name}: {e}")

                processed += 1
                _get_job_store().update(job_id, JobStatus.RUNNING, progress=processed / len(files))

            _cleanup_gpu_memory()

            _get_job_store().update(
                job_id, JobStatus.COMPLETED, progress=1.0,
                result={
                    "folder": str(folder),
                    "files_found": len(files),
                    "files_processed": processed,
                    "total_chunks": total_chunks,
                }
            )
        except Exception as e:
            logger.error(f"Folder ingest job {job_id} failed: {e}")
            _get_job_store().update(job_id, JobStatus.FAILED, error=str(e))

    background_tasks.add_task(_run_folder_job)

    return AsyncIngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Folder ingestion queued for '{request.folder_path}' ({request.collection_title}). Poll /ingest/jobs/{job_id}",
    )


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


@app.get("/collections")
async def list_collections():
    """List all RAG collections"""
    try:
        collections = _load_collections()
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
