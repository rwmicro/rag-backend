"""
Configuration settings for RAG Pipeline - MINIMAL VERSION
Only infrastructure settings that cannot be changed per API request
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    """
    Application settings with environment variable support

    Philosophy:
    - Infrastructure settings here (models, connections, paths)
    - Behavior settings in API requests (top_k, temperature, etc.)
    """

    # =============================================================================
    # SERVER CONFIGURATION (Infrastructure)
    # =============================================================================
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "*"

    # =============================================================================
    # PATHS (System structure)
    # =============================================================================
    DATA_DIR: str = "./data"
    CORPUS_DIR: str = "./data/corpus"
    CACHE_DIR: str = "./data/cache"

    # =============================================================================
    # EMBEDDING MODEL (Heavy - loaded at startup)
    # =============================================================================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 256
    # Required by every branch of create_embedding_model(). Cosine similarity is
    # computed as an inner product over unit vectors, so this must stay True
    # unless the vector store is changed to normalize on its own.
    NORMALIZE_EMBEDDINGS: bool = True

    # =============================================================================
    # VECTOR STORE (Database choice)
    # =============================================================================
    VECTOR_STORE_TYPE: str = "faiss"
    FAISS_INDEX_PATH: str = "./data/faiss/index.faiss"
    LANCEDB_URI: str = "./data/lancedb"
    SQLITE_VECTOR_DB_PATH: str = "./data/rag.db"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000

    # =============================================================================
    # LLM (Connection settings)
    # =============================================================================
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60

    # =============================================================================
    # RERANKER MODEL (Heavy - loaded at startup)
    # =============================================================================
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    RERANKER_DEVICE: str = "cuda"
    RERANKER_BATCH_SIZE: int = 64
    RERANKER_TYPE: str = "cross-encoder"   # "cross-encoder" or "llm"
    RERANKER_DOMAIN: str = "general"
    USE_RERANKING: bool = True

    # =============================================================================
    # CACHE (Disk-based cache - simple and efficient for local use)
    # =============================================================================
    USE_CACHE: bool = False
    CACHE_TTL: int = 3600

    # Redis support (optional, falls back to DiskCache)
    CACHE_TYPE: str = "disk"
    REDIS_URL: str = "redis://localhost:6379"

    # =============================================================================
    # RETRIEVAL
    # =============================================================================
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.5            # 0 = BM25 only, 1 = vector only
    ENABLE_ADAPTIVE_ALPHA: bool = False
    ADAPTIVE_ALPHA_MIN: float = 0.2      # Favor BM25 for keyword-ish queries
    ADAPTIVE_ALPHA_MAX: float = 0.95     # Favor vectors for semantic queries
    SCORE_NORMALIZATION_METHOD: str = "rrf"
    RRF_K: int = 60                      # Reciprocal Rank Fusion constant (standard in the literature)
    INITIAL_RETRIEVAL_K: int = 20
    FINAL_TOP_K: int = 5
    MIN_SIMILARITY_SCORE: float = 0.0
    DEDUP_SIMILARITY_THRESHOLD: float = 0.95   # Cosine similarity above which chunks are duplicates
    ENABLE_MULTI_HOP: bool = False

    # =============================================================================
    # MMR (Maximal Marginal Relevance)
    # =============================================================================
    ENABLE_MMR: bool = False
    MMR_LAMBDA: float = 0.5
    MMR_DIVERSITY_THRESHOLD: float = 0.5

    # =============================================================================
    # CONTEXT COMPRESSION
    # =============================================================================
    USE_COMPRESSION: bool = False
    MAX_CONTEXT_TOKENS: int = 4096

    # =============================================================================
    # QUERY ROUTING & CLASSIFICATION
    # =============================================================================
    ENABLE_AUTO_ROUTING: bool = False
    ROUTER_MODE: str = "simple"
    ENABLE_QUERY_CLASSIFICATION: bool = False
    QUERY_CLASSIFIER_USE_LLM: bool = False

    # =============================================================================
    # FALLBACK STRATEGIES
    # =============================================================================
    ENABLE_FALLBACK_STRATEGIES: bool = False
    RETRIEVAL_CONFIDENCE_HIGH: float = 0.8
    RETRIEVAL_CONFIDENCE_MEDIUM: float = 0.5
    RETRIEVAL_CONFIDENCE_MINIMUM: float = 0.2

    # Corrective RAG: rewrite the query and retry once when confidence is low.
    ENABLE_CORRECTIVE_RAG: bool = True
    CORRECTIVE_MAX_ATTEMPTS: int = 1  # retries beyond the initial attempt
    CORRECTIVE_TRIGGER_LEVEL: str = "low"  # "low" | "very_low" | "medium"
    CORRECTIVE_MERGE_METHOD: str = "max"  # "max" | "rrf"

    # =============================================================================
    # MODEL PRELOADING
    # =============================================================================
    PRELOAD_MODELS_ON_STARTUP: bool = True
    PRELOAD_MODELS: List[str] = []

    # =============================================================================
    # CONTEXTUAL EMBEDDINGS
    # =============================================================================
    USE_CONTEXTUAL_EMBEDDINGS: bool = False

    # =============================================================================
    # STORAGE PATHS
    # =============================================================================
    INDEX_DIR: str = "./data/index"
    FEEDBACK_DB_PATH: str = "./data/feedback.db"
    METADATA_DB_PATH: str = "./data/metadata.db"

    # =============================================================================
    # MISC
    # =============================================================================
    OLLAMA_HEALTH_CHECK_TIMEOUT: int = 5

    # =============================================================================
    # DEFAULT VALUES (Used when not specified in API requests)
    # =============================================================================
    # These are fallback values. Users should specify them in API requests.

    # Chunking defaults (for /ingest/file endpoint)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_SIZE: int = 100
    CHUNKING_STRATEGY: str = "semantic"

    # Retrieval defaults (for /query endpoint)
    TOP_K: int = 5
    RERANK_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.5

    # =============================================================================
    # COLLECTIONS DATABASE
    # =============================================================================
    COLLECTIONS_DB_PATH: str = "./data/collections.db"

    # =============================================================================
    # JOB STORE (Async ingestion)
    # =============================================================================
    JOBS_DB_PATH: str = "./data/jobs.db"

    # =============================================================================
    # DOCUMENT REGISTRY (Deduplication)
    # =============================================================================
    DOC_REGISTRY_DB_PATH: str = "./data/doc_registry.db"
    SKIP_DUPLICATE_INGESTION: bool = True

    # =============================================================================
    # GRAPH RAG PERSISTENCE
    # =============================================================================
    GRAPH_CACHE_DIR: str = "./data/graph_cache"
    USE_GRAPH_RAG_PERSISTENCE: bool = True

    # =============================================================================
    # QUERY OBSERVABILITY
    # =============================================================================
    ENABLE_QUERY_LOGGING: bool = True
    MAX_QUERY_LOGS: int = 1000
    QUERY_LOG_PATH: str = "./data/query_logs.jsonl"

    # =============================================================================
    # CONTEXTUAL SUMMARIES
    # =============================================================================
    CONTEXT_SUMMARY_MAX_TOKENS: int = 150   # ~2-3 sentences

    # =============================================================================
    # MULTILINGUAL PIPELINE (opt-in; consumed by create_multilingual_pipeline)
    # =============================================================================
    USE_MULTILINGUAL_EMBEDDINGS: bool = False
    MULTILINGUAL_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    ENABLE_MULTILINGUAL_BM25: bool = False
    BM25_USE_STEMMING: bool = True
    ENABLE_MULTILINGUAL_HYDE: bool = False
    ENABLE_MULTILINGUAL_QUERY_CLASSIFICATION: bool = False
    DEFAULT_LANGUAGE: str = "en"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


# Create settings instance
settings = Settings()


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    for directory in [
        settings.DATA_DIR,
        settings.CORPUS_DIR,
        settings.CACHE_DIR,
        os.path.dirname(settings.FAISS_INDEX_PATH)
        if settings.VECTOR_STORE_TYPE == "faiss"
        else None,
        settings.LANCEDB_URI if settings.VECTOR_STORE_TYPE == "lancedb" else None,
    ]:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


# Auto-create directories on import
ensure_directories()
