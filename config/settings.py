"""
Configuration settings for RAG Pipeline - MINIMAL VERSION
Only infrastructure settings that cannot be changed per API request
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
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
    LLM_TIMEOUT: int = 120

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
    CACHE_DIR: str = "./data/cache"

    # Note: System uses DiskCache by default (no Redis required)
    # To enable Redis later, add CACHE_TYPE and REDIS_URL (see REDIS-CACHE-STATUS.md)

    # =============================================================================
    # RETRIEVAL
    # =============================================================================
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.5            # 0 = BM25 only, 1 = vector only
    ENABLE_ADAPTIVE_ALPHA: bool = False
    SCORE_NORMALIZATION_METHOD: str = "rrf"
    INITIAL_RETRIEVAL_K: int = 20
    FINAL_TOP_K: int = 5
    MIN_SIMILARITY_SCORE: float = 0.0
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

    # =============================================================================
    # MODEL PRELOADING
    # =============================================================================
    PRELOAD_MODELS_ON_STARTUP: bool = False
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
    # HELPER METHODS
    # =============================================================================

    class Config:
        env_file = ".env"
        case_sensitive = True


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
