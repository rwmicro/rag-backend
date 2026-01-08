"""
Configuration settings for RAG Pipeline - MINIMAL VERSION
Only infrastructure settings that cannot be changed per API request
"""

from pydantic_settings import BaseSettings
from typing import Optional
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
    PORT: int = 8000
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
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 120

    # =============================================================================
    # RERANKER MODEL (Heavy - loaded at startup)
    # =============================================================================
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    RERANKER_DEVICE: str = "cuda"
    RERANKER_BATCH_SIZE: int = 64

    # =============================================================================
    # CACHE (Disk-based cache - simple and efficient for local use)
    # =============================================================================
    USE_CACHE: bool = False
    CACHE_TTL: int = 3600
    CACHE_DIR: str = "./data/cache"

    # Note: System uses DiskCache by default (no Redis required)
    # To enable Redis later, add CACHE_TYPE and REDIS_URL (see REDIS-CACHE-STATUS.md)

    # =============================================================================
    # DEFAULT VALUES (Used when not specified in API requests)
    # =============================================================================
    # These are fallback values. Users should specify them in API requests.

    # Chunking defaults (for /ingest/file endpoint)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
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
