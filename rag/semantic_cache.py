"""
Semantic Caching Module
Caches query results based on semantic similarity rather than exact match
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from loguru import logger

from .chunking import Chunk


@dataclass
class CachedQuery:
    """Represents a cached query with its results"""
    query: str
    query_embedding: np.ndarray
    chunks_with_scores: List[Tuple[Chunk, float]]
    timestamp: float
    top_k: int
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "query_embedding": self.query_embedding.tolist(),
            "chunks_with_scores": [
                {
                    "chunk": {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "chunk_id": chunk.chunk_id
                    },
                    "score": float(score)
                }
                for chunk, score in self.chunks_with_scores
            ],
            "timestamp": self.timestamp,
            "top_k": self.top_k,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedQuery':
        """Create from dictionary"""
        chunks_with_scores = [
            (
                Chunk(
                    content=item["chunk"]["content"],
                    metadata=item["chunk"]["metadata"],
                    chunk_id=item["chunk"]["chunk_id"]
                ),
                item["score"]
            )
            for item in data["chunks_with_scores"]
        ]

        return cls(
            query=data["query"],
            query_embedding=np.array(data["query_embedding"]),
            chunks_with_scores=chunks_with_scores,
            timestamp=data["timestamp"],
            top_k=data["top_k"],
            metadata=data.get("metadata", {})
        )


class SemanticCache:
    """
    Semantic query cache using embedding similarity

    Instead of exact query matching, this cache finds semantically similar queries
    and returns their cached results if similarity exceeds a threshold.
    """

    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.95,
        max_cache_size: int = 100,
        ttl_seconds: Optional[int] = 3600,  # 1 hour default
        cache_file: Optional[str] = None,
    ):
        """
        Initialize semantic cache

        Args:
            embedding_model: Embedding model to encode queries
            similarity_threshold: Minimum similarity to consider a cache hit (0.0-1.0)
            max_cache_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
            cache_file: Optional file path to persist cache
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache_file = cache_file

        # In-memory cache storage
        self.cache: List[CachedQuery] = []

        # Load from disk if cache file exists
        if cache_file and Path(cache_file).exists():
            self._load_from_disk()

        logger.info(f"Initialized semantic cache (threshold={similarity_threshold}, "
                   f"max_size={max_cache_size}, ttl={ttl_seconds}s)")

    def get(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[Chunk, float]]]:
        """
        Get cached results for a semantically similar query

        Args:
            query: Query string
            top_k: Number of results requested
            metadata_filter: Optional metadata filters used in the query

        Returns:
            Cached chunks_with_scores if similar query found, None otherwise
        """
        if not self.cache:
            return None

        # Encode query
        query_embedding = self.embedding_model.encode_single(query, is_query=True)

        # Find most similar cached query
        best_match = None
        best_similarity = 0.0

        current_time = time.time()

        for cached_query in self.cache:
            # Check if cache entry has expired
            if self.ttl_seconds and (current_time - cached_query.timestamp) > self.ttl_seconds:
                continue

            # Check if top_k matches (we only return exact top_k matches)
            if cached_query.top_k != top_k:
                continue

            # Check if metadata filters match
            if metadata_filter != cached_query.metadata.get("metadata_filter"):
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, cached_query.query_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_query

        # Return cached results if similarity exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            logger.info(f"Semantic cache HIT! Similarity: {best_similarity:.4f} for query: '{query[:50]}...' "
                       f"(matched: '{best_match.query[:50]}...')")
            return best_match.chunks_with_scores

        logger.debug(f"Semantic cache MISS. Best similarity: {best_similarity:.4f}")
        return None

    def set(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None
    ):
        """
        Cache query results

        Args:
            query: Query string
            chunks_with_scores: Retrieved chunks with scores
            top_k: Number of results
            metadata_filter: Optional metadata filters used
        """
        # Encode query
        query_embedding = self.embedding_model.encode_single(query, is_query=True)

        # Create cache entry
        cached_query = CachedQuery(
            query=query,
            query_embedding=query_embedding,
            chunks_with_scores=chunks_with_scores,
            timestamp=time.time(),
            top_k=top_k,
            metadata={"metadata_filter": metadata_filter}
        )

        # Add to cache
        self.cache.append(cached_query)

        # Enforce max cache size (LRU eviction)
        if len(self.cache) > self.max_cache_size:
            self.cache.pop(0)  # Remove oldest
            logger.debug(f"Evicted oldest cache entry (max size: {self.max_cache_size})")

        logger.debug(f"Cached query: '{query[:50]}...' (cache size: {len(self.cache)})")

        # Persist to disk if configured
        if self.cache_file:
            self._save_to_disk()

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Semantic cache cleared")

        if self.cache_file:
            self._save_to_disk()

    def evict_expired(self):
        """Remove expired cache entries"""
        if not self.ttl_seconds:
            return

        current_time = time.time()
        initial_size = len(self.cache)

        self.cache = [
            cached_query for cached_query in self.cache
            if (current_time - cached_query.timestamp) <= self.ttl_seconds
        ]

        removed = initial_size - len(self.cache)
        if removed > 0:
            logger.info(f"Evicted {removed} expired cache entries")
            if self.cache_file:
                self._save_to_disk()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {
                "size": 0,
                "max_size": self.max_cache_size,
                "threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds
            }

        current_time = time.time()
        active_count = sum(
            1 for cached_query in self.cache
            if not self.ttl_seconds or (current_time - cached_query.timestamp) <= self.ttl_seconds
        )

        return {
            "size": len(self.cache),
            "active": active_count,
            "max_size": self.max_cache_size,
            "threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _save_to_disk(self):
        """Persist cache to disk"""
        try:
            cache_data = [cached_query.to_dict() for cached_query in self.cache]

            # Ensure directory exists
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)

            logger.debug(f"Saved semantic cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save semantic cache: {e}")

    def _load_from_disk(self):
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            self.cache = [CachedQuery.from_dict(data) for data in cache_data]

            logger.info(f"Loaded {len(self.cache)} entries from semantic cache")

            # Evict expired entries immediately after loading
            self.evict_expired()
        except Exception as e:
            logger.error(f"Failed to load semantic cache: {e}")
            self.cache = []


# Global semantic cache instance (lazily initialized)
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache(embedding_model) -> SemanticCache:
    """Get or create the global semantic cache instance"""
    global _semantic_cache

    if _semantic_cache is None:
        from config.settings import settings

        cache_file = None
        if settings.USE_CACHE:
            cache_file = str(Path(settings.DATA_DIR) / "cache" / "semantic_cache.json")

        _semantic_cache = SemanticCache(
            embedding_model=embedding_model,
            similarity_threshold=getattr(settings, 'SEMANTIC_CACHE_THRESHOLD', 0.95),
            max_cache_size=getattr(settings, 'SEMANTIC_CACHE_MAX_SIZE', 100),
            ttl_seconds=getattr(settings, 'SEMANTIC_CACHE_TTL', 3600),
            cache_file=cache_file
        )

    return _semantic_cache
