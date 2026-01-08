"""
Caching Module
Implements embedding and query result caching
"""

from typing import Optional, Any, Dict
import hashlib
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
import diskcache
from loguru import logger

from config.settings import settings


class BaseCache(ABC):
    """Base class for caching"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache"""
        pass


class DiskCache(BaseCache):
    """Disk-based cache using diskcache"""

    def __init__(self, cache_dir: str = "./data/cache", size_limit: int = 1_000_000_000):
        """
        Initialize disk cache

        Args:
            cache_dir: Directory for cache storage
            size_limit: Max cache size in bytes (default 1GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = diskcache.Cache(
            str(self.cache_dir),
            size_limit=size_limit,
        )

        logger.info(f"Initialized disk cache at {cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        if ttl:
            self.cache.set(key, value, expire=ttl)
        else:
            self.cache.set(key, value)

    def delete(self, key: str) -> None:
        """Delete value from cache"""
        self.cache.delete(key)

    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": self.cache.volume(),
            "count": len(self.cache),
        }


class RedisCache(BaseCache):
    """Redis-based cache (for distributed caching)"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis cache

        Args:
            redis_url: Redis connection URL
        """
        try:
            import redis

            self.redis = redis.from_url(redis_url, decode_responses=False)
            self.redis.ping()  # Test connection

            logger.info(f"Connected to Redis cache: {redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        data = self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis with optional TTL"""
        data = pickle.dumps(value)
        if ttl:
            self.redis.setex(key, ttl, data)
        else:
            self.redis.set(key, data)

    def delete(self, key: str) -> None:
        """Delete value from Redis"""
        self.redis.delete(key)

    def clear(self) -> None:
        """Clear all cache (use with caution!)"""
        self.redis.flushdb()


class EmbeddingCache:
    """
    Specialized cache for embeddings
    Uses content hash as key for efficient lookup
    """

    def __init__(self, cache: BaseCache):
        self.cache = cache

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model"""
        # Use hash to avoid storing full text as key
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{model}:{text_hash}"

    def get(self, text: str, model: str) -> Optional[list]:
        """Get embedding from cache"""
        key = self._make_key(text, model)
        return self.cache.get(key)

    def set(self, text: str, model: str, embedding: list, ttl: Optional[int] = None) -> None:
        """Set embedding in cache"""
        key = self._make_key(text, model)
        self.cache.set(key, embedding, ttl)

    def clear(self) -> None:
        """Clear embedding cache"""
        # Only clear embedding keys
        # Note: This is a simplified version
        self.cache.clear()


class QueryCache:
    """
    Cache for query results
    Stores (query, results) pairs
    """

    def __init__(self, cache: BaseCache, ttl: int = 3600):
        self.cache = cache
        self.ttl = ttl

    def _make_key(self, query: str, top_k: int, collection: Optional[str] = None) -> str:
        """Create cache key from query parameters"""
        key_parts = [query, str(top_k)]
        if collection:
            key_parts.append(collection)

        key_str = ":".join(key_parts)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"query:{key_hash}"

    def get(
        self,
        query: str,
        top_k: int,
        collection: Optional[str] = None
    ) -> Optional[Any]:
        """Get query results from cache"""
        key = self._make_key(query, top_k, collection)
        return self.cache.get(key)

    def set(
        self,
        query: str,
        top_k: int,
        results: Any,
        collection: Optional[str] = None
    ) -> None:
        """Set query results in cache"""
        key = self._make_key(query, top_k, collection)
        self.cache.set(key, results, ttl=self.ttl)

    def clear(self) -> None:
        """Clear query cache"""
        self.cache.clear()


# Global cache instances
_cache_instance: Optional[BaseCache] = None
_embedding_cache_instance: Optional[EmbeddingCache] = None
_query_cache_instance: Optional[QueryCache] = None


def get_cache() -> BaseCache:
    """Get or create global cache instance"""
    global _cache_instance

    if _cache_instance is None:
        if settings.CACHE_TYPE == "redis":
            try:
                _cache_instance = RedisCache(settings.REDIS_URL)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache, falling back to disk: {e}")
                _cache_instance = DiskCache(settings.CACHE_DIR)
        else:
            _cache_instance = DiskCache(settings.CACHE_DIR)

    return _cache_instance


def get_embedding_cache() -> EmbeddingCache:
    """Get or create global embedding cache"""
    global _embedding_cache_instance

    if _embedding_cache_instance is None:
        _embedding_cache_instance = EmbeddingCache(get_cache())

    return _embedding_cache_instance


def get_query_cache() -> QueryCache:
    """Get or create global query cache"""
    global _query_cache_instance

    if _query_cache_instance is None:
        _query_cache_instance = QueryCache(
            get_cache(),
            ttl=settings.CACHE_TTL
        )

    return _query_cache_instance
