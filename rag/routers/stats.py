"""System- and query-level statistics endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings
from rag.schemas import StatsResponse

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Aggregate chunk/file counts across all loaded vector stores."""
    # vector_stores is a module-level cache populated in rag.main. Lazy import
    # avoids pulling ML deps at router-load time.
    from rag import main as main_module
    from rag.cache import get_query_cache

    try:
        total_chunks = 0
        total_files = 0
        for collection_id, store in main_module.vector_stores.items():
            try:
                s = store.get_stats()
                total_chunks += s.get("total_chunks", 0)
                total_files += s.get("total_files", 0)
            except Exception as e:
                logger.warning(f"Failed to get stats for collection {collection_id}: {e}")

        cache = get_query_cache().cache
        cache_stats = cache.stats() if hasattr(cache, "stats") else {}

        return StatsResponse(
            total_chunks=total_chunks,
            total_files=total_files,
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=settings.LLM_MODEL,
            vector_store_type=settings.VECTOR_STORE_TYPE,
            cache_stats=cache_stats,
        )
    except Exception as e:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/queries")
async def get_query_stats(limit: int = 100, collection_id: Optional[str] = None):
    """Recent query logs + aggregate stats, optionally filtered by collection."""
    from rag.observability import get_query_logger

    try:
        query_logger = get_query_logger()
        logs = (
            query_logger.get_logs_by_collection(collection_id, limit)
            if collection_id
            else query_logger.get_recent_logs(limit)
        )
        return {
            "logs": logs,
            "stats": query_logger.get_stats(),
            "total_logs": len(logs),
        }
    except Exception as e:
        logger.exception("Query stats error")
        raise HTTPException(status_code=500, detail=str(e))
