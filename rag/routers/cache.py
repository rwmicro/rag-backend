"""Cache-management endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter(tags=["cache"])


@router.delete("/cache")
async def clear_cache():
    """Clear the query-level cache."""
    try:
        from rag.cache import get_cache

        get_cache().clear()
        return {"success": True, "message": "Cache cleared"}
    except Exception as e:
        logger.exception("Cache clear error")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/graph")
async def clear_graph_cache(cache_name: Optional[str] = None):
    """Clear the global Graph-RAG cache (if initialized)."""
    # graph_rag lives as a module global in rag.main; look it up lazily to
    # avoid a circular import at module load time.
    try:
        from rag import main as main_module

        graph_rag = getattr(main_module, "graph_rag", None)
        if graph_rag is None:
            return {"success": False, "message": "Graph RAG not initialized"}
        graph_rag.clear_cache(cache_name)
        suffix = f"({cache_name}) " if cache_name else ""
        return {"success": True, "message": f"Graph cache {suffix}cleared"}
    except Exception as e:
        logger.exception("Graph cache clear error")
        raise HTTPException(status_code=500, detail=str(e))
