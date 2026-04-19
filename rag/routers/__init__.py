"""FastAPI routers, grouped by domain."""

from rag.routers.health import router as health_router
from rag.routers.jobs import router as jobs_router
from rag.routers.cache import router as cache_router
from rag.routers.config import router as config_router
from rag.routers.stats import router as stats_router
from rag.routers.models import router as models_router
from rag.routers.evaluate import router as evaluate_router

__all__ = [
    "health_router",
    "jobs_router",
    "cache_router",
    "config_router",
    "stats_router",
    "models_router",
    "evaluate_router",
]
