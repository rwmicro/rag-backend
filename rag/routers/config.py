"""Runtime/.env configuration endpoints.

Local, single-user deployment only. No auth — run behind a reverse proxy or
expose only on localhost if you care.
"""

from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings

router = APIRouter(tags=["config"])


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _REPO_ROOT / ".env"


_CATEGORIZED_KEYS = {
    "server": ["HOST", "PORT", "LOG_LEVEL"],
    "paths": ["DATA_DIR", "CORPUS_DIR", "INDEX_DIR"],
    "vector_store": ["VECTOR_STORE_TYPE", "LANCEDB_URI", "CHROMA_HOST", "CHROMA_PORT"],
    "embedding": [
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE",
        "NORMALIZE_EMBEDDINGS",
    ],
    "llm": ["LLM_BASE_URL", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS"],
    "chunking": ["CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNKING_STRATEGY"],
    "retrieval": [
        "TOP_K",
        "MIN_SIMILARITY_SCORE",
        "USE_HYBRID_SEARCH",
        "HYBRID_ALPHA",
    ],
    "reranking": ["USE_RERANKING", "RERANKER_MODEL", "RERANKER_TOP_K"],
    "compression": ["USE_COMPRESSION", "MAX_CONTEXT_TOKENS", "COMPRESSION_RATIO"],
}


@router.get("/config/env")
async def get_env_config():
    """Return current settings, grouped and flat, plus the .env path if present."""
    try:
        config = settings.model_dump()
        known = {k for keys in _CATEGORIZED_KEYS.values() for k in keys}

        structured_config: Dict[str, Dict[str, object]] = {
            group: {key: config.get(key) for key in keys}
            for group, keys in _CATEGORIZED_KEYS.items()
        }
        structured_config["other"] = {k: v for k, v in config.items() if k not in known}

        return {
            "config": structured_config,
            "flat_config": config,
            "file_path": str(_ENV_PATH) if _ENV_PATH.exists() else "Using defaults",
        }
    except Exception as e:
        logger.exception("Failed to read config")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/env")
async def update_env_config(updates: Dict[str, Dict[str, str]]):
    """Patch .env in-place, preserving comments and line ordering."""
    try:
        if not _ENV_PATH.exists():
            raise HTTPException(status_code=404, detail=".env file not found")

        with open(_ENV_PATH, "r") as f:
            lines = f.readlines()

        flat_updates: Dict[str, str] = {}
        for category_updates in updates.values():
            flat_updates.update(category_updates)

        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped or "=" not in stripped:
                new_lines.append(line)
                continue

            key = stripped.split("=", 1)[0].strip()
            if key not in flat_updates:
                new_lines.append(line)
                continue

            if "#" in stripped:
                comment = stripped.split("#", 1)[1]
                new_lines.append(f"{key}={flat_updates[key]}  # {comment}\n")
            else:
                new_lines.append(f"{key}={flat_updates[key]}\n")

        with open(_ENV_PATH, "w") as f:
            f.writelines(new_lines)

        logger.info(f"Updated .env configuration with {len(flat_updates)} changes")
        return {
            "success": True,
            "message": f"Updated {len(flat_updates)} configuration values",
            "note": "Restart the backend for changes to take effect",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update .env config")
        raise HTTPException(status_code=500, detail=str(e))
