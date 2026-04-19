"""Model discovery, health, and pull endpoints."""

import os
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings

router = APIRouter(tags=["models"])


@router.get("/models")
async def get_available_models(
    check_availability: bool = False,
    model_type: Optional[str] = None,
):
    """List embedding and reranker models, with optional availability probe."""
    try:
        from rag.model_registry import (
            EMBEDDING_MODELS,
            OLLAMA_EMBEDDING_MODELS,
            RERANKER_MODELS,
            to_dict,
        )
        from rag.model_validation import get_model_validator

        response: dict = {}

        if model_type in (None, "all", "embedding"):
            embedding_models = []

            for shortcut, info in EMBEDDING_MODELS.items():
                model_data = to_dict(info)
                model_data["shortcut"] = shortcut
                model_data["available"] = None
                if check_availability:
                    validator = get_model_validator()
                    is_valid, error_msg = validator.validate_embedding_model(
                        info.name, check_availability=True
                    )
                    model_data["available"] = is_valid
                    model_data["error"] = error_msg if not is_valid else None
                embedding_models.append(model_data)

            try:
                ollama_response = requests.get(
                    f"{settings.LLM_BASE_URL}/api/tags", timeout=5
                )
                if ollama_response.status_code == 200:
                    ollama_model_names = [
                        m["name"] for m in ollama_response.json().get("models", [])
                    ]
                    for model_name in [n for n in ollama_model_names if "embed" in n.lower()]:
                        info = None
                        for _, registry_info in OLLAMA_EMBEDDING_MODELS.items():
                            if registry_info.name == model_name or model_name.startswith(
                                registry_info.name
                            ):
                                info = registry_info
                                break
                        if info:
                            model_data = to_dict(info)
                            model_data["shortcut"] = None
                        else:
                            model_data = {
                                "name": model_name,
                                "dimension": 768,
                                "max_seq_length": 2048,
                                "model_type": "ollama",
                                "description": f"Ollama embedding model - {model_name}",
                                "size_mb": None,
                                "shortcut": None,
                            }
                        model_data["available"] = True
                        embedding_models.append(model_data)
                else:
                    logger.warning(
                        f"Failed to fetch Ollama models: {ollama_response.status_code}"
                    )
                    for shortcut, info in OLLAMA_EMBEDDING_MODELS.items():
                        model_data = to_dict(info)
                        model_data["shortcut"] = shortcut
                        model_data["available"] = False
                        embedding_models.append(model_data)
            except Exception as e:
                logger.warning(f"Could not query Ollama for models: {e}")
                for shortcut, info in OLLAMA_EMBEDDING_MODELS.items():
                    model_data = to_dict(info)
                    model_data["shortcut"] = shortcut
                    model_data["available"] = None
                    embedding_models.append(model_data)

            response["embedding_models"] = embedding_models

        if model_type in (None, "all", "reranker"):
            reranker_models = []
            for shortcut, info in RERANKER_MODELS.items():
                model_data = to_dict(info)
                model_data["shortcut"] = shortcut
                model_data["available"] = None
                if check_availability:
                    validator = get_model_validator()
                    is_valid, error_msg = validator.validate_reranker_model(info.name)
                    model_data["available"] = is_valid
                    model_data["error"] = error_msg if not is_valid else None
                reranker_models.append(model_data)
            response["reranker_models"] = reranker_models

        response["current_config"] = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
        }
        response["metadata"] = {
            "total_embedding_models": len(response.get("embedding_models", [])),
            "total_reranker_models": len(response.get("reranker_models", [])),
            "availability_checked": check_availability,
        }
        return response

    except Exception as e:
        logger.exception("Get models error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/models")
async def health_models():
    """Ollama reachability + cached model inventory."""
    try:
        health_info: dict = {
            "status": "ok",
            "ollama": {},
            "cached_models": {},
            "reranker_cache": {},
        }

        try:
            response = requests.get(
                f"{settings.LLM_BASE_URL}/api/tags",
                timeout=settings.OLLAMA_HEALTH_CHECK_TIMEOUT,
            )
            if response.status_code == 200:
                ollama_models = response.json().get("models", [])
                health_info["ollama"] = {
                    "status": "connected",
                    "url": settings.LLM_BASE_URL,
                    "models_available": len(ollama_models),
                    "models": [
                        {
                            "name": m.get("name"),
                            "size": m.get("size"),
                            "modified_at": m.get("modified_at"),
                        }
                        for m in ollama_models
                    ],
                }
            else:
                health_info["ollama"] = {
                    "status": "error",
                    "url": settings.LLM_BASE_URL,
                    "error": f"HTTP {response.status_code}",
                }
                health_info["status"] = "degraded"
        except Exception as e:
            health_info["ollama"] = {
                "status": "disconnected",
                "url": settings.LLM_BASE_URL,
                "error": str(e),
            }
            health_info["status"] = "degraded"

        # Reranker cache lives in rag.main; lazy import to avoid a circular import.
        from rag import main as main_module

        reranker_cache = getattr(main_module, "_reranker_cache", None) or {}
        health_info["reranker_cache"] = {
            "cached_models": list(reranker_cache.keys()),
            "count": len(reranker_cache),
        }

        try:
            cache_home = os.getenv("HF_HOME") or os.path.join(
                Path.home(), ".cache", "huggingface"
            )
            hub_cache = os.path.join(cache_home, "hub")
            if os.path.exists(hub_cache):
                cached_dirs = [d for d in os.listdir(hub_cache) if d.startswith("models--")]
                cached_model_names = [
                    d.replace("models--", "").replace("--", "/") for d in cached_dirs
                ]
                health_info["cached_models"] = {
                    "cache_path": hub_cache,
                    "models": cached_model_names,
                    "count": len(cached_model_names),
                }
            else:
                health_info["cached_models"] = {
                    "cache_path": hub_cache,
                    "models": [],
                    "count": 0,
                    "note": "Cache directory does not exist yet",
                }
        except Exception as e:
            health_info["cached_models"] = {"error": str(e)}

        health_info["config"] = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
            "preload_on_startup": settings.PRELOAD_MODELS_ON_STARTUP,
            "preload_models": settings.PRELOAD_MODELS,
        }
        return health_info

    except Exception as e:
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/pull")
async def pull_model(
    model_name: str,
    model_type: str = "embedding",
    auto_download: bool = True,
):
    """Ensure a model is downloaded/available on the host."""
    try:
        from rag.model_validation import ensure_model_available

        if model_type not in ("embedding", "reranker", "llm"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type '{model_type}'. Must be 'embedding', 'reranker', or 'llm'",
            )

        logger.info(f"Pull request for {model_type} model: {model_name}")
        is_available, error_msg = ensure_model_available(
            model_name=model_name, model_type=model_type, auto_download=auto_download
        )

        if is_available:
            return {
                "success": True,
                "message": f"Model '{model_name}' is available",
                "model_name": model_name,
                "model_type": model_type,
                "status": "ready",
            }
        if not auto_download:
            return {
                "success": False,
                "message": f"Model '{model_name}' is not available",
                "model_name": model_name,
                "model_type": model_type,
                "status": "not_available",
                "error": error_msg,
            }
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model '{model_name}': {error_msg}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pull model error")
        raise HTTPException(status_code=500, detail=str(e))
