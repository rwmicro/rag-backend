"""Liveness/readiness endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {
        "status": "ok",
        "message": "RAG Pipeline API",
        "version": "2.0.0",
    }


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "rag-backend",
        "version": "2.0.0",
    }
