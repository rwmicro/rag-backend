"""Async ingestion job status."""

from fastapi import APIRouter, HTTPException

from rag.app_state import get_job_store
from rag.schemas import JobStatusResponse

router = APIRouter(tags=["jobs"])


@router.get("/ingest/jobs/{job_id}", response_model=JobStatusResponse)
async def get_ingest_job_status(job_id: str):
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(**record)
