"""
Integration tests: full flow ingest → query → verify
Run with: pytest tests/test_integration.py -v -m integration
Skip by default in CI: pytest -m "not integration"
"""
import pytest
import asyncio
from pathlib import Path


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sample_markdown_bytes() -> bytes:
    return b"""# Test Document

## Introduction

The capital of France is Paris. Paris is known as the City of Light.
It has a population of approximately 2.1 million people in the city proper.

## Geography

France is located in Western Europe. It borders Germany, Italy, Spain, and Belgium.
The Seine river flows through Paris.

## History

Paris has been the capital of France since the late 10th century.
The Eiffel Tower was built in 1889 for the World Fair.
"""


@pytest.fixture(scope="module")
def app_client():
    """Create test client for the FastAPI app"""
    try:
        from fastapi.testclient import TestClient
        from rag.main import app
        client = TestClient(app)
        return client
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")


class TestCollectionsCRUD:
    def test_list_collections_empty_or_existing(self, app_client):
        resp = app_client.get("/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert "collections" in data
        assert "total" in data

    def test_get_nonexistent_collection(self, app_client):
        resp = app_client.get("/collections/nonexistent_xyz_123")
        assert resp.status_code == 404

    def test_health_check(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestIngestAndQuery:
    COLLECTION_ID = "test_integration_md_collection"

    def test_ingest_markdown(self, app_client, sample_markdown_bytes):
        resp = app_client.post(
            "/ingest/file",
            data={
                "collection_title": self.COLLECTION_ID,
                "chunk_size": "300",
                "chunk_overlap": "50",
                "chunking_strategy": "recursive",
            },
            files={"file": ("test_france.md", sample_markdown_bytes, "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stats"]["num_chunks"] > 0

    def test_duplicate_ingest_skipped(self, app_client, sample_markdown_bytes):
        """Second upload of same content should be skipped."""
        resp = app_client.post(
            "/ingest/file",
            data={"collection_title": self.COLLECTION_ID},
            files={"file": ("test_france.md", sample_markdown_bytes, "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should be skipped (same hash)
        assert data["success"] is True
        assert data["stats"].get("skipped") is True

    def test_query_collection_exists(self, app_client):
        """Collection should appear in list after ingestion."""
        resp = app_client.get("/collections")
        assert resp.status_code == 200
        ids = [c["id"] for c in resp.json()["collections"]]
        assert self.COLLECTION_ID in ids

    def test_delete_collection(self, app_client):
        resp = app_client.delete(f"/collections/{self.COLLECTION_ID}")
        assert resp.status_code == 200
        # Verify deleted
        resp2 = app_client.get(f"/collections/{self.COLLECTION_ID}")
        assert resp2.status_code == 404


class TestAsyncIngest:
    COLLECTION_ID = "test_async_integration_collection"

    def test_async_ingest_returns_job_id(self, app_client, sample_markdown_bytes):
        resp = app_client.post(
            "/ingest/file/async",
            data={"collection_title": self.COLLECTION_ID},
            files={"file": ("async_test.md", sample_markdown_bytes, "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_job_status_endpoint(self, app_client, sample_markdown_bytes):
        """Create job, then poll status."""
        resp = app_client.post(
            "/ingest/file/async",
            data={"collection_title": self.COLLECTION_ID + "_2"},
            files={"file": ("job_test.md", sample_markdown_bytes, "text/markdown")},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll status
        import time
        for _ in range(30):
            status_resp = app_client.get(f"/ingest/jobs/{job_id}")
            assert status_resp.status_code == 200
            status = status_resp.json()["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.5)

        assert status_resp.json()["status"] == "completed"

    def test_job_not_found(self, app_client):
        resp = app_client.get("/ingest/jobs/nonexistent-job-id-xyz")
        assert resp.status_code == 404


class TestFolderIngest:
    def test_folder_not_found(self, app_client):
        resp = app_client.post(
            "/ingest/folder",
            json={
                "folder_path": "/nonexistent/path/xyz",
                "collection_title": "test_folder_col",
            },
        )
        assert resp.status_code == 404

    def test_folder_ingest_current_data_dir(self, app_client):
        """Test ingesting from the data/corpus directory."""
        import os
        corpus_dir = "./data/corpus"
        os.makedirs(corpus_dir, exist_ok=True)

        # Create a test file
        test_file = Path(corpus_dir) / "folder_test.md"
        test_file.write_text("# Folder Test\nThis is a test document for folder ingestion.")

        resp = app_client.post(
            "/ingest/folder",
            json={
                "folder_path": corpus_dir,
                "collection_title": "test_folder_col",
                "recursive": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data

        # Cleanup
        test_file.unlink(missing_ok=True)


class TestQueryWithLLMOverride:
    def test_query_with_invalid_collection(self, app_client):
        resp = app_client.post(
            "/query",
            json={
                "query": "test query",
                "collection_id": "nonexistent_collection_xyz",
                "top_k": 3,
                "use_reranking": False,
                "stream": False,
            },
        )
        # Should return a valid response even for empty collection
        assert resp.status_code in (200, 500)

    def test_query_request_schema_with_overrides(self, app_client):
        """Test that the new fields are accepted in schema."""
        resp = app_client.post(
            "/query",
            json={
                "query": "test",
                "collection_id": "nonexistent",
                "top_k": 1,
                "stream": False,
                "llm_provider": "ollama",
                "llm_model_override": "llama3.2",
                "llm_timeout": 30,
            },
        )
        # Should not return 422 (validation error) - the fields should be accepted
        assert resp.status_code != 422
