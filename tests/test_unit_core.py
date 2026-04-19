"""Unit tests that don't require the heavy ML stack.

Covers the pure-Python pieces: schemas, job store, doc registry (incl.
chunk-level diff), collections DB, model-cache LRU eviction. These all run
against real SQLite tempfiles — no mocks.
"""

import os
import sys
import tempfile
import json
from collections import OrderedDict

import pytest

# Make the repo importable without `backend.` prefix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def test_query_request_defaults():
    from rag.schemas import QueryRequest

    qr = QueryRequest(query="hello")
    assert qr.query == "hello"
    assert qr.top_k == 5
    assert qr.stream is True
    assert qr.use_reranking is True
    assert qr.use_hybrid_search is True


def test_query_request_validates_top_k_range():
    from rag.schemas import QueryRequest

    with pytest.raises(Exception):
        QueryRequest(query="x", top_k=0)
    with pytest.raises(Exception):
        QueryRequest(query="x", top_k=51)


def test_query_request_llm_timeout_range():
    from rag.schemas import QueryRequest

    with pytest.raises(Exception):
        QueryRequest(query="x", llm_timeout=4)
    QueryRequest(query="x", llm_timeout=5)
    QueryRequest(query="x", llm_timeout=300)
    with pytest.raises(Exception):
        QueryRequest(query="x", llm_timeout=301)


def test_ingest_folder_request_round_trip():
    from rag.schemas import IngestFolderRequest

    payload = {"folder_path": "/tmp/docs", "collection_title": "Test"}
    req = IngestFolderRequest(**payload)
    assert req.recursive is True
    assert req.chunk_size == 1000
    assert req.chunking_strategy == "semantic"


# ---------------------------------------------------------------------------
# job store
# ---------------------------------------------------------------------------


def test_job_store_lifecycle(tmp_path):
    from rag.job_store import JobStore, JobStatus

    store = JobStore(str(tmp_path / "jobs.db"))
    jid = store.create_job()
    rec = store.get(jid)
    assert rec["status"] == "queued"

    store.update(jid, JobStatus.RUNNING, progress=0.3)
    assert store.get(jid)["status"] == "running"

    store.update(jid, JobStatus.COMPLETED, result={"chunks": 42})
    rec = store.get(jid)
    assert rec["status"] == "completed"
    assert rec["result"] == {"chunks": 42}


def test_job_store_mark_orphans_failed(tmp_path):
    """After a restart, queued/running jobs from the previous process
    should be flipped to failed so clients aren't left waiting forever."""
    from rag.job_store import JobStore, JobStatus

    db = str(tmp_path / "jobs.db")
    store = JobStore(db)
    queued = store.create_job()
    running = store.create_job()
    done = store.create_job()
    store.update(running, JobStatus.RUNNING)
    store.update(done, JobStatus.COMPLETED, result={"ok": True})

    # Simulate process restart
    store2 = JobStore(db)
    n = store2.mark_orphans_failed()
    assert n == 2

    assert store2.get(queued)["status"] == "failed"
    assert store2.get(running)["status"] == "failed"
    assert store2.get(done)["status"] == "completed"  # untouched


def test_job_store_get_missing_returns_none(tmp_path):
    from rag.job_store import JobStore

    store = JobStore(str(tmp_path / "jobs.db"))
    assert store.get("does-not-exist") is None


# ---------------------------------------------------------------------------
# doc registry + chunk-level diff
# ---------------------------------------------------------------------------


def test_doc_registry_file_dedup(tmp_path):
    from rag.doc_registry import DocRegistry

    reg = DocRegistry(str(tmp_path / "registry.db"))
    reg.upsert("col1", "a.pdf", "filehash1", ["c1", "c2"])
    assert reg.is_duplicate("col1", "a.pdf", "filehash1") is True
    assert reg.is_duplicate("col1", "a.pdf", "filehash2") is False
    assert reg.is_duplicate("col1", "b.pdf", "filehash1") is False


def test_doc_registry_diff_unchanged_only(tmp_path):
    """Same content → everything unchanged, nothing to re-embed."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    chunk_hashes = {"a.pdf-0": sha256_text("alpha"), "a.pdf-1": sha256_text("beta")}
    reg.upsert("col1", "a.pdf", "fh1", list(chunk_hashes.keys()), chunk_hashes)

    # Re-ingest identical content — tentative ids will differ but hashes match
    new_chunks = [
        ("a.pdf-0-new", sha256_text("alpha")),
        ("a.pdf-1-new", sha256_text("beta")),
    ]
    diff = reg.diff_chunks("col1", "a.pdf", new_chunks)
    assert sorted(diff.unchanged) == ["a.pdf-0", "a.pdf-1"]
    assert diff.added == []
    assert diff.removed == []
    assert diff.is_noop is True


def test_doc_registry_diff_partial_edit(tmp_path):
    """Middle chunk edited — first/last untouched."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    chunk_hashes = {
        "a.pdf-0": sha256_text("alpha"),
        "a.pdf-1": sha256_text("beta"),
        "a.pdf-2": sha256_text("gamma"),
    }
    reg.upsert("col1", "a.pdf", "fh1", list(chunk_hashes.keys()), chunk_hashes)

    new_chunks = [
        ("new-0", sha256_text("alpha")),
        ("new-1", sha256_text("beta-edited")),
        ("new-2", sha256_text("gamma")),
    ]
    diff = reg.diff_chunks("col1", "a.pdf", new_chunks)
    assert sorted(diff.unchanged) == ["a.pdf-0", "a.pdf-2"]
    assert diff.removed == ["a.pdf-1"]
    assert [h for _, h in diff.added] == [sha256_text("beta-edited")]
    assert diff.is_noop is False


def test_doc_registry_diff_legacy_record_triggers_full_rebuild(tmp_path):
    """Legacy rows (no chunk_hashes) diff as if nothing matched."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    # Pre-migration: chunk_hashes=None
    reg.upsert("col1", "a.pdf", "fh1", ["old-0", "old-1"])

    new_chunks = [("new-0", sha256_text("content-0")), ("new-1", sha256_text("content-1"))]
    diff = reg.diff_chunks("col1", "a.pdf", new_chunks)
    assert diff.unchanged == []
    assert diff.removed == []
    assert len(diff.added) == 2  # full embed


def test_doc_registry_diff_file_removed_chunks(tmp_path):
    """Trimmed document — old chunks that no longer appear are removed."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    chunk_hashes = {
        "a.pdf-0": sha256_text("alpha"),
        "a.pdf-1": sha256_text("beta"),
        "a.pdf-2": sha256_text("gamma"),
    }
    reg.upsert("col1", "a.pdf", "fh1", list(chunk_hashes.keys()), chunk_hashes)

    new_chunks = [("new-0", sha256_text("alpha"))]
    diff = reg.diff_chunks("col1", "a.pdf", new_chunks)
    assert diff.unchanged == ["a.pdf-0"]
    assert sorted(diff.removed) == ["a.pdf-1", "a.pdf-2"]
    assert diff.added == []


def test_doc_registry_schema_migration_adds_column(tmp_path):
    """Opening an old DB without chunk_hashes should add the column in place."""
    import sqlite3
    from rag.doc_registry import DocRegistry

    db_path = str(tmp_path / "legacy.db")
    # Simulate pre-migration schema
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE documents (
            collection_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_ids TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (collection_id, filename)
        )
    """)
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
        ("col1", "a.pdf", "fh1", json.dumps(["c1"]), "2026-01-01"),
    )
    conn.commit()
    conn.close()

    reg = DocRegistry(db_path)
    existing = reg.get_existing("col1", "a.pdf")
    assert existing is not None
    assert existing["chunk_ids"] == json.dumps(["c1"])
    assert "chunk_hashes" in existing  # migration added the column
    assert reg.get_chunk_hashes("col1", "a.pdf") == {}


# ---------------------------------------------------------------------------
# plan_ingest (incremental re-ingestion planner)
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Minimal duck-typed chunk for planner tests."""

    __slots__ = ("chunk_id", "content")

    def __init__(self, chunk_id, content):
        self.chunk_id = chunk_id
        self.content = content


def test_plan_ingest_first_time(tmp_path):
    """No prior record — everything is new, no diff, all chunks to embed."""
    from rag.doc_registry import DocRegistry

    reg = DocRegistry(str(tmp_path / "registry.db"))
    chunks = [_FakeChunk("a.pdf-0", "alpha"), _FakeChunk("a.pdf-1", "beta")]

    to_embed, diff, final_ids, final_hashes = reg.plan_ingest("col1", "a.pdf", chunks)

    assert diff is None
    assert [c.chunk_id for c in to_embed] == ["a.pdf-0", "a.pdf-1"]
    assert final_ids == ["a.pdf-0", "a.pdf-1"]
    assert set(final_hashes) == {"a.pdf-0", "a.pdf-1"}


def test_plan_ingest_noop_when_content_unchanged(tmp_path):
    """Same chunks re-submitted — nothing to embed, diff is a no-op."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    reg.upsert(
        "col1", "a.pdf", "fh1",
        ["a.pdf-0", "a.pdf-1"],
        {"a.pdf-0": sha256_text("alpha"), "a.pdf-1": sha256_text("beta")},
    )
    chunks = [_FakeChunk("a.pdf-0", "alpha"), _FakeChunk("a.pdf-1", "beta")]

    to_embed, diff, final_ids, final_hashes = reg.plan_ingest("col1", "a.pdf", chunks)

    assert to_embed == []
    assert diff is not None and diff.is_noop
    assert sorted(final_ids) == ["a.pdf-0", "a.pdf-1"]
    assert set(final_hashes) == {"a.pdf-0", "a.pdf-1"}


def test_plan_ingest_partial_edit_embeds_only_changed(tmp_path):
    """Middle chunk edited — only that chunk goes to embed; others kept."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    reg.upsert(
        "col1", "a.pdf", "fh1",
        ["a.pdf-0", "a.pdf-1", "a.pdf-2"],
        {
            "a.pdf-0": sha256_text("alpha"),
            "a.pdf-1": sha256_text("beta"),
            "a.pdf-2": sha256_text("gamma"),
        },
    )
    chunks = [
        _FakeChunk("a.pdf-0", "alpha"),
        _FakeChunk("a.pdf-1", "beta-edited"),
        _FakeChunk("a.pdf-2", "gamma"),
    ]

    to_embed, diff, final_ids, final_hashes = reg.plan_ingest("col1", "a.pdf", chunks)

    assert [c.chunk_id for c in to_embed] == ["a.pdf-1"]
    assert sorted(diff.unchanged) == ["a.pdf-0", "a.pdf-2"]
    assert diff.removed == ["a.pdf-1"]
    assert sorted(final_ids) == ["a.pdf-0", "a.pdf-1", "a.pdf-2"]
    # Unchanged hashes preserved from the old record; new chunk has new hash.
    assert final_hashes["a.pdf-0"] == sha256_text("alpha")
    assert final_hashes["a.pdf-1"] == sha256_text("beta-edited")


def test_plan_ingest_file_trimmed_returns_removed(tmp_path):
    """Document shrank — removed chunks bubble up so the caller can delete them."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    reg.upsert(
        "col1", "a.pdf", "fh1",
        ["a.pdf-0", "a.pdf-1", "a.pdf-2"],
        {
            "a.pdf-0": sha256_text("alpha"),
            "a.pdf-1": sha256_text("beta"),
            "a.pdf-2": sha256_text("gamma"),
        },
    )
    chunks = [_FakeChunk("a.pdf-0", "alpha")]

    to_embed, diff, final_ids, final_hashes = reg.plan_ingest("col1", "a.pdf", chunks)

    assert to_embed == []
    assert diff.unchanged == ["a.pdf-0"]
    assert sorted(diff.removed) == ["a.pdf-1", "a.pdf-2"]
    assert final_ids == ["a.pdf-0"]
    assert list(final_hashes) == ["a.pdf-0"]


def test_plan_ingest_legacy_record_forces_full_rebuild(tmp_path):
    """Legacy record (no chunk_hashes) → treat as full rebuild."""
    from rag.doc_registry import DocRegistry, sha256_text

    reg = DocRegistry(str(tmp_path / "registry.db"))
    reg.upsert("col1", "a.pdf", "fh1", ["old-0", "old-1"])  # no chunk_hashes
    chunks = [_FakeChunk("a.pdf-0", "alpha"), _FakeChunk("a.pdf-1", "beta")]

    to_embed, diff, final_ids, final_hashes = reg.plan_ingest("col1", "a.pdf", chunks)

    assert [c.chunk_id for c in to_embed] == ["a.pdf-0", "a.pdf-1"]
    assert diff.unchanged == []
    assert sorted(final_ids) == ["a.pdf-0", "a.pdf-1"]
    assert final_hashes["a.pdf-0"] == sha256_text("alpha")


# ---------------------------------------------------------------------------
# collections DB
# ---------------------------------------------------------------------------


def test_collections_db_crud(tmp_path):
    from rag.collections_db import CollectionsDB

    db = CollectionsDB(str(tmp_path / "cols.db"))
    db.upsert("c1", {"id": "c1", "title": "first", "file_count": 0})
    assert db.get("c1")["title"] == "first"

    db.upsert("c1", {"id": "c1", "title": "renamed", "file_count": 2})
    assert db.get("c1")["title"] == "renamed"
    assert db.get("c1")["file_count"] == 2

    all_cols = db.load_all()
    assert set(all_cols) == {"c1"}

    db.delete("c1")
    assert db.get("c1") is None


# ---------------------------------------------------------------------------
# LRU eviction helper (copy of the logic from rag/main.py — keeps this test
# decoupled from torch imports)
# ---------------------------------------------------------------------------


def test_lru_eviction_preserves_most_recent():
    cache: OrderedDict[str, str] = OrderedDict()
    max_size = 2

    def evict():
        while len(cache) > max_size:
            cache.popitem(last=False)

    cache["a"] = "A"
    cache["b"] = "B"
    evict()
    assert list(cache) == ["a", "b"]

    # Touch "a" — now "b" is oldest
    cache.move_to_end("a")
    cache["c"] = "C"
    evict()
    assert list(cache) == ["a", "c"]  # "b" was evicted
