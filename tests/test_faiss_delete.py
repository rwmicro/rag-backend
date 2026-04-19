"""FAISS delete_by_chunk_ids — proves the rebuild actually removes the vectors.

Skipped when FAISS isn't installed; this keeps the unit-test suite runnable on
lightweight envs while still covering the code path whenever the full deps are
present.
"""

import os
import sys

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chunking import Chunk
from rag.vectordb import FAISSStore


def _chunk(cid: str, vec, meta=None):
    c = Chunk(chunk_id=cid, content=f"content-{cid}", metadata=meta or {"filename": "a.pdf"})
    c.embedding = np.asarray(vec, dtype=np.float32).tolist()
    return c


def test_delete_by_chunk_ids_removes_only_targeted(tmp_path):
    index_path = str(tmp_path / "test.faiss")
    store = FAISSStore(dimension=4, index_path=index_path)

    store.add_chunks([
        _chunk("c0", [1.0, 0.0, 0.0, 0.0]),
        _chunk("c1", [0.0, 1.0, 0.0, 0.0]),
        _chunk("c2", [0.0, 0.0, 1.0, 0.0]),
        _chunk("c3", [0.0, 0.0, 0.0, 1.0]),
    ])
    assert store.index.ntotal == 4

    store.delete_by_chunk_ids(["c1", "c3"])

    assert store.index.ntotal == 2
    remaining = {m["chunk_id"] for m in store.chunks_metadata}
    assert remaining == {"c0", "c2"}

    # Querying the vector that *was* c1 should no longer surface c1.
    hits = store.search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), top_k=4)
    hit_ids = {c.chunk_id for c, _ in hits}
    assert "c1" not in hit_ids and "c3" not in hit_ids
    assert hit_ids.issubset({"c0", "c2"})


def test_delete_by_chunk_ids_noop_when_id_absent(tmp_path):
    index_path = str(tmp_path / "test.faiss")
    store = FAISSStore(dimension=3, index_path=index_path)
    store.add_chunks([_chunk("c0", [1.0, 0.0, 0.0])])

    store.delete_by_chunk_ids(["does-not-exist"])
    assert store.index.ntotal == 1


def test_delete_by_chunk_ids_empty_input_is_noop(tmp_path):
    index_path = str(tmp_path / "test.faiss")
    store = FAISSStore(dimension=3, index_path=index_path)
    store.add_chunks([_chunk("c0", [1.0, 0.0, 0.0])])

    store.delete_by_chunk_ids([])
    assert store.index.ntotal == 1


def test_delete_then_re_add_same_id_works(tmp_path):
    """Incremental re-ingest pattern: drop stale chunk, add the new one."""
    index_path = str(tmp_path / "test.faiss")
    store = FAISSStore(dimension=3, index_path=index_path)

    store.add_chunks([_chunk("c0", [1.0, 0.0, 0.0])])
    store.delete_by_chunk_ids(["c0"])
    assert store.index.ntotal == 0

    store.add_chunks([_chunk("c0", [0.0, 1.0, 0.0])])
    assert store.index.ntotal == 1
    assert store.chunks_metadata[0]["chunk_id"] == "c0"

    # Search should match the *new* vector direction, not the old one.
    hits = store.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)
    assert len(hits) == 1 and hits[0][0].chunk_id == "c0"
