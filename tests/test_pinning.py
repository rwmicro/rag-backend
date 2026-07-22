"""Document pinning: full-text injection that bypasses retrieval.

Covers the three layers: chunk reconstruction from the vector store, the
injection helper (dedup + token budget), and the pin/unpin persistence logic.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("faiss")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from rag.chunking import Chunk
from rag.vectordb import FAISSStore


def _chunk(cid, content, filename, index, vec=None):
    c = Chunk(
        chunk_id=cid,
        content=content,
        metadata={"filename": filename, "chunk_index": index},
    )
    c.embedding = list(vec or [1.0, 0.0, 0.0])
    return c


class TestGetChunksByFilename:
    def test_returns_document_chunks_in_order(self, tmp_path):
        store = FAISSStore(dimension=3, index_path=str(tmp_path / "t.faiss"))
        # Inserted out of order and interleaved with another file.
        store.add_chunks([
            _chunk("b1", "second part", "doc.md", 1),
            _chunk("x0", "other doc", "other.md", 0),
            _chunk("b0", "first part", "doc.md", 0),
        ])

        chunks = store.get_chunks_by_filename("doc.md")

        assert [c.content for c in chunks] == ["first part", "second part"]

    def test_unknown_filename_is_empty(self, tmp_path):
        store = FAISSStore(dimension=3, index_path=str(tmp_path / "t.faiss"))
        assert store.get_chunks_by_filename("nope.md") == []


class TestInjectPinnedChunks:
    @pytest.fixture
    def query_log(self):
        from rag.observability import create_query_log

        return create_query_log("q", "col")

    def _store(self, tmp_path):
        store = FAISSStore(dimension=3, index_path=str(tmp_path / "t.faiss"))
        store.add_chunks([
            _chunk("p0", "pinned intro", "pinned.md", 0),
            _chunk("p1", "pinned body", "pinned.md", 1),
            _chunk("r0", "retrieved stuff", "other.md", 0),
        ])
        return store

    def test_pinned_first_and_retrieved_kept(self, tmp_path, query_log):
        from rag.main import _inject_pinned_chunks

        store = self._store(tmp_path)
        retrieved = [(_chunk("r0", "retrieved stuff", "other.md", 0), 0.8)]

        result = _inject_pinned_chunks(store, ["pinned.md"], retrieved, query_log)

        contents = [c.content for c, _ in result]
        assert contents == ["pinned intro", "pinned body", "retrieved stuff"]
        # Pinned chunks carry a fixed top score.
        assert [s for _, s in result[:2]] == [1.0, 1.0]
        assert query_log.metadata["pinned_files"] == ["pinned.md"]

    def test_retrieved_chunks_from_pinned_file_are_deduplicated(
        self, tmp_path, query_log
    ):
        """The filterIdentifiers rule: no document twice in the context."""
        from rag.main import _inject_pinned_chunks

        store = self._store(tmp_path)
        retrieved = [
            (_chunk("p1", "pinned body", "pinned.md", 1), 0.9),
            (_chunk("r0", "retrieved stuff", "other.md", 0), 0.8),
        ]

        result = _inject_pinned_chunks(store, ["pinned.md"], retrieved, query_log)

        contents = [c.content for c, _ in result]
        assert contents.count("pinned body") == 1
        assert "retrieved stuff" in contents

    def test_token_budget_truncates(self, tmp_path, query_log, monkeypatch):
        from rag.main import _inject_pinned_chunks

        store = self._store(tmp_path)
        # Budget of 5 tokens ~= 20 chars: fits "pinned intro" (12) but not
        # "pinned body" on top of it.
        monkeypatch.setattr(settings, "PINNED_MAX_TOKENS", 5)

        result = _inject_pinned_chunks(store, ["pinned.md"], [], query_log)

        assert [c.content for c, _ in result] == ["pinned intro"]

    def test_store_without_support_is_a_noop(self, query_log):
        from rag.main import _inject_pinned_chunks

        class BareStore:
            pass

        retrieved = [(_chunk("r0", "retrieved stuff", "other.md", 0), 0.8)]
        assert (
            _inject_pinned_chunks(BareStore(), ["pinned.md"], retrieved, query_log)
            == retrieved
        )


class TestPinPersistence:
    def _make_collection(self, cid="pin-col"):
        from rag.main import _get_or_create_collection, _get_collections_db

        _get_or_create_collection(
            collection_id=cid,
            title="Pin test",
            llm_model="llama3.2",
            embedding_model="intfloat/multilingual-e5-large",
        )
        db = _get_collections_db()
        col = db.get(cid)
        col["files"] = ["a.md", "b.md"]
        db.upsert(cid, col)
        return db

    def test_pin_unpin_roundtrip(self):
        from rag.main import _set_document_pin

        db = self._make_collection()

        out = _set_document_pin("pin-col", "a.md", pinned=True)
        assert out["pinned_files"] == ["a.md"]
        # Idempotent.
        out = _set_document_pin("pin-col", "a.md", pinned=True)
        assert out["pinned_files"] == ["a.md"]

        out = _set_document_pin("pin-col", "a.md", pinned=False)
        assert out["pinned_files"] == []
        assert db.get("pin-col")["pinned_files"] == []

    def test_pin_unknown_document_404s(self):
        from fastapi import HTTPException
        from rag.main import _set_document_pin

        self._make_collection()
        with pytest.raises(HTTPException) as exc:
            _set_document_pin("pin-col", "ghost.md", pinned=True)
        assert exc.value.status_code == 404

    def test_get_pinned_files_reads_back(self):
        from rag.main import _get_pinned_files, _set_document_pin

        self._make_collection()
        _set_document_pin("pin-col", "b.md", pinned=True)
        assert _get_pinned_files("pin-col") == ["b.md"]
        assert _get_pinned_files(None) == []
        assert _get_pinned_files("does-not-exist") == []
