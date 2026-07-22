"""Guards born from the AnythingLLM comparison (2026-07).

Two silent quality killers:
1. Chunks longer than the embedder's max_seq_length are truncated at embedding
   time — retrieval never sees their second half. CHUNK_SIZE=1000 against
   e5-large's 512-token limit meant every default chunk was half-invisible.
2. No relevance floor: with MIN_SIMILARITY_SCORE=0 and min-max-normalized
   hybrid scores (top hit always 1.0), "the corpus has no answer" still
   produced a confident-looking context.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from rag.chunking import Chunk
from rag.embeddings import EmbeddingModel


@pytest.fixture
def captured_warnings():
    """Collect loguru WARNING+ messages emitted during the test."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


def _bare_embedding_model(max_seq_length):
    """An EmbeddingModel without __init__ (no model download) for guard tests."""
    m = object.__new__(EmbeddingModel)
    fake = MagicMock()
    fake.max_seq_length = max_seq_length
    # Fake tokenizer: one token per whitespace-separated word.
    fake.tokenizer = lambda text, truncation: {"input_ids": text.split()}
    m.model = fake
    return m


class TestTruncationGuard:
    def test_warns_when_text_exceeds_token_limit(self, captured_warnings):
        model = _bare_embedding_model(max_seq_length=8)

        # 20 words of >= 3 chars each -> passes the 2*max_seq char pre-filter
        # (> 16 chars) and exceeds the 8-token limit.
        long_text = "word " * 20
        model._warn_if_truncating([long_text])

        assert any("truncated" in msg for msg in captured_warnings)
        assert any("8-token limit" in msg for msg in captured_warnings)

    def test_silent_when_texts_fit(self, captured_warnings):
        model = _bare_embedding_model(max_seq_length=512)
        model._warn_if_truncating(["short text", "another short one"])
        assert captured_warnings == []

    def test_char_prefilter_avoids_tokenizing_short_texts(self):
        model = _bare_embedding_model(max_seq_length=512)
        model.model.tokenizer = MagicMock(
            side_effect=AssertionError("tokenizer must not run for short texts")
        )
        # 100 chars < 2 * 512 -> must never reach the tokenizer.
        model._warn_if_truncating(["x" * 100])

    def test_default_chunk_size_fits_e5_budget(self):
        """The CODE default must stay under e5-large's 512-token window, with
        room for the 'passage: ' prefix and a contextual-summary prefix.

        Instantiated without the env file on purpose: a local .env overriding
        CHUNK_SIZE is the developer's choice, but the shipped default has to be
        safe. (The first run of this test flagged exactly that: a leftover
        .env with CHUNK_SIZE=1000 silently reintroducing the truncation bug.)
        """
        from config.settings import Settings

        assert Settings(_env_file=None).CHUNK_SIZE <= 480


class TestRerankerFloor:
    @pytest.fixture
    def query_log(self):
        from rag.observability import create_query_log

        return create_query_log("q", None)

    def _chunks(self, scores):
        return [
            (Chunk(chunk_id=f"c{i}", content=f"content {i}", metadata={}), s)
            for i, s in enumerate(scores)
        ]

    def test_drops_chunks_below_floor(self, query_log, monkeypatch):
        from rag.main import _apply_reranker_floor

        monkeypatch.setattr(settings, "RERANKER_TYPE", "cross-encoder")
        monkeypatch.setattr(settings, "MIN_RERANKER_SCORE", 0.3)

        kept = _apply_reranker_floor(self._chunks([0.9, 0.31, 0.29, 0.05]), query_log)

        assert [s for _, s in kept] == [0.9, 0.31]
        assert query_log.retrieval.filtered_by_min_score == 2

    def test_can_drop_everything(self, query_log, monkeypatch):
        """An empty result is the point: better no context than noise."""
        from rag.main import _apply_reranker_floor

        monkeypatch.setattr(settings, "RERANKER_TYPE", "cross-encoder")
        monkeypatch.setattr(settings, "MIN_RERANKER_SCORE", 0.3)

        assert _apply_reranker_floor(self._chunks([0.1, 0.02]), query_log) == []

    def test_disabled_when_zero(self, query_log, monkeypatch):
        from rag.main import _apply_reranker_floor

        monkeypatch.setattr(settings, "MIN_RERANKER_SCORE", 0.0)
        chunks = self._chunks([0.1])
        assert _apply_reranker_floor(chunks, query_log) == chunks

    def test_skipped_for_llm_reranker(self, query_log, monkeypatch):
        """LLM reranker scores are not sigmoid outputs — the floor must not apply."""
        from rag.main import _apply_reranker_floor

        monkeypatch.setattr(settings, "RERANKER_TYPE", "llm")
        monkeypatch.setattr(settings, "MIN_RERANKER_SCORE", 0.3)
        chunks = self._chunks([0.1])
        assert _apply_reranker_floor(chunks, query_log) == chunks

    def test_accumulates_with_prior_filter(self, query_log, monkeypatch):
        from rag.main import _apply_reranker_floor

        monkeypatch.setattr(settings, "RERANKER_TYPE", "cross-encoder")
        monkeypatch.setattr(settings, "MIN_RERANKER_SCORE", 0.3)

        query_log.retrieval.filtered_by_min_score = 3  # pre-rerank filter
        _apply_reranker_floor(self._chunks([0.9, 0.1]), query_log)
        assert query_log.retrieval.filtered_by_min_score == 4
