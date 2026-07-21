"""
Tests for Retrieval Components
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunking import Chunk
from rag.retrieval import (
    normalize_minmax,
    normalize_zscore,
    reciprocal_rank_fusion,
    deduplicate_chunks
)


class TestScoreNormalization:
    """Test score normalization helpers (operate on plain score lists)."""

    def test_minmax_normalization(self):
        """Min-max rescales to [0, 1] with the extremes pinned."""
        normalized = normalize_minmax([0.9, 0.5, 0.1])

        assert all(0 <= score <= 1 for score in normalized)
        assert normalized[0] == 1.0  # highest -> 1
        assert normalized[2] == 0.0  # lowest  -> 0
        # Order is preserved (it maps element-wise, it does not sort).
        assert normalized == sorted(normalized, reverse=True)

    def test_minmax_constant_scores(self):
        """A flat list has no spread to rescale — everything collapses to 1.0."""
        assert normalize_minmax([0.4, 0.4, 0.4]) == [1.0, 1.0, 1.0]

    def test_zscore_normalization(self):
        """Z-score standardises then squashes through a sigmoid into [0, 1]."""
        normalized = normalize_zscore([0.9, 0.5, 0.5, 0.1])

        assert all(0 <= score <= 1 for score in normalized)
        # The sigmoid is centred on the mean, so the mean maps to ~0.5 —
        # NOT to 0, which is what a raw z-score would give.
        assert abs(np.mean(normalized) - 0.5) < 0.1
        assert normalized[0] > normalized[1]  # ranking preserved
        assert normalized[1] == normalized[2]  # equal inputs -> equal outputs

    def test_zscore_constant_scores(self):
        """Zero variance means no z-score is defined — fall back to 1.0."""
        assert normalize_zscore([0.4, 0.4, 0.4]) == [1.0, 1.0, 1.0]

    def test_empty_scores_normalization(self):
        """Both helpers tolerate an empty list."""
        assert normalize_minmax([]) == []
        assert normalize_zscore([]) == []


class TestReciprocalRankFusion:
    """Test RRF algorithm"""

    def test_rrf_basic(self):
        """Test basic RRF fusion"""
        # Two result lists with some overlap
        results1 = [
            (Chunk(chunk_id="A", content="content_a", metadata={}), 0.9),
            (Chunk(chunk_id="B", content="content_b", metadata={}), 0.8),
            (Chunk(chunk_id="C", content="content_c", metadata={}), 0.7),
        ]

        results2 = [
            (Chunk(chunk_id="B", content="content_b", metadata={}), 0.95),
            (Chunk(chunk_id="D", content="content_d", metadata={}), 0.85),
            (Chunk(chunk_id="A", content="content_a", metadata={}), 0.75),
        ]

        # Returns a dict: {chunk_id: (chunk, rrf_score)} — not a ranked list.
        fused = reciprocal_rank_fusion([results1, results2], k=60)

        assert set(fused.keys()) == {"A", "B", "C", "D"}

        # B is rank 2 then rank 1, so it beats everything appearing in one list only.
        scores = {cid: score for cid, (_, score) in fused.items()}
        assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["B"] > scores["C"]
        assert scores["B"] > scores["D"]
        # A appears in both lists too (rank 1 then rank 3).
        assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)

    def test_rrf_single_list(self):
        """Test RRF with single result list"""
        results = [
            (Chunk(chunk_id="A", content="content_a", metadata={}), 0.9),
            (Chunk(chunk_id="B", content="content_b", metadata={}), 0.8),
        ]

        fused = reciprocal_rank_fusion([results], k=60)

        assert len(fused) == 2
        # Rank order is preserved through the 1/(k+rank) weighting.
        assert fused["A"][1] > fused["B"][1]

    def test_rrf_empty_lists(self):
        """Test RRF with empty lists"""
        fused = reciprocal_rank_fusion([], k=60)
        assert len(fused) == 0

        fused = reciprocal_rank_fusion([[], []], k=60)
        assert len(fused) == 0


class TestDeduplication:
    """Test chunk deduplication"""

    def test_deduplicate_similar_chunks(self):
        """Test removing similar/duplicate chunks"""
        # Create chunks with embeddings
        chunks_with_scores = [
            (Chunk(
                chunk_id="chunk1",
                content="This is some content about AI.",
                metadata={},
                embedding=[0.1, 0.2, 0.3, 0.4]
            ), 0.9),
            (Chunk(
                chunk_id="chunk2",
                content="This is some content about AI.",  # Duplicate
                metadata={},
                embedding=[0.1, 0.2, 0.3, 0.4]
            ), 0.85),
            (Chunk(
                chunk_id="chunk3",
                content="Completely different topic about biology.",
                metadata={},
                embedding=[0.9, 0.8, 0.7, 0.6]
            ), 0.8),
        ]

        # With high threshold, duplicates should be removed
        deduplicated = deduplicate_chunks(chunks_with_scores, similarity_threshold=0.99)

        # Should keep only 2 chunks (chunk1 and chunk3)
        assert len(deduplicated) == 2

        # Highest scoring chunks should be kept
        chunk_ids = [chunk.chunk_id for chunk, _ in deduplicated]
        assert "chunk1" in chunk_ids
        assert "chunk3" in chunk_ids
        assert "chunk2" not in chunk_ids  # Duplicate, lower score

    def test_deduplicate_no_duplicates(self):
        """Test deduplication when no duplicates exist"""
        chunks_with_scores = [
            (Chunk(
                chunk_id="chunk1",
                content="Content A",
                metadata={},
                embedding=[0.1, 0.2, 0.3]
            ), 0.9),
            (Chunk(
                chunk_id="chunk2",
                content="Content B",
                metadata={},
                embedding=[0.9, 0.8, 0.7]
            ), 0.8),
        ]

        deduplicated = deduplicate_chunks(chunks_with_scores, similarity_threshold=0.9)

        # No duplicates, should keep all
        assert len(deduplicated) == 2

    def test_deduplicate_without_embeddings(self):
        """Test deduplication when chunks don't have embeddings"""
        chunks_with_scores = [
            (Chunk(chunk_id="chunk1", content="Content A", metadata={}), 0.9),
            (Chunk(chunk_id="chunk2", content="Content B", metadata={}), 0.8),
        ]

        # Should not crash, just return original
        deduplicated = deduplicate_chunks(chunks_with_scores, similarity_threshold=0.9)

        assert len(deduplicated) == 2

    def test_deduplicate_threshold_effect(self):
        """A LOWER threshold deduplicates more aggressively, not less.

        similarity_threshold is the cosine similarity at which two chunks are
        considered duplicates, so lowering it makes more pairs qualify.
        """
        # These two vectors have a cosine similarity of exactly 0.8.
        chunks_with_scores = [
            (Chunk(chunk_id="chunk1", content="Similar content", metadata={}, embedding=[1.0, 0.0, 0.0]), 0.9),
            (Chunk(chunk_id="chunk2", content="Similar content", metadata={}, embedding=[0.8, 0.6, 0.0]), 0.85),
        ]

        # Threshold above the pair's similarity -> not duplicates -> both kept.
        deduplicated_strict = deduplicate_chunks(
            chunks_with_scores, similarity_threshold=0.95
        )
        assert len(deduplicated_strict) == 2

        # Threshold below it -> treated as duplicates -> only the best-scoring kept.
        deduplicated_loose = deduplicate_chunks(
            chunks_with_scores, similarity_threshold=0.5
        )
        assert len(deduplicated_loose) == 1
        assert deduplicated_loose[0][0].chunk_id == "chunk1"


class TestChunkScoring:
    """Test chunk scoring functionality"""

    def test_score_descending_order(self):
        """Test that chunks are returned in score-descending order"""
        chunks = [
            (Chunk(chunk_id="1", content="content1", metadata={}), 0.5),
            (Chunk(chunk_id="2", content="content2", metadata={}), 0.9),
            (Chunk(chunk_id="3", content="content3", metadata={}), 0.7),
        ]

        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)

        scores = [score for _, score in sorted_chunks]
        assert scores == [0.9, 0.7, 0.5]

    def test_score_filtering(self):
        """Test filtering chunks by minimum score"""
        chunks = [
            (Chunk(chunk_id="1", content="content1", metadata={}), 0.9),
            (Chunk(chunk_id="2", content="content2", metadata={}), 0.5),
            (Chunk(chunk_id="3", content="content3", metadata={}), 0.2),
        ]

        min_score = 0.4
        filtered = [(chunk, score) for chunk, score in chunks if score >= min_score]

        assert len(filtered) == 2
        assert all(score >= min_score for _, score in filtered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
