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
    normalize_scores,
    reciprocal_rank_fusion,
    deduplicate_chunks
)


class TestScoreNormalization:
    """Test score normalization methods"""

    def test_minmax_normalization(self):
        """Test min-max normalization"""
        chunks = [
            (Chunk("1", "content1", {}), 0.9),
            (Chunk("2", "content2", {}), 0.5),
            (Chunk("3", "content3", {}), 0.1),
        ]

        normalized = normalize_scores(chunks, method="minmax")

        # Check that scores are in [0, 1] range
        scores = [score for _, score in normalized]
        assert all(0 <= score <= 1 for score in scores)

        # Highest original score should be 1.0
        assert normalized[0][1] == 1.0

        # Lowest original score should be 0.0
        assert normalized[2][1] == 0.0

    def test_zscore_normalization(self):
        """Test z-score normalization"""
        chunks = [
            (Chunk("1", "content1", {}), 0.9),
            (Chunk("2", "content2", {}), 0.5),
            (Chunk("3", "content3", {}), 0.5),
            (Chunk("4", "content4", {}), 0.1),
        ]

        normalized = normalize_scores(chunks, method="zscore")

        # After z-score, mean should be ~0
        scores = [score for _, score in normalized]
        mean_score = np.mean(scores)
        assert abs(mean_score) < 0.1  # Close to 0

    def test_rrf_normalization(self):
        """Test Reciprocal Rank Fusion scoring"""
        chunks = [
            (Chunk("1", "content1", {}), 0.9),
            (Chunk("2", "content2", {}), 0.5),
            (Chunk("3", "content3", {}), 0.1),
        ]

        normalized = normalize_scores(chunks, method="rrf", k=60)

        # RRF scores should be positive
        scores = [score for _, score in normalized]
        assert all(score > 0 for score in scores)

        # Scores should be in descending order (preserved from original)
        assert scores == sorted(scores, reverse=True)

    def test_empty_chunks_normalization(self):
        """Test normalization with empty chunk list"""
        chunks = []

        for method in ["minmax", "zscore", "rrf"]:
            normalized = normalize_scores(chunks, method=method)
            assert len(normalized) == 0


class TestReciprocalRankFusion:
    """Test RRF algorithm"""

    def test_rrf_basic(self):
        """Test basic RRF fusion"""
        # Two result lists with some overlap
        results1 = [
            (Chunk("A", "content_a", {}), 0.9),
            (Chunk("B", "content_b", {}), 0.8),
            (Chunk("C", "content_c", {}), 0.7),
        ]

        results2 = [
            (Chunk("B", "content_b", {}), 0.95),
            (Chunk("D", "content_d", {}), 0.85),
            (Chunk("A", "content_a", {}), 0.75),
        ]

        fused = reciprocal_rank_fusion([results1, results2], k=60)

        # B should rank high (top in both lists)
        chunk_ids = [chunk.chunk_id for chunk, _ in fused]
        assert "B" in chunk_ids

        # All unique chunks should be in result
        assert len(set(chunk_ids)) == 4  # A, B, C, D

    def test_rrf_single_list(self):
        """Test RRF with single result list"""
        results = [
            (Chunk("A", "content_a", {}), 0.9),
            (Chunk("B", "content_b", {}), 0.8),
        ]

        fused = reciprocal_rank_fusion([results], k=60)

        # Should return same order
        assert len(fused) == 2
        assert fused[0][0].chunk_id == "A"
        assert fused[1][0].chunk_id == "B"

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
                "chunk1",
                "This is some content about AI.",
                {},
                embedding=[0.1, 0.2, 0.3, 0.4]
            ), 0.9),
            (Chunk(
                "chunk2",
                "This is some content about AI.",  # Duplicate
                {},
                embedding=[0.1, 0.2, 0.3, 0.4]
            ), 0.85),
            (Chunk(
                "chunk3",
                "Completely different topic about biology.",
                {},
                embedding=[0.9, 0.8, 0.7, 0.6]
            ), 0.8),
        ]

        # With high threshold, duplicates should be removed
        deduplicated = deduplicate_chunks(chunks_with_scores, threshold=0.99)

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
                "chunk1",
                "Content A",
                {},
                embedding=[0.1, 0.2, 0.3]
            ), 0.9),
            (Chunk(
                "chunk2",
                "Content B",
                {},
                embedding=[0.9, 0.8, 0.7]
            ), 0.8),
        ]

        deduplicated = deduplicate_chunks(chunks_with_scores, threshold=0.9)

        # No duplicates, should keep all
        assert len(deduplicated) == 2

    def test_deduplicate_without_embeddings(self):
        """Test deduplication when chunks don't have embeddings"""
        chunks_with_scores = [
            (Chunk("chunk1", "Content A", {}), 0.9),
            (Chunk("chunk2", "Content B", {}), 0.8),
        ]

        # Should not crash, just return original
        deduplicated = deduplicate_chunks(chunks_with_scores, threshold=0.9)

        assert len(deduplicated) == 2

    def test_deduplicate_threshold_effect(self):
        """Test that threshold affects deduplication"""
        chunks_with_scores = [
            (Chunk(
                "chunk1",
                "Similar content",
                {},
                embedding=[0.1, 0.2, 0.3]
            ), 0.9),
            (Chunk(
                "chunk2",
                "Similar content",
                {},
                embedding=[0.11, 0.21, 0.31]  # Very similar
            ), 0.85),
        ]

        # High threshold - should remove duplicate
        deduplicated_high = deduplicate_chunks(chunks_with_scores, threshold=0.95)
        assert len(deduplicated_high) == 1

        # Low threshold - should keep both
        deduplicated_low = deduplicate_chunks(chunks_with_scores, threshold=0.5)
        assert len(deduplicated_low) == 2


class TestChunkScoring:
    """Test chunk scoring functionality"""

    def test_score_descending_order(self):
        """Test that chunks are returned in score-descending order"""
        chunks = [
            (Chunk("1", "content1", {}), 0.5),
            (Chunk("2", "content2", {}), 0.9),
            (Chunk("3", "content3", {}), 0.7),
        ]

        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)

        scores = [score for _, score in sorted_chunks]
        assert scores == [0.9, 0.7, 0.5]

    def test_score_filtering(self):
        """Test filtering chunks by minimum score"""
        chunks = [
            (Chunk("1", "content1", {}), 0.9),
            (Chunk("2", "content2", {}), 0.5),
            (Chunk("3", "content3", {}), 0.2),
        ]

        min_score = 0.4
        filtered = [(chunk, score) for chunk, score in chunks if score >= min_score]

        assert len(filtered) == 2
        assert all(score >= min_score for _, score in filtered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
