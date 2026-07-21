
import pytest
from unittest.mock import MagicMock
import numpy as np

from rag.retrieval import HybridRetriever, normalize_minmax, normalize_zscore
from rag.chunking import Chunk
from config.settings import settings

class TestRetrievalScoring:
    
    @pytest.fixture
    def mock_components(self):
        vector_store = MagicMock()
        embedding_model = MagicMock()
        return vector_store, embedding_model

    def test_normalize_minmax(self):
        # Test basic normalization
        scores = [10.0, 20.0, 30.0]
        normalized = normalize_minmax(scores)
        assert normalized == [0.0, 0.5, 1.0]

        # Test single value
        scores = [10.0]
        normalized = normalize_minmax(scores)
        assert normalized == [1.0]

        # Test uniform values
        scores = [10.0, 10.0, 10.0]
        normalized = normalize_minmax(scores)
        assert normalized == [1.0, 1.0, 1.0]

        # Test empty
        assert normalize_minmax([]) == []

    def test_normalize_zscore(self):
        # Test basic z-score
        scores = [10.0, 20.0, 30.0]
        # Mean=20, Std=8.16
        # Z = [-1.22, 0, 1.22]
        # Sigmoidlike should map to ~[0,1]
        normalized = normalize_zscore(scores)
        assert len(normalized) == 3
        assert 0.0 <= min(normalized) <= 1.0
        assert 0.0 <= max(normalized) <= 1.0
        # Check order is preserved
        assert normalized[0] < normalized[1] < normalized[2]

    def test_hybrid_retrieval_minmax_integration(self, mock_components):
        """Verify that minmax normalization is actually used when configured"""
        vector_store, embedding_model = mock_components
        retriever = HybridRetriever(vector_store, embedding_model, alpha=0.5)

        # Mock vector search results
        vector_results = [
            (Chunk(chunk_id="1", content="content1", metadata={}), 0.8),
            (Chunk(chunk_id="2", content="content2", metadata={}), 0.5),
        ]
        # Configure parent Retriever.retrieve mock
        with pytest.MonkeyPatch.context():
            # We mock the super().retrieve method by mocking the vector_store.search actually
            # since the base Retriever calls vector_store.search
            vector_store.search.return_value = vector_results
            embedding_model.encode_single.return_value = np.array([0.1])

            # Mock BM25 search
            retriever._bm25_search = MagicMock(return_value=[
                (Chunk(chunk_id="1", content="content1", metadata={}), 10.0), # High BM25
                (Chunk(chunk_id="3", content="content3", metadata={}), 5.0),
            ])
            
            # Since we are mocking internal methods we need to be careful. 
            # Ideally we rely on the component contracts.
            
            # Let's call retrieve with forced minmax
            results = retriever.retrieve("test", top_k=2, build_bm25=False, normalization_method="minmax")
            
            # Chunk 1:
            # Vector raw: 0.8. Vector is [0.8, 0.5] -> norm [1.0, 0.0]
            # Chunk 1 vector norm = 1.0 * 0.5 (alpha) = 0.5
            # BM25 raw: [10.0, 5.0] -> norm [1.0, 0.0]
            # Chunk 1 bm25 norm = 1.0 * 0.5 (1-alpha) = 0.5
            # Total Chunk 1 = 1.0
            
            # Chunk 2:
            # Vector norm = 0.0 * 0.5 = 0.0
            # BM25: Not present -> 0
            # Total Chunk 2 = 0.0
            
            # Chunk 3: 
            # Vector: Not present -> 0
            # BM25 norm = 0.0
            # Total Chunk 3 = 0.0
            
            assert results[0][0].chunk_id == "1"
            assert results[0][1] == 1.0
            
    def test_default_normalization_method_is_rrf(self, mock_components):
        """retrieve() falls back to settings.SCORE_NORMALIZATION_METHOD, which is 'rrf'.

        The two methods are distinguishable by score magnitude: RRF yields
        1/(RRF_K + rank), while min-max yields alpha-weighted values near 1.
        """
        assert settings.SCORE_NORMALIZATION_METHOD == "rrf"

        vector_store, embedding_model = mock_components
        retriever = HybridRetriever(vector_store, embedding_model)

        vector_store.search.return_value = [
            (Chunk(chunk_id="1", content="c", metadata={}), 0.9)
        ]
        embedding_model.encode_single.return_value = np.array([0.1])
        retriever._bm25_search = MagicMock(return_value=[])

        results = retriever.retrieve("test", top_k=1, build_bm25=False)

        # Single result, rank 1, present in one list only.
        assert results[0][1] == pytest.approx(1 / (settings.RRF_K + 1))

    def test_explicit_minmax_overrides_the_default(self, mock_components):
        """Passing normalization_method explicitly wins over the setting."""
        vector_store, embedding_model = mock_components
        retriever = HybridRetriever(vector_store, embedding_model, alpha=0.7)

        vector_store.search.return_value = [
            (Chunk(chunk_id="1", content="c", metadata={}), 0.9)
        ]
        embedding_model.encode_single.return_value = np.array([0.1])
        retriever._bm25_search = MagicMock(return_value=[])

        results = retriever.retrieve(
            "test", top_k=1, build_bm25=False, normalization_method="minmax"
        )

        # A lone vector score normalises to 1.0, weighted by alpha; no BM25 side.
        assert results[0][1] == pytest.approx(0.7)
