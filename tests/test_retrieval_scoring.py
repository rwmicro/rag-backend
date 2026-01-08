
import pytest
from unittest.mock import MagicMock
import numpy as np

from backend.rag.retrieval import HybridRetriever, normalize_minmax, normalize_zscore, reciprocal_rank_fusion
from backend.rag.chunking import Chunk
from backend.config.settings import settings

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
            (Chunk("1", "content1"), 0.8),
            (Chunk("2", "content2"), 0.5),
        ]
        # Configure parent Retriever.retrieve mock
        with pytest.MonkeyPatch.context() as m:
            # We mock the super().retrieve method by mocking the vector_store.search actually
            # since the base Retriever calls vector_store.search
            vector_store.search.return_value = vector_results
            embedding_model.encode_single.return_value = np.array([0.1])

            # Mock BM25 search
            retriever._bm25_search = MagicMock(return_value=[
                (Chunk("1", "content1"), 10.0), # High BM25
                (Chunk("3", "content3"), 5.0),
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
            
    def test_default_settings_usage(self, mock_components):
        """Verify that the retrieve method uses the default setting when None is passed"""
        vector_store, embedding_model = mock_components
        retriever = HybridRetriever(vector_store, embedding_model)
        
        # We need to verify what normalization method is used inside.
        # This is slightly white-box.
        
        # But we can check if it behaves like normalized scores (high) or RRF scores (low)
        vector_store.search.return_value = [(Chunk("1", "c"), 0.9)]
        embedding_model.encode_single.return_value = np.array([0.1])
        retriever._bm25_search = MagicMock(return_value=[])
        
        # Force the settings to be minmax for this test context, though we updated the file
        # The imported settings object might need reloading or patching if we want to be safe, 
        # but since we edited the file on disk, if the test process starts fresh it will pick it up.
        # However, for this specific test method, let's verify assumptions.
        
        results = retriever.retrieve("test", top_k=1, build_bm25=False)
        score = results[0][1]
        
        # If minmax, score should be roughly alpha * 1.0 = 0.7 (default alpha)
        # If RRF, score should be 1/(60+1) ~ 0.016
        
        # With single result in minmax:
        # Vector scores: [0.9] -> minmax -> [1.0] (handle single value case)
        # Score = 1.0 * 0.7 = 0.7
        
        assert score > 0.1, f"Score {score} is too low, expected MinMax behavior (default)"
