"""
Tests for RAG Pipeline Service Components
Demonstrates how the refactored service layer improves testability
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag.pipeline_service import (
    QueryContextualizer,
    AdaptiveAlphaCalculator,
    SemanticCacheHandler,
    RetrieverFactory,
)


class TestQueryContextualizer:
    """Tests for QueryContextualizer"""

    def test_contextualize_without_history(self):
        """Should return original query when no history"""
        query = "What is machine learning?"
        result = QueryContextualizer.contextualize(
            query=query,
            conversation_history=None,
            llm_generator=Mock(),
            query_log=Mock(),
        )
        assert result == query

    def test_contextualize_with_history(self):
        """Should contextualize query with conversation history"""
        query = "How does it work?"
        history = [
            Mock(role="user", content="Tell me about neural networks"),
            Mock(role="assistant", content="Neural networks are..."),
        ]

        llm_generator = Mock()
        llm_generator.contextualize_query.return_value = "How do neural networks work?"

        query_log = Mock()

        result = QueryContextualizer.contextualize(
            query=query,
            conversation_history=history,
            llm_generator=llm_generator,
            query_log=query_log,
        )

        assert result == "How do neural networks work?"
        assert query_log.contextualized_query == "How do neural networks work?"
        llm_generator.contextualize_query.assert_called_once()


class TestAdaptiveAlphaCalculator:
    """Tests for AdaptiveAlphaCalculator"""

    def test_keyword_query_high_alpha(self):
        """Keyword queries should get high alpha (favor vector search)"""
        query = "Find document with id:12345"
        alpha = AdaptiveAlphaCalculator.calculate(query)
        assert alpha == 0.9

    def test_semantic_query_low_alpha(self):
        """Semantic queries should get low alpha (favor BM25)"""
        query = "Explain how transformers work"
        alpha = AdaptiveAlphaCalculator.calculate(query)
        assert alpha == 0.6

    def test_comparative_query_balanced_alpha(self):
        """Comparative queries should get balanced alpha"""
        query = "Compare GPT-3 versus GPT-4"
        alpha = AdaptiveAlphaCalculator.calculate(query)
        assert alpha == 0.7

    def test_default_alpha_fallback(self):
        """Unknown query types should use default alpha"""
        query = "Some random query"
        default = 0.75
        alpha = AdaptiveAlphaCalculator.calculate(query, default_alpha=default)
        assert alpha == default

    def test_case_insensitive_matching(self):
        """Should match patterns case-insensitively"""
        query = "EXPLAIN the concept"
        alpha = AdaptiveAlphaCalculator.calculate(query)
        assert alpha == 0.6


class TestSemanticCacheHandler:
    """Tests for SemanticCacheHandler"""

    @patch("rag.pipeline_service.settings")
    def test_cache_disabled(self, mock_settings):
        """Should return None when cache is disabled"""
        mock_settings.USE_CACHE = False

        result = SemanticCacheHandler.check_cache(
            query="test query",
            top_k=10,
            metadata_filter=None,
            embedding_model=Mock(),
            query_log=Mock(),
        )

        assert result is None

    @patch("rag.pipeline_service.settings")
    @patch("rag.pipeline_service.get_semantic_cache")
    def test_cache_hit(self, mock_get_cache, mock_settings):
        """Should return cached results on cache hit"""
        mock_settings.USE_CACHE = True

        cached_results = [("chunk1", 0.9), ("chunk2", 0.8)]
        mock_cache = Mock()
        mock_cache.get.return_value = cached_results
        mock_get_cache.return_value = mock_cache

        query_log = Mock()
        query_log.metadata = {}
        query_log.retrieval = Mock()

        result = SemanticCacheHandler.check_cache(
            query="test query",
            top_k=10,
            metadata_filter=None,
            embedding_model=Mock(),
            query_log=query_log,
        )

        assert result == cached_results
        assert query_log.metadata["semantic_cache_hit"] is True
        assert query_log.retrieval.final_candidates == 2

    @patch("rag.pipeline_service.settings")
    @patch("rag.pipeline_service.get_semantic_cache")
    def test_cache_miss(self, mock_get_cache, mock_settings):
        """Should return None on cache miss"""
        mock_settings.USE_CACHE = True

        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        result = SemanticCacheHandler.check_cache(
            query="test query",
            top_k=10,
            metadata_filter=None,
            embedding_model=Mock(),
            query_log=Mock(),
        )

        assert result is None


class TestRetrieverFactory:
    """Tests for RetrieverFactory"""

    @patch("rag.pipeline_service.settings")
    @patch("rag.pipeline_service.HybridRetriever")
    def test_create_hybrid_retriever(self, mock_hybrid_retriever, mock_settings):
        """Should create HybridRetriever when hybrid search is enabled"""
        mock_settings.USE_HYBRID_SEARCH = True
        mock_settings.HYBRID_ALPHA = 0.7
        mock_settings.ENABLE_ADAPTIVE_ALPHA = False

        request = Mock()
        request.use_hybrid_search = True
        request.enable_adaptive_alpha = False

        query_log = Mock()
        query_log.metadata = {}

        RetrieverFactory.create(
            request=request,
            vector_store=Mock(),
            embedding_model=Mock(),
            query="test",
            query_log=query_log,
        )

        mock_hybrid_retriever.assert_called_once()
        call_kwargs = mock_hybrid_retriever.call_args[1]
        assert call_kwargs["alpha"] == 0.7

    @patch("rag.pipeline_service.settings")
    @patch("rag.pipeline_service.Retriever")
    def test_create_standard_retriever(self, mock_retriever, mock_settings):
        """Should create standard Retriever when hybrid search is disabled"""
        mock_settings.USE_HYBRID_SEARCH = False

        request = Mock()
        request.use_hybrid_search = False

        RetrieverFactory.create(
            request=request,
            vector_store=Mock(),
            embedding_model=Mock(),
            query="test",
            query_log=Mock(),
        )

        mock_retriever.assert_called_once()


# Integration test example
class TestPipelineIntegration:
    """Integration tests for pipeline components working together"""

    def test_full_pipeline_flow(self):
        """Test that components can be chained together"""
        # This would be a full integration test
        # showing how the service layer components work together
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
