"""
RAG Pipeline Service
Breaks down the massive _execute_rag_pipeline() function into smaller, testable components
"""

from typing import List, Tuple, Dict, Any, Optional
import time
from loguru import logger

from .chunking import Chunk
from .generation import LLMGenerator
from config.settings import settings


class SemanticCacheHandler:
    """Handles semantic cache operations"""

    @staticmethod
    def check_cache(
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict],
        embedding_model,
        query_log: Any,
    ) -> Optional[List[Tuple[Chunk, float]]]:
        """
        Check semantic cache for cached results.

        Returns:
            Cached results if found, None otherwise
        """
        if not settings.USE_CACHE:
            return None

        try:
            from .cache import get_semantic_cache

            semantic_cache = get_semantic_cache(embedding_model)

            cached_results = semantic_cache.get(
                query=query, top_k=top_k, metadata_filter=metadata_filter
            )

            if cached_results:
                logger.info(f"Semantic cache HIT for query: '{query[:50]}...'")
                query_log.metadata["semantic_cache_hit"] = True
                query_log.retrieval.final_candidates = len(cached_results)
                return cached_results

            logger.debug("Semantic cache MISS")
            return None

        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}")
            return None


class QueryContextualizer:
    """Handles query contextualization with conversation history"""

    @staticmethod
    def contextualize(
        query: str,
        conversation_history: Optional[List],
        llm_generator: LLMGenerator,
        query_log: Any,
    ) -> str:
        """
        Contextualize query using conversation history.

        Returns:
            Contextualized query string
        """
        if not conversation_history:
            return query

        try:
            # Convert Pydantic models to dicts if needed
            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation_history
            ]

            contextualized_query = llm_generator.contextualize_query(
                query, history_dicts
            )

            logger.info(f"Contextualized query: {contextualized_query}")
            query_log.contextualized_query = contextualized_query

            return contextualized_query

        except Exception as e:
            logger.warning(f"Query contextualization failed: {e}, using original query")
            return query


class QueryRouter:
    """Handles automatic query routing"""

    @staticmethod
    def apply_routing(query: str, request: Any, query_router, query_log: Any) -> None:
        """
        Apply automatic query routing to determine retrieval strategy.

        Modifies request in-place with routing decisions.
        """
        if not (request.auto_route and settings.ENABLE_AUTO_ROUTING and query_router):
            return

        logger.info("Applying automatic query routing...")

        try:
            query_type, auto_strategy = query_router.route(query)

            # Override request parameters with router's strategy
            request.use_hybrid_search = auto_strategy.use_hybrid_search
            request.use_multi_query = auto_strategy.use_multi_query
            request.use_hyde = auto_strategy.use_hyde
            request.use_graph_rag = auto_strategy.use_graph_rag
            request.use_reranking = auto_strategy.use_reranking
            request.num_query_variations = auto_strategy.num_query_variations
            request.num_hypothetical_docs = auto_strategy.num_hypothetical_docs
            request.hyde_fusion = auto_strategy.hyde_fusion
            request.graph_expansion_depth = auto_strategy.graph_expansion_depth
            request.graph_alpha = auto_strategy.graph_alpha

            # Log routing decision
            query_log.routing_decision = {
                "query_type": query_type.value,
                "strategy": auto_strategy.to_dict(),
                "mode": settings.ROUTER_MODE,
            }

            logger.info(
                f"Router selected {query_type.value}: "
                f"hybrid={auto_strategy.use_hybrid_search}, "
                f"multi_query={auto_strategy.use_multi_query}, "
                f"hyde={auto_strategy.use_hyde}"
            )

        except Exception as e:
            logger.warning(f"Query routing failed: {e}, using request parameters")


class AdaptiveAlphaCalculator:
    """Calculates adaptive alpha for hybrid search based on query characteristics"""

    @staticmethod
    def calculate(query: str, default_alpha: float = 0.7) -> float:
        """
        Calculate optimal alpha based on query type.

        Args:
            query: Query text
            default_alpha: Default alpha if no pattern matches

        Returns:
            Optimal alpha value (0.0-1.0)
        """
        try:
            query_lower = query.lower()

            # Keyword-heavy queries (IDs, codes, exact terms) → favor vector search
            if any(
                pattern in query_lower
                for pattern in ["id:", "code:", "exact:", "specific:", "#"]
            ):
                alpha = 0.9
                logger.info(f"Detected keyword/exact-match query → alpha={alpha}")
                return alpha

            # Semantic/conceptual queries → favor BM25
            if any(
                pattern in query_lower
                for pattern in ["explain", "describe", "what is", "how does", "why"]
            ):
                alpha = 0.6
                logger.info(f"Detected semantic query → alpha={alpha}")
                return alpha

            # Comparative queries → balanced alpha
            if any(
                pattern in query_lower
                for pattern in ["compare", "difference", "versus", "vs"]
            ):
                alpha = 0.7
                logger.info(f"Detected comparative query → alpha={alpha}")
                return alpha

            # Default
            logger.debug(f"Using default alpha={default_alpha}")
            return default_alpha

        except Exception as e:
            logger.warning(
                f"Alpha calculation failed: {e}, using default={default_alpha}"
            )
            return default_alpha


class MultilingualPipelineHandler:
    """Handles multilingual retrieval pipeline"""

    @staticmethod
    def execute(
        query: str,
        request: Any,
        vector_store,
        llm_generator: LLMGenerator,
        query_log: Any,
    ) -> Tuple[List[Tuple[Chunk, float]], float]:
        """
        Execute multilingual retrieval pipeline.

        Returns:
            Tuple of (chunks_with_scores, retrieval_start_time)
        """
        logger.info("Using multilingual retrieval pipeline...")
        retrieval_start = time.time()

        from .multilingual_pipeline import create_multilingual_pipeline

        # Create multilingual pipeline
        ml_pipeline = create_multilingual_pipeline(
            vector_store=vector_store,
            llm_generator=llm_generator,
            enable_all_features=False,
        )

        # Configure pipeline from request
        ml_pipeline.use_multilingual_embeddings = request.use_multilingual_embeddings
        ml_pipeline.use_multilingual_bm25 = request.use_multilingual_bm25
        ml_pipeline.use_multilingual_hyde = request.use_multilingual_hyde
        ml_pipeline.use_multilingual_classifier = request.use_multilingual_classifier

        # Detect language if requested
        detected_language = None
        if request.detect_language:
            detected_language, language_confidence = ml_pipeline.detect_language(query)
            logger.info(
                f"Detected language: {detected_language} (confidence: {language_confidence:.2f})"
            )
            query_log.metadata["detected_language"] = detected_language
            query_log.metadata["language_confidence"] = float(language_confidence)

        # Use provided language or detected language
        query_language = request.query_language or detected_language

        # Classify query with multilingual patterns
        if request.use_multilingual_classifier:
            query_analysis = ml_pipeline.classify_query(query, language=query_language)
            logger.info(f"Query type: {query_analysis['query_type']}")
            query_log.metadata["query_classification"] = query_analysis

        # Determine initial top_k
        initial_top_k = request.top_k
        if request.use_reranking:
            initial_top_k = settings.INITIAL_RETRIEVAL_K
        elif request.use_graph_rag:
            initial_top_k = request.top_k * 3

        # Retrieve
        chunks_with_scores = ml_pipeline.retrieve(
            query=query,
            top_k=initial_top_k,
            use_hyde=request.use_multilingual_hyde,
            use_bm25=request.use_multilingual_bm25,
            fusion_method="rrf",
            metadata_filter=request.metadata_filter,
            language=query_language,
        )

        logger.info(f"Multilingual pipeline retrieved {len(chunks_with_scores)} chunks")
        query_log.retrieval.initial_candidates = len(chunks_with_scores)

        return chunks_with_scores, retrieval_start


class RetrieverFactory:
    """Creates appropriate retriever based on request parameters"""

    @staticmethod
    def create(request: Any, vector_store, embedding_model, query: str, query_log: Any):
        """
        Create appropriate retriever (Hybrid or Standard).

        Returns:
            Retriever instance
        """
        from .retrieval import HybridRetriever, Retriever

        if request.use_hybrid_search or settings.USE_HYBRID_SEARCH:
            # Calculate alpha
            hybrid_alpha = settings.HYBRID_ALPHA

            if request.enable_adaptive_alpha and settings.ENABLE_ADAPTIVE_ALPHA:
                logger.info("Using adaptive alpha for hybrid search...")
                hybrid_alpha = AdaptiveAlphaCalculator.calculate(
                    query, default_alpha=settings.HYBRID_ALPHA
                )
                query_log.metadata["adaptive_alpha"] = hybrid_alpha

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                alpha=hybrid_alpha,
            )
            logger.info(f"Created HybridRetriever with alpha={hybrid_alpha}")

        else:
            retriever = Retriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
            )
            logger.info("Created standard Retriever")

        return retriever


class RetrievalStrategyExecutor:
    """Executes different retrieval strategies (HyDE, multi-query, standard, etc.)"""

    @staticmethod
    def execute(
        query: str, request: Any, retriever, llm_generator: LLMGenerator, query_log: Any
    ) -> List[Tuple[Chunk, float]]:
        """
        Execute retrieval strategy based on request parameters.

        Returns:
            List of (chunk, score) tuples
        """
        # Determine initial top_k for retrieval
        initial_top_k = request.top_k

        if request.use_reranking:
            initial_top_k = settings.INITIAL_RETRIEVAL_K
        elif request.use_graph_rag:
            initial_top_k = request.top_k * 3

        # Execute appropriate retrieval strategy
        if request.use_hyde:
            chunks_with_scores = RetrievalStrategyExecutor._execute_hyde(
                query, request, retriever, llm_generator, initial_top_k, query_log
            )
        elif request.use_multi_query:
            chunks_with_scores = RetrievalStrategyExecutor._execute_multi_query(
                query, request, retriever, llm_generator, initial_top_k, query_log
            )
        else:
            chunks_with_scores = RetrievalStrategyExecutor._execute_standard(
                query, request, retriever, initial_top_k, query_log
            )

        return chunks_with_scores

    @staticmethod
    def _execute_hyde(
        query: str,
        request: Any,
        retriever,
        llm_generator: LLMGenerator,
        top_k: int,
        query_log: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Execute HyDE retrieval strategy"""
        from .hyde import HyDE

        logger.info("Using HyDE retrieval...")
        hyde = HyDE(llm_generator=llm_generator, retriever=retriever)

        chunks_with_scores = hyde.retrieve(
            query=query,
            top_k=top_k,
            num_hypothetical_docs=request.num_hypothetical_docs,
            fusion_method=request.hyde_fusion,
            metadata_filter=request.metadata_filter,
        )

        logger.info(f"HyDE retrieved {len(chunks_with_scores)} chunks")
        query_log.retrieval.strategy = "hyde"
        query_log.retrieval.initial_candidates = len(chunks_with_scores)

        return chunks_with_scores

    @staticmethod
    def _execute_multi_query(
        query: str,
        request: Any,
        retriever,
        llm_generator: LLMGenerator,
        top_k: int,
        query_log: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Execute multi-query retrieval strategy"""
        logger.info("Using multi-query retrieval...")

        # Generate query variations
        variations = llm_generator.generate_query_variations(
            query, num_variations=request.num_query_variations
        )
        logger.info(f"Generated {len(variations)} query variations")

        # Retrieve for each variation
        all_chunks = []
        for variation in variations:
            chunks = retriever.retrieve(
                query=variation,
                top_k=top_k // len(variations),
                metadata_filter=request.metadata_filter,
            )
            all_chunks.extend(chunks)

        # Deduplicate and re-score using RRF
        from .retrieval import reciprocal_rank_fusion

        chunks_with_scores = reciprocal_rank_fusion([all_chunks], k=60)
        chunks_with_scores = chunks_with_scores[:top_k]

        logger.info(f"Multi-query retrieved {len(chunks_with_scores)} chunks")
        query_log.retrieval.strategy = "multi_query"
        query_log.retrieval.initial_candidates = len(chunks_with_scores)

        return chunks_with_scores

    @staticmethod
    def _execute_standard(
        query: str, request: Any, retriever, top_k: int, query_log: Any
    ) -> List[Tuple[Chunk, float]]:
        """Execute standard retrieval strategy"""
        logger.info("Using standard retrieval...")

        chunks_with_scores = retriever.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=request.metadata_filter,
        )

        logger.info(f"Retrieved {len(chunks_with_scores)} chunks")
        query_log.retrieval.strategy = "standard"
        query_log.retrieval.initial_candidates = len(chunks_with_scores)

        return chunks_with_scores


class PostRetrievalProcessor:
    """Handles post-retrieval processing: reranking, filtering, compression"""

    @staticmethod
    def process(
        chunks_with_scores: List[Tuple[Chunk, float]],
        query: str,
        request: Any,
        embedding_model,
        query_log: Any,
    ) -> List[Tuple[Chunk, float]]:
        """
        Apply post-retrieval processing.

        Returns:
            Processed chunks with scores
        """
        # Apply reranking if enabled
        if request.use_reranking and chunks_with_scores:
            chunks_with_scores = PostRetrievalProcessor._apply_reranking(
                chunks_with_scores, query, request, query_log
            )

        # Apply score filtering
        if request.min_score and request.min_score > 0:
            chunks_with_scores = PostRetrievalProcessor._filter_by_score(
                chunks_with_scores, request.min_score, query_log
            )

        # Deduplicate
        chunks_with_scores = PostRetrievalProcessor._deduplicate(
            chunks_with_scores, query_log
        )

        # Apply compression if enabled
        if request.use_compression and chunks_with_scores:
            chunks_with_scores = PostRetrievalProcessor._apply_compression(
                chunks_with_scores, query, embedding_model, query_log
            )

        return chunks_with_scores

    @staticmethod
    def _apply_reranking(
        chunks_with_scores: List[Tuple[Chunk, float]],
        query: str,
        request: Any,
        query_log: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Apply reranking"""
        from .retrieval import get_reranker

        logger.info("Applying reranking...")

        try:
            reranker = get_reranker(model_name=settings.RERANKER_MODEL)

            chunks_with_scores = reranker.rerank(
                query=query,
                chunks_with_scores=chunks_with_scores,
                top_k=request.top_k,
            )

            logger.info(f"Reranked to {len(chunks_with_scores)} chunks")
            query_log.retrieval.reranking_applied = True

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original ranking")

        return chunks_with_scores

    @staticmethod
    def _filter_by_score(
        chunks_with_scores: List[Tuple[Chunk, float]], min_score: float, query_log: Any
    ) -> List[Tuple[Chunk, float]]:
        """Filter chunks by minimum score"""
        original_count = len(chunks_with_scores)

        chunks_with_scores = [
            (chunk, score) for chunk, score in chunks_with_scores if score >= min_score
        ]

        filtered_count = original_count - len(chunks_with_scores)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} chunks below min_score={min_score}")
            query_log.retrieval.score_filtered_count = filtered_count

        return chunks_with_scores

    @staticmethod
    def _deduplicate(
        chunks_with_scores: List[Tuple[Chunk, float]], query_log: Any
    ) -> List[Tuple[Chunk, float]]:
        """Remove duplicate chunks"""
        seen_content = set()
        unique_chunks = []
        duplicate_count = 0

        for chunk, score in chunks_with_scores:
            content_key = chunk.content.strip().lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_chunks.append((chunk, score))
            else:
                duplicate_count += 1

        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate chunks")
            query_log.retrieval.duplicates_removed = duplicate_count

        return unique_chunks

    @staticmethod
    def _apply_compression(
        chunks_with_scores: List[Tuple[Chunk, float]],
        query: str,
        embedding_model,
        query_log: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Apply context compression"""
        from .compression import ContextCompressor

        logger.info("Applying context compression...")

        try:
            compressor = ContextCompressor(embedding_model=embedding_model)

            compressed_chunks = compressor.compress(
                query=query,
                chunks_with_scores=chunks_with_scores,
                compression_ratio=0.7,
            )

            logger.info(f"Compressed to {len(compressed_chunks)} chunks")
            query_log.retrieval.compression_applied = True

            return compressed_chunks

        except Exception as e:
            logger.warning(f"Compression failed: {e}, using uncompressed chunks")
            return chunks_with_scores
