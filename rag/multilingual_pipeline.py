"""
Multilingual Retrieval Pipeline

Integrates all multilingual RAG components:
- Language detection
- Multilingual embeddings
- Multilingual BM25 tokenization
- Multilingual NER
- Multilingual query classification
- Multilingual HyDE

Usage:
    pipeline = MultilingualRetrievalPipeline(
        embedding_model=multilingual_embedding_model,
        vector_store=vector_store,
        llm_generator=llm_generator
    )

    results = pipeline.retrieve("Qu'est-ce que le machine learning?", top_k=10)
    # Automatically detects French, uses French-specific processing
"""

from typing import List, Tuple, Optional, Dict, Any
from loguru import logger

from .chunking import Chunk
from .embeddings import MultilingualEmbeddingModel, create_embedding_model
from .retrieval import MultilingualBM25Index, MultilingualTokenizer
from .query_classifier import MultilingualQueryClassifier
from .hyde import MultilingualHyDE
from .graph_rag import MultilingualNER
from .language_detection import get_language_detector
from config.settings import settings


class MultilingualRetrievalPipeline:
    """
    Complete Multilingual RAG Pipeline

    Combines all multilingual components for seamless cross-lingual retrieval:
    1. Automatic language detection
    2. Language-specific query classification
    3. Multilingual embedding generation
    4. Language-aware BM25 search
    5. Multilingual HyDE (optional)
    6. Cross-lingual result fusion

    Features:
    - Query in one language, retrieve docs in any language
    - Language-specific optimization (tokenization, NER, query patterns)
    - Automatic fallback to English for unsupported languages
    - Support for 100+ languages via multilingual-e5 embeddings

    Example:
        >>> pipeline = MultilingualRetrievalPipeline(...)
        >>> # Query in French
        >>> results = pipeline.retrieve("Qu'est-ce que l'apprentissage automatique?")
        >>> # Returns relevant docs in French, English, or any other language
    """

    def __init__(
        self,
        embedding_model=None,
        vector_store=None,
        llm_generator=None,
        use_multilingual_embeddings: bool = True,
        use_multilingual_bm25: bool = True,
        use_multilingual_hyde: bool = False,
        use_multilingual_classifier: bool = True,
        default_language: str = "en"
    ):
        """
        Initialize multilingual retrieval pipeline

        Args:
            embedding_model: Embedding model (will create multilingual if None)
            vector_store: Vector store for similarity search
            llm_generator: LLM for HyDE and classification
            use_multilingual_embeddings: Use multilingual-e5-large
            use_multilingual_bm25: Use language-specific BM25 tokenization
            use_multilingual_hyde: Use multilingual HyDE
            use_multilingual_classifier: Use multilingual query classification
            default_language: Default language for fallback
        """
        self.default_language = default_language

        # Initialize language detector
        self.language_detector = get_language_detector()
        logger.info("Language detector initialized")

        # Initialize embedding model
        if embedding_model is None and use_multilingual_embeddings:
            logger.info("Creating multilingual embedding model...")
            self.embedding_model = create_embedding_model(
                model_name=settings.MULTILINGUAL_EMBEDDING_MODEL,
                use_multilingual=True,
                device=settings.EMBEDDING_DEVICE
            )
        else:
            self.embedding_model = embedding_model

        self.vector_store = vector_store

        # Initialize multilingual BM25
        self.use_multilingual_bm25 = use_multilingual_bm25
        if use_multilingual_bm25:
            self.bm25_index = MultilingualBM25Index(
                language_detector=self.language_detector,
                tokenizer=MultilingualTokenizer(),
                use_stemming=settings.BM25_USE_STEMMING
            )
            logger.info("Multilingual BM25 index initialized")
        else:
            self.bm25_index = None

        # Initialize multilingual query classifier
        self.use_multilingual_classifier = use_multilingual_classifier
        if use_multilingual_classifier:
            self.query_classifier = MultilingualQueryClassifier(
                use_llm=False,
                default_language=default_language
            )
            logger.info("Multilingual query classifier initialized")
        else:
            self.query_classifier = None

        # Initialize multilingual HyDE
        self.use_multilingual_hyde = use_multilingual_hyde
        if use_multilingual_hyde and llm_generator:
            self.hyde = MultilingualHyDE(
                embedding_model=self.embedding_model,
                llm_generator=llm_generator,
                default_language=default_language
            )
            logger.info("Multilingual HyDE initialized")
        else:
            self.hyde = None

        # Initialize multilingual NER
        self.ner = MultilingualNER(default_language=default_language)
        logger.info("Multilingual NER initialized")

        logger.info(f"MultilingualRetrievalPipeline initialized (default_language={default_language})")

    def detect_language(self, query: str) -> Tuple[str, float]:
        """
        Detect query language

        Args:
            query: Query text

        Returns:
            Tuple of (language_code, confidence)
        """
        return self.language_detector.detect(query)

    def classify_query(self, query: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify query with language-specific patterns

        Args:
            query: Query text
            language: Optional language code (auto-detected if None)

        Returns:
            Query analysis dict with query type, entities, etc.
        """
        if not self.query_classifier:
            return {"query_type": "UNKNOWN", "language": language or self.default_language}

        analysis = self.query_classifier.classify(query, language=language)

        return {
            "query_type": analysis.query_type.value,
            "confidence": analysis.confidence,
            "entities": analysis.entities,
            "has_negation": analysis.has_negation,
            "suggested_strategy": analysis.suggested_strategy,
            "language": analysis.suggested_params.get("query_language", self.default_language)
        }

    def build_bm25_index(self, chunks: List[Chunk], default_language: Optional[str] = None):
        """
        Build multilingual BM25 index from chunks

        Args:
            chunks: List of document chunks
            default_language: Default language for chunks without detection
        """
        if not self.bm25_index:
            logger.warning("Multilingual BM25 not enabled")
            return

        self.bm25_index.build_index(
            chunks,
            default_language=default_language or self.default_language
        )
        logger.info(f"Built multilingual BM25 index with {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hyde: Optional[bool] = None,
        use_bm25: Optional[bool] = None,
        fusion_method: str = "rrf",
        metadata_filter: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant documents using multilingual pipeline

        Args:
            query: Query in any supported language
            top_k: Number of results to return
            use_hyde: Whether to use HyDE (None = auto-decide based on query complexity)
            use_bm25: Whether to use BM25 (None = use if available)
            fusion_method: How to combine vector + BM25 results ("rrf", "average", "max")
            metadata_filter: Optional metadata filters
            language: Optional language code (auto-detected if None)

        Returns:
            List of (chunk, score) tuples
        """
        # Step 1: Detect language
        if language is None:
            language, confidence = self.detect_language(query)
            logger.info(f"Detected language: {language} (confidence: {confidence:.2f})")
        else:
            confidence = 1.0

        # Step 2: Classify query
        query_analysis = self.classify_query(query, language=language)
        logger.info(f"Query type: {query_analysis['query_type']}")

        # Step 3: Decide whether to use HyDE
        should_use_hyde = use_hyde if use_hyde is not None else (
            self.use_multilingual_hyde and
            query_analysis['query_type'] in ['ANALYTICAL', 'PROCEDURAL', 'COMPARATIVE']
        )

        # Step 4: Retrieve using appropriate strategy
        if should_use_hyde and self.hyde:
            # Use multilingual HyDE
            logger.info("Using multilingual HyDE retrieval")
            results = self.hyde.retrieve(
                query=query,
                vector_store=self.vector_store,
                top_k=top_k,
                fusion_method="average",
                metadata_filter=metadata_filter,
                language=language
            )

        else:
            # Use vector search + optional BM25
            logger.info("Using multilingual vector search")

            # Vector search
            query_embedding = self.embedding_model.encode_single(query, is_query=True)
            vector_results = self.vector_store.search(
                query_embedding,
                top_k=top_k * 2 if use_bm25 else top_k,
                metadata_filter=metadata_filter
            )

            # BM25 search (if enabled)
            should_use_bm25 = use_bm25 if use_bm25 is not None else (
                self.use_multilingual_bm25 and self.bm25_index is not None
            )

            if should_use_bm25:
                logger.info("Adding multilingual BM25 results")
                bm25_results = self.bm25_index.search(
                    query,
                    top_k=top_k * 2,
                    query_language=language
                )

                # Fuse results
                if fusion_method == "rrf":
                    from .fusion import reciprocal_rank_fusion
                    results = reciprocal_rank_fusion(
                        [vector_results, bm25_results],
                        top_k=top_k
                    )
                elif fusion_method == "max":
                    # Combine and take max scores
                    combined = {}
                    for chunk, score in vector_results + bm25_results:
                        chunk_id = chunk.chunk_id
                        if chunk_id not in combined or score > combined[chunk_id][1]:
                            combined[chunk_id] = (chunk, score)
                    results = sorted(combined.values(), key=lambda x: x[1], reverse=True)[:top_k]
                else:
                    # Simple concatenation
                    results = vector_results[:top_k]
            else:
                results = vector_results[:top_k]

        logger.info(f"Retrieved {len(results)} results for query in {language}")
        return results

    def extract_entities(self, text: str, language: Optional[str] = None) -> List[Any]:
        """
        Extract entities from text using multilingual NER

        Args:
            text: Text to extract entities from
            language: Optional language code (auto-detected if None)

        Returns:
            List of Entity objects
        """
        return self.ner.extract_entities(text, language=language)


def create_multilingual_pipeline(
    vector_store=None,
    llm_generator=None,
    enable_all_features: bool = False
) -> MultilingualRetrievalPipeline:
    """
    Factory function to create multilingual pipeline

    Args:
        vector_store: Vector store instance
        llm_generator: LLM generator instance
        enable_all_features: Enable all multilingual features (for testing)

    Returns:
        Configured MultilingualRetrievalPipeline
    """
    # Use settings to determine what to enable
    use_multilingual_embeddings = enable_all_features or settings.USE_MULTILINGUAL_EMBEDDINGS
    use_multilingual_bm25 = enable_all_features or settings.ENABLE_MULTILINGUAL_BM25
    use_multilingual_hyde = enable_all_features or settings.ENABLE_MULTILINGUAL_HYDE
    use_multilingual_classifier = enable_all_features or settings.ENABLE_MULTILINGUAL_QUERY_CLASSIFICATION

    pipeline = MultilingualRetrievalPipeline(
        vector_store=vector_store,
        llm_generator=llm_generator,
        use_multilingual_embeddings=use_multilingual_embeddings,
        use_multilingual_bm25=use_multilingual_bm25,
        use_multilingual_hyde=use_multilingual_hyde,
        use_multilingual_classifier=use_multilingual_classifier,
        default_language=settings.DEFAULT_LANGUAGE
    )

    logger.info("Multilingual pipeline created via factory")
    return pipeline
