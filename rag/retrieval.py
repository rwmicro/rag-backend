"""
Advanced Retrieval Module
Implements hybrid search, reranking, and multi-query retrieval
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import json
import re
from rank_bm25 import BM25Okapi
from loguru import logger

from .chunking import Chunk
from .vectordb import VectorStore
from .embeddings import EmbeddingModel
from config.settings import settings

# Optional multilingual tokenization imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.debug(
        "Jieba not available for Chinese tokenization. Install with: pip install jieba"
    )

try:
    import MeCab

    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    logger.debug(
        "MeCab not available for Japanese tokenization. Install with: pip install mecab-python3"
    )

try:
    from konlpy.tag import Okt

    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    logger.debug(
        "KoNLPy not available for Korean tokenization. Install with: pip install konlpy"
    )


def normalize_minmax(scores: List[float]) -> List[float]:
    """
    Min-max normalization: scales scores to [0, 1] range

    Args:
        scores: List of raw scores

    Returns:
        List of normalized scores in [0, 1]
    """
    if not scores or len(scores) == 0:
        return []

    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()

    if max_score - min_score == 0:
        return [1.0] * len(scores)

    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()


def normalize_zscore(scores: List[float]) -> List[float]:
    """
    Z-score normalization: standardizes scores to have mean=0, std=1
    Then shifts to positive range for combining

    Args:
        scores: List of raw scores

    Returns:
        List of normalized scores
    """
    if not scores or len(scores) == 0:
        return []

    scores_array = np.array(scores)
    mean = scores_array.mean()
    std = scores_array.std()

    if std == 0:
        return [1.0] * len(scores)

    # Standardize
    z_scores = (scores_array - mean) / std

    # Shift to positive range using sigmoid-like transformation
    # This maps z-scores to [0, 1] range approximately
    normalized = 1 / (1 + np.exp(-z_scores))

    return normalized.tolist()


class MultilingualTokenizer:
    """
    Multilingual Tokenizer for BM25

    Supports language-specific tokenization strategies:
    - European languages: NLTK Snowball stemmers + stopwords
    - Chinese: Jieba word segmentation
    - Japanese: MeCab morphological analysis
    - Korean: KoNLPy (Okt tokenizer)
    - Thai/Lao/Burmese/Khmer: Character n-grams (no word boundaries)
    - Fallback: Simple whitespace + lowercase

    Usage:
        tokenizer = MultilingualTokenizer()
        tokens = tokenizer.tokenize("Hello world", language="en")
    """

    def __init__(self, download_nltk_data: bool = True):
        """
        Initialize multilingual tokenizer

        Args:
            download_nltk_data: Whether to download NLTK data if not present
        """
        self.stemmers = {}
        self.stop_words = {}

        # Language support mapping
        self.stemmer_languages = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "nl": "dutch",
            "ru": "russian",
            "no": "norwegian",
            "sv": "swedish",
            "da": "danish",
            "fi": "finnish",
            "hu": "hungarian",
            "ro": "romanian",
            "tr": "turkish",
            "ar": "arabic",
        }

        # Initialize NLTK if available
        if NLTK_AVAILABLE and download_nltk_data:
            self._init_nltk()

        # Initialize MeCab for Japanese
        self.mecab = None
        if MECAB_AVAILABLE:
            try:
                self.mecab = MeCab.Tagger()
                logger.info("MeCab initialized for Japanese tokenization")
            except Exception as e:
                logger.warning(f"Failed to initialize MeCab: {e}")

        # Initialize KoNLPy for Korean
        self.okt = None
        if KONLPY_AVAILABLE:
            try:
                self.okt = Okt()
                logger.info("KoNLPy (Okt) initialized for Korean tokenization")
            except Exception as e:
                logger.warning(f"Failed to initialize KoNLPy: {e}")

    def _init_nltk(self):
        """Initialize NLTK data (stemmers and stopwords)"""
        try:
            # Download required NLTK data
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                logger.info("Downloading NLTK stopwords...")
                nltk.download("stopwords", quiet=True)

            # Initialize stemmers and stopwords for supported languages
            for lang_code, lang_name in self.stemmer_languages.items():
                try:
                    self.stemmers[lang_code] = SnowballStemmer(lang_name)
                    self.stop_words[lang_code] = set(stopwords.words(lang_name))
                except Exception as e:
                    logger.debug(f"Could not load NLTK resources for {lang_name}: {e}")

            logger.info(f"NLTK initialized for {len(self.stemmers)} languages")

        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")

    def tokenize(
        self, text: str, language: str = "en", use_stemming: bool = True
    ) -> List[str]:
        """
        Tokenize text based on language

        Args:
            text: Text to tokenize
            language: ISO 639-1 language code
            use_stemming: Whether to apply stemming (for supported languages)

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Chinese tokenization
        if language == "zh" and JIEBA_AVAILABLE:
            tokens = list(jieba.cut(text.lower()))
            # Remove whitespace tokens
            return [t for t in tokens if t.strip()]

        # Japanese tokenization
        if language == "ja" and self.mecab:
            try:
                # MeCab returns space-separated tokens
                result = self.mecab.parse(text).strip()
                tokens = []
                for line in result.split("\n"):
                    if line and "\t" in line:
                        surface = line.split("\t")[0]
                        if surface:
                            tokens.append(surface.lower())
                return tokens
            except Exception as e:
                logger.warning(
                    f"MeCab tokenization failed: {e}, falling back to simple split"
                )
                return text.lower().split()

        # Korean tokenization
        if language == "ko" and self.okt:
            try:
                tokens = self.okt.morphs(text, stem=use_stemming)
                return [t.lower() for t in tokens]
            except Exception as e:
                logger.warning(
                    f"KoNLPy tokenization failed: {e}, falling back to simple split"
                )
                return text.lower().split()

        # Character n-grams for languages without word boundaries
        if language in ["th", "lo", "my", "km"]:
            # Use character bigrams for basic tokenization
            text_clean = text.lower().replace(" ", "")
            return [text_clean[i : i + 2] for i in range(len(text_clean) - 1)]

        # European languages with NLTK stemming
        if language in self.stemmers and NLTK_AVAILABLE:
            # Simple tokenization (split on whitespace and punctuation)
            tokens = re.findall(r"\b\w+\b", text.lower())

            # Remove stopwords
            if language in self.stop_words:
                tokens = [t for t in tokens if t not in self.stop_words[language]]

            # Apply stemming if enabled
            if use_stemming:
                stemmer = self.stemmers[language]
                tokens = [stemmer.stem(t) for t in tokens]

            return tokens

        # Fallback: Simple tokenization (whitespace + lowercase)
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens


class MultilingualBM25Index:
    """
    Multilingual BM25 Index with automatic language detection

    Builds separate BM25 indices per language for optimal performance,
    or a single multilingual index if documents are mixed.

    Features:
    - Automatic language detection per document
    - Language-specific tokenization
    - Supports 100+ languages via MultilingualTokenizer
    - Fallback to simple tokenization for unsupported languages

    Usage:
        from .language_detection import get_language_detector
        detector = get_language_detector()
        bm25 = MultilingualBM25Index(detector)
        bm25.build_index(chunks)
        results = bm25.search("query text", top_k=10)
    """

    def __init__(
        self,
        language_detector=None,
        tokenizer: Optional[MultilingualTokenizer] = None,
        use_stemming: bool = True,
    ):
        """
        Initialize multilingual BM25 index

        Args:
            language_detector: LanguageDetector instance (optional, will import if None)
            tokenizer: MultilingualTokenizer instance (optional, will create if None)
            use_stemming: Whether to use stemming for supported languages
        """
        # Lazy import to avoid circular dependencies
        if language_detector is None:
            try:
                from .language_detection import get_language_detector

                self.language_detector = get_language_detector()
            except ImportError:
                logger.warning(
                    "Language detector not available, will use default language"
                )
                self.language_detector = None
        else:
            self.language_detector = language_detector

        # Initialize tokenizer
        self.tokenizer = tokenizer or MultilingualTokenizer()
        self.use_stemming = use_stemming

        # BM25 indices and corpora
        self.bm25_index = None
        self.corpus_chunks = []
        self.corpus_languages = []  # Track language for each chunk

        logger.info("MultilingualBM25Index initialized")

    def build_index(self, chunks: List[Chunk], default_language: str = "en"):
        """
        Build BM25 index from chunks with automatic language detection

        Args:
            chunks: List of chunks to index
            default_language: Default language if detection fails
        """
        if not chunks:
            logger.warning("Cannot build BM25 index: no chunks provided")
            self.bm25_index = None
            self.corpus_chunks = []
            self.corpus_languages = []
            return

        tokenized_corpus = []
        languages = []

        for chunk in chunks:
            # Detect language
            if self.language_detector:
                # Use first 500 chars for detection (more reliable than full chunk)
                sample_text = chunk.content[:500]
                detected_lang = self.language_detector.detect_with_fallback(
                    sample_text, default=default_language, min_confidence=0.5
                )
            else:
                detected_lang = default_language

            languages.append(detected_lang)

            # Tokenize with language-specific strategy
            tokens = self.tokenizer.tokenize(
                chunk.content, language=detected_lang, use_stemming=self.use_stemming
            )
            tokenized_corpus.append(tokens)

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.corpus_chunks = chunks
        self.corpus_languages = languages

        # Log language distribution
        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        logger.info(f"Built multilingual BM25 index with {len(chunks)} chunks")
        logger.info(
            f"Language distribution: {dict(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))}"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        query_language: Optional[str] = None,
        default_language: str = "en",
    ) -> List[Tuple[Chunk, float]]:
        """
        Search using BM25 with multilingual tokenization

        Args:
            query: Search query
            top_k: Number of results to return
            query_language: Query language (auto-detected if None)
            default_language: Default language if detection fails

        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25_index is None or len(self.corpus_chunks) == 0:
            return []

        # Detect query language
        if query_language is None and self.language_detector:
            query_language = self.language_detector.detect_with_fallback(
                query, default=default_language, min_confidence=0.5
            )
        elif query_language is None:
            query_language = default_language

        logger.debug(f"Query language: {query_language}")

        # Tokenize query
        tokenized_query = self.tokenizer.tokenize(
            query, language=query_language, use_stemming=self.use_stemming
        )

        if not tokenized_query:
            logger.warning("Query tokenization resulted in empty tokens")
            return []

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Handle empty scores
        if len(scores) == 0:
            return []

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return chunks with scores
        results = [
            (self.corpus_chunks[idx], float(scores[idx]))
            for idx in top_indices
            if idx < len(self.corpus_chunks) and scores[idx] > 0
        ]

        logger.debug(
            f"BM25 search returned {len(results)} results (query_lang={query_language})"
        )

        return results


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    chunks_with_scores: List[Tuple[Chunk, float]],
    top_k: int = 10,
    lambda_param: float = 0.7,
    diversity_threshold: float = 0.8,
) -> List[Tuple[Chunk, float]]:
    """
    Maximal Marginal Relevance (MMR) for diversity-aware ranking

    MMR balances relevance to query with diversity among selected results.
    Formula:
        MMR = argmax[D_i ∈ D \\ S] [λ × Sim(D_i, Q) - (1-λ) × max[D_j ∈ S] Sim(D_i, D_j)]

    Where:
        - D: Set of all candidate documents
        - S: Set of already selected documents
        - Q: Query
        - λ (lambda): Trade-off parameter (0-1)
          - λ=1: Pure relevance (no diversity)
          - λ=0: Pure diversity (no relevance)
          - λ=0.7: Balanced (default, recommended)

    Args:
        query_embedding: Query embedding vector
        chunks_with_scores: Initial retrieval results (relevance-sorted)
        top_k: Number of diverse results to select
        lambda_param: Trade-off between relevance (λ) and diversity (1-λ)
        diversity_threshold: Only enforce diversity if similarity > threshold

    Returns:
        MMR-ranked list of (chunk, mmr_score) tuples
    """
    if not chunks_with_scores or len(chunks_with_scores) <= 1:
        return chunks_with_scores[:top_k]

    # Extract chunks and embeddings
    candidates = []
    candidate_embeddings = []

    for chunk, relevance_score in chunks_with_scores:
        if chunk.embedding is not None:
            candidates.append((chunk, relevance_score))
            candidate_embeddings.append(chunk.embedding)
        else:
            logger.warning(f"Chunk {chunk.chunk_id} missing embedding, skipping MMR")

    if not candidates:
        logger.warning("No chunks with embeddings for MMR, returning original results")
        return chunks_with_scores[:top_k]

    # Convert to numpy arrays
    candidate_embeddings = np.array(candidate_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    candidate_norms = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # Compute relevance scores (query-document similarity)
    # Shape: (num_candidates,)
    relevance_scores = np.dot(candidate_norms, query_norm.T).flatten()

    # Normalize relevance scores to [0, 1]
    if relevance_scores.max() > relevance_scores.min():
        relevance_scores = (relevance_scores - relevance_scores.min()) / (
            relevance_scores.max() - relevance_scores.min()
        )

    # MMR selection algorithm
    selected_indices = []
    selected_embeddings = []
    remaining_indices = list(range(len(candidates)))

    # Select top_k documents iteratively
    for _ in range(min(top_k, len(candidates))):
        if not remaining_indices:
            break

        mmr_scores = []

        for idx in remaining_indices:
            # Relevance component: λ × Sim(D_i, Q)
            relevance = lambda_param * relevance_scores[idx]

            # Diversity component: (1-λ) × max[D_j ∈ S] Sim(D_i, D_j)
            diversity_penalty = 0.0

            if selected_embeddings:
                # Compute similarity with all selected documents
                selected_array = np.array(selected_embeddings)
                candidate_norm = candidate_norms[idx : idx + 1]

                similarities = np.dot(selected_array, candidate_norm.T).flatten()
                max_similarity = similarities.max()

                # Only apply penalty if similarity exceeds threshold
                if max_similarity > diversity_threshold:
                    diversity_penalty = (1 - lambda_param) * max_similarity

            # MMR score
            mmr_score = relevance - diversity_penalty
            mmr_scores.append((idx, mmr_score))

        # Select document with highest MMR score
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])

        selected_indices.append(best_idx)
        selected_embeddings.append(candidate_embeddings[best_idx])
        remaining_indices.remove(best_idx)

    # Build result list with MMR scores
    mmr_results = []
    for rank, idx in enumerate(selected_indices):
        chunk, original_score = candidates[idx]
        # Use rank-based score (higher rank = higher score)
        mmr_score = 1.0 - (rank / len(selected_indices))
        mmr_results.append((chunk, mmr_score))

    logger.info(
        f"MMR diversification: {len(mmr_results)}/{len(candidates)} selected "
        f"(λ={lambda_param}, threshold={diversity_threshold})"
    )

    return mmr_results


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Chunk, float]]], k: int = 60
) -> Dict[str, Tuple[Chunk, float]]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists
    Formula: score = sum(1 / (k + rank_i)) across all lists

    Args:
        results_list: List of ranked result lists [(chunk, score), ...]
        k: Constant for RRF formula (default: 60, standard in literature)

    Returns:
        Dictionary mapping chunk_id to (chunk, rrf_score)
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for results in results_list:
        # Results are already sorted by score (descending)
        for rank, (chunk, _) in enumerate(results, start=1):
            chunk_id = chunk.chunk_id

            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = chunk

    # Convert to (chunk, score) tuples
    return {
        chunk_id: (chunk_map[chunk_id], score) for chunk_id, score in rrf_scores.items()
    }


def deduplicate_chunks(
    chunks_with_scores: List[Tuple[Chunk, float]],
    similarity_threshold: float = 0.95,  # Increased from 0.90 to keep more diverse results
    embedding_model=None,
) -> List[Tuple[Chunk, float]]:
    """
    Remove duplicate or highly similar chunks based on content similarity

    Args:
        chunks_with_scores: List of (chunk, score) tuples
        similarity_threshold: Cosine similarity threshold (default: 0.95)
        embedding_model: Optional embedding model for computing similarities
                        If None, uses simple text-based similarity

    Returns:
        Deduplicated list of (chunk, score) tuples
    """
    if not chunks_with_scores or len(chunks_with_scores) <= 1:
        return chunks_with_scores

    # If chunks have embeddings, use them for similarity computation
    if chunks_with_scores[0][0].embedding is not None:
        # Use existing embeddings
        embeddings = np.array([chunk.embedding for chunk, _ in chunks_with_scores])
    elif embedding_model is not None:
        # Generate embeddings on the fly
        texts = [chunk.content for chunk, _ in chunks_with_scores]
        embeddings = embedding_model.encode(texts)
    else:
        # Fallback: use simple token overlap (Jaccard similarity)
        logger.warning("No embeddings available for deduplication, using token overlap")
        return _deduplicate_by_token_overlap(chunks_with_scores, similarity_threshold)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)

    # Compute pairwise cosine similarity matrix
    similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

    # Keep track of which chunks to keep
    keep_indices = []
    removed_count = 0

    for i in range(len(chunks_with_scores)):
        # Check if this chunk is similar to any already kept chunk
        is_duplicate = False

        for kept_idx in keep_indices:
            if similarity_matrix[i, kept_idx] > similarity_threshold:
                # This chunk is too similar to an already kept chunk
                is_duplicate = True
                removed_count += 1
                logger.debug(
                    f"Removing duplicate chunk {i} (similarity {similarity_matrix[i, kept_idx]:.3f} "
                    f"with chunk {kept_idx})"
                )
                break

        if not is_duplicate:
            keep_indices.append(i)

    # Return deduplicated chunks (preserving order and scores)
    deduplicated = [chunks_with_scores[i] for i in keep_indices]

    if removed_count > 0:
        logger.info(
            f"Deduplication removed {removed_count} similar chunks "
            f"({len(deduplicated)}/{len(chunks_with_scores)} remaining)"
        )

    return deduplicated


def _deduplicate_by_token_overlap(
    chunks_with_scores: List[Tuple[Chunk, float]],
    threshold: float = 0.90,
) -> List[Tuple[Chunk, float]]:
    """
    Fallback deduplication using token-based Jaccard similarity

    Args:
        chunks_with_scores: List of (chunk, score) tuples
        threshold: Jaccard similarity threshold

    Returns:
        Deduplicated list
    """

    def jaccard_similarity(text1: str, text2: str) -> float:
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union) if union else 0.0

    keep_indices = []
    removed_count = 0

    for i, (chunk_i, score_i) in enumerate(chunks_with_scores):
        is_duplicate = False

        for kept_idx in keep_indices:
            chunk_kept = chunks_with_scores[kept_idx][0]
            similarity = jaccard_similarity(chunk_i.content, chunk_kept.content)

            if similarity > threshold:
                is_duplicate = True
                removed_count += 1
                break

        if not is_duplicate:
            keep_indices.append(i)

    deduplicated = [chunks_with_scores[i] for i in keep_indices]

    if removed_count > 0:
        logger.info(f"Token-based deduplication removed {removed_count} chunks")

    return deduplicated


class AdaptiveHybridSearch:
    """
    Adaptive Hybrid Search with dynamic alpha adjustment

    Dynamically adjusts the alpha parameter (vector vs BM25 weight) based on query characteristics:
    - Query length and complexity
    - Presence of entities and technical terms
    - Question words indicating semantic vs keyword search
    - Quoted terms requiring exact matching
    - Query type from classification

    Formula:
        α_adaptive = base_α + Σ(adjustments)
        final_α = clip(α_adaptive, α_min, α_max)

    Where:
        - base_α: Starting point (default 0.7)
        - adjustments: Query-specific modifications
        - α_min, α_max: Bounds (0.2, 0.95)

    Higher α (→ 1.0) = More vector search (semantic understanding)
    Lower α (→ 0.0) = More BM25 (keyword matching)
    """

    def __init__(
        self,
        base_alpha: float = 0.7,
        alpha_min: float = 0.2,
        alpha_max: float = 0.95,
        enable_adaptation: bool = True,
    ):
        """
        Initialize adaptive hybrid search

        Args:
            base_alpha: Default alpha value (0.7 = 70% vector, 30% BM25)
            alpha_min: Minimum alpha (0.2 = favor BM25 for keyword queries)
            alpha_max: Maximum alpha (0.95 = favor vector for semantic queries)
            enable_adaptation: Whether to enable adaptive alpha (False = use base_alpha)
        """
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.enable_adaptation = enable_adaptation

        # Patterns for query analysis
        self._compile_patterns()

        logger.info(
            f"Initialized AdaptiveHybridSearch (base_α={base_alpha}, "
            f"range=[{alpha_min}, {alpha_max}], adaptive={enable_adaptation})"
        )

    def _compile_patterns(self):
        """Compile regex patterns for query analysis"""
        # Question words (favor semantic/vector search)
        self.question_words = (
            r"\b(what|why|how|which|when|where|who|explain|describe|compare)\b"
        )

        # Exact match indicators (favor BM25/keyword search)
        self.exact_match_indicators = r'"[^"]+"|\b(id|code|number|version|sku|isbn)\b'

        # Technical terms pattern (simple heuristic: CamelCase, snake_case, kebab-case)
        self.technical_terms = (
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b\w+_\w+\b|\b\w+-\w+\b"
        )

        # Entities (capitalized words, likely proper nouns - favor semantic)
        self.entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"

        # Numbers and dates (favor BM25 for exact matching)
        self.numeric_pattern = r"\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d+\.?\d*\b"

    def compute_adaptive_alpha(
        self,
        query: str,
        query_type: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute adaptive alpha based on query characteristics

        Args:
            query: User query string
            query_type: Optional query type from classification
            query_analysis: Optional analysis from query classifier

        Returns:
            Tuple of (adaptive_alpha, adjustment_details)
        """
        if not self.enable_adaptation:
            return self.base_alpha, {"reason": "adaptation_disabled"}

        # Start with base alpha
        alpha = self.base_alpha
        adjustments = {}

        # 1. Query Length Analysis
        # Longer queries (>8 words) are usually more semantic/contextual → favor vector
        query_words = query.split()
        word_count = len(query_words)

        if word_count > 12:
            adjustment = 0.10
            alpha += adjustment
            adjustments["long_query"] = adjustment
        elif word_count > 8:
            adjustment = 0.05
            alpha += adjustment
            adjustments["medium_query"] = adjustment
        elif word_count <= 3:
            # Very short queries might be keyword-based → favor BM25
            adjustment = -0.10
            alpha += adjustment
            adjustments["short_query"] = adjustment

        # 2. Question Words (favor semantic understanding)
        if re.search(self.question_words, query.lower()):
            adjustment = 0.10
            alpha += adjustment
            adjustments["question_words"] = adjustment

        # 3. Exact Match Indicators (favor BM25)
        exact_matches = re.findall(self.exact_match_indicators, query, re.IGNORECASE)
        if exact_matches:
            # Strong shift toward BM25 for exact matches
            adjustment = -0.20
            alpha += adjustment
            adjustments["exact_match"] = {
                "adjustment": adjustment,
                "matches": exact_matches,
            }

        # 4. Technical Terms (favor BM25 for precise matching)
        technical_terms = re.findall(self.technical_terms, query)
        if technical_terms:
            # Technical terms often need exact matching
            adjustment = -0.10
            alpha += adjustment
            adjustments["technical_terms"] = {
                "adjustment": adjustment,
                "count": len(technical_terms),
            }

        # 5. Entity Detection (favor vector for semantic entity understanding)
        entities = re.findall(self.entity_pattern, query)
        # Filter out question words
        entities = [
            e
            for e in entities
            if e.lower() not in ["what", "why", "how", "which", "when", "where", "who"]
        ]

        if len(entities) >= 2:
            adjustment = 0.08
            alpha += adjustment
            adjustments["entities"] = {"adjustment": adjustment, "count": len(entities)}

        # 6. Numeric/Date Patterns (favor BM25 for exact matching)
        numeric_matches = re.findall(self.numeric_pattern, query)
        if numeric_matches:
            adjustment = -0.08
            alpha += adjustment
            adjustments["numeric_data"] = {
                "adjustment": adjustment,
                "count": len(numeric_matches),
            }

        # 7. Query Type Adjustments (from classifier)
        if query_type:
            type_adjustments = {
                "factoid": 0.05,  # Slightly favor vector for factual questions
                "analytical": 0.15,  # Strong favor for vector (complex reasoning)
                "comparative": 0.12,  # Favor vector for comparisons
                "procedural": 0.08,  # Favor vector for "how-to"
                "navigational": -0.05,  # Slight favor for BM25 (finding sections)
                "aggregative": 0.10,  # Favor vector for summarization
                "exact_match": -0.25,  # Strong favor for BM25
                "temporal": -0.05,  # Slight favor for BM25 (dates)
                "negation": 0.10,  # Favor vector (semantic negation)
                "multi_hop": 0.15,  # Strong favor for vector (reasoning)
                "conversational": 0.08,  # Favor vector for context
            }

            if query_type.lower() in type_adjustments:
                adjustment = type_adjustments[query_type.lower()]
                alpha += adjustment
                adjustments["query_type"] = {
                    "type": query_type,
                    "adjustment": adjustment,
                }

        # 8. Query Analysis Integration (from query_classifier)
        if query_analysis:
            # Use confidence to weight adjustments
            confidence = query_analysis.get("confidence", 1.0)

            # Has negation? Favor vector for semantic understanding
            if query_analysis.get("has_negation"):
                adjustment = 0.12 * confidence
                alpha += adjustment
                adjustments["has_negation"] = adjustment

            # Requires exact match? Favor BM25
            if query_analysis.get("requires_exact_match"):
                adjustment = -0.20 * confidence
                alpha += adjustment
                adjustments["requires_exact_match"] = adjustment

            # Temporal markers? Favor BM25 for date matching
            temporal_markers = query_analysis.get("temporal_markers", [])
            if temporal_markers:
                adjustment = -0.08 * confidence
                alpha += adjustment
                adjustments["temporal_markers"] = {
                    "adjustment": adjustment,
                    "markers": temporal_markers,
                }

        # 9. Quoted strings (exact phrase matching → strong BM25 preference)
        quoted_strings = re.findall(r'"([^"]+)"', query)
        if quoted_strings:
            adjustment = -0.15
            alpha += adjustment
            adjustments["quoted_strings"] = {
                "adjustment": adjustment,
                "count": len(quoted_strings),
            }

        # Clip alpha to valid range
        alpha_clipped = np.clip(alpha, self.alpha_min, self.alpha_max)

        # Log adjustment details
        adjustment_summary = {
            "original_alpha": self.base_alpha,
            "computed_alpha": alpha,
            "final_alpha": alpha_clipped,
            "clipped": alpha != alpha_clipped,
            "adjustments": adjustments,
            "query_length": word_count,
        }

        logger.info(
            f"Adaptive α: {self.base_alpha:.2f} → {alpha_clipped:.2f} "
            f"(adjustments: {len(adjustments)})"
        )
        logger.debug(f"Alpha adjustments: {adjustments}")

        return alpha_clipped, adjustment_summary

    def explain_alpha(self, adjustment_details: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of alpha computation

        Args:
            adjustment_details: Details from compute_adaptive_alpha

        Returns:
            Explanation string
        """
        explanation = f"Alpha computation for hybrid search:\n"
        explanation += f"  Base α: {adjustment_details['original_alpha']:.2f}\n"

        adjustments = adjustment_details.get("adjustments", {})
        if adjustments:
            explanation += f"  Adjustments applied:\n"
            for key, value in adjustments.items():
                if isinstance(value, dict) and "adjustment" in value:
                    explanation += f"    • {key}: {value['adjustment']:+.2f}\n"
                else:
                    explanation += f"    • {key}: {value:+.2f}\n"

        explanation += f"  Final α: {adjustment_details['final_alpha']:.2f}\n"
        explanation += f"  Interpretation: "

        final_alpha = adjustment_details["final_alpha"]
        if final_alpha >= 0.8:
            explanation += "Strong vector search (semantic understanding)"
        elif final_alpha >= 0.6:
            explanation += "Balanced toward vector search"
        elif final_alpha >= 0.4:
            explanation += "Balanced hybrid search"
        elif final_alpha >= 0.25:
            explanation += "Balanced toward BM25 (keyword matching)"
        else:
            explanation += "Strong BM25 search (exact keyword matching)"

        return explanation


class Retriever:
    """
    Basic retriever using vector similarity search
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        apply_mmr: bool = False,
        mmr_lambda: Optional[float] = None,
        mmr_diversity_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filters (e.g., {"file_type": "pdf"})
            apply_mmr: Whether to apply MMR diversity enforcement
            mmr_lambda: MMR lambda parameter (uses settings if None)
            mmr_diversity_threshold: MMR diversity threshold (uses settings if None)

        Returns:
            List of (chunk, score) tuples
        """
        # Encode query (dense)
        query_embedding = self.embedding_model.encode_single(query, is_query=True)

        # Retrieve more candidates if MMR is enabled
        retrieval_k = top_k * 2 if apply_mmr else top_k

        # Search vector store with optional filter
        results = self.vector_store.search(
            query_embedding, retrieval_k, metadata_filter=metadata_filter
        )

        # Apply MMR if enabled
        if apply_mmr and results:
            lambda_param = mmr_lambda if mmr_lambda is not None else settings.MMR_LAMBDA
            diversity_threshold = (
                mmr_diversity_threshold
                if mmr_diversity_threshold is not None
                else settings.MMR_DIVERSITY_THRESHOLD
            )

            results = maximal_marginal_relevance(
                query_embedding=query_embedding,
                chunks_with_scores=results,
                top_k=top_k,
                lambda_param=lambda_param,
                diversity_threshold=diversity_threshold,
            )

        if metadata_filter:
            logger.info(
                f"Retrieved {len(results)} chunks for query (filtered by {metadata_filter})"
            )
        else:
            logger.info(f"Retrieved {len(results)} chunks for query")

        for i, (chunk, score) in enumerate(results):
            logger.debug(
                f"  [{i + 1}] Score: {score:.4f} | ID: {chunk.chunk_id} | Content: {chunk.content[:100].replace(chr(10), ' ')}..."
            )
        return results


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining vector search with BM25 keyword search
    Uses Reciprocal Rank Fusion (RRF) for score combination

    Supports adaptive alpha adjustment based on query characteristics
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        alpha: float = 0.7,  # Weight for vector search (1-alpha for BM25)
        enable_adaptive_alpha: bool = False,  # Enable adaptive alpha computation
        adaptive_alpha_config: Optional[
            Dict[str, Any]
        ] = None,  # Config for AdaptiveHybridSearch
    ):
        super().__init__(vector_store, embedding_model)
        self.alpha = alpha
        self.enable_adaptive_alpha = enable_adaptive_alpha

        # BM25 index (built lazily)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[Chunk] = []

        # Adaptive alpha support
        if enable_adaptive_alpha:
            config = adaptive_alpha_config or {}
            self.adaptive_search = AdaptiveHybridSearch(
                base_alpha=alpha,
                alpha_min=config.get("alpha_min", settings.ADAPTIVE_ALPHA_MIN),
                alpha_max=config.get("alpha_max", settings.ADAPTIVE_ALPHA_MAX),
                enable_adaptation=True,
            )
            logger.info("Adaptive alpha enabled for HybridRetriever")
        else:
            self.adaptive_search = None

    def _build_bm25_index(self, chunks: List[Chunk]):
        """Build BM25 index from chunks"""
        if not chunks:
            logger.warning("Cannot build BM25 index: no chunks provided")
            self.bm25_index = None
            self.bm25_corpus = []
            return

        # Tokenize corpus
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = chunks

        logger.info(f"Built BM25 index with {len(chunks)} chunks")

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Perform BM25 search"""
        if self.bm25_index is None or len(self.bm25_corpus) == 0:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Handle empty scores
        if len(scores) == 0:
            return []

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return chunks with scores
        results = [
            (self.bm25_corpus[idx], float(scores[idx]))
            for idx in top_indices
            if idx < len(self.bm25_corpus)
        ]

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        build_bm25: bool = True,
        normalization_method: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        query_type: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
        apply_mmr: bool = False,
        mmr_lambda: Optional[float] = None,
        mmr_diversity_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve using hybrid search (vector + BM25)

        Args:
            query: Search query
            top_k: Number of results to return
            build_bm25: Whether to build BM25 index from vector results
            normalization_method: Score normalization method ("minmax", "zscore", "rrf")
                                 If None, uses settings.SCORE_NORMALIZATION_METHOD
            metadata_filter: Optional metadata filters
            query_type: Optional query type from classifier (for adaptive alpha)
            query_analysis: Optional query analysis dict (for adaptive alpha)
            apply_mmr: Whether to apply MMR diversity enforcement
            mmr_lambda: MMR lambda parameter (uses settings if None)
            mmr_diversity_threshold: MMR diversity threshold (uses settings if None)

        Returns:
            List of (chunk, score) tuples
        """
        # Compute adaptive alpha if enabled
        alpha_to_use = self.alpha
        alpha_details = None

        if self.enable_adaptive_alpha and self.adaptive_search:
            alpha_to_use, alpha_details = self.adaptive_search.compute_adaptive_alpha(
                query=query,
                query_type=query_type,
                query_analysis=query_analysis,
            )
            logger.info(f"Using adaptive α={alpha_to_use:.2f} (base={self.alpha:.2f})")

            # Log explanation in debug mode
            if alpha_details and logger.level <= 10:  # DEBUG level
                explanation = self.adaptive_search.explain_alpha(alpha_details)
                logger.debug(f"\n{explanation}")
        else:
            alpha_to_use = self.alpha

        # Use configured normalization method if not specified
        if normalization_method is None:
            normalization_method = settings.SCORE_NORMALIZATION_METHOD

        # Step 1: Vector search (get more results for fusion)
        vector_results = super().retrieve(
            query, top_k * 2, metadata_filter=metadata_filter
        )

        # Step 2: Build BM25 index if needed
        if build_bm25 and (
            self.bm25_index is None or len(self.bm25_corpus) != len(vector_results)
        ):
            chunks = [chunk for chunk, _ in vector_results]
            self._build_bm25_index(chunks)

        # Step 3: BM25 search
        bm25_results = self._bm25_search(query, top_k * 2)

        # Step 4: Combine using configured normalization method
        if normalization_method == "rrf":
            # Reciprocal Rank Fusion
            combined_dict = reciprocal_rank_fusion(
                results_list=[vector_results, bm25_results], k=settings.RRF_K
            )
            # Sort by RRF score
            results = sorted(
                [(chunk, score) for chunk, score in combined_dict.values()],
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            logger.info(
                f"Hybrid search (RRF): {len(results)} results (k={settings.RRF_K})"
            )

        else:
            # Min-max or Z-score normalization with weighted combination
            combined_scores = {}

            # Normalize vector scores
            if vector_results:
                vector_scores = [score for _, score in vector_results]

                if normalization_method == "minmax":
                    normalized_vector = normalize_minmax(vector_scores)
                elif normalization_method == "zscore":
                    normalized_vector = normalize_zscore(vector_scores)
                else:
                    raise ValueError(
                        f"Unknown normalization method: {normalization_method}"
                    )

                for (chunk, _), norm_score in zip(vector_results, normalized_vector):
                    combined_scores[chunk.chunk_id] = {
                        "chunk": chunk,
                        "score": alpha_to_use * norm_score,
                    }

            # Normalize BM25 scores
            if bm25_results:
                bm25_scores = [score for _, score in bm25_results]

                if normalization_method == "minmax":
                    normalized_bm25 = normalize_minmax(bm25_scores)
                elif normalization_method == "zscore":
                    normalized_bm25 = normalize_zscore(bm25_scores)
                else:
                    raise ValueError(
                        f"Unknown normalization method: {normalization_method}"
                    )

                for (chunk, _), norm_score in zip(bm25_results, normalized_bm25):
                    if chunk.chunk_id in combined_scores:
                        combined_scores[chunk.chunk_id]["score"] += (
                            1 - alpha_to_use
                        ) * norm_score
                    else:
                        combined_scores[chunk.chunk_id] = {
                            "chunk": chunk,
                            "score": (1 - alpha_to_use) * norm_score,
                        }

            # Sort by combined score
            sorted_results = sorted(
                combined_scores.values(), key=lambda x: x["score"], reverse=True
            )[:top_k]

            results = [(item["chunk"], item["score"]) for item in sorted_results]

            alpha_info = f"α={alpha_to_use:.2f}"
            if self.enable_adaptive_alpha and alpha_to_use != self.alpha:
                alpha_info += f" (adapted from {self.alpha:.2f})"

            logger.info(
                f"Hybrid search ({normalization_method}): {len(results)} results ({alpha_info})"
            )
            for i, (chunk, score) in enumerate(results):
                logger.debug(
                    f"  [Hybrid {i + 1}] Score: {score:.4f} | ID: {chunk.chunk_id} | Content: {chunk.content[:100].replace(chr(10), ' ')}..."
                )

        # Apply MMR diversity enforcement if enabled
        if apply_mmr and results:
            query_embedding = self.embedding_model.encode_single(query, is_query=True)

            lambda_param = mmr_lambda if mmr_lambda is not None else settings.MMR_LAMBDA
            diversity_threshold = (
                mmr_diversity_threshold
                if mmr_diversity_threshold is not None
                else settings.MMR_DIVERSITY_THRESHOLD
            )

            results = maximal_marginal_relevance(
                query_embedding=query_embedding,
                chunks_with_scores=results,
                top_k=top_k,
                lambda_param=lambda_param,
                diversity_threshold=diversity_threshold,
            )

        return results


class Reranker:
    """
    Reranker using cross-encoder models
    Provides more accurate relevance scoring
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        Initialize reranker

        Args:
            model_name: Cross-encoder model name
        """
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        logger.info(f"Loading reranker model: {model_name}")

        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"✓ Loaded reranker: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
        apply_deduplication: bool = True,
        dedup_threshold: Optional[float] = None,
        embedding_model=None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank chunks using cross-encoder with optional deduplication

        Args:
            query: Search query
            chunks_with_scores: Initial retrieval results
            top_k: Number of results to return (None = return all)
            apply_deduplication: Whether to deduplicate similar chunks
            dedup_threshold: Similarity threshold for deduplication (uses settings if None)
            embedding_model: Embedding model for deduplication (optional)

        Returns:
            Reranked and deduplicated list of (chunk, score) tuples
        """
        if not chunks_with_scores:
            return []

        # Prepare query-document pairs
        pairs = [[query, chunk.content] for chunk, _ in chunks_with_scores]

        # Get reranking scores
        rerank_scores = self.model.predict(pairs)

        # Combine with chunks
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(chunks_with_scores, rerank_scores)
        ]

        # Sort by reranked score
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Reranked {len(chunks_with_scores)} chunks")

        # Apply deduplication before final top-k selection
        if apply_deduplication:
            threshold = (
                dedup_threshold
                if dedup_threshold is not None
                else settings.DEDUP_SIMILARITY_THRESHOLD
            )
            reranked = deduplicate_chunks(
                reranked,
                similarity_threshold=threshold,
                embedding_model=embedding_model,
            )

        # Return top-k after deduplication
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"Final results after deduplication: {len(reranked)} chunks")
        for i, (chunk, score) in enumerate(reranked):
            logger.debug(
                f"  [Reranked {i + 1}] Score: {score:.4f} | ID: {chunk.chunk_id} | Content: {chunk.content[:100].replace(chr(10), ' ')}..."
            )
        return reranked


class LLMReranker:
    """
    LLM-based Reranker using generative models
    Supports "Smart Prompts" for table/entity awareness and domain specificity
    """

    def __init__(self, llm_generator, domain: str = "general"):
        """
        Initialize LLM reranker

        Args:
            llm_generator: LLMGenerator instance
            domain: Domain context (e.g., "medical", "legal")
        """
        self.llm_generator = llm_generator
        self.domain = domain
        logger.info(f"Initialized LLMReranker with domain: {domain}")

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
        apply_deduplication: bool = True,
        dedup_threshold: Optional[float] = None,
        embedding_model=None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank using LLM evaluation
        """
        if not chunks_with_scores:
            return []

        # 1. Prepare candidates
        candidates_text = ""
        for i, (chunk, _) in enumerate(chunks_with_scores):
            # Truncate content for reasonable prompt size
            content_preview = chunk.content[:500].replace("\n", " ")
            candidates_text += f"ID: {i}\nCONTENT: {content_preview}...\n\n"

        # 2. Construct Smart Prompt
        prompt = f"""You are an expert reranker in the {self.domain} domain.
Your task is to score the relevance of the following document chunks to the user query.

User Query: "{query}"

Relevance Criteria:
1. **Direct Answer**: Does the chunk answer the query directly?
2. **Tabular Data**: Pay SPECIAL ATTENTION to chunks containing tables or structured data matching the query.
3. **Entity Match**: Prioritize chunks specifically mentioning entities (names, codes, dates) from the query.
4. **Domain Context**: Evaluate relevance based on standard {self.domain} practices.

Candidate Chunks:
{candidates_text}

INSTRUCTIONS:
- Assign a score from 0.0 to 1.0 for each chunk ID.
- 1.0 = Perfect match / Direct answer.
- 0.0 = Totally irrelevant.
- Return ONLY a JSON list of objects: [{{"id": 0, "score": 0.9}}, ...]
- Do not add any explanation.
"""

        # 3. Call LLM
        try:
            logger.info(
                f"LLM Reranking {len(chunks_with_scores)} chunks (Domain: {self.domain})"
            )
            response = self.llm_generator.generate(
                prompt, max_tokens=500, temperature=0.0
            )

            # Clean response (handle markdown code blocks if any)
            response = response.replace("```json", "").replace("```", "").strip()

            # Parse scores
            scores_list = json.loads(response)

            # Map back to chunks
            reranked_results = []
            for item in scores_list:
                idx = item.get("id")
                score = float(item.get("score", 0.0))

                if idx is not None and 0 <= idx < len(chunks_with_scores):
                    chunk = chunks_with_scores[idx][0]
                    reranked_results.append((chunk, score))

            # Add any missing chunks with 0 score (fallback)
            ranked_ids = {item.get("id") for item in scores_list}
            for i, (chunk, _) in enumerate(chunks_with_scores):
                if i not in ranked_ids:
                    reranked_results.append((chunk, 0.0))

            # Sort
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            reranked = reranked_results

        except Exception as e:
            logger.error(f"LLM Reranking failed: {e}. Falling back to original order.")
            reranked = chunks_with_scores

        logger.info(f"LLM Reranked {len(reranked)} chunks")

        # Apply deduplication logic (same as standard Reranker)
        if apply_deduplication:
            threshold = (
                dedup_threshold
                if dedup_threshold is not None
                else settings.DEDUP_SIMILARITY_THRESHOLD
            )
            reranked = deduplicate_chunks(
                reranked,
                similarity_threshold=threshold,
                embedding_model=embedding_model,
            )

        # Return top-k
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(f"Final results after LLM reranking: {len(reranked)} chunks")
        for i, (chunk, score) in enumerate(reranked):
            logger.debug(
                f"  [LLM Reranked {i + 1}] Score: {score:.4f} | ID: {chunk.chunk_id} | Content: {chunk.content[:100].replace(chr(10), ' ')}..."
            )

        return reranked


class MultiQueryRetriever:
    """
    Multi-query retrieval with query expansion
    Generates multiple query variations for better coverage
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_generator=None,  # Optional LLM for query generation
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator

    def _generate_query_variations(
        self, query: str, num_variations: int = 3
    ) -> List[str]:
        """
        Generate query variations
        Uses simple heuristics or LLM if available
        """
        variations = [query]  # Include original

        if self.llm_generator:
            # Use LLM to generate variations
            prompt = f"""Generate {num_variations} alternative phrasings of this question:
"{query}"

Return only the questions, one per line."""

            try:
                response = self.llm_generator.generate(prompt, max_tokens=200)
                generated = response.strip().split("\n")
                variations.extend([q.strip() for q in generated if q.strip()])
            except Exception as e:
                logger.warning(f"Failed to generate query variations: {e}")

        # Simple variations (if LLM not available)
        if len(variations) == 1:
            # Add question words
            if not query.lower().startswith(
                ("what", "how", "why", "when", "where", "who")
            ):
                variations.append(f"What is {query}?")
                variations.append(f"How does {query} work?")

        return variations[: num_variations + 1]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        num_variations: int = 2,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve using multiple query variations

        Args:
            query: Original query
            top_k: Number of final results
            num_variations: Number of query variations to generate
            metadata_filter: Optional metadata filters

        Returns:
            Fused and ranked results
        """
        # Generate variations
        queries = self._generate_query_variations(query, num_variations)
        logger.info(f"Generated {len(queries)} query variations")

        # Retrieve for each query
        all_results: Dict[str, Tuple[Chunk, List[float]]] = {}

        for q in queries:
            results = self.retriever.retrieve(
                q, top_k * 2, metadata_filter=metadata_filter
            )

            for chunk, score in results:
                if chunk.chunk_id not in all_results:
                    all_results[chunk.chunk_id] = (chunk, [])
                all_results[chunk.chunk_id][1].append(score)

        # Combine scores (max or average)
        combined_results = [
            (chunk, max(scores))  # Or use np.mean(scores)
            for chunk, scores in all_results.values()
        ]

        # Sort and return top-k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
