"""
Contrastive Retrieval for Negation Handling

Handles queries with negation by retrieving both positive and negative examples,
then filtering or re-ranking based on contrastive scoring.
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
import numpy as np
import re
from loguru import logger

from .chunking import Chunk
from .embeddings import EmbeddingModel
from config.settings import settings


@dataclass
class NegationAnalysis:
    """Analysis of negation in a query"""
    has_negation: bool
    negation_type: str  # "explicit", "implicit", "exclusion", "contrast"
    negated_terms: List[str]
    positive_query: str  # Query with negation removed
    negative_query: Optional[str]  # Query focusing on negated terms
    confidence: float


class NegationDetector:
    """
    Detects and analyzes negation in queries

    Handles multiple negation patterns:
    - Explicit negation: "not", "no", "never", "without"
    - Exclusion: "exclude", "except", "other than"
    - Contrast: "instead of", "rather than", "alternative to"
    """

    def __init__(self):
        """Initialize negation detector with patterns"""
        # Explicit negation words
        self.negation_words = {
            "not", "no", "none", "never", "nothing", "nobody", "nowhere",
            "neither", "nor", "without", "lacking", "absent", "excluding"
        }

        # Exclusion patterns
        self.exclusion_patterns = [
            r'\bexclude\b',
            r'\bexcept\b',
            r'\bbut not\b',
            r'\bother than\b',
            r'\baside from\b',
            r'\bapart from\b',
            r'\brather than\b',
            r'\binstead of\b',
        ]

        # Contrast patterns
        self.contrast_patterns = [
            r'\balternative to\b',
            r'\bdifferent from\b',
            r'\bopposite of\b',
            r'\bcontrary to\b',
            r'\bversus\b',
            r'\bvs\.?\b',
        ]

        logger.info("Initialized NegationDetector")

    def analyze(self, query: str) -> NegationAnalysis:
        """
        Analyze query for negation

        Args:
            query: User query

        Returns:
            NegationAnalysis with detected negation information
        """
        query_lower = query.lower()
        has_negation = False
        negation_type = None
        negated_terms = []
        confidence = 0.0

        # 1. Check for explicit negation words
        query_words = set(query_lower.split())
        negation_found = query_words & self.negation_words

        if negation_found:
            has_negation = True
            negation_type = "explicit"
            confidence = 0.9

            # Extract negated terms (words after negation words)
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in self.negation_words and i + 1 < len(words):
                    # Get 1-3 words after negation word
                    negated_terms.extend(words[i+1:min(i+4, len(words))])

            logger.debug(f"Explicit negation detected: {negation_found}")

        # 2. Check for exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.search(pattern, query_lower):
                has_negation = True
                negation_type = "exclusion"
                confidence = max(confidence, 0.85)

                # Extract terms after exclusion phrase
                match = re.search(pattern + r'\s+(\w+(?:\s+\w+)*)', query_lower)
                if match:
                    negated_terms.append(match.group(1))

                logger.debug(f"Exclusion pattern detected: {pattern}")
                break

        # 3. Check for contrast patterns
        if not has_negation:
            for pattern in self.contrast_patterns:
                if re.search(pattern, query_lower):
                    has_negation = True
                    negation_type = "contrast"
                    confidence = 0.75

                    # Extract contrasting terms
                    match = re.search(pattern + r'\s+(\w+(?:\s+\w+)*)', query_lower)
                    if match:
                        negated_terms.append(match.group(1))

                    logger.debug(f"Contrast pattern detected: {pattern}")
                    break

        # 4. Generate positive and negative queries
        positive_query = query
        negative_query = None

        if has_negation and negated_terms:
            # Remove negation words and terms for positive query
            positive_query = self._remove_negation(query, negated_terms)

            # Create negative query focusing on negated terms
            negative_query = " ".join(negated_terms)

        if has_negation:
            logger.info(
                f"Negation analysis: type={negation_type}, "
                f"confidence={confidence:.2f}, "
                f"negated_terms={negated_terms}"
            )

        return NegationAnalysis(
            has_negation=has_negation,
            negation_type=negation_type or "none",
            negated_terms=negated_terms,
            positive_query=positive_query,
            negative_query=negative_query,
            confidence=confidence,
        )

    def _remove_negation(self, query: str, negated_terms: List[str]) -> str:
        """
        Remove negation words and negated terms from query

        Args:
            query: Original query
            negated_terms: Terms to remove

        Returns:
            Cleaned query
        """
        # Remove negation words
        words = query.split()
        filtered_words = [
            word for word in words
            if word.lower() not in self.negation_words
        ]

        # Remove negated terms
        result = " ".join(filtered_words)

        # Remove exclusion/contrast phrases
        for pattern in self.exclusion_patterns + self.contrast_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = " ".join(result.split())

        return result if result else query


class ContrastiveRetriever:
    """
    Contrastive retrieval for handling negation queries

    Strategy:
    1. Detect negation in query
    2. Generate positive query (what to find) and negative query (what to avoid)
    3. Retrieve candidates using positive query
    4. Retrieve negative examples using negative query
    5. Filter or re-rank by penalizing results similar to negative examples

    Formula for contrastive scoring:
        score_contrastive = score_positive - β × max(sim(result, neg_example))

    Where:
        - score_positive: Original relevance score
        - β: Penalty weight (default 0.5)
        - sim(result, neg_example): Similarity to negative examples
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        base_retriever,
        penalty_weight: float = 0.5,
        negative_threshold: float = 0.7,
    ):
        """
        Initialize contrastive retriever

        Args:
            embedding_model: Embedding model for computing similarities
            base_retriever: Base retriever to use for searching
            penalty_weight: Weight for negative similarity penalty (β)
            negative_threshold: Similarity threshold for considering negative match
        """
        self.embedding_model = embedding_model
        self.base_retriever = base_retriever
        self.penalty_weight = penalty_weight
        self.negative_threshold = negative_threshold
        self.negation_detector = NegationDetector()

        logger.info(
            f"Initialized ContrastiveRetriever "
            f"(β={penalty_weight}, threshold={negative_threshold})"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        apply_contrastive: bool = True,
    ) -> Tuple[List[Tuple[Chunk, float]], Optional[NegationAnalysis]]:
        """
        Retrieve with contrastive negation handling

        Args:
            query: User query
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
            apply_contrastive: Whether to apply contrastive filtering

        Returns:
            Tuple of (results, negation_analysis)
        """
        # 1. Analyze query for negation
        negation_analysis = self.negation_detector.analyze(query)

        # 2. If no negation or not applying contrastive, use standard retrieval
        if not apply_contrastive or not negation_analysis.has_negation:
            results = self.base_retriever.retrieve(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            return results, negation_analysis

        logger.info(
            f"Applying contrastive retrieval for {negation_analysis.negation_type} negation"
        )

        # 3. Retrieve positive candidates (what we want)
        positive_query = negation_analysis.positive_query
        positive_results = self.base_retriever.retrieve(
            query=positive_query,
            top_k=top_k * 3,  # Retrieve more for filtering
            metadata_filter=metadata_filter,
        )

        if not positive_results:
            logger.warning("No positive results found")
            return [], negation_analysis

        # 4. Retrieve negative examples (what we want to avoid)
        negative_examples = []
        if negation_analysis.negative_query:
            negative_query = negation_analysis.negative_query
            negative_results = self.base_retriever.retrieve(
                query=negative_query,
                top_k=min(10, top_k),  # Fewer negative examples
                metadata_filter=metadata_filter,
            )
            negative_examples = [chunk for chunk, _ in negative_results]

            logger.debug(f"Retrieved {len(negative_examples)} negative examples")

        # 5. Apply contrastive scoring
        contrastive_results = self._apply_contrastive_scoring(
            positive_results=positive_results,
            negative_examples=negative_examples,
            negated_terms=negation_analysis.negated_terms,
        )

        # 6. Sort and return top_k
        contrastive_results.sort(key=lambda x: x[1], reverse=True)
        final_results = contrastive_results[:top_k]

        logger.info(
            f"Contrastive retrieval: {len(final_results)} results "
            f"(filtered from {len(positive_results)})"
        )

        return final_results, negation_analysis

    def _apply_contrastive_scoring(
        self,
        positive_results: List[Tuple[Chunk, float]],
        negative_examples: List[Chunk],
        negated_terms: List[str],
    ) -> List[Tuple[Chunk, float]]:
        """
        Apply contrastive scoring to filter/re-rank results

        Args:
            positive_results: Initial retrieval results
            negative_examples: Chunks representing what to avoid
            negated_terms: Terms that were negated in query

        Returns:
            Re-scored results
        """
        if not negative_examples and not negated_terms:
            return positive_results

        contrastive_results = []

        # Get embeddings for negative examples
        negative_embeddings = []
        if negative_examples:
            for neg_chunk in negative_examples:
                if neg_chunk.embedding is not None:
                    negative_embeddings.append(neg_chunk.embedding)

        if negative_embeddings:
            negative_embeddings = np.array(negative_embeddings)
            # Normalize for cosine similarity
            negative_norms = negative_embeddings / (
                np.linalg.norm(negative_embeddings, axis=1, keepdims=True) + 1e-8
            )

        # Process each positive result
        for chunk, score in positive_results:
            # Start with original score
            contrastive_score = score

            # Penalty 1: Embedding similarity to negative examples
            if negative_embeddings and chunk.embedding is not None:
                chunk_embedding = np.array(chunk.embedding).reshape(1, -1)
                chunk_norm = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)

                # Compute similarities with all negative examples
                similarities = np.dot(negative_norms, chunk_norm.T).flatten()
                max_neg_similarity = similarities.max()

                # Apply penalty if similarity exceeds threshold
                if max_neg_similarity > self.negative_threshold:
                    penalty = self.penalty_weight * max_neg_similarity
                    contrastive_score -= penalty

                    logger.debug(
                        f"Chunk {chunk.chunk_id}: "
                        f"negative similarity={max_neg_similarity:.3f}, "
                        f"penalty={penalty:.3f}"
                    )

            # Penalty 2: Content-based filtering (negated terms)
            if negated_terms:
                chunk_content_lower = chunk.content.lower()
                negated_term_count = 0

                for term in negated_terms:
                    term_lower = term.lower()
                    # Count occurrences of negated term
                    if term_lower in chunk_content_lower:
                        negated_term_count += chunk_content_lower.count(term_lower)

                # Apply content-based penalty
                if negated_term_count > 0:
                    content_penalty = min(0.3, negated_term_count * 0.1)
                    contrastive_score -= content_penalty

                    logger.debug(
                        f"Chunk {chunk.chunk_id}: "
                        f"negated term count={negated_term_count}, "
                        f"content penalty={content_penalty:.3f}"
                    )

            # Only keep results with positive score
            if contrastive_score > 0:
                contrastive_results.append((chunk, contrastive_score))

        return contrastive_results

    def explain_contrastive_scoring(
        self,
        negation_analysis: NegationAnalysis,
        original_count: int,
        filtered_count: int,
    ) -> str:
        """
        Generate explanation of contrastive scoring

        Args:
            negation_analysis: Negation analysis result
            original_count: Original result count
            filtered_count: Filtered result count

        Returns:
            Explanation string
        """
        explanation = "=== Contrastive Retrieval Report ===\n"
        explanation += f"Negation Type: {negation_analysis.negation_type}\n"
        explanation += f"Confidence: {negation_analysis.confidence:.2f}\n"

        if negation_analysis.negated_terms:
            explanation += f"Negated Terms: {', '.join(negation_analysis.negated_terms)}\n"

        explanation += f"\nPositive Query: \"{negation_analysis.positive_query}\"\n"

        if negation_analysis.negative_query:
            explanation += f"Negative Query: \"{negation_analysis.negative_query}\"\n"

        explanation += f"\nResults: {filtered_count} (filtered from {original_count})\n"
        explanation += f"Penalty Weight: β={self.penalty_weight}\n"
        explanation += f"Negative Threshold: {self.negative_threshold}\n"

        explanation += "\nScoring Formula:\n"
        explanation += "  score_contrastive = score_positive - β × max(sim(result, neg_example))\n"
        explanation += "  - Also penalized if negated terms appear in content\n"

        explanation += "=" * 36

        return explanation
