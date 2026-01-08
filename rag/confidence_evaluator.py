"""
Confidence Evaluation and Fallback Strategies

Evaluates retrieval quality and applies fallback strategies when confidence is low.
"""

from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
from loguru import logger

from .chunking import Chunk
from config.settings import settings


class ConfidenceLevel(Enum):
    """Confidence levels for retrieval results"""
    HIGH = "high"           # Confidence >= high_threshold
    MEDIUM = "medium"       # medium_threshold <= confidence < high_threshold
    LOW = "low"             # minimum_threshold <= confidence < medium_threshold
    VERY_LOW = "very_low"   # confidence < minimum_threshold


class FallbackStrategy(Enum):
    """Available fallback strategies"""
    EXPAND_SEARCH = "expand_search"           # Retrieve more results (increase top_k)
    SWITCH_STRATEGY = "switch_strategy"       # Try different retrieval strategy
    RELAX_FILTERS = "relax_filters"          # Remove or loosen metadata filters
    MULTI_QUERY = "multi_query"              # Generate query variations
    FULL_TEXT_SEARCH = "full_text_search"    # Fall back to BM25-only search
    GRAPH_EXPANSION = "graph_expansion"      # Use graph relationships to expand
    BROADER_CONTEXT = "broader_context"      # Retrieve parent/sibling chunks
    NONE = "none"                            # No fallback needed


@dataclass
class ConfidenceEvaluation:
    """Result of confidence evaluation"""
    confidence_score: float
    confidence_level: ConfidenceLevel
    metrics: Dict[str, float]
    issues: List[str]
    suggested_fallbacks: List[FallbackStrategy]
    should_apply_fallback: bool


class ConfidenceEvaluator:
    """
    Evaluates retrieval confidence using multiple metrics

    Metrics considered:
    1. Score distribution: Are top results significantly better than rest?
    2. Score magnitude: Are absolute scores high enough?
    3. Result count: Did we get enough results?
    4. Score variance: Are results diverse or all similar?
    5. Top-K gap: Is there a gap between top result and others?
    """

    def __init__(
        self,
        high_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None,
        minimum_threshold: Optional[float] = None,
        enable_fallback: bool = True,
    ):
        """
        Initialize confidence evaluator

        Args:
            high_threshold: Threshold for high confidence (default from settings)
            medium_threshold: Threshold for medium confidence (default from settings)
            minimum_threshold: Minimum acceptable confidence (default from settings)
            enable_fallback: Whether to enable fallback strategies
        """
        self.high_threshold = high_threshold or settings.RETRIEVAL_CONFIDENCE_HIGH
        self.medium_threshold = medium_threshold or settings.RETRIEVAL_CONFIDENCE_MEDIUM
        self.minimum_threshold = minimum_threshold or settings.RETRIEVAL_CONFIDENCE_MINIMUM
        self.enable_fallback = enable_fallback and settings.ENABLE_FALLBACK_STRATEGIES

        logger.info(
            f"Initialized ConfidenceEvaluator "
            f"(thresholds: high={self.high_threshold}, "
            f"medium={self.medium_threshold}, "
            f"min={self.minimum_threshold}, "
            f"fallback={self.enable_fallback})"
        )

    def evaluate(
        self,
        results: List[Tuple[Chunk, float]],
        query: str,
        expected_min_results: int = 3,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceEvaluation:
        """
        Evaluate confidence of retrieval results

        Args:
            results: Retrieved chunks with scores
            query: Original query
            expected_min_results: Minimum expected number of results
            query_analysis: Optional query analysis for context

        Returns:
            ConfidenceEvaluation with score, level, and suggested fallbacks
        """
        if not results:
            return ConfidenceEvaluation(
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                metrics={"no_results": 1.0},
                issues=["No results returned"],
                suggested_fallbacks=[
                    FallbackStrategy.EXPAND_SEARCH,
                    FallbackStrategy.RELAX_FILTERS,
                    FallbackStrategy.MULTI_QUERY,
                ],
                should_apply_fallback=True,
            )

        # Compute individual metrics
        metrics = {}
        issues = []

        # 1. Score Magnitude Metric
        # Check if top scores are high enough
        scores = [score for _, score in results]
        top_score = scores[0] if scores else 0.0
        avg_score = np.mean(scores) if scores else 0.0

        metrics["top_score"] = top_score
        metrics["avg_score"] = avg_score

        if top_score < 0.5:
            issues.append(f"Low top score: {top_score:.2f}")
            metrics["score_magnitude"] = 0.3
        elif top_score < 0.7:
            metrics["score_magnitude"] = 0.6
        else:
            metrics["score_magnitude"] = 1.0

        # 2. Score Distribution Metric
        # Check if there's a clear winner or all scores are similar
        if len(scores) > 1:
            score_std = np.std(scores)
            score_range = max(scores) - min(scores)

            metrics["score_std"] = score_std
            metrics["score_range"] = score_range

            # Good: High variance means clear differentiation
            # Bad: Low variance means all results equally mediocre
            if score_std < 0.05:
                issues.append(f"Low score variance: {score_std:.3f} (results not differentiated)")
                metrics["score_distribution"] = 0.4
            elif score_std < 0.15:
                metrics["score_distribution"] = 0.7
            else:
                metrics["score_distribution"] = 1.0
        else:
            metrics["score_distribution"] = 0.5  # Neutral for single result

        # 3. Top-K Gap Metric
        # Is there a significant gap between top result and 2nd result?
        if len(scores) >= 2:
            gap = scores[0] - scores[1]
            metrics["top_gap"] = gap

            if gap > 0.2:
                # Strong winner - high confidence
                metrics["top_gap_score"] = 1.0
            elif gap > 0.1:
                # Moderate gap
                metrics["top_gap_score"] = 0.7
            elif gap < 0.03:
                # Very close scores - uncertain
                issues.append(f"Minimal gap between top results: {gap:.3f}")
                metrics["top_gap_score"] = 0.3
            else:
                metrics["top_gap_score"] = 0.5
        else:
            metrics["top_gap_score"] = 0.5  # Neutral

        # 4. Result Count Metric
        # Did we get enough results?
        result_count = len(results)
        metrics["result_count"] = result_count

        if result_count < expected_min_results:
            issues.append(f"Few results: {result_count} < {expected_min_results}")
            metrics["result_count_score"] = 0.4
        elif result_count < expected_min_results * 2:
            metrics["result_count_score"] = 0.7
        else:
            metrics["result_count_score"] = 1.0

        # 5. Score Decay Metric
        # Do scores decay smoothly or drop sharply?
        if len(scores) >= 3:
            # Compute decay rate from top to 3rd result
            decay_rate = (scores[0] - scores[2]) / max(scores[0], 0.01)
            metrics["decay_rate"] = decay_rate

            if decay_rate > 0.7:
                # Sharp drop - only top result is good
                issues.append(f"Sharp score decay: {decay_rate:.2f} (weak alternatives)")
                metrics["score_decay"] = 0.4
            elif decay_rate < 0.3:
                # Gentle decay - multiple good results
                metrics["score_decay"] = 1.0
            else:
                metrics["score_decay"] = 0.7
        else:
            metrics["score_decay"] = 0.5

        # 6. Query-Specific Adjustments
        # Adjust confidence based on query characteristics
        query_confidence_modifier = 1.0

        if query_analysis:
            # Low confidence queries (ambiguous, broad)
            if query_analysis.get("query_type") == "aggregative":
                # Aggregative queries are inherently harder
                query_confidence_modifier = 0.9
            elif query_analysis.get("query_type") == "multi_hop":
                # Multi-hop requires multiple pieces of info
                query_confidence_modifier = 0.85
            elif query_analysis.get("query_type") == "exact_match":
                # Exact match should have high confidence if found
                if top_score < 0.8:
                    issues.append("Exact match query with low score")
                    query_confidence_modifier = 0.7

        metrics["query_confidence_modifier"] = query_confidence_modifier

        # Aggregate confidence score
        # Weighted combination of metrics
        confidence_score = (
            0.30 * metrics["score_magnitude"]
            + 0.20 * metrics["score_distribution"]
            + 0.20 * metrics["top_gap_score"]
            + 0.15 * metrics["result_count_score"]
            + 0.15 * metrics["score_decay"]
        ) * query_confidence_modifier

        metrics["confidence_score"] = confidence_score

        # Determine confidence level
        if confidence_score >= self.high_threshold:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= self.medium_threshold:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= self.minimum_threshold:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW

        # Determine suggested fallback strategies
        suggested_fallbacks = self._suggest_fallbacks(
            confidence_level=confidence_level,
            metrics=metrics,
            issues=issues,
            query_analysis=query_analysis,
        )

        should_apply_fallback = (
            self.enable_fallback
            and confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        )

        logger.info(
            f"Confidence evaluation: {confidence_level.value} "
            f"(score={confidence_score:.2f}, issues={len(issues)})"
        )

        if issues:
            logger.debug(f"Confidence issues: {', '.join(issues)}")

        if suggested_fallbacks:
            logger.debug(f"Suggested fallbacks: {[f.value for f in suggested_fallbacks]}")

        return ConfidenceEvaluation(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            metrics=metrics,
            issues=issues,
            suggested_fallbacks=suggested_fallbacks,
            should_apply_fallback=should_apply_fallback,
        )

    def _suggest_fallbacks(
        self,
        confidence_level: ConfidenceLevel,
        metrics: Dict[str, float],
        issues: List[str],
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[FallbackStrategy]:
        """
        Suggest appropriate fallback strategies based on confidence issues

        Args:
            confidence_level: Evaluated confidence level
            metrics: Computed metrics
            issues: List of identified issues
            query_analysis: Optional query analysis

        Returns:
            List of suggested fallback strategies (ordered by priority)
        """
        if confidence_level == ConfidenceLevel.HIGH:
            return [FallbackStrategy.NONE]

        fallbacks = []

        # Issue: Few results
        if metrics.get("result_count", 0) < 3:
            fallbacks.append(FallbackStrategy.EXPAND_SEARCH)
            fallbacks.append(FallbackStrategy.RELAX_FILTERS)

        # Issue: Low scores across the board
        if metrics.get("avg_score", 0) < 0.4:
            fallbacks.append(FallbackStrategy.MULTI_QUERY)
            fallbacks.append(FallbackStrategy.SWITCH_STRATEGY)

        # Issue: Sharp score decay (only one good result)
        if metrics.get("decay_rate", 0) > 0.7:
            fallbacks.append(FallbackStrategy.BROADER_CONTEXT)
            fallbacks.append(FallbackStrategy.GRAPH_EXPANSION)

        # Issue: Low top score
        if metrics.get("top_score", 0) < 0.5:
            if query_analysis and query_analysis.get("requires_exact_match"):
                # For exact matches, try full-text search
                fallbacks.append(FallbackStrategy.FULL_TEXT_SEARCH)
            else:
                fallbacks.append(FallbackStrategy.SWITCH_STRATEGY)
                fallbacks.append(FallbackStrategy.MULTI_QUERY)

        # Query-specific fallbacks
        if query_analysis:
            query_type = query_analysis.get("query_type")

            if query_type == "multi_hop":
                # Multi-hop benefits from graph expansion
                fallbacks.append(FallbackStrategy.GRAPH_EXPANSION)

            elif query_type == "exact_match":
                # Exact matches benefit from full-text search
                fallbacks.append(FallbackStrategy.FULL_TEXT_SEARCH)

            elif query_type == "aggregative":
                # Aggregative queries need more results
                fallbacks.append(FallbackStrategy.EXPAND_SEARCH)

        # Remove duplicates while preserving order
        seen = set()
        unique_fallbacks = []
        for fallback in fallbacks:
            if fallback not in seen:
                seen.add(fallback)
                unique_fallbacks.append(fallback)

        # Limit to top 3 fallback strategies
        return unique_fallbacks[:3]

    def format_report(self, evaluation: ConfidenceEvaluation) -> str:
        """
        Generate human-readable confidence report

        Args:
            evaluation: ConfidenceEvaluation result

        Returns:
            Formatted report string
        """
        report = "=== Retrieval Confidence Report ===\n"
        report += f"Confidence Score: {evaluation.confidence_score:.2f}\n"
        report += f"Confidence Level: {evaluation.confidence_level.value.upper()}\n\n"

        report += "Metrics:\n"
        for key, value in evaluation.metrics.items():
            if isinstance(value, float):
                report += f"  • {key}: {value:.3f}\n"
            else:
                report += f"  • {key}: {value}\n"

        if evaluation.issues:
            report += f"\nIssues ({len(evaluation.issues)}):\n"
            for issue in evaluation.issues:
                report += f"  ⚠ {issue}\n"

        if evaluation.suggested_fallbacks:
            report += f"\nSuggested Fallbacks:\n"
            for i, fallback in enumerate(evaluation.suggested_fallbacks, 1):
                report += f"  {i}. {fallback.value}\n"

        report += f"\nApply Fallback: {'Yes' if evaluation.should_apply_fallback else 'No'}\n"
        report += "=" * 37

        return report
