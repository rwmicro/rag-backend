"""Corrective RAG loop (CRAG-style): grade, rewrite, retry, merge.

Pattern: after the initial retrieval, grade the result set with a
ConfidenceEvaluator. If confidence is low, ask the LLM to rewrite the query,
re-retrieve, and merge the two result sets so the final ranking benefits from
both attempts.

The loop is model-agnostic: it never instantiates an embedding model and
never assumes a specific reranker — it operates entirely through callables
the caller supplies, so the frontend's per-collection embedding choice is
preserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from .chunking import Chunk
    from .confidence_evaluator import ConfidenceEvaluator


# Numeric ranks for confidence levels (higher = more confident).
_LEVEL_RANK = {"very_low": 1, "low": 2, "medium": 3, "high": 4}


@dataclass
class CorrectiveAttempt:
    query: str
    confidence_score: float
    confidence_level: str
    num_results: int
    issues: List[str]
    rewrite_used: bool


@dataclass
class CorrectiveTrace:
    """Observability payload for /stats/queries and downstream debugging."""

    triggered: bool
    final_strategy: str = "initial"  # "initial" | "rewritten" | "merged"
    rewritten_query: Optional[str] = None
    attempts: List[CorrectiveAttempt] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "final_strategy": self.final_strategy,
            "rewritten_query": self.rewritten_query,
            "attempts": [asdict(a) for a in self.attempts],
        }


_REWRITE_PROMPT = """The retrieval query below produced poor results. Rewrite it to be \
more specific and retrieval-friendly: expand acronyms, add synonyms or domain \
terms, prefer concrete nouns over pronouns. Keep it a single line, under 30 \
words.

Issues with the current results:
{issues}

Original query:
{query}

Rewritten query (output the query only, no preamble, no quotes):"""


class CorrectiveLoop:
    """Single-retry corrective retrieval loop.

    One retry is the production sweet spot: most quality gain happens on the
    first rewrite, and a second retry rarely helps but doubles latency.
    """

    def __init__(
        self,
        evaluator: "ConfidenceEvaluator",
        llm_generator: Any,
        max_attempts: int = 1,
        trigger_level: str = "low",
        merge_method: str = "max",
    ):
        """
        Args:
            evaluator: ConfidenceEvaluator instance.
            llm_generator: anything with `.generate(prompt, max_tokens, temperature)`.
            max_attempts: number of retries after the initial attempt (0 disables).
            trigger_level: retry when initial confidence is at-or-below this level.
            merge_method: "max" (keep best score per chunk) or "rrf" (rank fusion).
        """
        self.evaluator = evaluator
        self.llm_generator = llm_generator
        self.max_attempts = max(0, int(max_attempts))
        self._trigger_rank = _LEVEL_RANK.get(trigger_level, 2)
        if merge_method not in ("max", "rrf"):
            raise ValueError(f"unknown merge_method: {merge_method}")
        self.merge_method = merge_method

    def run(
        self,
        query: str,
        initial_results: List[Tuple[Chunk, float]],
        retry_retrieve: Callable[[str], List[Tuple[Chunk, float]]],
        top_k: Optional[int] = None,
    ) -> Tuple[List[Tuple[Chunk, float]], CorrectiveTrace]:
        """Run the corrective loop on top of an initial retrieval.

        Args:
            query: the query that produced `initial_results`.
            initial_results: ranked (chunk, score) tuples from the first retrieval.
            retry_retrieve: callable invoked with the rewritten query if a retry
                is triggered. Should return ranked (chunk, score) tuples in the
                same score space as `initial_results`.
            top_k: optional cap on the merged result list.

        Returns:
            (final_results, trace) — final_results uses the original score scale.
        """
        trace = CorrectiveTrace(triggered=False)

        evaluation = self.evaluator.evaluate(initial_results, query)
        trace.attempts.append(
            CorrectiveAttempt(
                query=query,
                confidence_score=float(evaluation.confidence_score),
                confidence_level=evaluation.confidence_level.value,
                num_results=len(initial_results),
                issues=list(evaluation.issues),
                rewrite_used=False,
            )
        )

        initial_rank = _LEVEL_RANK.get(evaluation.confidence_level.value, 4)
        if self.max_attempts == 0 or initial_rank > self._trigger_rank:
            return initial_results, trace

        trace.triggered = True
        rewritten = self._rewrite_query(query, evaluation.issues)
        if not rewritten:
            return initial_results, trace

        trace.rewritten_query = rewritten
        logger.info(f"Corrective rewrite: {query!r} -> {rewritten!r}")

        try:
            retry_results = retry_retrieve(rewritten)
        except Exception as e:
            logger.warning(f"Corrective retry retrieval failed: {e}")
            return initial_results, trace

        retry_eval = self.evaluator.evaluate(retry_results, rewritten)
        trace.attempts.append(
            CorrectiveAttempt(
                query=rewritten,
                confidence_score=float(retry_eval.confidence_score),
                confidence_level=retry_eval.confidence_level.value,
                num_results=len(retry_results),
                issues=list(retry_eval.issues),
                rewrite_used=True,
            )
        )

        if not retry_results:
            trace.final_strategy = "initial"
            return initial_results, trace
        if not initial_results:
            trace.final_strategy = "rewritten"
            return retry_results[:top_k] if top_k else retry_results, trace

        merged = self._merge(initial_results, retry_results)
        if top_k:
            merged = merged[:top_k]
        trace.final_strategy = "merged"
        return merged, trace

    def _rewrite_query(self, query: str, issues: List[str]) -> Optional[str]:
        if self.llm_generator is None:
            return None
        issues_text = "\n".join(f"- {i}" for i in issues) or "- low retrieval scores"
        prompt = _REWRITE_PROMPT.format(query=query, issues=issues_text)
        try:
            rewritten = self.llm_generator.generate(
                prompt=prompt, max_tokens=120, temperature=0.3
            )
        except Exception as e:
            logger.warning(f"Corrective rewrite LLM call failed: {e}")
            return None

        rewritten = (rewritten or "").strip().strip('"').strip("'").strip()
        # Strip a leading "Rewritten query:" if the model echoed the prompt.
        for prefix in ("Rewritten query:", "Query:", "Output:"):
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()

        if not rewritten or rewritten.lower() == query.strip().lower():
            return None
        return rewritten

    def _merge(
        self,
        a: List[Tuple[Chunk, float]],
        b: List[Tuple[Chunk, float]],
    ) -> List[Tuple[Chunk, float]]:
        if self.merge_method == "rrf":
            from .retrieval import reciprocal_rank_fusion

            fused = reciprocal_rank_fusion([a, b])
            return sorted(fused.values(), key=lambda x: x[1], reverse=True)

        # "max": union by chunk_id, keep the highest score so downstream
        # MIN_SIMILARITY_SCORE / reranking semantics stay intact.
        by_id: Dict[str, Tuple[Chunk, float]] = {}
        for chunk, score in list(a) + list(b):
            current = by_id.get(chunk.chunk_id)
            if current is None or score > current[1]:
                by_id[chunk.chunk_id] = (chunk, score)
        return sorted(by_id.values(), key=lambda x: x[1], reverse=True)
