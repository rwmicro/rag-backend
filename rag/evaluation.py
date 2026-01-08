"""
Evaluation Harness for RAG Pipeline
Provides comprehensive metrics and benchmarking for retrieval and generation quality
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from loguru import logger
import json
import csv
from datetime import datetime

from .chunking import Chunk


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EvaluationSample:
    """Single evaluation sample with query and ground truth"""
    query: str
    relevant_chunk_ids: List[str]  # Ground truth relevant chunks
    expected_answer: Optional[str] = None  # Optional ground truth answer
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation"""
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # NDCG@k for different k values
    recall_at_k: Dict[int, float]  # Recall@k
    precision_at_k: Dict[int, float]  # Precision@k
    map_score: float  # Mean Average Precision


@dataclass
class GenerationMetrics:
    """Metrics for generation evaluation"""
    relevance_score: float  # 0-1, how relevant is the answer
    faithfulness_score: float  # 0-1, is answer faithful to context
    completeness_score: float  # 0-1, does answer cover the question
    avg_length: float  # Average answer length in tokens


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    retrieval_metrics: RetrievalMetrics
    generation_metrics: Optional[GenerationMetrics] = None
    sample_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# Retrieval Metrics
# ============================================================================

def calculate_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Calculate Reciprocal Rank for a single query

    Args:
        retrieved_ids: List of retrieved chunk IDs (in order)
        relevant_ids: List of relevant chunk IDs (ground truth)

    Returns:
        Reciprocal rank (0 if no relevant docs found)
    """
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_mrr(
    results: List[Tuple[List[str], List[str]]]
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries

    Args:
        results: List of (retrieved_ids, relevant_ids) tuples

    Returns:
        Mean Reciprocal Rank
    """
    if not results:
        return 0.0

    rrs = [calculate_reciprocal_rank(ret, rel) for ret, rel in results]
    return np.mean(rrs)


def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Precision@k

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of relevant chunk IDs
        k: Cutoff position

    Returns:
        Precision@k
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)

    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Recall@k

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of relevant chunk IDs
        k: Cutoff position

    Returns:
        Recall@k
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)

    return relevant_in_top_k / len(relevant_ids)


def calculate_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of relevant chunk IDs
        k: Cutoff position

    Returns:
        NDCG@k
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            # Binary relevance (1 if relevant, 0 otherwise)
            dcg += 1.0 / np.log2(rank + 1)

    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for rank in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / np.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_average_precision(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Calculate Average Precision for a single query

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of relevant chunk IDs

    Returns:
        Average Precision
    """
    if not relevant_ids:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            precision_sum += precision_at_rank

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant_ids)


def calculate_map(
    results: List[Tuple[List[str], List[str]]]
) -> float:
    """
    Calculate Mean Average Precision across multiple queries

    Args:
        results: List of (retrieved_ids, relevant_ids) tuples

    Returns:
        Mean Average Precision
    """
    if not results:
        return 0.0

    aps = [calculate_average_precision(ret, rel) for ret, rel in results]
    return np.mean(aps)


# ============================================================================
# Evaluation Class
# ============================================================================

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG pipeline
    """

    def __init__(self, retriever=None, llm_generator=None):
        """
        Initialize evaluator

        Args:
            retriever: Retriever instance for evaluation
            llm_generator: LLM generator for LLM-as-judge evaluation
        """
        self.retriever = retriever
        self.llm_generator = llm_generator

    def evaluate_retrieval(
        self,
        samples: List[EvaluationSample],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance

        Args:
            samples: List of evaluation samples
            k_values: K values for @k metrics

        Returns:
            RetrievalMetrics object
        """
        if not samples or not self.retriever:
            raise ValueError("Samples and retriever required for evaluation")

        logger.info(f"Evaluating retrieval on {len(samples)} samples...")

        # Collect results for each sample
        results = []

        for sample in samples:
            # Retrieve chunks
            retrieved = self.retriever.retrieve(sample.query, top_k=max(k_values))
            retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]

            results.append((retrieved_ids, sample.relevant_chunk_ids))

        # Calculate metrics
        mrr = calculate_mrr(results)
        map_score = calculate_map(results)

        # Calculate @k metrics
        ndcg_at_k = {}
        recall_at_k = {}
        precision_at_k = {}

        for k in k_values:
            ndcg_scores = [calculate_ndcg_at_k(ret, rel, k) for ret, rel in results]
            recall_scores = [calculate_recall_at_k(ret, rel, k) for ret, rel in results]
            precision_scores = [calculate_precision_at_k(ret, rel, k) for ret, rel in results]

            ndcg_at_k[k] = np.mean(ndcg_scores)
            recall_at_k[k] = np.mean(recall_scores)
            precision_at_k[k] = np.mean(precision_scores)

        logger.info(f"Retrieval evaluation complete: MRR={mrr:.4f}, MAP={map_score:.4f}")

        return RetrievalMetrics(
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            map_score=map_score,
        )

    def evaluate_generation(
        self,
        samples: List[EvaluationSample],
        generated_answers: List[str],
        contexts: List[List[Chunk]],
    ) -> GenerationMetrics:
        """
        Evaluate generation quality using LLM-as-judge

        Args:
            samples: List of evaluation samples
            generated_answers: List of generated answers
            contexts: List of context chunks used for each answer

        Returns:
            GenerationMetrics object
        """
        if not self.llm_generator:
            raise ValueError("LLM generator required for generation evaluation")

        logger.info(f"Evaluating generation quality on {len(samples)} samples...")

        relevance_scores = []
        faithfulness_scores = []
        completeness_scores = []
        lengths = []

        for sample, answer, context_chunks in zip(samples, generated_answers, contexts):
            # Evaluate relevance
            relevance = self._evaluate_relevance(sample.query, answer)
            relevance_scores.append(relevance)

            # Evaluate faithfulness
            context_text = "\n\n".join([c.content for c in context_chunks])
            faithfulness = self._evaluate_faithfulness(context_text, answer)
            faithfulness_scores.append(faithfulness)

            # Evaluate completeness (if expected answer available)
            if sample.expected_answer:
                completeness = self._evaluate_completeness(
                    sample.query,
                    sample.expected_answer,
                    answer
                )
                completeness_scores.append(completeness)

            # Track length
            lengths.append(len(answer.split()))

        logger.info(f"Generation evaluation complete")

        return GenerationMetrics(
            relevance_score=np.mean(relevance_scores),
            faithfulness_score=np.mean(faithfulness_scores),
            completeness_score=np.mean(completeness_scores) if completeness_scores else 0.0,
            avg_length=np.mean(lengths),
        )

    def _evaluate_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate relevance of answer to query using LLM-as-judge

        Args:
            query: Original query
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        prompt = f"""Evaluate how relevant this answer is to the query on a scale of 0.0 to 1.0.

Query: {query}

Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0, where:
- 1.0 = Perfectly relevant, directly answers the query
- 0.5 = Partially relevant, touches on the topic
- 0.0 = Not relevant at all

Score:"""

        try:
            response = self.llm_generator.generate(prompt, max_tokens=10, temperature=0.0)
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning(f"Failed to evaluate relevance: {e}")
            return 0.5

    def _evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Evaluate faithfulness of answer to context using LLM-as-judge

        Args:
            context: Retrieved context
            answer: Generated answer

        Returns:
            Faithfulness score (0-1)
        """
        prompt = f"""Evaluate how faithful this answer is to the given context on a scale of 0.0 to 1.0.

Context:
{context[:2000]}

Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0, where:
- 1.0 = Answer is fully supported by context, no hallucinations
- 0.5 = Partially supported, some unsupported claims
- 0.0 = Not supported at all, hallucinated

Score:"""

        try:
            response = self.llm_generator.generate(prompt, max_tokens=10, temperature=0.0)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Failed to evaluate faithfulness: {e}")
            return 0.5

    def _evaluate_completeness(
        self,
        query: str,
        expected_answer: str,
        generated_answer: str
    ) -> float:
        """
        Evaluate completeness of answer using LLM-as-judge

        Args:
            query: Original query
            expected_answer: Expected/ideal answer
            generated_answer: Generated answer

        Returns:
            Completeness score (0-1)
        """
        prompt = f"""Evaluate how complete this answer is compared to the expected answer on a scale of 0.0 to 1.0.

Query: {query}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

Respond with ONLY a number between 0.0 and 1.0, where:
- 1.0 = Covers all key points from expected answer
- 0.5 = Covers some key points
- 0.0 = Misses all key points

Score:"""

        try:
            response = self.llm_generator.generate(prompt, max_tokens=10, temperature=0.0)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Failed to evaluate completeness: {e}")
            return 0.5

    def generate_synthetic_qa_pairs(
        self,
        chunks: List[Chunk],
        num_questions: int = 3,
    ) -> List[EvaluationSample]:
        """
        Generate synthetic Q&A pairs from chunks for evaluation

        Args:
            chunks: List of chunks to generate questions from
            num_questions: Number of questions to generate per chunk

        Returns:
            List of evaluation samples
        """
        if not self.llm_generator:
            raise ValueError("LLM generator required for synthetic Q&A generation")

        logger.info(f"Generating {num_questions} questions per chunk from {len(chunks)} chunks...")

        samples = []

        for chunk in chunks:
            prompt = f"""Generate {num_questions} specific questions that can be answered using ONLY the information in this text. Questions should be diverse and test different aspects.

Text:
{chunk.content}

Generate {num_questions} questions (one per line):"""

            try:
                response = self.llm_generator.generate(prompt, max_tokens=200, temperature=0.7)
                questions = [q.strip() for q in response.strip().split("\n") if q.strip()]

                for question in questions[:num_questions]:
                    samples.append(EvaluationSample(
                        query=question,
                        relevant_chunk_ids=[chunk.chunk_id],
                        metadata={"source": "synthetic", "chunk_id": chunk.chunk_id}
                    ))

            except Exception as e:
                logger.warning(f"Failed to generate questions for chunk {chunk.chunk_id}: {e}")

        logger.info(f"Generated {len(samples)} synthetic Q&A pairs")
        return samples

    def compare_strategies(
        self,
        samples: List[EvaluationSample],
        strategies: Dict[str, Any],
    ) -> Dict[str, EvaluationResult]:
        """
        A/B comparison of different retrieval strategies

        Args:
            samples: List of evaluation samples
            strategies: Dict of strategy_name -> strategy_config

        Returns:
            Dict of strategy_name -> EvaluationResult
        """
        logger.info(f"Comparing {len(strategies)} strategies on {len(samples)} samples...")

        results = {}

        for strategy_name, strategy_config in strategies.items():
            logger.info(f"Evaluating strategy: {strategy_name}")

            # Configure retriever with strategy
            # (Implementation depends on retriever interface)

            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(samples)

            results[strategy_name] = EvaluationResult(
                retrieval_metrics=retrieval_metrics,
                sample_count=len(samples),
                metadata={"strategy": strategy_name, "config": strategy_config}
            )

        logger.info(f"Strategy comparison complete")
        return results

    def export_results(
        self,
        results: EvaluationResult,
        output_path: str,
        format: str = "json"
    ):
        """
        Export evaluation results to file

        Args:
            results: Evaluation results
            output_path: Output file path
            format: Output format ("json" or "csv")
        """
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results.to_dict(), f, indent=2)
            logger.info(f"Results exported to {output_path} (JSON)")

        elif format == "csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Write retrieval metrics
                writer.writerow(["Metric", "Value"])
                writer.writerow(["MRR", results.retrieval_metrics.mrr])
                writer.writerow(["MAP", results.retrieval_metrics.map_score])

                for k, value in results.retrieval_metrics.ndcg_at_k.items():
                    writer.writerow([f"NDCG@{k}", value])

                for k, value in results.retrieval_metrics.recall_at_k.items():
                    writer.writerow([f"Recall@{k}", value])

                for k, value in results.retrieval_metrics.precision_at_k.items():
                    writer.writerow([f"Precision@{k}", value])

                # Write generation metrics if available
                if results.generation_metrics:
                    writer.writerow([])
                    writer.writerow(["Relevance", results.generation_metrics.relevance_score])
                    writer.writerow(["Faithfulness", results.generation_metrics.faithfulness_score])
                    writer.writerow(["Completeness", results.generation_metrics.completeness_score])
                    writer.writerow(["Avg Length", results.generation_metrics.avg_length])

            logger.info(f"Results exported to {output_path} (CSV)")

        else:
            raise ValueError(f"Unknown format: {format}")
