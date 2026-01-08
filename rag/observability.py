"""
Observability and Logging Module
Provides structured logging for RAG pipeline queries
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time
import json
import os
import uuid
from functools import wraps
from collections import deque
from loguru import logger

from config.settings import settings


@dataclass
class TimingMetrics:
    """Timing metrics for different pipeline stages"""
    embedding_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    reranking_ms: Optional[float] = None
    deduplication_ms: Optional[float] = None
    compression_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class RetrievalMetrics:
    """Metrics for the retrieval stage"""
    strategy_used: List[str] = field(default_factory=list)
    initial_candidates: int = 0
    final_candidates: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    vector_scores: List[float] = field(default_factory=list)
    bm25_scores: List[float] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    final_scores: List[float] = field(default_factory=list)
    deduplication_removed: int = 0
    normalization_method: Optional[str] = None


@dataclass
class GenerationMetrics:
    """Metrics for the generation stage"""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class QueryLog:
    """Complete query log entry"""
    query_id: str
    timestamp: str
    query: str
    contextualized_query: Optional[str] = None
    routing_decision: Optional[Dict[str, Any]] = None
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    collection_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class QueryLogger:
    """
    Centralized query logger for observability
    Maintains in-memory logs and writes to disk
    """

    def __init__(self, max_logs: int = 1000):
        """
        Initialize query logger

        Args:
            max_logs: Maximum number of logs to keep in memory
        """
        self.max_logs = max_logs
        self.logs: deque = deque(maxlen=max_logs)
        self.log_file = settings.QUERY_LOG_PATH

        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_query(self, query_log: QueryLog):
        """
        Log a query

        Args:
            query_log: QueryLog object to log
        """
        if not settings.ENABLE_QUERY_LOGGING:
            return

        # Add to in-memory logs
        self.logs.append(query_log)

        # Write to disk (JSONL format)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(query_log.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write query log to disk: {e}")

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent query logs

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of query log dictionaries
        """
        recent_logs = list(self.logs)[-limit:]
        return [log.to_dict() for log in recent_logs]

    def get_logs_by_collection(self, collection_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get logs for a specific collection

        Args:
            collection_id: Collection ID to filter by
            limit: Maximum number of logs to return

        Returns:
            List of query log dictionaries
        """
        filtered_logs = [
            log for log in self.logs
            if log.collection_id == collection_id
        ][-limit:]

        return [log.to_dict() for log in filtered_logs]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics from logs

        Returns:
            Dictionary with statistics
        """
        if not self.logs:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "avg_retrieval_latency_ms": 0,
                "avg_generation_latency_ms": 0,
            }

        total_latencies = [log.timing.total_ms for log in self.logs if log.timing.total_ms]
        retrieval_latencies = [log.timing.retrieval_ms for log in self.logs if log.timing.retrieval_ms]
        generation_latencies = [log.timing.generation_ms for log in self.logs if log.timing.generation_ms]

        return {
            "total_queries": len(self.logs),
            "avg_latency_ms": sum(total_latencies) / len(total_latencies) if total_latencies else 0,
            "avg_retrieval_latency_ms": sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0,
            "avg_generation_latency_ms": sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0,
            "avg_candidates_retrieved": sum(log.retrieval.initial_candidates for log in self.logs) / len(self.logs),
            "avg_final_candidates": sum(log.retrieval.final_candidates for log in self.logs) / len(self.logs),
        }


# Global query logger instance
_query_logger: Optional[QueryLogger] = None


def get_query_logger() -> QueryLogger:
    """Get or create the global query logger"""
    global _query_logger

    if _query_logger is None:
        _query_logger = QueryLogger(max_logs=settings.MAX_QUERY_LOGS)

    return _query_logger


def create_query_log(query: str, collection_id: Optional[str] = None) -> QueryLog:
    """
    Create a new query log entry

    Args:
        query: Original query string
        collection_id: Optional collection ID

    Returns:
        New QueryLog object
    """
    return QueryLog(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        query=query,
        collection_id=collection_id,
    )


def timing_decorator(stage_name: str):
    """
    Decorator to measure execution time of a function

    Args:
        stage_name: Name of the pipeline stage (e.g., "retrieval", "generation")

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.debug(f"{stage_name} completed in {elapsed_ms:.2f}ms")

            # Store timing in result if it's a tuple with query_log
            if isinstance(result, tuple) and len(result) >= 2:
                # Assume last element is query_log
                query_log = result[-1]
                if isinstance(query_log, QueryLog):
                    setattr(query_log.timing, f"{stage_name}_ms", elapsed_ms)

            return result

        return wrapper
    return decorator


class QueryContext:
    """
    Context manager for tracking query execution
    Automatically logs timing and metrics
    """

    def __init__(self, query: str, collection_id: Optional[str] = None):
        """
        Initialize query context

        Args:
            query: Original query
            collection_id: Optional collection ID
        """
        self.query_log = create_query_log(query, collection_id)
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self.query_log

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log"""
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.query_log.timing.total_ms = elapsed_ms

        # Log any errors
        if exc_type is not None:
            self.query_log.error = f"{exc_type.__name__}: {str(exc_val)}"

        # Save to logger
        get_query_logger().log_query(self.query_log)

        # Don't suppress exceptions
        return False


def measure_time(stage_name: str):
    """
    Context manager for measuring time of a code block

    Args:
        stage_name: Name of the stage being measured

    Usage:
        with measure_time("retrieval") as timer:
            # ... do work ...
            timer.elapsed_ms  # Access elapsed time
    """
    class Timer:
        def __init__(self):
            self.elapsed_ms = 0
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed_ms = (time.time() - self.start_time) * 1000
            logger.debug(f"{stage_name} completed in {self.elapsed_ms:.2f}ms")
            return False

    return Timer()
