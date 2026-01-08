"""
Feedback Loop and Logging Infrastructure

Tracks RAG performance metrics, user feedback, and system events for:
1. Performance monitoring
2. Quality improvement
3. Debugging and troubleshooting
4. A/B testing and experimentation
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import json
import sqlite3
from pathlib import Path
from loguru import logger
import hashlib

from config.settings import settings


class EventType(Enum):
    """Types of events to log"""
    QUERY = "query"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    USER_FEEDBACK = "user_feedback"
    ERROR = "error"
    PERFORMANCE = "performance"


class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"           # 1-5 stars
    RELEVANCE = "relevance"     # Specific chunk relevance
    CORRECTION = "correction"   # User provides correction
    REPORT = "report"           # Report issue


@dataclass
class QueryEvent:
    """Log entry for query event"""
    event_id: str
    timestamp: str
    event_type: str
    query: str
    query_hash: str
    query_type: Optional[str] = None
    query_length: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvent:
    """Log entry for retrieval event"""
    event_id: str
    timestamp: str
    event_type: str
    query_hash: str
    retrieval_strategy: str
    top_k: int
    result_count: int
    retrieval_time_ms: float
    top_score: Optional[float] = None
    avg_score: Optional[float] = None
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationEvent:
    """Log entry for answer generation event"""
    event_id: str
    timestamp: str
    event_type: str
    query_hash: str
    answer_length: int
    generation_time_ms: float
    model_name: Optional[str] = None
    verification_status: Optional[str] = None
    verification_confidence: Optional[float] = None
    context_chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackEvent:
    """Log entry for user feedback"""
    event_id: str
    timestamp: str
    event_type: str
    query_hash: str
    feedback_type: str
    feedback_value: Any  # Rating, boolean, or text
    chunk_id: Optional[str] = None
    comment: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackLogger:
    """
    Comprehensive logging system for RAG pipeline

    Features:
    - Event logging to SQLite database
    - Performance metrics tracking
    - User feedback collection
    - Query analytics
    - Error tracking
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        enable_logging: bool = True,
        log_queries: bool = True,
        log_retrievals: bool = True,
        log_generations: bool = True,
        log_feedback: bool = True,
    ):
        """
        Initialize feedback logger

        Args:
            db_path: Path to SQLite database
            enable_logging: Master switch for logging
            log_queries: Log query events
            log_retrievals: Log retrieval events
            log_generations: Log generation events
            log_feedback: Log user feedback
        """
        self.db_path = db_path or settings.FEEDBACK_DB_PATH
        self.enable_logging = enable_logging
        self.log_queries = log_queries
        self.log_retrievals = log_retrievals
        self.log_generations = log_generations
        self.log_feedback = log_feedback

        # Create database directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        if self.enable_logging:
            self._init_database()
            logger.info(f"FeedbackLogger initialized (db={self.db_path})")
        else:
            logger.info("FeedbackLogger disabled")

    def _init_database(self):
        """Initialize SQLite database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                query TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                query_type TEXT,
                query_length INTEGER,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT
            )
        """)

        # Create index on query_hash for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_hash
            ON query_events(query_hash)
        """)

        # Create index on timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_timestamp
            ON query_events(timestamp)
        """)

        # Retrieval events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                retrieval_strategy TEXT NOT NULL,
                top_k INTEGER,
                result_count INTEGER,
                retrieval_time_ms REAL,
                top_score REAL,
                avg_score REAL,
                confidence_score REAL,
                confidence_level TEXT,
                chunk_ids TEXT,
                metadata TEXT,
                FOREIGN KEY (query_hash) REFERENCES query_events(query_hash)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_query_hash
            ON retrieval_events(query_hash)
        """)

        # Generation events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                answer_length INTEGER,
                generation_time_ms REAL,
                model_name TEXT,
                verification_status TEXT,
                verification_confidence REAL,
                context_chunk_count INTEGER,
                metadata TEXT,
                FOREIGN KEY (query_hash) REFERENCES query_events(query_hash)
            )
        """)

        # Feedback events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                feedback_value TEXT,
                chunk_id TEXT,
                comment TEXT,
                user_id TEXT,
                metadata TEXT,
                FOREIGN KEY (query_hash) REFERENCES query_events(query_hash)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_query_hash
            ON feedback_events(query_hash)
        """)

        # Performance metrics table (aggregated)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

        logger.debug("Database initialized with tables and indices")

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())

    def _compute_query_hash(self, query: str) -> str:
        """Compute hash of query for tracking"""
        return hashlib.md5(query.encode()).hexdigest()

    def log_query(
        self,
        query: str,
        query_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log query event

        Args:
            query: User query
            query_type: Classified query type
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata

        Returns:
            query_hash for linking related events
        """
        if not self.enable_logging or not self.log_queries:
            return self._compute_query_hash(query)

        event = QueryEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=EventType.QUERY.value,
            query=query,
            query_hash=self._compute_query_hash(query),
            query_type=query_type,
            query_length=len(query.split()),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )

        self._write_event("query_events", event)
        return event.query_hash

    def log_retrieval(
        self,
        query_hash: str,
        retrieval_strategy: str,
        result_count: int,
        retrieval_time_ms: float,
        top_k: int = 10,
        top_score: Optional[float] = None,
        avg_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        confidence_level: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log retrieval event

        Args:
            query_hash: Hash of query (from log_query)
            retrieval_strategy: Strategy used (hybrid, vector, etc.)
            result_count: Number of results retrieved
            retrieval_time_ms: Retrieval time in milliseconds
            top_k: Requested number of results
            top_score: Score of top result
            avg_score: Average score of results
            confidence_score: Confidence evaluation score
            confidence_level: Confidence level (high/medium/low)
            chunk_ids: IDs of retrieved chunks
            metadata: Additional metadata
        """
        if not self.enable_logging or not self.log_retrievals:
            return

        event = RetrievalEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=EventType.RETRIEVAL.value,
            query_hash=query_hash,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            result_count=result_count,
            retrieval_time_ms=retrieval_time_ms,
            top_score=top_score,
            avg_score=avg_score,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            chunk_ids=chunk_ids or [],
            metadata=metadata or {},
        )

        self._write_event("retrieval_events", event)

    def log_generation(
        self,
        query_hash: str,
        answer_length: int,
        generation_time_ms: float,
        model_name: Optional[str] = None,
        verification_status: Optional[str] = None,
        verification_confidence: Optional[float] = None,
        context_chunk_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log answer generation event

        Args:
            query_hash: Hash of query
            answer_length: Length of generated answer (words)
            generation_time_ms: Generation time in milliseconds
            model_name: LLM model name
            verification_status: Answer verification status
            verification_confidence: Verification confidence score
            context_chunk_count: Number of context chunks used
            metadata: Additional metadata
        """
        if not self.enable_logging or not self.log_generations:
            return

        event = GenerationEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=EventType.GENERATION.value,
            query_hash=query_hash,
            answer_length=answer_length,
            generation_time_ms=generation_time_ms,
            model_name=model_name,
            verification_status=verification_status,
            verification_confidence=verification_confidence,
            context_chunk_count=context_chunk_count,
            metadata=metadata or {},
        )

        self._write_event("generation_events", event)

    def log_feedback(
        self,
        query_hash: str,
        feedback_type: FeedbackType,
        feedback_value: Any,
        chunk_id: Optional[str] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log user feedback

        Args:
            query_hash: Hash of query
            feedback_type: Type of feedback
            feedback_value: Feedback value (rating, boolean, etc.)
            chunk_id: Optional chunk ID (for chunk-specific feedback)
            comment: Optional text comment
            user_id: Optional user identifier
            metadata: Additional metadata
        """
        if not self.enable_logging or not self.log_feedback:
            return

        event = FeedbackEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=EventType.USER_FEEDBACK.value,
            query_hash=query_hash,
            feedback_type=feedback_type.value,
            feedback_value=feedback_value,
            chunk_id=chunk_id,
            comment=comment,
            user_id=user_id,
            metadata=metadata or {},
        )

        self._write_event("feedback_events", event)

    def _write_event(self, table_name: str, event):
        """
        Write event to database

        Args:
            table_name: Name of table
            event: Event dataclass instance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Convert dataclass to dict
            event_dict = asdict(event)

            # Convert lists/dicts to JSON strings
            for key, value in event_dict.items():
                if isinstance(value, (list, dict)):
                    event_dict[key] = json.dumps(value)

            # Generate column names and placeholders
            columns = list(event_dict.keys())
            placeholders = ["?" for _ in columns]

            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """

            cursor.execute(query, list(event_dict.values()))
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to write event to {table_name}: {e}")

    def get_query_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get query analytics

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of results

        Returns:
            List of query analytics
        """
        if not self.enable_logging:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("timestamp <= ?")
            params.append(end_date)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
            SELECT query_hash, query, query_type, COUNT(*) as count
            FROM query_events
            WHERE {where_sql}
            GROUP BY query_hash
            ORDER BY count DESC
            LIMIT ?
        """

        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_performance_metrics(
        self,
        metric_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated performance metrics

        Args:
            metric_type: Filter by metric type (retrieval/generation)
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary of metrics
        """
        if not self.enable_logging:
            return {}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        metrics = {}

        # Retrieval metrics
        if not metric_type or metric_type == "retrieval":
            query = """
                SELECT
                    AVG(retrieval_time_ms) as avg_retrieval_time,
                    AVG(result_count) as avg_result_count,
                    AVG(top_score) as avg_top_score,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(*) as total_retrievals
                FROM retrieval_events
            """

            cursor.execute(query)
            row = cursor.fetchone()
            metrics["retrieval"] = dict(row) if row else {}

        # Generation metrics
        if not metric_type or metric_type == "generation":
            query = """
                SELECT
                    AVG(generation_time_ms) as avg_generation_time,
                    AVG(answer_length) as avg_answer_length,
                    AVG(verification_confidence) as avg_verification_confidence,
                    COUNT(*) as total_generations
                FROM generation_events
            """

            cursor.execute(query)
            row = cursor.fetchone()
            metrics["generation"] = dict(row) if row else {}

        # Feedback metrics
        query = """
            SELECT
                feedback_type,
                COUNT(*) as count
            FROM feedback_events
            GROUP BY feedback_type
        """

        cursor.execute(query)
        feedback_stats = {row["feedback_type"]: row["count"] for row in cursor.fetchall()}
        metrics["feedback"] = feedback_stats

        conn.close()
        return metrics

    def get_low_confidence_queries(
        self,
        threshold: float = 0.5,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get queries with low confidence scores

        Args:
            threshold: Confidence threshold
            limit: Maximum results

        Returns:
            List of low confidence queries
        """
        if not self.enable_logging:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT
                q.query,
                r.confidence_score,
                r.confidence_level,
                r.result_count
            FROM query_events q
            JOIN retrieval_events r ON q.query_hash = r.query_hash
            WHERE r.confidence_score < ?
            ORDER BY r.confidence_score ASC
            LIMIT ?
        """

        cursor.execute(query, (threshold, limit))
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return results

    def close(self):
        """Close logger (cleanup)"""
        logger.info("FeedbackLogger closed")
