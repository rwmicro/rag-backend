"""Job store for tracking async ingestion tasks"""
import sqlite3
import json
import uuid
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn().execute("""
            CREATE TABLE IF NOT EXISTS ingest_jobs (
                job_id      TEXT PRIMARY KEY,
                status      TEXT NOT NULL DEFAULT 'queued',
                progress    REAL,
                result      TEXT,
                error       TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conn().execute(
            "INSERT INTO ingest_jobs (job_id, status, created_at, updated_at) VALUES (?, 'queued', ?, ?)",
            (job_id, now, now)
        )
        return job_id

    def update(self, job_id: str, status: JobStatus, progress: Optional[float] = None,
               result: Optional[Dict] = None, error: Optional[str] = None):
        now = datetime.now(timezone.utc).isoformat()
        self._conn().execute("""
            UPDATE ingest_jobs
            SET status=?, progress=?, result=?, error=?, updated_at=?
            WHERE job_id=?
        """, (
            status.value, progress,
            json.dumps(result) if result else None,
            error, now, job_id,
        ))

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM ingest_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("result"):
            d["result"] = json.loads(d["result"])
        return d
