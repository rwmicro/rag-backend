"""Document hash registry for deduplication and change detection"""
import sqlite3
import hashlib
import json
import threading
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timezone


class DocRegistry:
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
            CREATE TABLE IF NOT EXISTS documents (
                collection_id   TEXT NOT NULL,
                filename        TEXT NOT NULL,
                file_hash       TEXT NOT NULL,
                chunk_ids       TEXT NOT NULL,
                ingested_at     TEXT NOT NULL,
                PRIMARY KEY (collection_id, filename)
            )
        """)

    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def get_existing(self, collection_id: str, filename: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM documents WHERE collection_id=? AND filename=?",
            (collection_id, filename)
        ).fetchone()
        return dict(row) if row else None

    def upsert(self, collection_id: str, filename: str, file_hash: str, chunk_ids: List[str]):
        now = datetime.now(timezone.utc).isoformat()
        self._conn().execute("""
            INSERT INTO documents (collection_id, filename, file_hash, chunk_ids, ingested_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(collection_id, filename) DO UPDATE SET
                file_hash = excluded.file_hash,
                chunk_ids = excluded.chunk_ids,
                ingested_at = excluded.ingested_at
        """, (collection_id, filename, file_hash, json.dumps(chunk_ids), now))

    def get_chunk_ids(self, collection_id: str, filename: str) -> List[str]:
        existing = self.get_existing(collection_id, filename)
        if not existing:
            return []
        return json.loads(existing["chunk_ids"])

    def is_duplicate(self, collection_id: str, filename: str, file_hash: str) -> bool:
        existing = self.get_existing(collection_id, filename)
        return existing is not None and existing["file_hash"] == file_hash

    def delete(self, collection_id: str, filename: str):
        self._conn().execute(
            "DELETE FROM documents WHERE collection_id=? AND filename=?",
            (collection_id, filename)
        )
