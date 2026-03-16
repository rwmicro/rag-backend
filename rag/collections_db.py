"""SQLite-backed collections store - drop-in replacement for collections.json"""
import sqlite3
import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class CollectionsDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn().execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id          TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

    def load_all(self) -> Dict[str, Any]:
        rows = self._conn().execute(
            "SELECT id, data FROM collections ORDER BY created_at"
        ).fetchall()
        return {row["id"]: json.loads(row["data"]) for row in rows}

    def get(self, collection_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT data FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def upsert(self, collection_id: str, data: Dict[str, Any]):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        blob = json.dumps(data)
        existing = self._conn().execute(
            "SELECT created_at FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        self._conn().execute("""
            INSERT INTO collections (id, data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at
        """, (collection_id, blob, created_at, now))

    def delete(self, collection_id: str):
        self._conn().execute(
            "DELETE FROM collections WHERE id = ?", (collection_id,)
        )

    def migrate_from_json(self, json_path: str):
        import os
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path) as f:
                old = json.load(f)
            for cid, data in old.items():
                self.upsert(cid, data)
            logger.info(f"Migrated {len(old)} collections from {json_path} to SQLite")
        except Exception as e:
            logger.warning(f"Migration from JSON failed: {e}")
