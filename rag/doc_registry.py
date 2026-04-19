"""Document hash registry for deduplication and change detection.

Tracks two levels of hashes:
  * file-level — quick "has anything changed at all?" check.
  * chunk-level — on a changed file, lets us re-embed only the chunks whose
    content actually changed.

Schema is forward/backward compatible: `chunk_ids` is retained; the new
`chunk_hashes` column stores a `{chunk_id: content_hash}` JSON map.
"""
import sqlite3
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone


def sha256_text(text: str) -> str:
    """Stable content hash for a chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IngestionDiff:
    """What to do with a document whose file-hash changed.

    - unchanged: chunk_ids whose content hash matches an existing chunk → keep as-is
    - added:     (tentative_chunk_id, content_hash) pairs to newly embed
    - removed:   chunk_ids in the old registry but no longer present → delete
    """

    unchanged: List[str]
    added: List[Tuple[str, str]]
    removed: List[str]

    @property
    def is_noop(self) -> bool:
        return not self.added and not self.removed


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
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                collection_id   TEXT NOT NULL,
                filename        TEXT NOT NULL,
                file_hash       TEXT NOT NULL,
                chunk_ids       TEXT NOT NULL,
                ingested_at     TEXT NOT NULL,
                PRIMARY KEY (collection_id, filename)
            )
        """)
        # Additive migration — safe to run on existing DBs.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "chunk_hashes" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN chunk_hashes TEXT")

    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def get_existing(self, collection_id: str, filename: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM documents WHERE collection_id=? AND filename=?",
            (collection_id, filename),
        ).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        collection_id: str,
        filename: str,
        file_hash: str,
        chunk_ids: List[str],
        chunk_hashes: Optional[Dict[str, str]] = None,
    ):
        """Record or update the hash fingerprint of a file + its chunks.

        `chunk_hashes` maps chunk_id → content_hash. If omitted, incremental
        re-ingest diffs fall back to "rebuild everything" for this document.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn().execute(
            """
            INSERT INTO documents (collection_id, filename, file_hash, chunk_ids, chunk_hashes, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(collection_id, filename) DO UPDATE SET
                file_hash = excluded.file_hash,
                chunk_ids = excluded.chunk_ids,
                chunk_hashes = excluded.chunk_hashes,
                ingested_at = excluded.ingested_at
            """,
            (
                collection_id,
                filename,
                file_hash,
                json.dumps(chunk_ids),
                json.dumps(chunk_hashes) if chunk_hashes else None,
                now,
            ),
        )

    def get_chunk_ids(self, collection_id: str, filename: str) -> List[str]:
        existing = self.get_existing(collection_id, filename)
        if not existing:
            return []
        return json.loads(existing["chunk_ids"])

    def get_chunk_hashes(self, collection_id: str, filename: str) -> Dict[str, str]:
        """Return {chunk_id: content_hash} for a previously ingested file."""
        existing = self.get_existing(collection_id, filename)
        if not existing or not existing.get("chunk_hashes"):
            return {}
        return json.loads(existing["chunk_hashes"])

    def diff_chunks(
        self,
        collection_id: str,
        filename: str,
        new_chunks: List[Tuple[str, str]],
    ) -> IngestionDiff:
        """Compute an incremental diff against a prior ingestion.

        Args:
            new_chunks: list of (tentative_chunk_id, content_hash) from freshly
                chunking the updated document. `tentative_chunk_id` is what
                you'd use if this were a first-time ingest — it's replaced by
                the stored chunk_id on matches.

        Returns:
            IngestionDiff with:
              - unchanged: existing chunk_ids whose content_hash still matches
              - added:     new (chunk_id, content_hash) pairs to embed
              - removed:   existing chunk_ids whose hash no longer appears

        Semantics:
            - Content-addressed: a chunk is "the same" iff its content_hash
              is unchanged, regardless of position in the document.
            - If no prior chunk_hashes exist (legacy record, pre-migration),
              everything is treated as added + removed (full rebuild).
        """
        old_map = self.get_chunk_hashes(collection_id, filename)
        new_hashes = {h for _, h in new_chunks}

        unchanged = [cid for cid, h in old_map.items() if h in new_hashes]
        removed = [cid for cid, h in old_map.items() if h not in new_hashes]

        kept_hashes = {old_map[cid] for cid in unchanged}
        added = [(cid, h) for cid, h in new_chunks if h not in kept_hashes]

        return IngestionDiff(unchanged=unchanged, added=added, removed=removed)

    def is_duplicate(self, collection_id: str, filename: str, file_hash: str) -> bool:
        existing = self.get_existing(collection_id, filename)
        return existing is not None and existing["file_hash"] == file_hash

    def delete(self, collection_id: str, filename: str):
        self._conn().execute(
            "DELETE FROM documents WHERE collection_id=? AND filename=?",
            (collection_id, filename),
        )

    def plan_ingest(
        self,
        collection_id: str,
        filename: str,
        chunks: List,
    ) -> Tuple[List, Optional[IngestionDiff], List[str], Dict[str, str]]:
        """Plan an (incremental) ingest against the prior registry state.

        Returns:
            chunks_to_embed:    subset of `chunks` whose content hash is new.
            diff:               IngestionDiff vs. prior record, or None for first-time ingest.
            final_chunk_ids:    combined id list (unchanged + newly added) to persist.
            final_chunk_hashes: combined {chunk_id: content_hash} map to persist.

        `chunks` must expose `chunk_id` and `content` attributes.
        """
        new_hashes_by_id = {c.chunk_id: sha256_text(c.content) for c in chunks}

        if not self.get_existing(collection_id, filename):
            return list(chunks), None, list(new_hashes_by_id.keys()), dict(new_hashes_by_id)

        diff = self.diff_chunks(collection_id, filename, list(new_hashes_by_id.items()))
        added_hashes = {h for _, h in diff.added}
        chunks_to_embed = [c for c in chunks if new_hashes_by_id[c.chunk_id] in added_hashes]

        old_hashes = self.get_chunk_hashes(collection_id, filename)
        final_ids = list(diff.unchanged) + [c.chunk_id for c in chunks_to_embed]
        final_hashes = {cid: old_hashes[cid] for cid in diff.unchanged}
        final_hashes.update(
            {c.chunk_id: new_hashes_by_id[c.chunk_id] for c in chunks_to_embed}
        )
        return chunks_to_embed, diff, final_ids, final_hashes
