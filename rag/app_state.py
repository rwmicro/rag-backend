"""Lazy singletons for persistent stores.

Extracted from rag/main.py so routers can reach the stores without
re-importing main (which would cause circular imports).
"""

import os
from typing import Optional

from config.settings import settings
from rag.collections_db import CollectionsDB
from rag.job_store import JobStore
from rag.doc_registry import DocRegistry


COLLECTIONS_FILE = os.path.join(settings.DATA_DIR, "collections.json")

_collections_db: Optional[CollectionsDB] = None
_job_store: Optional[JobStore] = None
_doc_registry: Optional[DocRegistry] = None


def get_collections_db() -> CollectionsDB:
    global _collections_db
    if _collections_db is None:
        _collections_db = CollectionsDB(settings.COLLECTIONS_DB_PATH)
        _collections_db.migrate_from_json(COLLECTIONS_FILE)
    return _collections_db


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore(settings.JOBS_DB_PATH)
    return _job_store


def get_doc_registry() -> DocRegistry:
    global _doc_registry
    if _doc_registry is None:
        _doc_registry = DocRegistry(settings.DOC_REGISTRY_DB_PATH)
    return _doc_registry
