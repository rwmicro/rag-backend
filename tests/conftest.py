"""Shared test fixtures.

The stores under `rag.app_state` are lazily-created module-level singletons that
point at the paths in `config.settings`. Without isolation, every test run writes
into the developer's real ./data/*.db and leaks state into the next run — the
suite would only be green the first time it was ever executed.

Two scopes are offered because the suite has two shapes of test:
  * unit tests want a clean slate per test  -> `isolate_data_stores` (autouse)
  * the end-to-end flow in test_integration.py is a *sequence* (ingest, then
    re-ingest to check dedup, then query, then delete) whose steps share one
    collection, so it isolates once per module instead — see the
    `integration` marker opt-out below.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings  # noqa: E402

# Settings attributes that point at a per-run SQLite file or directory.
_DB_PATH_SETTINGS = (
    "COLLECTIONS_DB_PATH",
    "JOBS_DB_PATH",
    "DOC_REGISTRY_DB_PATH",
    "METADATA_DB_PATH",
    "FEEDBACK_DB_PATH",
    "SQLITE_VECTOR_DB_PATH",
)


@contextmanager
def redirect_data_stores(data_dir: Path):
    """Point every persistent store at `data_dir` and reset the singletons."""
    import rag.app_state as app_state

    data_dir.mkdir(parents=True, exist_ok=True)

    saved_settings = {}
    for name in (*_DB_PATH_SETTINGS, "DATA_DIR", "FAISS_INDEX_PATH"):
        if hasattr(settings, name):
            saved_settings[name] = getattr(settings, name)

    for name in _DB_PATH_SETTINGS:
        if name in saved_settings:
            filename = Path(saved_settings[name]).name
            object.__setattr__(settings, name, str(data_dir / filename))
    object.__setattr__(settings, "DATA_DIR", str(data_dir))
    object.__setattr__(settings, "FAISS_INDEX_PATH", str(data_dir / "faiss" / "index.faiss"))

    # Drop cached singletons so they are rebuilt against the temp paths.
    saved_singletons = {
        name: getattr(app_state, name)
        for name in ("_collections_db", "_job_store", "_doc_registry")
    }
    for name in saved_singletons:
        setattr(app_state, name, None)

    # main.py caches vector stores per collection id; a stale entry would point
    # at the previous test's index file.
    main_module = None
    try:
        import rag.main as main_module

        main_module.vector_stores.clear()
    except Exception:
        main_module = None

    try:
        yield
    finally:
        for name, value in saved_settings.items():
            object.__setattr__(settings, name, value)
        for name, value in saved_singletons.items():
            setattr(app_state, name, value)
        if main_module is not None:
            main_module.vector_stores.clear()


@pytest.fixture(autouse=True)
def isolate_data_stores(request, tmp_path):
    """Fresh data dir per test, except for the sequential integration flow."""
    if request.node.get_closest_marker("integration"):
        # test_integration.py isolates once per module (its steps build on each
        # other); resetting between them would delete the collection under test.
        yield
        return

    with redirect_data_stores(tmp_path / "data"):
        yield
