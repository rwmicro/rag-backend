"""Guard against `settings.X` references that no attribute backs.

config/settings.py describes itself as a "MINIMAL VERSION" and has been trimmed
before while call sites kept referencing the removed fields. Because pydantic
raises AttributeError only when the line actually runs, such a gap stays
invisible until the feature is exercised at runtime — NORMALIZE_EMBEDDINGS broke
every embedding-model creation path, and RRF_K broke the default hybrid search.
"""

import re
from pathlib import Path

from config.settings import Settings

PROJECT_ROOT = Path(__file__).parent.parent
SETTINGS_ATTR = re.compile(r"\bsettings\.([A-Z][A-Z0-9_]+)")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "build", "dist"}


def _iter_source_files():
    for path in PROJECT_ROOT.rglob("*.py"):
        if SKIP_DIRS.isdisjoint(path.parts):
            yield path


def test_every_referenced_setting_exists():
    declared = set(Settings.model_fields)

    missing = {}
    for path in _iter_source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in SETTINGS_ATTR.finditer(line):
                name = match.group(1)
                if name not in declared:
                    location = f"{path.relative_to(PROJECT_ROOT)}:{lineno}"
                    missing.setdefault(name, []).append(location)

    assert not missing, "settings referenced but not declared in Settings:\n" + "\n".join(
        f"  {name} -> {', '.join(locations)}" for name, locations in sorted(missing.items())
    )
