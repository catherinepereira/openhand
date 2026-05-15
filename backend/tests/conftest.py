"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Tests run with openhand/ on sys.path so backend.* imports work the same
# way they do under uvicorn (launched from the repo root), regardless of
# where pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT
