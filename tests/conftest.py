"""
tests/conftest.py
=================
Shared pytest fixtures for the atrium-alto-postprocess test suite.

Nothing here requires ML models, GPU, or network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure repo root is importable ────────────────────────────────────────
# pytest.ini already sets pythonpath = . for pytest ≥ 7.
# This guard keeps things working with older pytest versions too.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))