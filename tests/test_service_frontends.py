"""
tests/test_service_frontends.py — the two upload forms accept exactly the extensions
the text-lines readers support (#31 Phase 4).

The `accept=` lists are static HTML, so they drifted from text_formats.READERS (ten
supported extensions were missing). Equality, not a subset check, so a removed kind
fails too. No FastAPI needed: the pages are read as text.
"""

import re
from pathlib import Path

import pytest

import text_formats

ROOT = Path(__file__).resolve().parent.parent
_ACCEPT = re.compile(r'id="fileInput"[^>]*\baccept="([^"]*)"', re.S)


@pytest.mark.parametrize("page", ["service/frontend/index.html", "service/frontend-lindat/index.html"])
def test_upload_form_accepts_exactly_the_supported_extensions(page):
    html = (ROOT / page).read_text(encoding="utf-8")
    match = _ACCEPT.search(html)
    assert match, f"{page}: no accept= list on #fileInput"
    accepted = {ext.strip().lower() for ext in match.group(1).split(",") if ext.strip()}
    assert accepted == set(text_formats.supported_extensions())
