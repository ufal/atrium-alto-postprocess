"""
tests/test_calibration.py
=========================
Real-data regression net for #3, driving the harvested calibration fixtures
through the production per-line path (recategorize_from_csv._rescore_row) with
frozen ppl / lang_score. We assert the contract, not borderline labels:

  * clean confident prose          -> Clear
  * readable degraded text         -> never Trash
  * confident-garbage (hard sweep) -> Trash
  * any garbage                    -> never Clear
  * numeric/stamp content          -> Non-text (pre-filter)
  * high-rot clean Czech           -> never Trash (rot false-positive guard)

Multi-token / interspersed inverted garbage that only the page-level sweep can
reclassify is intentionally NOT asserted per-line (see test_page_postprocess).
"""

import sys
import types
from pathlib import Path

import pytest

# Stub the GPU/ML stack before importing the tool (it imports langID_classify).
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from recategorize_from_csv import _load_lang_config, _rescore_row  # noqa: E402

from tests.calibration_fixtures import (  # noqa: E402
    CLEAR,
    NOISY,
    NON_TEXT,
    ROT_FALSE_POSITIVE_GUARDS,
    TRASH_GARBAGE,
)
from text_util_langID import pre_filter_line  # noqa: E402

_EXPECTED, _KNOWN = _load_lang_config(str(_ROOT / "config_langID.txt"))


def _categ(text, ppl, lang_score):
    """Faithful per-line category via the production re-scorer. original_lang is
    set to ces (trusted): for these low-score garbage lines the remap CAP is a
    no-op, and the hard sweep keys off orig_lang_score, which we preserve."""
    row = {
        "text": text,
        "original_text": text,
        "original_lang": "ces_Latn",
        "orig_lang_score": "0.0" if lang_score is None else f"{lang_score}",
        "perplex": "0.0" if ppl is None else f"{ppl}",
        "categ": "Noisy",
        "word_count": str(len(text.split())),
    }
    return _rescore_row(row, _EXPECTED, _KNOWN)["categ"]


@pytest.mark.parametrize("text,ppl,ls,exp,note", CLEAR, ids=lambda f: f if isinstance(f, str) else "")
def test_clean_prose_is_clear(text, ppl, ls, exp, note):
    assert _categ(text, ppl, ls) == "Clear", note


@pytest.mark.parametrize("text,ppl,ls,exp,note", NOISY)
def test_readable_text_never_trashed(text, ppl, ls, exp, note):
    # 0.85 may legitimately lift some of these to Clear; the locked invariant is
    # that readable Czech is NEVER Trashed.
    assert _categ(text, ppl, ls) != "Trash", note


_HARD_SWEEP = [f for f in TRASH_GARBAGE if f[1] is not None and f[2] is not None and f[2] < 0.45 and f[1] > 1000.0]


@pytest.mark.parametrize("text,ppl,ls,exp,note", _HARD_SWEEP)
def test_confident_garbage_is_trash(text, ppl, ls, exp, note):
    assert _categ(text, ppl, ls) == "Trash", note


@pytest.mark.parametrize("text,ppl,ls,exp,note", TRASH_GARBAGE)
def test_garbage_never_clear(text, ppl, ls, exp, note):
    assert _categ(text, ppl, ls) != "Clear", note


@pytest.mark.parametrize("text,ppl,ls,exp,note", NON_TEXT)
def test_numeric_stamp_content_filtered(text, ppl, ls, exp, note):
    cat, _ = pre_filter_line(text)
    assert cat in ("Non-text", "Empty"), note


@pytest.mark.parametrize("text,ppl,ls,exp,note", ROT_FALSE_POSITIVE_GUARDS)
def test_high_rot_clean_czech_never_trashed(text, ppl, ls, exp, note):
    assert _categ(text, ppl, ls) != "Trash", note
