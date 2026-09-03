import sys
import types
from pathlib import Path

import pytest

# Stub the GPU/ML stack before importing the tool (it imports classify_TEXT),
# mirroring tests/test_calibration.py.
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
    ROT_FALSE_POSITIVE_GUARDS,
    TRASH_INVERTED,
)
from text_util import override_constants, pre_filter_line  # noqa: E402
from tools.const_importance_sweep import SEARCH_SPACE  # noqa: E402

_EXPECTED, _KNOWN = _load_lang_config(str(_ROOT / "setup" / "config.txt"))


def _process_mocked_line(text: str, ppl: float, orig_lang_score: float) -> str:
    """Categorise one line through the REAL production path, models aside.

    This used to hand-roll its own copy of the pipeline, and the copy was
    wrong in a way that mattered: it fed ``categorize_line`` the ``remap_lang``
    CAP (``LANG_SCORE_REMAP``, i.e. the score *recorded in the CSV*) where
    production feeds the **two-tier trust score**. Those are different numbers
    -- for the ``oueussd`` fixture, 0.75 against 0.9163 -- so every assertion in
    this module was being made about a value the pipeline never computes.

    It now goes through ``recategorize_from_csv._rescore_row``, which delegates
    to ``classify_TEXT.score_line``: the same single scoring step the live
    pipeline runs. ``original_lang`` is ces (expected/trusted), matching the
    convention already used by ``tests/test_calibration.py::_categ`` -- the
    trust tier is then 1.0, so ``orig_lang_score`` reaches the structural
    guards unscaled. That is the strict choice here: it is the *largest*
    language score the tiers can produce, which makes the "must be Trash"
    assertions harder to satisfy, not easier.

    Constants are read at call time inside the scorer, so the
    ``override_constants`` sweeps below are honoured.
    """
    action, clean_text = pre_filter_line(text)
    if action != "Process":
        return action

    row = {
        "text": clean_text,
        "original_text": text,
        "original_lang": "ces_Latn",
        "orig_lang_score": "0.0" if orig_lang_score is None else f"{orig_lang_score}",
        "perplex": "0.0" if ppl is None else f"{ppl}",
        "categ": "Noisy",
        "word_count": str(len(clean_text.split())),
    }
    return _rescore_row(row, _EXPECTED, _KNOWN)["categ"]


@pytest.mark.parametrize("text, ppl, lang_score, expected, note", ROT_FALSE_POSITIVE_GUARDS + CLEAR)
def test_clean_czech_never_demoted_to_trash_default_config(text, ppl, lang_score, expected, note):
    """Invariant at default config: Clean and highly-rotated valid Czech stays out of Trash."""
    categ = _process_mocked_line(text, ppl, lang_score)
    assert categ != "Trash", f"False positive demotion at default config: {note}"


@pytest.mark.parametrize("text, ppl, lang_score, expected, note", TRASH_INVERTED)
def test_inverted_trash_stays_trash_at_default(text, ppl, lang_score, expected, note):
    """Ensure the rotation guard remains strictly effective at default configurations."""
    categ = _process_mocked_line(text, ppl, lang_score)
    assert categ == "Trash", f"Failed to catch inverted trash at default config: {note}"


@pytest.mark.parametrize("text, ppl, lang_score, expected, note", ROT_FALSE_POSITIVE_GUARDS + CLEAR)
def test_clean_czech_tuning_robustness_swept_bounds(text, ppl, lang_score, expected, note):
    """
    Tuning-robustness gate: sweep critical garbage/rotation constants across their allowed
    const_importance_sweep search space boundaries to ensure parameter tuning won't
    accidentally demote valid Czech text.
    """
    params_to_sweep = [
        "ROT_RATIO_INVERTED_MIN",
        "WEIRD_RATIO_INVERTED_MIN",
        "PPL_INVERTED_MIN",
        "SUSPICIOUS_ROT_RATIO",
        "CATEG_GARBAGE_DENSITY_HIGH",
        "SUSPICIOUS_WQX_RATIO",
        "INVERTED_WEIRD_PENALTY",
    ]

    for param in params_to_sweep:
        for bound in ("low", "high"):
            val = SEARCH_SPACE[param][bound]
            with override_constants({param: val}):
                categ = _process_mocked_line(text, ppl, lang_score)
                assert categ != "Trash", f"Regression with {param}={val} ({bound}): {note}"
