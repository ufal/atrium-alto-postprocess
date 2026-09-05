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

# Stub the GPU/ML stack before importing the tool (it imports classify_TEXT).
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from recategorize_from_csv import _load_lang_config, _rescore_row  # noqa: E402

import classify_TEXT as LC  # noqa: E402
from tests.calibration_fixtures import (  # noqa: E402
    ALLCAPS_HEADLINE,
    CLEAR,
    HEADLINE_NUMBERED,
    NOISY,
    NON_TEXT,
    NOTATION_SHORT,
    ROT_FALSE_POSITIVE_GUARDS,
    SHORT_EXCEPTIONS,
    TRASH_GARBAGE,
    VOCABULARY_SHORT,
)
from text_util import pre_filter_line  # noqa: E402

_EXPECTED, _KNOWN = _load_lang_config(str(_ROOT / "setup" / "config.txt"))


def _categ(text, ppl, lang_score, original_lang="ces_Latn"):
    """Faithful per-line category via the production re-scorer.

    ``original_lang`` defaults to ces (expected, trust tier 1.0), which is right
    for the Czech fixtures: the remap cap is a no-op on them and the hard sweep
    keys off ``orig_lang_score``, which is preserved either way.

    It is a **parameter** rather than a constant because hardcoding it silently
    rescored every non-Czech fixture. Production sends an unrecognised language
    through ``TRUST_TIER_UNKNOWN`` (0.50), so FastText's ``isl`` at 0.56 reaches
    the guards as 0.28, not 0.56 — below ``LANG_SCORE_REMAP`` where the pinned
    value sat above it. Fixtures whose real prediction is not Czech must pass
    their own language or they are not testing what the pipeline does. This is
    the same class of harness bug already fixed in
    ``tests/test_rotation_regression.py``.
    """
    row = {
        "text": text,
        "original_text": text,
        "original_lang": original_lang,
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


# ---------------------------------------------------------------------------
# (#3 2026-07-02 DanaKriv calibration pass) forgiven-headline floor + the
# all-caps headline guard. Locked as `!= Trash` / `== Process`, mirroring the
# conservative style above: the 0.80 boundary may legitimately lift some of
# these further to Clear, and the invariant under test is the floor, not the
# exact band.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,ppl,ls,exp,note", HEADLINE_NUMBERED)
def test_headline_numbered_never_trashed(text, ppl, ls, exp, note):
    assert _categ(text, ppl, ls) != "Trash", note


@pytest.mark.parametrize("text,ppl,ls,exp,note", SHORT_EXCEPTIONS)
def test_short_exceptions_never_trashed(text, ppl, ls, exp, note):
    assert _categ(text, ppl, ls) != "Trash", note


@pytest.mark.parametrize("text,ppl,ls,exp,note", ALLCAPS_HEADLINE)
def test_allcaps_headline_word_is_scored(text, ppl, ls, exp, note):
    cat, _ = pre_filter_line(text)
    assert cat == "Process", note


# ── Issue #30: the short diacritic-free population ──────────────────────────


@pytest.mark.parametrize("text,ppl,ls,lang,exp,note", VOCABULARY_SHORT)
def test_short_vocabulary_current_behaviour_is_pinned(text, ppl, ls, lang, exp, note):
    """Freezes what the pipeline does to short domain vocabulary TODAY.

    Not an assertion that Trash is correct — issue #30 exists because it very
    likely is not. Pinning it makes any change visible in review, and makes the
    effect of a proposed gate on `rule_short_garbage` measurable rather than
    asserted.
    """
    assert _categ(text, ppl, ls, original_lang=lang) == exp, note


@pytest.mark.parametrize("text,ppl,ls,lang,exp,note", NOTATION_SHORT)
def test_short_notation_is_recovered(text, ppl, ls, lang, exp, note):
    """`is_domain_notation()` must keep notation out of `rule_short_garbage`.

    These would all be Trash without the predicate — `II/C` at 6e7 perplexity is
    the extreme case, and it survives only because notation is exempt from
    `rule_extreme_ppl` and `rule_absolute_ppl`.
    """
    assert _categ(text, ppl, ls, original_lang=lang) == exp, note


def test_fixture_languages_reach_the_guards_through_the_trust_tier():
    """Regression lock on the harness bug, anchored to the MECHANISM.

    Asserts on `trust_lang_score` rather than on a category, for two reasons.
    It is the actual thing the harness got wrong — hardcoding
    `original_lang="ces_Latn"` gives trust tier 1.0 where production applies
    TRUST_TIER_UNKNOWN (0.50). And it stays meaningful however issue #30 is
    resolved: a category-level assertion goes vacuous the moment a change makes
    all three fixtures agree again, which is precisely when the lock is most
    needed.

    `Equus caballus` is the clearest case: FastText says ast @ 0.77, so the
    guards see 0.385 — below LANG_SCORE_REMAP (0.75). Scored as Czech it would
    arrive as 0.77, above it, and the fixture would read as green-on-master.
    That artifact is why two of the three candidates offered in issue #30 looked
    non-discriminating.
    """
    for text, ppl, ls, lang, _exp, _note in VOCABULARY_SHORT:
        sig = LC.score_line(
            text_content=text,
            original_text=text,
            original_lang=lang,
            original_lang_score=ls,
            perplexity=ppl,
            known_lang_bases=_KNOWN,
            expected_langs=_EXPECTED,
        )
        as_czech = LC.score_line(
            text_content=text,
            original_text=text,
            original_lang="ces_Latn",
            original_lang_score=ls,
            perplexity=ppl,
            known_lang_bases=_KNOWN,
            expected_langs=_EXPECTED,
        )

        assert sig["trust_lang_score"] == pytest.approx(ls * LC.TRUST_TIER_UNKNOWN), (
            f"{text!r}: {lang} should be an unknown base and take TRUST_TIER_UNKNOWN"
        )
        assert as_czech["trust_lang_score"] == pytest.approx(ls), (
            f"{text!r}: ces is an expected language and should be unscaled"
        )
        assert sig["trust_lang_score"] < as_czech["trust_lang_score"]
