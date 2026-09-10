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
    TRASH_INVERTED,
    VOCABULARY_SHORT,
)
from text_util import _has_shape_garbage_evidence, _has_strong_garbage_evidence, pre_filter_line  # noqa: E402

_EXPECTED, _KNOWN = _load_lang_config(str(_ROOT / "setup" / "config.txt"))


def _row_values(row):
    """Fixture rows may be plain tuples OR pytest.param() ParameterSets.

    Issue #30 wraps the `oueussd` row of TRASH_INVERTED in ``pytest.param(...)``
    to carry a strict xfail. ``pytest.param`` returns a ParameterSet, a 3-field
    NamedTuple, so ``len(row)`` on that row silently becomes 3 and any 5-tuple
    unpack over the list raises a TypeError naming neither the list nor the row.

    ``@pytest.mark.parametrize`` unwraps ParameterSets itself, so only code that
    reads a fixture list positionally OUTSIDE parametrize needs this. Everything
    that does goes through here.
    """
    return tuple(getattr(row, "values", row))


# The declared shape of every fixture list. Nine are 5-tuples
# ``(text, ppl, orig_lang_score, expected_categ, note)``; the two issue-#30
# lists carry the FastText language at index 3, because scoring them as Czech
# grants trust tier 1.0 where production applies TRUST_TIER_UNKNOWN — see the
# comment block above VOCABULARY_SHORT in tests/calibration_fixtures.py.
_FIXTURE_ARITIES = [
    ("CLEAR", 5),
    ("NOISY", 5),
    ("TRASH_GARBAGE", 5),
    ("TRASH_INVERTED", 5),
    ("NON_TEXT", 5),
    ("ROT_FALSE_POSITIVE_GUARDS", 5),
    ("HEADLINE_NUMBERED", 5),
    ("SHORT_EXCEPTIONS", 5),
    ("ALLCAPS_HEADLINE", 5),
    ("NOTATION_SHORT", 6),
    ("VOCABULARY_SHORT", 6),
]


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
    for row in VOCABULARY_SHORT:
        text, ppl, ls, lang, _exp, _note = _row_values(row)
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


def test_notation_is_convicted_by_hard_sweep_when_language_also_fails():
    """The behavioural half of the hard-sweep decision (issue #30, §K).

    @david-spacil measured that with `SHORT_PPL_CAP` lifted, `II/C`, `1 ks` and
    `Reg.Bez.Aussig.` still route to `trash_hard_sweep`. Pinned here as intended
    behaviour: notation is exempt from the two routes that convict on perplexity
    alone, but not from `rule_hard_sweep`, which needs FastText to have failed
    independently (`orig_lang_score < HARD_SWEEP_LANG_MAX`).

    The justification is measured in
    `test_text_utils.py::TestNotationIsNotExemptFromHardSweep` — the predicate
    accepts most capitalised dot-chained garbage, so it cannot carry a hard-sweep
    exemption on its own.
    """
    for text in ("II/C", "1 ks", "Reg.Bez.Aussig."):
        assert (
            LC.score_line(
                text_content=text,
                original_text=text,
                original_lang="ces_Latn",
                original_lang_score=0.30,  # below HARD_SWEEP_LANG_MAX
                perplexity=6.0e7,
                known_lang_bases=_KNOWN,
                expected_langs=_EXPECTED,
                apply_short_cap=False,
            )["reason"]
            == "trash_hard_sweep"
        ), text


def test_notation_survives_perplexity_when_language_is_placed():
    """The other side of the same gate: hard sweep needs BOTH witnesses.

    When FastText does place the line, notation survives arbitrarily high
    perplexity — which is the whole point of exempting it from
    `rule_extreme_ppl` and `rule_absolute_ppl`. `II/C` measures around 6e7 and is
    correct; without the exemption that number alone would convict it.
    """
    for text in ("II/C", "1 ks", "Reg.Bez.Aussig."):
        categ = LC.score_line(
            text_content=text,
            original_text=text,
            original_lang="ces_Latn",
            original_lang_score=0.60,  # above HARD_SWEEP_LANG_MAX
            perplexity=6.0e7,
            known_lang_bases=_KNOWN,
            expected_langs=_EXPECTED,
            apply_short_cap=False,
        )["categ"]
        assert categ != "Trash", f"{text} was convicted on perplexity alone"


@pytest.mark.parametrize("name, arity", _FIXTURE_ARITIES)
def test_every_fixture_row_has_its_declared_arity(name, arity):
    """Structural guard on tests/calibration_fixtures.py.

    Two shapes coexist there (5-tuple and 6-tuple, the language at index 3), and
    since issue #30 a row may additionally be a ``pytest.param(...)`` wrapper.
    Both facts are invisible until something unpacks a row and gets a TypeError
    that names neither the list nor the row. This is the test that names them.

    It is also why the ALL_FIXTURES aggregate was removed: concatenating lists
    of two different arities, one of which may contain ParameterSets, is a shape
    error waiting for its first consumer.
    """
    import tests.calibration_fixtures as CF

    rows = getattr(CF, name)
    assert rows, f"{name} is empty"
    for row in rows:
        values = _row_values(row)
        assert len(values) == arity, f"{name}: expected {arity} fields, got {len(values)} in {row!r}"
        assert isinstance(values[0], str), f"{name}: first field must be the text, got {values[0]!r}"


def test_strong_evidence_is_false_on_the_entire_disputed_population():
    """The mechanism behind issue #30, asserted where a category cannot hide it.

    `_has_strong_garbage_evidence()` is the second witness six §9 rules require,
    and the one proposed as a gate on `rule_short_garbage`. On the short
    diacritic-free population that issue #30 is about it returns False on all
    six of its clauses at once:

      * `gibberish_present`  — 0; these are not vowel-dominated tokens
      * `valid_word_ratio <= 0.20` — the ratio is 1.0, because
        `compute_valid_ratio()` is shape-only (length >= 3, >= 70% alphabetic,
        no strange character, no mid-word uppercase) and every line here passes
      * `lang_score <= 0.20 and orig_lang_score <= 0.50` — fails on the FIRST
        conjunct: the trust tier lands these at 0.28-0.92, not below 0.20
      * `garbage_density >= 0.35` — 0.0
      * `weird_ratio >= 0.75` and the `<= 0.40 / >= 0.40` pair — weird_ratio is
        ~0.28-0.35, under both

    So gating gate 6 behind it is, ON THIS POPULATION, equivalent to deleting
    the rule — not "adding a second witness", because there is no second witness
    here to consult. Asserted on the predicate rather than on a category
    deliberately: it holds before that gate lands and after it, so it stays a
    meaningful lock through the change instead of flipping with the fixtures.
    """
    for row in VOCABULARY_SHORT:
        text, ppl, ls, lang, _exp, note = _row_values(row)
        sig = LC.score_line(
            text_content=text,
            original_text=text,
            original_lang=lang,
            original_lang_score=ls,
            perplexity=ppl,
            known_lang_bases=_KNOWN,
            expected_langs=_EXPECTED,
        )
        assert sig["valid_word_ratio"] == 1.0, (
            f"{text!r}: compute_valid_ratio is shape-only, which is what suppresses the witness"
        )
        assert not _has_strong_garbage_evidence(
            text,
            valid_word_ratio=sig["valid_word_ratio"],
            lang_score=sig["trust_lang_score"],
            orig_lang_score=ls,  # classify_TEXT passes the RAW FastText score here
            gibberish_present=(sig["gibberish"] + sig["weird_wx"]) > 0,
            garbage_density=sig["garbage_density"],
            weird_ratio=sig["word_weird"],
            is_upright_czech=sig["is_upright_czech"],
        ), note


def test_strong_evidence_is_also_false_on_the_garbage_it_should_catch():
    """The other half of the same finding, and the reason #30 is a real trade.

    `oueussd` is genuine OCR garbage and shares every signal with the domain
    vocabulary above, so the predicate misses it too. A gate that lifts
    `malakofauna` therefore lifts `oueussd` with it; the strict xfail on that
    fixture in tests/test_rotation_regression.py is the record of that debt.

    Pinned here so the symmetry is a stated fact rather than something a reader
    has to rediscover from two files.
    """
    text, ppl, ls, _exp, note = _row_values(TRASH_INVERTED[-1])
    assert text == "oueussd", "fixture moved — update this test rather than the assertion"
    sig = LC.score_line(
        text_content=text,
        original_text=text,
        original_lang="ces_Latn",
        original_lang_score=ls,
        perplexity=ppl,
        known_lang_bases=_KNOWN,
        expected_langs=_EXPECTED,
    )
    assert not _has_strong_garbage_evidence(
        text,
        valid_word_ratio=sig["valid_word_ratio"],
        lang_score=sig["trust_lang_score"],
        orig_lang_score=ls,  # classify_TEXT passes the RAW FastText score here
        gibberish_present=(sig["gibberish"] + sig["weird_wx"]) > 0,
        garbage_density=sig["garbage_density"],
        weird_ratio=sig["word_weird"],
        is_upright_czech=sig["is_upright_czech"],
    ), note


def _second_witness(text, ppl, ls, lang):
    """The disjunction the short-line garbage route will evaluate (issue #30).

    Exactly `_has_strong_garbage_evidence(...) or _has_shape_garbage_evidence(...)`,
    on the signal vector `classify_TEXT.score_line` produces for the line. Kept
    here rather than in the module under test because the left operand needs the
    full production vector, which is what this file already builds.
    """
    sig = LC.score_line(
        text_content=text,
        original_text=text,
        original_lang=lang,
        original_lang_score=ls,
        perplexity=ppl,
        known_lang_bases=_KNOWN,
        expected_langs=_EXPECTED,
    )
    strong = _has_strong_garbage_evidence(
        text,
        valid_word_ratio=sig["valid_word_ratio"],
        lang_score=sig["trust_lang_score"],
        orig_lang_score=ls,
        gibberish_present=(sig["gibberish"] + sig["weird_wx"]) > 0,
        garbage_density=sig["garbage_density"],
        weird_ratio=sig["word_weird"],
        is_upright_czech=sig["is_upright_czech"],
    )
    return strong or _has_shape_garbage_evidence(text)


def test_the_disjunction_the_gate_will_evaluate_separates_the_population():
    """Composition test for the issue #30 second witness, on production vectors.

    `_has_shape_garbage_evidence()` is read by the short-line garbage route as a
    second disjunct beside `_has_strong_garbage_evidence()` (gate 6, #30 D15),
    behind `SHORT_GARBAGE_WITNESS_ENABLE` which is still off by default. This
    pins the composed expression on the same signal vectors production computes,
    independently of the flag — so the condition stays tested whether or not the
    call site is currently live. The call site itself is covered by
    tests/test_short_garbage_witness_wiring.py.

    Both halves matter and neither is redundant:

      * the strong predicate is False on ALL of these lines, garbage included
        (test_strong_evidence_is_false_on_the_entire_disputed_population), so
        the disjunction is carried entirely by the shape witness here;
      * the shape witness is False on all the vocabulary, so it cannot recover
        the rule by convicting everything.

    The residue is asserted as part of the contract rather than left out:
    `edelite` is garbage that neither witness reaches, and pretending otherwise
    is what a lexicon-free approach would have to do.
    """
    garbage = [("oueussd", 850.00, 0.9163, "ces_Latn"), ("sektlll", 850.00, 0.60, "ces_Latn")]
    for text, ppl, ls, lang in garbage:
        assert _second_witness(text, ppl, ls, lang), f"{text!r}: garbage must have a second witness"

    for row in VOCABULARY_SHORT:
        text, ppl, ls, lang, _exp, note = _row_values(row)
        assert not _second_witness(text, ppl, ls, lang), f"{note}: vocabulary must not"

    for row in NOTATION_SHORT:
        text, ppl, ls, lang, _exp, note = _row_values(row)
        assert not _second_witness(text, ppl, ls, lang), f"{note}: notation must not"

    assert not _second_witness("edelite", 850.00, 0.60, "ces_Latn"), (
        "the phonotactically legal residue is out of reach by construction — issue #30 D14"
    )
