"""
tests/test_scoring_single_source.py

Locks the two production scoring paths to ONE implementation.

Why this exists
---------------
``classify_TEXT.py`` (the live batch pipeline),
``tools/recategorize_from_csv.py`` (the offline re-scorer) and
``service/text_inference.py`` (the FastAPI ``/process`` endpoint) used to each
carry a hand-maintained copy of the "assemble per-line signals, then
categorise" step.
``tests/test_pipeline_parity.py`` says it plainly: *"parity is maintained by
hand, not by construction"* — the re-scorer only agreed with production because
someone kept patching it to match (its own ``ALIGNMENT FIX`` / ``RESTORE
PARITY`` comments are the scar tissue).

The service copy had drifted furthest: it never applied ``remap_lang``, the
two-tier trust scaling or ``SHORT_PPL_CAP``, and never passed
``orig_lang_score`` at all -- leaving it at the 1.0 default, which silently
disabled ``rule_hard_sweep``, ``rule_extreme_ppl`` and ``rule_wqx_rot``. The
API returned Noisy for lines the pipeline calls Trash.

All three now call ``classify_TEXT.score_line`` / ``row_from_signals``. These
tests hold that line: they fail if a future change reintroduces a second copy,
or edits one path's behaviour without the others.

The round-trip test is the important one. It reproduces what actually happens
in production — score a line, write it to a CSV row, then re-score that row
offline — and asserts the re-scored row is identical to the original. That is
the property offline re-measurement depends on: a diff between a re-scored
corpus and a shipped batch must reflect the change under test, not scoring
drift between the two code paths.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import recategorize_from_csv as R  # noqa: E402

import classify_TEXT as LC  # noqa: E402

# Base codes the collections actually expect/trust, mirroring setup/config.txt.
EXPECTED_LANGS = ["ces"]
KNOWN_BASES = frozenset({"ces", "deu", "eng", "fra", "pol", "ita", "slk"})

# (text, original_lang, orig_lang_score, perplexity)
#
# Chosen to exercise every branch score_line() has to get right: the trust
# tiers (expected / trusted-foreign / unknown), the SHORT_PPL_CAP path at
# wc <= 2 versus uncapped at wc >= 3, rotated and inverted lines, structured
# archaeological lines, and the short diacritic-free lines that rule_short_garbage
# fires on.
CASES = [
    ("v klášteře Strahovském.", "ces_Latn", 1.0000, 119.50),
    ("republiky československé", "ces_Latn", 0.9136, 15.88),
    ("Opomenulé nebo opozděné ohlášení trestú se pokutou peněžitou", "ces_Latn", 1.0000, 153.00),
    ("oueussd", "isl_Latn", 0.9163, 4820.00),
    ("noywqued noqnsoa es yasoq yuasvyo quqpzodo oqou onuauodo", "ces_Latn", 0.7500, 856.00),
    ("malakofauna", "isl_Latn", 0.9163, 1210.00),
    ("Equus caballus", "lat_Latn", 0.8800, 640.00),
    ("Occipitale", "ita_Latn", 0.7700, 990.00),
    ("hr. komponenta 4", "ces_Latn", 0.6200, 780.00),
    ("Maxilla+dentes", "deu_Latn", 0.5400, 1450.00),
    ("rozm. 12 x 4 cm", "ces_Latn", 0.8100, 410.00),
    ("Alyrý cvod nede % Agrgr oAOrt", "slk_Latn", 0.6658, 15168.00),
    ("2742/2%", "ces_Latn", 0.3000, 2100.00),
    ("Ji r", "vie_Latn", 0.4100, 3300.00),
    ("' ' \" k4že /olonbka,\"3 Ege 94%", "ces_Latn", 0.2013, 1648.00),
]


def _live_row(text, lang, lang_score, ppl, *, line_num=1):
    """Exactly what the live pipeline builds for one scored line."""
    sig = LC.score_line(
        text_content=text,
        original_text=text,
        original_lang=lang,
        original_lang_score=lang_score,
        perplexity=ppl,
        known_lang_bases=KNOWN_BASES,
        expected_langs=EXPECTED_LANGS,
    )
    return LC.row_from_signals(
        sig,
        file_id="CTX000000001",
        page_id=1,
        line_num=line_num,
        text_content=text,
        original_text=text,
        split_ws="",
        split_we="",
        original_lang=lang,
        original_lang_score=lang_score,
    )


@pytest.mark.parametrize("text, lang, lang_score, ppl", CASES)
def test_offline_rescore_reproduces_live_row_exactly(text, lang, lang_score, ppl):
    """live scoring -> CSV row -> offline re-score must be a fixed point.

    This is the property ``tools/recategorize_from_csv.py`` exists to provide:
    re-scoring a stored batch with unchanged constants must return that batch,
    so any diff observed while evaluating a proposed change is attributable to
    the change alone.
    """
    live = _live_row(text, lang, lang_score, ppl)
    rescored = R._rescore_row(dict(live), EXPECTED_LANGS, KNOWN_BASES)

    for column in LC.CSV_HEADER:
        assert rescored[column] == live[column], (
            f"offline re-score drifted from the live pipeline on {column!r} "
            f"for {text!r}: live={live[column]!r} rescored={rescored[column]!r}"
        )


@pytest.mark.parametrize("text, lang, lang_score, ppl", CASES)
def test_rescore_is_idempotent(text, lang, lang_score, ppl):
    """Re-scoring twice changes nothing — no slow drift over repeated passes."""
    once = R._rescore_row(dict(_live_row(text, lang, lang_score, ppl)), EXPECTED_LANGS, KNOWN_BASES)
    twice = R._rescore_row(dict(once), EXPECTED_LANGS, KNOWN_BASES)

    for column in LC.CSV_HEADER:
        assert twice[column] == once[column], f"re-scoring is not idempotent on {column!r} for {text!r}"


def test_both_paths_call_the_same_scorer():
    """The re-scorer must delegate, not carry its own copy of the signal block.

    Guards against the failure mode this module was written for: someone
    "fixes" the offline path by pasting the live pipeline's signal assembly
    back into ``_rescore_row`` and the two drift apart again.
    """
    assert R.score_line is LC.score_line
    assert R.row_from_signals is LC.row_from_signals

    source = Path(R.__file__).read_text()
    body = source[source.index("def _rescore_row") : source.index("def _coerce_locators")]

    assert "score_line(" in body, "_rescore_row no longer delegates to the shared scorer"
    for duplicated in ("compute_quality_score(", "categorize_line(", "TRUST_TIER_UNKNOWN"):
        assert duplicated not in body, (
            f"_rescore_row has re-grown its own copy of {duplicated!r}; "
            "it must go through classify_TEXT.score_line instead"
        )


def test_trust_tier_is_applied_not_the_remap_cap():
    """The scorer must see the two-tier trust score, not the stored remap cap.

    These are different numbers and confusing them is the single easiest way to
    silently de-calibrate the structural guards: an unknown-language line is
    handed ``orig * TRUST_TIER_UNKNOWN``, while the ``lang_score`` recorded in
    the CSV is the ``remap_lang`` cap. Both are returned by ``score_line`` and
    they must stay distinct.
    """
    sig = LC.score_line(
        text_content="malakofauna",
        original_text="malakofauna",
        original_lang="isl_Latn",
        original_lang_score=0.9163,
        perplexity=1210.0,
        known_lang_bases=KNOWN_BASES,
        expected_langs=EXPECTED_LANGS,
    )

    assert sig["trust_lang_score"] == pytest.approx(0.9163 * LC.TRUST_TIER_UNKNOWN)
    # The stored score is the remap cap for a Latin-script unknown language.
    from text_util import LANG_SCORE_REMAP

    assert sig["lang_score"] == pytest.approx(LANG_SCORE_REMAP)
    assert sig["trust_lang_score"] != pytest.approx(sig["lang_score"])

    # An expected-language line keeps its raw score in both.
    czech = LC.score_line(
        text_content="republiky československé",
        original_text="republiky československé",
        original_lang="ces_Latn",
        original_lang_score=0.9136,
        perplexity=15.88,
        known_lang_bases=KNOWN_BASES,
        expected_langs=EXPECTED_LANGS,
    )
    assert czech["trust_lang_score"] == pytest.approx(0.9136)


def test_density_and_vowels_ride_the_pre_repair_text():
    """``garbage_density`` / ``vowel_ratio`` must come from ``original_text``.

    Every other signal is computed on the cleaned ``text_content``. Getting this
    backwards lets line-repair hide exactly the noise those two signals exist to
    measure, and it is invisible whenever the two texts happen to be equal —
    which is most of the corpus, so a test has to force them apart.
    """
    dirty = "kontext ###@@@!!!"
    clean = "kontext"

    sig = LC.score_line(
        text_content=clean,
        original_text=dirty,
        original_lang="ces_Latn",
        original_lang_score=0.8,
        perplexity=400.0,
        known_lang_bases=KNOWN_BASES,
        expected_langs=EXPECTED_LANGS,
    )

    from text_util import compute_garbage_density, compute_vowel_ratio

    assert sig["garbage_density"] == pytest.approx(compute_garbage_density(dirty))
    assert sig["vowel_ratio"] == pytest.approx(compute_vowel_ratio(dirty))
    # ...while word_count follows the cleaned line.
    assert sig["word_count"] == 1


def test_short_ppl_cap_applies_to_one_and_two_token_lines_only():
    """``SHORT_PPL_CAP`` must be applied identically by whoever calls the scorer.

    Both paths used to apply this themselves; it now lives in one place. The
    boundary is ``wc <= 2``, and the capped value is what gets both scored and
    stored.
    """
    kw = dict(
        original_lang="isl_Latn",
        original_lang_score=0.9,
        perplexity=9999.0,
        known_lang_bases=KNOWN_BASES,
        expected_langs=EXPECTED_LANGS,
    )

    one = LC.score_line(text_content="oueussd", original_text="oueussd", **kw)
    two = LC.score_line(text_content="oueussd nupoy", original_text="oueussd nupoy", **kw)
    three = LC.score_line(text_content="oueussd nupoy yoysqu", original_text="oueussd nupoy yoysqu", **kw)

    assert one["perplex"] == LC.SHORT_PPL_CAP
    assert two["perplex"] == LC.SHORT_PPL_CAP
    assert three["perplex"] == 9999.0


# ---------------------------------------------------------------------------
# The API endpoint is the third consumer and must not drift either.
# ---------------------------------------------------------------------------


def _mock_ft(lang: str, score: float):
    from unittest.mock import MagicMock

    ft = MagicMock()
    ft.predict.return_value = ([[f"__label__{lang}"]], [[score]])
    return ft


@pytest.mark.parametrize("text, lang, lang_score, ppl", CASES)
def test_service_endpoint_agrees_with_the_pipeline(text, lang, lang_score, ppl):
    """``/process`` must return the category the batch pipeline would assign.

    The service resolves its own language configuration, so this also pins that
    it resolves the *same* one the pipeline uses.
    """
    pytest.importorskip("lxml")
    from service.text_inference import _classify_line, _lang_config

    expected_langs, known_bases = _lang_config()
    api = _classify_line(
        text,
        ppl,
        ft_model=_mock_ft(lang, lang_score),
        ppl_model=None,
        tokenizer=None,
        device="cpu",
    )
    sig = LC.score_line(
        text_content=text,
        original_text=text,
        original_lang=lang,
        original_lang_score=lang_score,
        perplexity=ppl,
        known_lang_bases=known_bases,
        expected_langs=expected_langs,
    )

    assert api["category"] == sig["categ"], f"service/pipeline category drift on {text!r}"
    assert api["quality_score"] == pytest.approx(round(sig["quality_score"], 4))
    assert api["perplexity"] == pytest.approx(round(sig["perplex"], 2))


def test_service_reaches_the_lang_gated_trash_routes():
    """Regression: ``orig_lang_score`` must reach ``categorize_line``.

    The service used to omit it, so the argument sat at its 1.0 default and
    three Trash routes keyed on low language confidence could never fire from
    the API. This line is confidently garbage with an unconfident language
    prediction; the hard sweep must catch it.
    """
    pytest.importorskip("lxml")
    from service.text_inference import _classify_line

    out = _classify_line(
        "kfjs qmwx zzpl vvbn",
        5200.0,
        ft_model=_mock_ft("vie", 0.30),
        ppl_model=None,
        tokenizer=None,
        device="cpu",
    )
    assert out["category"] == "Trash"
    assert out["orig_lang_score"] == pytest.approx(0.30)


def test_service_applies_the_short_ppl_cap():
    """``SHORT_PPL_CAP`` was never applied by the service; it must be now."""
    pytest.importorskip("lxml")
    from service.text_inference import _classify_line

    out = _classify_line(
        "oueussd",
        9999.0,
        ft_model=_mock_ft("isl", 0.92),
        ppl_model=None,
        tokenizer=None,
        device="cpu",
    )
    assert out["perplexity"] == pytest.approx(LC.SHORT_PPL_CAP)


def test_service_does_not_carry_its_own_signal_block():
    """Guard against the service re-growing a private copy of the scorer."""
    import service.text_inference as SVC

    assert SVC.score_line is LC.score_line

    source = Path(SVC.__file__).read_text()
    body = source[source.index("def _classify_line(") : source.index("# Module-level singleton")]

    assert "score_line(" in body, "_classify_line no longer delegates to the shared scorer"
    for duplicated in ("compute_quality_score(", "categorize_line(", "analyze_rotation_signals("):
        assert duplicated not in body, (
            f"_classify_line has re-grown its own copy of {duplicated!r}; "
            "it must go through classify_TEXT.score_line instead"
        )
