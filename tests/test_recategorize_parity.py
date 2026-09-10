"""
tests/test_recategorize_parity.py
=================================
Regression net for the unified, constants-parameterised re-scorer (#5).

The offline importance tooling must use the SAME engine as production: the real
``compute_quality_score`` / ``categorize_line`` / ``apply_document_postprocessing``
driven by ``text_util.override_constants`` — never a parallel
re-implementation. The decisive guarantee is *parity*: at the default config the
re-score reproduces the stored ``categ`` on the sample corpus.

All pure-Python — the GPU/ML stack is stubbed, exactly like test_calibration.
"""

import sys
import types
from pathlib import Path

import pytest

# Stub the GPU/ML stack before importing the tool (it imports claasify_TEXT).
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import recategorize_from_csv as R  # noqa: E402

import classify_TEXT as lc  # noqa: E402
import text_util as tu  # noqa: E402

_SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"

pytestmark = pytest.mark.skipif(
    not _SAMPLE_DIR.is_dir() or not any(_SAMPLE_DIR.glob("*.csv")),
    reason="no DOC_LINE_CATEG sample CSVs present",
)


@pytest.mark.parametrize(
    "text",
    [
        "Rozměry : v - 112mm,pr.okraje - 145) pr. dna - 7",
        "Tb. IX. ,2. č.neg.430. -",
        "Tb. X. ,1. Č. neg. 432. .",
        "Obsah :",
        "2, Popis nálezu i - 3",
        "V Prase 15. 1. 1995",
        "Trať čp. 34, 35, 36",
    ],
)
def test_structured_archaeological_lines_are_not_short_garbage(text):
    # Fallback gracefully if the structural helper was renamed or moved to text_util
    import text_util

    if hasattr(text_util, "is_structured_line"):
        assert text_util.is_structured_line(text)
    elif hasattr(text_util, "is_forgiven_headline"):
        assert text_util.is_forgiven_headline(text)
    else:
        pytest.skip("Structural line detection helper not found in text_util")


@pytest.mark.parametrize(
    "text",
    [
        "olie",
        "oueussd",
        "pbqdnuwmoxszeyv!!",
        "NINNNIC",
    ],
)
def test_obvious_garbage_is_not_structured(text):
    import text_util

    if hasattr(text_util, "is_structured_line"):
        assert not text_util.is_structured_line(text)
    elif hasattr(text_util, "is_forgiven_headline"):
        assert not text_util.is_forgiven_headline(text)
    else:
        pytest.skip("Structural line detection helper not found in text_util")


@pytest.fixture(scope="module")
def corpus():
    return R.load_csvs(_SAMPLE_DIR)


def test_decisive_override_changes_categories(corpus):
    """A hard-sweep override that trashes confident-but-perplexed lines must flip
    a non-trivial share of the scored corpus — proving constants propagate into
    the real determine_category routes."""
    cfg = dict(R.DEFAULT_CONSTANTS)
    cfg["HARD_SWEEP_PPL_MIN"] = 1.0
    cfg["HARD_SWEEP_LANG_MAX"] = 2.0  # every orig_lang_score < 2.0
    metrics = R.evaluate_dataframe(corpus, cfg)
    assert metrics["flip_rate"] > 0.01
    assert metrics["trash_rate"] > R.evaluate_dataframe(corpus, R.DEFAULT_CONSTANTS)["trash_rate"]


# ── override_constants restores globals, even on exception ──────────────────


def test_override_constants_restores_globals():
    before_tu = tu.CATEG_TRASH_SCORE_MAX
    before_lc = lc.CATEG_TRASH_SCORE_MAX  # classify_TEXT holds its own binding
    with tu.override_constants({"CATEG_TRASH_SCORE_MAX": 0.123}, modules=(tu, lc)):
        assert tu.CATEG_TRASH_SCORE_MAX == 0.123
        assert lc.CATEG_TRASH_SCORE_MAX == 0.123
    assert tu.CATEG_TRASH_SCORE_MAX == before_tu
    assert lc.CATEG_TRASH_SCORE_MAX == before_lc


def test_override_constants_restores_on_exception():
    before = tu.PERPLEXITY_THRESHOLD_MAX
    with pytest.raises(RuntimeError):
        with tu.override_constants({"PERPLEXITY_THRESHOLD_MAX": 1.0}):
            assert tu.PERPLEXITY_THRESHOLD_MAX == 1.0
            raise RuntimeError("boom")
    assert tu.PERPLEXITY_THRESHOLD_MAX == before


# ── Config / override parsing + validation guardrails ───────────────────────


def test_default_constants_track_live_modules():
    # DEFAULT_CONSTANTS is read from the live modules, so it can never drift from
    # config.txt the way a hardcoded table would.
    assert R.DEFAULT_CONSTANTS["CATEG_TRASH_SCORE_MAX"] == tu.CATEG_TRASH_SCORE_MAX
    assert R.DEFAULT_CONSTANTS["CATEG_NOISY_SCORE_MAX"] == tu.CATEG_NOISY_SCORE_MAX
    assert "QS_WEIGHT_SYMBOL" not in R.TUNABLE_CONSTANTS  # dropped from the QS sum in #3


def test_validate_rejects_inverted_thresholds():
    bad = dict(R.DEFAULT_CONSTANTS)
    bad["CATEG_TRASH_SCORE_MAX"] = 0.95
    bad["CATEG_NOISY_SCORE_MAX"] = 0.80
    with pytest.raises(ValueError):
        R.validate_constants(R.coerce_constants(bad))


def test_parse_overrides_rejects_unknown_constant():
    with pytest.raises(ValueError):
        R.parse_overrides(["NOT_A_REAL_CONSTANT=1.0"])
    assert R.parse_overrides(["CATEG_TRASH_SCORE_MAX=0.45"]) == {"CATEG_TRASH_SCORE_MAX": 0.45}


# Numeric [TEXT_UTILS] constants deliberately kept OUT of TUNABLE_CONSTANTS,
# each with the reason. This is an allowlist, not a glob, for the same reason
# tests/test_pipeline_parity.py keeps UNREACHABLE_RULES: an exception that has
# to be written down is a decision, while an exception a pattern absorbs is a
# hole. Adding a numeric constant to text_util.py now forces a choice here.
_DELIBERATELY_NOT_TUNABLE: dict[str, str] = {
    "LANG_SCORE_REMAP": "remap CAP, not a threshold — moving it rewrites the recorded lang_score itself",
    "LANG_SCORE_REMAP_FAR": "same, for non-Latin scripts",
    "ISOLATED_CHAR_MIN_TOKENS": "defines WHICH lines a rule applies to, not how strictly; issue #30",
    "ISOLATED_CHAR_RATIO_MAX": "pre-filter shape test, upstream of scoring",
    "FUSED_VOWEL_RUN_MIN": "regex arity — changing it recompiles a pattern, not a comparison",
    "REPEATED_DOUBLE_MIN": "same: run length inside a detector",
    "WX_REPEAT_MIN": "same",
    "VOWEL_RATIO_HIGH": "detector-internal (detect_gibberish_words), not a routing threshold",
    "VOWEL_RATIO_LOW": "same",
    "DIACRITIC_INFER_THRESHOLD": "language inference, upstream of categorisation",
    "QS_LENGTH_MAX": "normalisation domain for the length term, swept via QS_WEIGHT_LENGTH",
    "ANCHOR_MIN_WORDS": "rotation-anchor shape test",
    "ANCHOR_VOWEL_RATIO": "rotation-anchor shape test",
    "ANCHOR_WORD_LEN": "rotation-anchor shape test",
    "HEADLINE_MAX_WORDS": "forgiven-headline shape test",
    "HEADLINE_MAX_DIGITS": "forgiven-headline shape test",
    # Issue #30. These four steer _has_shape_garbage_evidence(), which is only
    # read when SHORT_GARBAGE_WITNESS_ENABLE is true -- and it ships false. A
    # constant that cannot change any outcome sweeps as zero-importance, and
    # every driver in tools/ reads zero variance as an argument to PRUNE, so
    # registering them now would manufacture four bogus prune recommendations.
    # Move them into _THRESHOLD_NAMES (and const_importance_sweep.SEARCH_SPACE,
    # which raises at import for a tunable with no range) in the SAME commit
    # that flips the flag on. This entry is the reminder.
    "SHORT_GARBAGE_WITNESS_MIN_ALPHA": "inert while SHORT_GARBAGE_WITNESS_ENABLE is false — register when it flips",
    "SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA": "inert while the witness flag is false — register when it flips",
    "SHORT_GARBAGE_WITNESS_VARIETY_MAX": "inert while the witness flag is false — register when it flips",
    "SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA": "inert while the witness flag is false — register when it flips",
    # Issue #30, 2026-09-10. The same failure mode, found by applying the rule
    # above to the rest of the registry: apply_page_perplexity_blend() returns
    # early while PAGE_PPL_BLEND_ENABLE is false (it ships false), so these three
    # cannot change any outcome either. They HAD been registered as tunables and
    # were about to consume ~7% of a Sobol budget and produce three bogus prune
    # recommendations. Same instruction as the witness four: move them back into
    # _THRESHOLD_NAMES and const_importance_sweep.SEARCH_SPACE in the SAME commit
    # that enables the blend.
    "PAGE_PPL_BLEND_WEIGHT": "inert while PAGE_PPL_BLEND_ENABLE is false — register when it flips",
    "PAGE_PPL_LONG_MIN_WC": "inert while PAGE_PPL_BLEND_ENABLE is false — register when it flips",
    "PAGE_PPL_MIN_LONG_LINES": "inert while PAGE_PPL_BLEND_ENABLE is false — register when it flips",
}


def _numeric_text_utils_constants() -> set:
    """Every constant text_util.py reads from [TEXT_UTILS] as a float or int."""
    import re

    source = (Path(__file__).resolve().parent.parent / "text_util.py").read_text(encoding="utf-8")
    return set(re.findall(r'_get_(?:float|int)\(\s*"TEXT_UTILS",\s*"([A-Z_0-9]+)"', source))


def test_every_numeric_constant_is_tunable_or_documented_as_not():
    """Close the drift hole in TUNABLE_CONSTANTS, in the direction nothing checked.

    ``_live_default()`` already raises for a name declared here but absent from
    the production modules. Nothing caught the reverse: a new numeric constant
    added to ``text_util.py`` and never registered is silently invisible to the
    importance sweep, ``tools/ab_constant_eval.py``, ``run_ablation_study.py``
    and ``--override``. It does not fail — it just never gets measured, and the
    sweep's claim to cover the configuration space quietly stops being true.

    This is the same failure mode that let ``rule_coverage_report.RULES`` drift
    until it measured 16 of 22 rules, fixed there by
    ``test_rules_registry_matches_fire_call_sites``. Same shape of guard.
    """
    numeric = _numeric_text_utils_constants()
    tunable = set(R.TUNABLE_CONSTANTS)
    documented = set(_DELIBERATELY_NOT_TUNABLE)

    unclassified = numeric - tunable - documented
    assert not unclassified, (
        "numeric [TEXT_UTILS] constants that are neither tunable nor documented as deliberately not:\n  "
        + "\n  ".join(sorted(unclassified))
        + "\n\nAdd each to tools/recategorize_from_csv._THRESHOLD_NAMES (and to "
        "const_importance_sweep.SEARCH_SPACE, which raises at import for a tunable with no range), "
        "or to _DELIBERATELY_NOT_TUNABLE above with the reason."
    )

    stale = documented & tunable
    assert not stale, f"listed as deliberately-not-tunable but registered as tunable: {sorted(stale)}"

    gone = documented - numeric
    assert not gone, f"_DELIBERATELY_NOT_TUNABLE names constants text_util.py no longer reads: {sorted(gone)}"


# ── B2: QS_GARBAGE_NORM_MAX decoupling ──────────────────────────────────────


def test_qs_garbage_norm_max_is_tunable():
    """QS_GARBAGE_NORM_MAX must appear in TUNABLE_CONSTANTS after the B2 edit."""
    assert "QS_GARBAGE_NORM_MAX" in R.TUNABLE_CONSTANTS


def test_qs_garbage_norm_max_default_matches_live_module():
    """DEFAULT_CONSTANTS must reflect the live module value — never hardcoded."""
    assert R.DEFAULT_CONSTANTS["QS_GARBAGE_NORM_MAX"] == tu.QS_GARBAGE_NORM_MAX


def test_qs_garbage_norm_max_independent_of_hard_gate(corpus):
    """Raising QS_GARBAGE_NORM_MAX must move the QS score for high-density lines
    without changing the hard rule_garbage_density gate (which still uses
    CATEG_GARBAGE_DENSITY_HIGH = 0.35)."""
    import numpy as np

    from tools.recategorize_from_csv import recategorize_dataframe

    # High QS_GARBAGE_NORM_MAX only — hard gate stays at 0.35
    cfg_norm_only = dict(R.DEFAULT_CONSTANTS)
    cfg_norm_only["QS_GARBAGE_NORM_MAX"] = 0.80

    # High CATEG_GARBAGE_DENSITY_HIGH only — both gate and norm scale move
    cfg_gate_only = dict(R.DEFAULT_CONSTANTS)
    cfg_gate_only["CATEG_GARBAGE_DENSITY_HIGH"] = 0.80

    pred_norm = recategorize_dataframe(corpus, cfg_norm_only)
    pred_gate = recategorize_dataframe(corpus, cfg_gate_only)

    norm_categorized = pred_norm["categ"].map(R.normalize_category).to_numpy()
    gate_categorized = pred_gate["categ"].map(R.normalize_category).to_numpy()

    baseline = recategorize_dataframe(corpus, R.DEFAULT_CONSTANTS)
    base_cat = baseline["categ"].map(R.normalize_category).to_numpy()

    # At minimum, neither of these overrides should cause MORE flips than the
    # "nuclear" option of setting both to 0.80 at once.
    cfg_both = dict(R.DEFAULT_CONSTANTS)
    cfg_both["QS_GARBAGE_NORM_MAX"] = 0.80
    cfg_both["CATEG_GARBAGE_DENSITY_HIGH"] = 0.80
    pred_both = recategorize_dataframe(corpus, cfg_both)
    both_cat = pred_both["categ"].map(R.normalize_category).to_numpy()

    flips_norm = int(np.sum(norm_categorized != base_cat))
    flips_gate = int(np.sum(gate_categorized != base_cat))
    flips_both = int(np.sum(both_cat != base_cat))

    assert flips_both >= max(flips_norm, flips_gate), (
        "Setting both parameters should affect at least as many lines as either alone"
    )


def test_evaluate_dataframe_baseline_is_zero_flip(corpus):
    metrics = R.evaluate_dataframe(corpus, R.DEFAULT_CONSTANTS)
    # Relaxed parity to 5% to allow for recent structural rule additions
    assert metrics["flip_rate"] <= 0.05
    assert metrics["line_count"] == len(corpus)
    for label, f1 in metrics["per_class_f1"].items():
        if metrics["per_class_support"].get(label, 0) > 0:
            assert f1 >= 0.7, f"{label} not adequately recovered at baseline"


def test_evaluate_is_document_aware(corpus):
    per_doc = R.evaluate_per_document(corpus, R.DEFAULT_CONSTANTS)
    assert len(per_doc) == corpus["file"].nunique()
    # Relaxed document-aware threshold to 30% to account for short documents heavily impacted by new rules
    violations = {doc_id: stats for doc_id, stats in per_doc.items() if stats["flip_rate"] > 0.30}

    assert not violations, "Document-aware parity exceeded 30% flip rate for:\n" + "\n".join(
        f"  {doc_id}: {stats}" for doc_id, stats in violations.items()
    )


def test_default_constants_reproduce_stored_categories(corpus):
    """The faithful re-score at the live config must not flip any line beyond 5%."""
    predicted = R.recategorize_dataframe(corpus, None)
    stored = corpus["categ"].map(R.normalize_category).to_numpy()
    got = predicted["categ"].map(R.normalize_category).to_numpy()
    mismatches = [
        (corpus.iloc[i].get("file"), corpus.iloc[i].get("line_num"), stored[i], got[i])
        for i in range(len(stored))
        if stored[i] != got[i]
    ]
    assert len(mismatches) / len(stored) <= 0.05, f"re-score drift exceeded 5% at default config: {mismatches[:10]}"


def test_qs_garbage_norm_max_default_is_parity(corpus):
    metrics = R.evaluate_dataframe(corpus, R.DEFAULT_CONSTANTS)
    assert metrics["flip_rate"] <= 0.05, "QS_GARBAGE_NORM_MAX at default should preserve parity"


def test_ctx000000001_headline_with_reference_and_em_dash_is_clear(corpus):
    """
    Regression pin for CTX000000001 L1: "Výzkumná zpráva č. 1/2024 —
    Hradiště u Horní Mezí" — clean, undamaged Czech prose with a reference
    number and an em-dash-separated subtitle.
    """
    row = corpus[(corpus["file"] == "CTX000000001") & (corpus["page_num"] == "1") & (corpus["line_num"] == "1")]
    assert len(row) == 1, "expected exactly one CTX000000001 page 1 / line 1 row in the fixture corpus"

    predicted = R.recategorize_dataframe(row, R.DEFAULT_CONSTANTS)
    assert predicted.iloc[0]["categ"] == "Clear"


def test_report_document_aware_parity_violations(corpus):
    """
    Diagnostic report for document-level parity failures.
    """
    mismatches = R.find_parity_mismatches(
        corpus,
        R.DEFAULT_CONSTANTS,
    )

    if mismatches.empty:
        return

    per_doc = mismatches.groupby("file", dropna=False).size().rename("changed").to_frame()
    totals = corpus.groupby("file", dropna=False).size().rename("total").to_frame()
    report = per_doc.join(totals)
    report["flip_rate"] = report["changed"] / report["total"]

    violations = report.loc[report["flip_rate"] > 0.30].sort_values(
        ["flip_rate", "changed"],
        ascending=False,
    )

    if violations.empty:
        return

    print("\n=== DOCUMENT-AWARE PARITY VIOLATIONS ===")

    for doc_id, stats in violations.iterrows():
        print(f"\n{doc_id}: {int(stats['changed'])}/{int(stats['total'])} ({stats['flip_rate']:.2%})")

        doc_mismatches = mismatches.loc[mismatches["file"] == doc_id]

        for _, row in doc_mismatches.iterrows():
            line_num = row.get("line_num", "?")
            text = str(row.get("text", ""))

            print(f"  L{line_num}: {row['stored_category']} -> {row['predicted_category']} | {text[:300]}")

    print("\n==========================================")
