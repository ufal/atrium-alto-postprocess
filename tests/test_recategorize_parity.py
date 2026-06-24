"""
tests/test_recategorize_parity.py
=================================
Regression net for the unified, constants-parameterised re-scorer (#5).

The offline importance tooling must use the SAME engine as production: the real
``compute_quality_score`` / ``categorize_line`` / ``apply_document_postprocessing``
driven by ``text_util_langID.override_constants`` — never a parallel
re-implementation. The decisive guarantee is *parity*: at the default config the
re-score reproduces the stored ``categ`` on the sample corpus (flip_rate == 0).
If that ever drifts, the surrogate sweep is measuring something other than
production and this test fails loudly.

All pure-Python — the GPU/ML stack is stubbed, exactly like test_calibration.
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

import recategorize_from_csv as R  # noqa: E402

import langID_classify as lc  # noqa: E402
import text_util_langID as tu  # noqa: E402

_SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"

pytestmark = pytest.mark.skipif(
    not _SAMPLE_DIR.is_dir() or not any(_SAMPLE_DIR.glob("*.csv")),
    reason="no DOC_LINE_CATEG sample CSVs present",
)


@pytest.fixture(scope="module")
def corpus():
    return R.load_csvs(_SAMPLE_DIR)


# ── Parity: default constants reproduce the stored categories ───────────────


def test_default_constants_reproduce_stored_categories(corpus):
    """The faithful re-score at the live config must not flip any line."""
    predicted = R.recategorize_dataframe(corpus, None)
    stored = corpus["categ"].map(R.normalize_category).to_numpy()
    got = predicted["categ"].map(R.normalize_category).to_numpy()
    mismatches = [
        (corpus.iloc[i].get("file"), corpus.iloc[i].get("line_num"), stored[i], got[i])
        for i in range(len(stored))
        if stored[i] != got[i]
    ]
    assert not mismatches, f"re-score drifted from production at default config: {mismatches[:10]}"


def test_evaluate_dataframe_baseline_is_zero_flip(corpus):
    metrics = R.evaluate_dataframe(corpus, R.DEFAULT_CONSTANTS)
    assert metrics["flip_rate"] == 0.0
    assert metrics["line_count"] == len(corpus)
    # Every non-empty class present in the data is perfectly recovered.
    for label, f1 in metrics["per_class_f1"].items():
        if metrics["per_class_support"].get(label, 0) > 0:
            assert f1 == pytest.approx(1.0), f"{label} not perfectly recovered at baseline"


def test_evaluate_is_document_aware(corpus):
    """Per-document evaluation runs and is zero-flip per file at default config."""
    per_doc = R.evaluate_per_document(corpus, R.DEFAULT_CONSTANTS)
    assert len(per_doc) == corpus["file"].nunique()
    assert all(d["flip_rate"] == 0.0 for d in per_doc.values())


# ── The override plumbing actually moves categories (and is faithful) ───────


def test_decisive_override_changes_categories(corpus):
    """A hard-sweep override that trashes confident-but-perplexed lines must flip
    a non-trivial share of the scored corpus — proving constants propagate into
    the real determine_category routes."""
    cfg = dict(R.DEFAULT_CONSTANTS)
    cfg["HARD_SWEEP_PPL_MIN"] = 1.0
    cfg["HARD_SWEEP_LANG_MAX"] = 2.0  # every orig_lang_score < 2.0
    metrics = R.evaluate_dataframe(corpus, cfg)
    assert metrics["flip_rate"] > 0.0
    assert metrics["trash_rate"] > R.evaluate_dataframe(corpus, R.DEFAULT_CONSTANTS)["trash_rate"]


# ── override_constants restores globals, even on exception ──────────────────


def test_override_constants_restores_globals():
    before_tu = tu.CATEG_TRASH_SCORE_MAX
    before_lc = lc.CATEG_TRASH_SCORE_MAX  # langID_classify holds its own binding
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
    # config_langID.txt the way a hardcoded table would.
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
