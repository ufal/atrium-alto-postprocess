"""
tests/test_rule_coverage.py
===========================
Tests for the B5 rule-fire coverage instrumentation.

Verifies four properties:

1. **Parity** — with ``RULE_FIRE_COUNTS = None`` (the default) the engine
   output is byte-identical to the pre-instrumentation behaviour.  The
   existing ``test_recategorize_parity.py`` already covers the full-corpus
   parity guarantee; these tests cover the instrumentation itself.

2. **Correct fire registration** — inside ``rule_fire_capture()`` a crafted
   line that is known to trip a specific rule increments exactly that rule's
   counter and leaves all others at zero.

3. **Context-manager stack safety** — nested ``rule_fire_capture()`` calls
   restore the outer context on exit; an exception inside the block does not
   permanently enable counting.

4. **End-to-end smoke** — ``run_coverage`` completes on the sample fixture
   without error and returns a plausible result dict.

All tests are pure-Python; the GPU/ML stack is stubbed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Stub the GPU/ML stack before any production imports.
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
for _p in (str(_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import text_util_langID as tu  # noqa: E402
from text_util_langID import (  # noqa: E402
    _fire,
    override_constants,
    rule_fire_capture,
)

_SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"
_HAS_SAMPLES = _SAMPLE_DIR.is_dir() and any(_SAMPLE_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# 1. Parity — RULE_FIRE_COUNTS = None is the default
# ---------------------------------------------------------------------------


def test_rule_fire_counts_default_is_none():
    """The global sentinel must be None outside a capture block."""
    assert tu.RULE_FIRE_COUNTS is None


def test_fire_noop_outside_capture():
    """_fire() must be a no-op when RULE_FIRE_COUNTS is None."""
    _fire("rule_hard_sweep")
    # No exception, no side-effect.
    assert tu.RULE_FIRE_COUNTS is None


def test_categorize_line_output_unchanged_by_instrumentation():
    """categorize_line() must return the same result with and without a
    capture block active — instrumentation must be transparent."""
    from text_util_langID import categorize_line

    kwargs = dict(
        qs=0.3,
        txt="random gibberish wqx xyz",
        wc=4,
        vowel_ratio=0.1,
        perplexity=5000.0,
        weird_ratio=0.8,
        valid_word_ratio=0.1,
        lang_score=0.2,
        orig_lang_score=0.2,
        gibberish_present=True,
        garbage_density=0.1,
        is_upright_czech=False,
        ghost_dominated=False,
    )

    result_outside = categorize_line(**kwargs)

    with rule_fire_capture():
        result_inside = categorize_line(**kwargs)

    assert result_outside == result_inside, (
        f"categorize_line changed output when inside rule_fire_capture(): {result_outside} vs {result_inside}"
    )


# ---------------------------------------------------------------------------
# 2. Correct fire registration
# ---------------------------------------------------------------------------


def test_fire_increments_counter():
    """_fire() must increment the right key when inside a capture block."""
    with rule_fire_capture() as counts:
        _fire("rule_hard_sweep")
        _fire("rule_hard_sweep")
        _fire("penalty_wqx_rot")

    assert counts["rule_hard_sweep"] == 2
    assert counts["penalty_wqx_rot"] == 1
    assert counts.get("rule_allcaps", 0) == 0


def test_rule_fire_capture_yields_live_dict():
    """The yielded dict is the live RULE_FIRE_COUNTS — mutations inside the
    block are immediately visible through the yielded reference."""
    with rule_fire_capture() as counts:
        assert tu.RULE_FIRE_COUNTS is counts
        _fire("rule_extreme_ppl")
        assert counts["rule_extreme_ppl"] == 1


def test_hard_sweep_fires_for_low_lang_high_ppl():
    """A line with very low lang_score and extreme perplexity should trip
    rule_hard_sweep (the first rule in determine_category)."""
    from text_util_langID import categorize_line

    with rule_fire_capture() as counts:
        categ, _ = categorize_line(
            qs=0.2,
            txt="wqx bqd zze",
            wc=3,
            vowel_ratio=0.05,
            perplexity=99000.0,
            weird_ratio=0.9,
            valid_word_ratio=0.0,
            lang_score=0.1,
            orig_lang_score=0.1,  # < HARD_SWEEP_LANG_MAX (0.45)
            gibberish_present=True,
            garbage_density=0.05,
            is_upright_czech=False,
            ghost_dominated=False,
        )

    assert categ == "Trash"
    # rule_hard_sweep fires first and returns immediately — only it should fire.
    assert counts.get("rule_hard_sweep", 0) == 1
    # No later rules should have fired (short-circuit).
    for rule in (
        "rule_extreme_ppl",
        "rule_absolute_ppl",
        "rule_inverted",
        "rule_allcaps",
        "rule_garbage_density",
    ):
        assert counts.get(rule, 0) == 0, f"{rule} should not fire after rule_hard_sweep"


def test_lowppl_clear_fires_for_low_perplexity():
    """A line with very low perplexity and enough words should trip
    rule_lowppl_clear and be classified Clear."""
    from text_util_langID import categorize_line

    with rule_fire_capture() as counts:
        categ, _ = categorize_line(
            qs=0.85,
            txt="Toto je velmi dobrý český text.",
            wc=6,
            vowel_ratio=0.40,
            perplexity=10.0,  # < LOWPPL_CLEAR_MAX (50.0)
            weird_ratio=0.05,
            valid_word_ratio=0.95,
            lang_score=0.92,
            orig_lang_score=0.92,
            gibberish_present=False,
            garbage_density=0.02,
            is_upright_czech=True,
            ghost_dominated=False,
        )

    assert categ == "Clear"
    assert counts.get("rule_lowppl_clear", 0) == 1


def test_penalty_wqx_rot_fires_and_depresses_qs():
    """The WQX/rotation penalty must fire and lower qs for a suspicious line."""
    from text_util_langID import categorize_line

    # A line with high rot_ratio and low lang_score (not upright Czech)
    # so that penalty_wqx_rot applies.
    # orig_lang_score < 0.75 required; wqx_ratio > 0.10 OR rot_ratio > 0.50.
    with rule_fire_capture() as counts:
        categ_penalised, qs_penalised = categorize_line(
            qs=0.90,  # high starting qs — penalty will drag it down
            txt="wqx bqd mow nuw",  # w/q/x rich, rotatable-heavy
            wc=4,
            vowel_ratio=0.10,
            perplexity=300.0,
            weird_ratio=0.5,
            valid_word_ratio=0.1,
            lang_score=0.30,
            orig_lang_score=0.30,  # < 0.75
            gibberish_present=True,
            garbage_density=0.1,
            is_upright_czech=False,
            ghost_dominated=False,
        )

    assert counts.get("penalty_wqx_rot", 0) == 1, "penalty_wqx_rot should have fired"
    # qs should have been depressed (aligned_score <= original qs)
    assert qs_penalised < 0.90, f"Expected qs < 0.90 after penalty, got {qs_penalised}"


# ---------------------------------------------------------------------------
# 3. Context-manager stack safety
# ---------------------------------------------------------------------------


def test_capture_restores_none_after_exit():
    """RULE_FIRE_COUNTS must return to None after the capture block exits."""
    with rule_fire_capture():
        assert tu.RULE_FIRE_COUNTS is not None
    assert tu.RULE_FIRE_COUNTS is None


def test_nested_capture_restores_outer():
    """Nested rule_fire_capture() calls must stack correctly."""
    with rule_fire_capture() as outer_counts:
        _fire("rule_hard_sweep")
        with rule_fire_capture() as inner_counts:
            _fire("rule_extreme_ppl")
            assert inner_counts.get("rule_extreme_ppl", 0) == 1
            assert inner_counts.get("rule_hard_sweep", 0) == 0
        # After inner exits, outer context is restored.
        assert tu.RULE_FIRE_COUNTS is outer_counts
        _fire("rule_hard_sweep")

    assert outer_counts["rule_hard_sweep"] == 2
    assert outer_counts.get("rule_extreme_ppl", 0) == 0
    assert tu.RULE_FIRE_COUNTS is None


def test_capture_restores_on_exception():
    """An exception inside rule_fire_capture() must still restore RULE_FIRE_COUNTS."""
    assert tu.RULE_FIRE_COUNTS is None
    with pytest.raises(RuntimeError):
        with rule_fire_capture():
            assert tu.RULE_FIRE_COUNTS is not None
            raise RuntimeError("boom")
    assert tu.RULE_FIRE_COUNTS is None


def test_disabled_rules_override_suppresses_fire():
    """When a rule is in DISABLED_RULES (via override_constants), its _fire()
    call is never reached — so no count is registered."""
    from text_util_langID import categorize_line

    with override_constants({"DISABLED_RULES": frozenset(["rule_hard_sweep"])}):
        with rule_fire_capture() as counts:
            categorize_line(
                qs=0.2,
                txt="wqx bqd zze",
                wc=3,
                vowel_ratio=0.05,
                perplexity=99000.0,
                weird_ratio=0.9,
                valid_word_ratio=0.0,
                lang_score=0.1,
                orig_lang_score=0.1,
                gibberish_present=True,
                garbage_density=0.05,
                is_upright_czech=False,
                ghost_dominated=False,
            )

    assert counts.get("rule_hard_sweep", 0) == 0, "rule_hard_sweep should NOT fire when it is in DISABLED_RULES"


# ---------------------------------------------------------------------------
# 4. End-to-end smoke on the fixture corpus
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_smoke():
    """run_coverage must complete without error on the smoke fixture and return
    a dict with all 14 rules, each having the expected keys."""
    from rule_coverage_report import RULES, run_coverage

    results = run_coverage(
        raw_path=str(_SAMPLE_DIR),
        skip_loo=True,  # skip LOO for speed in unit tests
        quiet=True,
    )

    assert set(results.keys()) == set(RULES), f"Unexpected rule keys: {set(results.keys()) ^ set(RULES)}"
    for rule, data in results.items():
        assert "fire_count" in data, f"Missing fire_count for {rule}"
        assert "fire_rate" in data, f"Missing fire_rate for {rule}"
        assert "decisive_count" in data, f"Missing decisive_count for {rule}"
        assert "clear_loss" in data, f"Missing clear_loss for {rule}"
        assert "class" in data, f"Missing class for {rule}"
        assert data["class"] in {"DEAD", "REDUNDANT-HERE", "LOAD-BEARING"}, (
            f"Unexpected class value for {rule}: {data['class']}"
        )
        assert isinstance(data["fire_count"], int)
        assert isinstance(data["fire_rate"], float)
        assert data["fire_rate"] >= 0.0


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_with_loo_smoke():
    """run_coverage with LOO enabled must complete and return non-negative
    decisive_count and clear_loss for every rule."""
    from rule_coverage_report import RULES, run_coverage

    results = run_coverage(
        raw_path=str(_SAMPLE_DIR),
        skip_loo=False,
        quiet=True,
    )

    for rule in RULES:
        assert results[rule]["decisive_count"] >= 0
        assert results[rule]["clear_loss"] >= 0
        # clear_loss can never exceed decisive_count
        assert results[rule]["clear_loss"] <= results[rule]["decisive_count"], (
            f"clear_loss > decisive_count for {rule}: {results[rule]['clear_loss']} > {results[rule]['decisive_count']}"
        )


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_json_output(tmp_path):
    """run_coverage must write valid JSON to the --output path."""
    import json

    from rule_coverage_report import run_coverage

    out_file = tmp_path / "rule_coverage.json"
    run_coverage(
        raw_path=str(_SAMPLE_DIR),
        output_path=str(out_file),
        skip_loo=True,
        quiet=True,
    )

    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert "n_lines" in payload
    assert "n_scored" in payload
    assert "rules" in payload
    assert len(payload["rules"]) == 14
