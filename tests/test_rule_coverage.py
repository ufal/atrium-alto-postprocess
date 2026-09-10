"""
tests/test_rule_coverage.py
===========================
Tests for the B5 rule-fire coverage instrumentation.

Verifies four properties:

1. **Parity** — with ``RULE_FIRE_COUNTS = None`` (the default) the engine
   output is byte-identical to the pre-instrumentation behaviour.
2. **Correct fire registration** — inside ``rule_fire_capture()`` a crafted
   line increments exactly that rule's counter.
3. **Context-manager stack safety** — nested calls stack properly.
4. **End-to-end smoke** — ``run_coverage`` completes on the sample fixture.

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

import text_util as tu  # noqa: E402
from text_util import (  # noqa: E402
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
    assert tu.RULE_FIRE_COUNTS is None


def test_categorize_line_output_unchanged_by_instrumentation():
    """categorize_line() must return the same result with and without a
    capture block active — instrumentation must be transparent."""
    from text_util import categorize_line

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
        _fire("rule_wqx_rot")

    assert counts["rule_hard_sweep"] == 2
    assert counts["rule_wqx_rot"] == 1
    assert counts.get("rule_allcaps", 0) == 0


def test_rule_fire_capture_yields_live_dict():
    """The yielded dict is the live RULE_FIRE_COUNTS."""
    with rule_fire_capture() as counts:
        assert tu.RULE_FIRE_COUNTS is counts
        _fire("rule_extreme_ppl")
        assert counts["rule_extreme_ppl"] == 1


def test_hard_sweep_fires_for_low_lang_high_ppl():
    """A line with very low lang_score and extreme perplexity should trip
    rule_hard_sweep (the first rule in determine_category)."""
    from text_util import categorize_line

    with rule_fire_capture() as counts:
        categ, _ = categorize_line(
            qs=0.2,
            txt="klm klm klm",
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

    assert categ == "Trash"
    assert counts.get("rule_hard_sweep", 0) == 1

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
    from text_util import categorize_line

    with rule_fire_capture() as counts:
        categ, _ = categorize_line(
            qs=0.85,
            txt="Toto je velmi dobrý český text.",
            wc=6,
            vowel_ratio=0.40,
            perplexity=10.0,
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
    """When a rule is in DISABLED_RULES, its _fire() call is never reached."""
    from text_util import categorize_line

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
    a dict with all registered rules."""
    from rule_coverage_report import RULES, run_coverage

    results = run_coverage(
        raw_path=str(_SAMPLE_DIR),
        skip_loo=True,
        quiet=True,
    )

    assert set(results.keys()) == set(RULES), f"Unexpected rule keys: {set(results.keys()) ^ set(RULES)}"
    for _rule, data in results.items():
        assert "fire_count" in data
        assert "fire_rate" in data
        assert "decisive_count" in data
        assert "clear_loss" in data
        assert "class" in data
        assert data["class"] in {"DEAD", "REDUNDANT-HERE", "LOAD-BEARING"}
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
        assert results[rule]["clear_loss"] <= results[rule]["decisive_count"]


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_json_output(tmp_path):
    """run_coverage must write valid JSON to the --output path."""
    import json

    from rule_coverage_report import RULES, run_coverage

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
    # Dynamically match RULES length so test survives when new rules are added.
    assert len(payload["rules"]) == len(RULES)


# ---------------------------------------------------------------------------
# 5. Registry / call-site parity
# ---------------------------------------------------------------------------


def test_rules_registry_matches_fire_call_sites():
    """rule_coverage_report.RULES must equal the _fire() call-sites in text_util.

    This used to be a "keep in sync" comment with nothing enforcing it, and it
    drifted: the five rules introduced by the issue #30 work (rule_short_line,
    rule_damaged_token, rule_reference_floor, rule_bigram_run,
    rule_fragment_tokens) never made it into the registry, so every coverage
    report and every ablation sweep run after PR #32 quietly measured 16 of the
    21 rules. A stale registry does not fail loudly — it just under-reports,
    which is why this needs a test rather than a comment.
    """
    import re

    from rule_coverage_report import RULES

    source = (_ROOT / "text_util.py").read_text(encoding="utf-8")
    call_sites = set(re.findall(r'_fire\("([a-z_]+)"\)', source))

    declared = set(RULES)
    assert declared == call_sites, (
        f"registry out of sync with text_util.py\n"
        f"  fired but not declared: {sorted(call_sites - declared)}\n"
        f"  declared but never fired: {sorted(declared - call_sites)}"
    )
    assert len(RULES) == len(declared), "RULES contains duplicates"


def test_ablation_rule_lists_are_subsets_of_the_registry():
    """The ablation drivers must name rules that actually exist.

    ``override_constants({"DISABLED_RULES": frozenset([name])})`` matches by
    string. A name no ``_fire()`` site produces therefore disables nothing, and
    the driver reports the resulting zero flips / zero clear-loss as
    "**PRUNE** (Signal variance ~ 0)" -- an argument to delete a rule that was
    never switched off.

    That is not hypothetical. Four entries in both lists were still spelled
    ``penalty_*`` from before those gates were renamed to ``rule_*``, so every
    ablation report since carried four such rows, and
    ``rule_coverage_report._PENALTY_RULES`` (which partitioned on the same
    prefix) had been the empty list the whole time. Neither list was covered by
    a test -- only ``rule_coverage_report.RULES`` was, by the test above.

    Asserted as a subset rather than as equality: choosing to ablate a subset is
    a legitimate decision (a cheaper sweep), while naming a rule that does not
    exist never is.
    """
    from greedy_backward_elimination import CANDIDATE_RULES
    from rule_coverage_report import RULES
    from run_ablation_study import RULES_TO_ABLATE

    registry = set(RULES)
    assert set(RULES_TO_ABLATE) <= registry, (
        f"run_ablation_study.RULES_TO_ABLATE names rules that never fire: {sorted(set(RULES_TO_ABLATE) - registry)}"
    )
    assert set(CANDIDATE_RULES) <= registry, (
        f"greedy_backward_elimination.CANDIDATE_RULES names rules that never fire: "
        f"{sorted(set(CANDIDATE_RULES) - registry)}"
    )
    assert len(RULES_TO_ABLATE) == len(set(RULES_TO_ABLATE)), "RULES_TO_ABLATE contains duplicates"


# ---------------------------------------------------------------------------
# (#30 B2) Word-count attribution of rule fires.
# ---------------------------------------------------------------------------


def test_wc_breakdown_attributes_fires_to_word_counts():
    """Every fire lands in exactly one word-count bucket, and buckets sum to totals.

    @david-spacil reported that "59.4% of hard-sweep-family firings land exactly
    on `wc == 3`" and that it was "not measured further, just noting it". Nothing
    in the repository could reproduce that shape of figure, so it stayed an
    anecdote for six weeks. This pins the instrument that can.
    """
    import tools.rule_coverage_report as RC

    counts = RC.run_wc_breakdown(str(_SAMPLE_DIR), config_path=str(_ROOT / "setup" / "config.txt"), quiet=True)

    assert set(counts) == set(RC.RULES), "the breakdown must cover the same registry as the coverage report"
    for rule, buckets in counts.items():
        assert set(buckets) == set(RC.WC_BUCKETS), f"{rule} has unexpected buckets: {sorted(buckets)}"
        assert all(v >= 0 for v in buckets.values())

    assert any(sum(b.values()) for b in counts.values()), "no rule fired at all; the capture is not wired"


def test_wc_breakdown_totals_agree_with_the_coverage_pass():
    """The two instruments must not disagree about which rules fire.

    They measure the same thing by different routes -- the coverage pass scores
    document-by-document, the breakdown line-by-line -- so a rule that fires in
    one and not the other means one of them is lying. Fire COUNTS may legitimately
    differ (the coverage pass applies document post-processing, the per-line
    breakdown does not), so only the fired/not-fired sets are compared.
    """
    import tools.rule_coverage_report as RC

    by_wc = RC.run_wc_breakdown(str(_SAMPLE_DIR), config_path=str(_ROOT / "setup" / "config.txt"), quiet=True)
    coverage = RC.run_coverage(
        str(_SAMPLE_DIR), config_path=str(_ROOT / "setup" / "config.txt"), quiet=True, skip_loo=True
    )

    fired_by_wc = {r for r, b in by_wc.items() if sum(b.values())}
    fired_by_coverage = {r for r, v in coverage.items() if v["fire_count"]}
    assert fired_by_wc == fired_by_coverage, (
        f"instruments disagree on which rules fire: only in --by-wc {sorted(fired_by_wc - fired_by_coverage)}, "
        f"only in coverage {sorted(fired_by_coverage - fired_by_wc)}"
    )
