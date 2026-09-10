"""
tests/test_tool_config_resolution.py
====================================
Guards on how the offline tools resolve a ``--config`` argument.

Every test here pins a defect that produced a confident, wrong, *silent* answer.
That is this repository's documented recurring failure mode -- three test
harnesses have fed ``categorize_line`` a value production never computes -- and
these are the same mistake relocated into ``tools/``.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recategorize_from_csv import (  # noqa: E402
    _load_lang_config,
    short_cap_arms_hard_sweep,
)

import classify_TEXT as LC  # noqa: E402
import text_util as TU  # noqa: E402

SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"
REAL_CONFIG = _ROOT / "setup" / "config.txt"


def _variant_config(tmp_path: Path, **overrides: str) -> Path:
    """A copy of setup/config.txt with some constants changed."""
    text = REAL_CONFIG.read_text(encoding="utf-8")
    for key, value in overrides.items():
        out = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
                out.append(f"{key} = {value}")
            else:
                out.append(line)
        text = "\n".join(out) + "\n"
    path = tmp_path / "variant_config.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# D-A: rule_coverage_report must MEASURE the config it is handed.
# ---------------------------------------------------------------------------


def test_coverage_report_honours_config_thresholds(tmp_path):
    """`--config` must reach the thresholds, not only the language lists.

    It used to call `_load_lang_config` and nothing else, so every threshold came
    from whatever `text_util` imported at start-up. Two runs pointed at configs
    with different `MOSTLY_READABLE_VALID_MIN` produced byte-identical JSON.

    This report is what classifies rules DEAD / REDUNDANT-HERE / LOAD-BEARING and
    what `tools/RULE_COVERAGE.md` gates retirement decisions on, so measuring the
    wrong configuration is not a cosmetic problem.
    """
    from rule_coverage_report import run_coverage

    variant = _variant_config(tmp_path, MOSTLY_READABLE_VALID_MIN="0.10", LOWPPL_CLEAR_MAX="120.0")

    base = run_coverage(str(SAMPLE_DIR), config_path=str(REAL_CONFIG), quiet=True, skip_loo=True)
    moved = run_coverage(str(SAMPLE_DIR), config_path=str(variant), quiet=True, skip_loo=True)

    base_fires = {k: v["fire_count"] for k, v in base.items()}
    moved_fires = {k: v["fire_count"] for k, v in moved.items()}
    assert base_fires != moved_fires, (
        "coverage is identical under two different configs; --config is not reaching the thresholds"
    )


def test_coverage_report_rejects_an_invalid_config(tmp_path):
    """An invalid config must fail, not produce a clean table.

    The same file that makes `recategorize_from_csv` raise
    `CATEG_TRASH_SCORE_MAX must be < CATEG_NOISY_SCORE_MAX` used to yield a
    perfectly ordinary coverage report.
    """
    from rule_coverage_report import run_coverage

    bad = _variant_config(tmp_path, CATEG_TRASH_SCORE_MAX="0.95")
    with pytest.raises(ValueError, match="CATEG_TRASH_SCORE_MAX"):
        run_coverage(str(SAMPLE_DIR), config_path=str(bad), quiet=True, skip_loo=True)


# ---------------------------------------------------------------------------
# D-B: one set of language fallbacks, and no silent degradation.
# ---------------------------------------------------------------------------


def test_offline_and_production_share_one_language_fallback():
    """The offline copy had drifted from the shipped one by exactly `slk`.

    A Slovak line then reached the guards at TRUST_TIER_UNKNOWN (0.50) offline
    and TRUST_TIER_TRUSTED (0.85) in production -- the trust-tier divergence
    class already fixed three times in this repository's test harnesses.
    """
    assert "slk" in TU.DEFAULT_TRUSTED_FOREIGN_LANGS
    _expected, known_bases = _load_lang_config(None)
    assert "slk" in known_bases, "the built-in offline fallback lost Slovak again"

    # And the production module resolves from the same named constants.
    assert "slk" in TU._TRUSTED_FOREIGN_LANG_BASES
    assert LC.DEFAULT_TRUSTED_FOREIGN_LANGS is TU.DEFAULT_TRUSTED_FOREIGN_LANGS


def test_shipped_config_agrees_with_the_fallback_on_slovak():
    """Belt and braces: the fallback and the shipped file must not disagree."""
    _expected, from_file = _load_lang_config(str(REAL_CONFIG))
    _expected2, from_default = _load_lang_config(None)
    assert from_file == from_default, (
        f"setup/config.txt and the built-in fallback resolve differently: {sorted(from_file)} vs {sorted(from_default)}"
    )


@pytest.mark.parametrize("bad_path", ["config.txt", "setup/nope.txt"])
def test_a_missing_config_raises_instead_of_degrading(bad_path):
    """`configparser.read()` ignores a missing path; that must not be the behaviour.

    Every documented command in this repo said `--config config.txt`, and no
    such file has existed at the repo root since it moved under `setup/`. The
    silent fallback changed the trust tier of every non-expected language.
    """
    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        _load_lang_config(bad_path)


def test_a_config_without_a_classify_section_raises(tmp_path):
    partial = tmp_path / "partial.txt"
    partial.write_text("[TEXT_UTILS]\nCATEG_TRASH_SCORE_MAX = 0.55\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[CLASSIFY\]"):
        _load_lang_config(str(partial))


# ---------------------------------------------------------------------------
# D-C / D-D: documented commands and defaults must be runnable from the repo root.
# ---------------------------------------------------------------------------


def test_documented_config_paths_exist():
    """Docs must not tell a contributor to pass a file that does not exist."""
    for doc in (_ROOT / "tools" / "SWEEP_NOTES.md", _ROOT / "tools" / "RULE_COVERAGE.md"):
        for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(("#", ">")) or "lutsai@stargate" in line:
                continue  # comments and the archived run log are records, not instructions
            if "--config config.txt" in line:
                pytest.fail(f"{doc.name}:{i} tells the reader to pass a non-existent config.txt")


def test_tool_defaults_resolve_from_the_repo_root():
    """Every driver's default paths must work when run as documented, from the root."""
    import argparse

    import greedy_backward_elimination as GBE
    from greedy_backward_elimination import main as _greedy_main  # noqa: F401

    parser_src = Path(GBE.__file__).read_text(encoding="utf-8")
    assert '"../setup/config.txt"' not in parser_src, "greedy default config escapes the repo root"
    assert 'Path("../data_samples/DOC_LINE_CATEG")' not in parser_src, "greedy default input escapes the repo root"
    assert isinstance(argparse.ArgumentParser, type)


# ---------------------------------------------------------------------------
# B7: the cap dependency a joint sweep can cross.
# ---------------------------------------------------------------------------


def test_shipped_config_keeps_hard_sweep_unarmed_at_short_lines():
    """SHORT_PPL_CAP < HARD_SWEEP_PPL_MIN is load-bearing, not incidental.

    Issue #30 measured 58,427 notation lines that are `Clear` ONLY because the
    cap flattens their perplexity below the hard-sweep floor -- 50,221 of them
    sitting exactly at the cap. Anyone moving either constant inherits that
    dependency, so it is pinned here with the reason attached.
    """
    assert not short_cap_arms_hard_sweep(), (
        "SHORT_PPL_CAP now sits above HARD_SWEEP_PPL_MIN. That arms rule_hard_sweep at "
        "wc <= 2 for the first time and puts ~58,427 notation lines in reach of Trash. "
        "If this is deliberate, measure notation Clear-loss and update this test with the evidence."
    )


def test_the_joint_move_is_detected_as_arming_the_route():
    """The configuration the tuner can reach is exactly the one worth flagging."""
    assert short_cap_arms_hard_sweep({"SHORT_PPL_CAP": 950.0, "HARD_SWEEP_PPL_MIN": 500.0})


# ---------------------------------------------------------------------------
# A swept constant that no decision path reads is a wasted dimension.
# ---------------------------------------------------------------------------


def test_weird_ratio_inverted_min_is_still_decision_inert():
    """`WEIRD_RATIO_INVERTED_MIN` is loaded and swept but read by nothing.

    setup/config.txt says so and asks for a team ack before removal. This test
    pins the claim so it cannot quietly stop being true (or quietly stay true
    while someone assumes it was fixed): it occupies 1 of 40 sweep dimensions,
    and every importance ranking prices it as noise.
    """
    src = (_ROOT / "text_util.py").read_text(encoding="utf-8")
    reads = [
        i
        for i, line in enumerate(src.splitlines(), 1)
        if "WEIRD_RATIO_INVERTED_MIN" in line and not line.lstrip().startswith("#")
    ]
    assert len(reads) == 1, (
        f"WEIRD_RATIO_INVERTED_MIN is now referenced at lines {reads} in text_util.py. "
        "If it was wired into a decision path, delete this test; if it gained another "
        "definition, that is a bug."
    )
