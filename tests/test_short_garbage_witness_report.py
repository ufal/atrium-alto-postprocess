"""
Fast, model-free tests for `tools/short_garbage_witness_report.py` (issue #30).

The report exists because `_has_shape_garbage_evidence()` is wired but disabled
(`SHORT_GARBAGE_WITNESS_ENABLE` defaults to false), so re-scoring a collection
cannot show what it WOULD reach. Asking the predicate directly is the only way to
measure its exposure before the flag is flipped. Two properties are worth pinning:

  * the report's per-clause diagnosis must not drift from the predicate it
    describes — the module asserts this per line, and
    `test_clause_diagnosis_matches_the_predicate` exercises that assertion
    across the whole #30 fixture population rather than one line at a time;
  * the report must not become a second scoring path. `test_no_signal_reconstruction`
    is the guard: the same class of drift that
    `tests/test_scoring_single_source.py` pins for the three real scorers.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import text_util as tu  # noqa: E402

_TOOL_PATH = _ROOT / "tools" / "short_garbage_witness_report.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("short_garbage_witness_report", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R = _load_tool()


# The #30 population, both sides. Kept local rather than imported from
# calibration_fixtures because these are inputs to a *predicate*, not scored
# fixtures, and they carry no ppl/lang columns.
_WITNESSED = ["oueussd", "sektlll", "cuxoaid", "Tthts I", "vansasaasasa", "NINNNIC", "rragment", "IDIDIDIDIDIDUOID"]
_NOT_WITNESSED = [
    "malakofauna",
    "diapozitiv",
    "Equus caballus",
    "Occipitale",
    "Kaaden",
    "Pinii",
    "kontext",
    "Uniocrassus",
    "Skelettmaterial",
    "Ossa tarsi",
    "radius prox.sin.",
    "Reg.Bez.Aussig.",
    "1 ks",
    "II/C",
    "I-VIII-c",
    "vrstva 3",
    "ctvrtek",
    "Hannah",
    "Schifffahrt",
    "mm",
    "edelite",
]


@pytest.mark.parametrize("text", _WITNESSED + _NOT_WITNESSED)
def test_clause_diagnosis_matches_the_predicate(text):
    """`classify_line` asserts internally; this proves it holds on both sides.

    A clause list that is non-empty exactly when the predicate is True is what
    makes the report's "which clause fired" column trustworthy. If the predicate
    grows a clause the report does not know about, this goes red rather than
    silently under-reporting.
    """
    verdict = R.classify_line(text)
    assert bool(verdict["clauses"]) == verdict["witness"]
    assert verdict["witness"] == tu._has_shape_garbage_evidence(text)


@pytest.mark.parametrize("text", _WITNESSED)
def test_witnessed_lines_name_a_clause(text):
    verdict = R.classify_line(text)
    assert verdict["witness"], f"{text!r} should be witnessed"
    assert verdict["clauses"], f"{text!r} is witnessed but names no clause"
    for clause in verdict["clauses"].split(","):
        assert clause in R._CLAUSE_ORDER


@pytest.mark.parametrize("text", _NOT_WITNESSED)
def test_vocabulary_and_notation_are_not_witnessed(text):
    verdict = R.classify_line(text)
    assert not verdict["witness"], f"{text!r} must not be witnessed (clauses={verdict['clauses']})"


def test_route_eligibility_is_the_text_only_half_of_the_entry_condition():
    """Eligibility must mirror gate 6's text-only terms, and nothing more."""
    assert R.classify_line("oueussd")["route_eligible"]
    # a diacritic escapes the route entirely
    assert not R.classify_line("oueussdá")["route_eligible"]
    # notation is exempt
    assert not R.classify_line("II/C")["route_eligible"]
    # too many tokens for the route
    long_line = " ".join(["oueussd"] * (tu.ISOLATED_CHAR_MIN_TOKENS + 1))
    assert not R.classify_line(long_line)["route_eligible"]
    # ...but the witness itself is length-agnostic, and the report must not
    # conflate the two: eligibility is about the ROUTE, the witness is about the
    # spelling.
    assert R.classify_line(long_line)["witness"]


def test_no_signal_reconstruction():
    """The report must never grow a second scoring path.

    `lang_score` in a DOC_LINE_CATEG CSV is the `remap_lang` cap, not the
    two-tier trust score the rules read; deriving routing decisions from the
    stored columns is the harness bug already fixed in
    tests/test_rotation_regression.py and tests/test_calibration.py::_categ.
    Re-scoring belongs to tools/recategorize_from_csv.py, which reuses
    classify_TEXT.score_line. Source-inspected for the same reason
    tests/test_scoring_single_source.py inspects its subjects.
    """
    source = _TOOL_PATH.read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    for forbidden in (
        "compute_quality_score(",
        "categorize_line(",
        "determine_category(",
        "score_line(",
        "TRUST_TIER_UNKNOWN",
        "trust_lang_score",
    ):
        assert forbidden not in body, (
            f"{_TOOL_PATH.name} references {forbidden!r}: it is re-scoring, or reconstructing a signal. "
            "It must stay on text-only predicates."
        )


def test_reads_a_doc_line_categ_csv_and_writes_candidates(tmp_path):
    src = tmp_path / "CTX000000000.csv"
    with src.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["categ", "text", "word_count"])
        writer.writerow(["Trash", "oueussd", "1"])
        writer.writerow(["Clear", "malakofauna", "1"])
        writer.writerow(["Clear", "v klášteře Strahovském.", "3"])
        writer.writerow(["Trash", "", "0"])

    out = tmp_path / "candidates.csv"
    assert R.main([str(src), "--out", str(out)]) == 0

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["text"] for r in rows] == ["oueussd"], "only witnessed lines belong in the candidate file"
    assert rows[0]["categ_current"] == "Trash"
    assert rows[0]["gold_categ"] == "", "gold column must ship blank so it can be filled blind"


def test_plain_lines_mode(tmp_path):
    probe = tmp_path / "probe.txt"
    probe.write_text("oueussd\nmalakofauna\n\n", encoding="utf-8")
    assert R.main(["--lines", str(probe)]) == 0


def test_flag_state_does_not_change_the_report():
    """The report reads the predicate directly, so the ship-inert flag is moot.

    This is the property that makes the tool usable *before* the flag flips —
    and the reason its header says the flag does not affect it.
    """
    before = R.classify_line("oueussd")
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_ENABLE": True}):
        after = R.classify_line("oueussd")
    assert before == after
