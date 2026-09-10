"""
tests/test_gold_objective.py
============================
Guards on the gold-scoring path (`--gold-column`) and on the frame alignment the
offline diff report depends on.

Both halves exist for the same reason: this repository's recurring failure mode
is a harness that returns a confident number about something other than what it
claims to measure. A gold column that silently falls back to the pipeline's own
labels, or a diff report that compares row N against row M, are that failure mode
in two new places.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recategorize_from_csv import (  # noqa: E402
    GOLD_COLUMN_DEFAULT,
    _gold_report,
    evaluate_dataframe,
    load_csvs,
    rescore_csv,
)

GOLD_DIR = _ROOT / "tools" / "gold"
SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"


# ---------------------------------------------------------------------------
# The gold column changes the objective, and never silently.
# ---------------------------------------------------------------------------


def test_example_gold_set_exists_and_is_per_document():
    """One CSV per document, matching the DOC_LINE_CATEG convention.

    Several documents in one CSV would put their lines on a shared page and let
    page-level post-processing act across a boundary production never sees.
    """
    csvs = sorted(GOLD_DIR.glob("*.csv"))
    assert csvs, "tools/gold/ has no CSVs; run tools/gold/build_example_gold.py"
    for path in csvs:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert GOLD_COLUMN_DEFAULT in df.columns, f"{path.name} has no {GOLD_COLUMN_DEFAULT} column"
        assert df["file"].nunique() == 1, f"{path.name} mixes {df['file'].nunique()} documents"
        assert df["file"].iloc[0] == path.stem, f"{path.name} does not match its `file` value"


def test_missing_gold_column_raises_instead_of_scoring_perfectly():
    """The whole point of the flag is that it cannot quietly no-op.

    Falling back to the stored categories would score predictions against
    themselves, i.e. a perfect score, which is worse than a crash because nobody
    goes looking for the cause of good news.
    """
    df = load_csvs(SAMPLE_DIR)
    with pytest.raises(KeyError, match="gold_categ"):
        evaluate_dataframe(df, None, gold_category_column="gold_categ")


def test_gold_objective_differs_from_the_self_referential_one():
    """Scoring against gold must not reproduce scoring against `categ`.

    If these agreed, the flag would be decorative. They differ here because the
    example gold set records `oueussd` as `Trash` while the merged #30 gate lets
    it reach `Clear` -- the accepted debt, made visible by the objective.
    """
    df = load_csvs(GOLD_DIR)
    self_ref = evaluate_dataframe(df, None)
    vs_gold = evaluate_dataframe(df, None, gold_category_column=GOLD_COLUMN_DEFAULT)

    assert self_ref["flip_rate"] == pytest.approx(0.0, abs=1e-9), (
        "the re-score is expected to reproduce the stored categories exactly; "
        "if this fails, the parity guarantee is broken and nothing below is meaningful"
    )
    assert vs_gold["flip_rate"] > 0.0, "gold scoring collapsed onto the self-referential answer"
    assert vs_gold["gold_column"] == GOLD_COLUMN_DEFAULT


def test_gold_report_carries_the_incumbent_comparison():
    """`baseline_vs_gold` is the number that decides adoption.

    Beating the pipeline on the pipeline's own labels means nothing; the trial
    has to beat the shipped labels against humans.
    """
    df = load_csvs(GOLD_DIR)
    metrics = evaluate_dataframe(df, None, gold_category_column=GOLD_COLUMN_DEFAULT)
    assert "baseline_vs_gold" in metrics
    assert "gold_delta_macro_f1" in metrics
    # Nothing was tuned, so the trial IS the incumbent and the delta is zero.
    assert metrics["gold_delta_macro_f1"] == pytest.approx(0.0, abs=1e-9)


def test_blank_gold_cells_are_skipped_not_scored():
    """A partially annotated collection must score only its annotated rows."""
    old, new = rescore_csv(sorted(GOLD_DIR.glob("*.csv"))[0])
    full = _gold_report(old, new, GOLD_COLUMN_DEFAULT)
    assert full is not None and full["n"] == len(old)

    blanked = old.copy()
    blanked[GOLD_COLUMN_DEFAULT] = ""
    assert _gold_report(blanked, new, GOLD_COLUMN_DEFAULT) == {"n": 0}

    partial = old.copy()
    partial.loc[partial.index[1:], GOLD_COLUMN_DEFAULT] = ""
    assert _gold_report(partial, new, GOLD_COLUMN_DEFAULT)["n"] == 1


# ---------------------------------------------------------------------------
# Frame alignment: the diff report must be about categories, never row order.
# ---------------------------------------------------------------------------


def test_diff_report_is_invariant_to_on_disk_row_order(tmp_path):
    """Reversing a document's stored row order must not invent category changes.

    Before this was fixed, `rescore_csv` sorted only the `old` frame by
    (page_num, line_num) while `new` came back in input order, and `_report`
    compared them positionally. On a 9-line sample document, reversing the rows
    produced "4 line(s) changed category" while the category COUNTS stayed
    identical -- the signature of a misalignment rather than a real flip.

    This matters beyond cosmetics: `total lines changed category: 0` is the
    parity signal quoted in tools/SWEEP_NOTES.md as evidence that the offline
    re-scorer reproduces production.
    """
    src = SAMPLE_DIR / "CTX000000002.csv"
    original = pd.read_csv(src, dtype=str, keep_default_na=False)

    reversed_path = tmp_path / src.name
    original.iloc[::-1].to_csv(reversed_path, index=False)

    old, new = rescore_csv(reversed_path)

    assert list(old.index) == list(new.index), "rescore_csv returned misaligned frames"
    changed = int((old["categ"].to_numpy() != new["categ"].to_numpy()).sum())
    assert changed == 0, (
        f"{changed} phantom category change(s) from row order alone; "
        "the frames are misaligned, not the categories different"
    )


def test_rescore_csv_returns_frames_sharing_an_index(tmp_path):
    """A weaker, structural version of the guard above, on the shipped samples."""
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        old, new = rescore_csv(path)
        assert list(old.index) == list(new.index), f"{path.name}: index mismatch"
