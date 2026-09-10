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
    GOLD_SIDECAR_KEYS,
    _gold_report,
    annotated_mask,
    attach_gold_sidecar,
    document_decade,
    evaluate_dataframe,
    load_csvs,
    rescore_csv,
)

GOLD_DIR = _ROOT / "tools" / "gold"
SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"
SIDECAR_DIR = GOLD_DIR / "sidecars"
ISSUE30_SIDECAR = SIDECAR_DIR / "issue30_gold_2067.csv"


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


# ---------------------------------------------------------------------------
# Key-indexed gold sidecars (`--gold-sidecar`).
# ---------------------------------------------------------------------------


def test_sidecars_are_not_reachable_as_per_document_gold():
    """A sidecar in `tools/gold/` corrupts every driver that globs the directory.

    The drivers treat each `*.csv` under an `--input-dir` as a scoreable
    per-document gold set. A sidecar is neither per-document nor scoreable on its
    own -- it has no `categ` column -- so one sitting beside `GOLD_CLEAR.csv`
    poisons `load_csvs(GOLD_DIR)` and takes the per-document guard above with it.
    This is the guard on the fix, not a style rule.
    """
    assert ISSUE30_SIDECAR.exists(), "the issue-#30 sidecar is missing"
    stray = [p.name for p in GOLD_DIR.glob("*.csv") if "categ" not in pd.read_csv(p, nrows=0).columns]
    assert not stray, f"sidecar-shaped CSVs directly in tools/gold/: {stray}"


def test_issue30_sidecar_has_the_documented_shape():
    """The contract GOLD.md states, asserted rather than described.

    2,067 rows over 816 documents, split 508 / 1,559 by `gold_source`, no
    duplicate keys. The counts are the delivery's own and are what every figure
    quoted from this file assumes.
    """
    df = pd.read_csv(ISSUE30_SIDECAR, dtype=str, keep_default_na=False)

    assert list(df.columns) == ["file", "page_num", "line_num", "gold_categ", "gold_source"]
    assert len(df) == 2067
    assert df["file"].nunique() == 816
    assert not df.duplicated(subset=list(GOLD_SIDECAR_KEYS)).any(), "duplicate keys would double-count"
    assert df["gold_source"].value_counts().to_dict() == {"calibration_1567": 1559, "issue30_508": 508}
    assert set(df["gold_categ"]) <= {"Clear", "Noisy", "Trash", "Non-text", "Empty"}


def test_sidecar_join_survives_a_locator_type_mismatch():
    """The one real failure mode: str page_num against int page_num.

    A sidecar is read as text; a delivered batch may carry its locators as
    integers. Without the coercion both sides are well-formed, the join matches
    nothing, and the run reports "no gold" instead of failing -- this
    repository's recurring bug in a new place.
    """
    gold = pd.read_csv(ISSUE30_SIDECAR, dtype=str, keep_default_na=False).head(3)
    batch = pd.DataFrame(
        {
            "file": list(gold["file"]),
            "page_num": [int(x) for x in gold["page_num"]],
            "line_num": [int(x) for x in gold["line_num"]],
            "text": ["x"] * 3,
            "categ": ["Clear"] * 3,
        },
        index=[7, 8, 9],
    )

    out = attach_gold_sidecar(batch, ISSUE30_SIDECAR, verbose=False)

    assert list(out[GOLD_COLUMN_DEFAULT]) == list(gold["gold_categ"])
    assert list(out.index) == [7, 8, 9], "index must survive; _gold_report realigns on it"
    assert len(out) == len(batch), "a left join must not add or drop rows"


def test_sidecar_keeps_unmatched_rows_and_leaves_their_gold_blank():
    """Partial matches are correct, not a failure.

    Only 484 of the issue-#30 508 keys are still in the changed population, so a
    join that dropped unmatched rows would quietly change the population being
    scored. Blank gold cells are already skipped by `_gold_report`.
    """
    gold = pd.read_csv(ISSUE30_SIDECAR, dtype=str, keep_default_na=False).head(2)
    batch = pd.DataFrame(
        {
            "file": list(gold["file"]) + ["NOT_ANNOTATED"],
            "page_num": [int(x) for x in gold["page_num"]] + [1],
            "line_num": [int(x) for x in gold["line_num"]] + [1],
            "text": ["x"] * 3,
            "categ": ["Clear"] * 3,
        }
    )

    out = attach_gold_sidecar(batch, ISSUE30_SIDECAR, verbose=False)

    assert len(out) == 3
    assert str(out[GOLD_COLUMN_DEFAULT].iloc[2]) in ("nan", "", "None")


def test_sidecar_matching_nothing_raises_instead_of_scoring_no_gold():
    """The silent version of this mistake is a run that looks like it used gold."""
    batch = pd.DataFrame({"file": ["NO_SUCH_DOC"], "page_num": [1], "line_num": [1], "text": ["x"], "categ": ["Clear"]})

    with pytest.raises(ValueError, match="matched 0 of"):
        attach_gold_sidecar(batch, ISSUE30_SIDECAR, verbose=False)


def test_sidecar_missing_a_key_column_raises(tmp_path):
    """Same rule as `--gold-column`: malformed input is an error, not a fallback."""
    bad = tmp_path / "bad.csv"
    bad.write_text("file,gold_categ\nDOC,Clear\n", encoding="utf-8")
    batch = pd.DataFrame({"file": ["DOC"], "page_num": [1], "line_num": [1], "text": ["x"], "categ": ["Clear"]})

    with pytest.raises(ValueError, match="missing key column"):
        attach_gold_sidecar(batch, bad, verbose=False)


def test_joined_sidecar_scores_through_the_normal_gold_report():
    """End to end: sidecar -> join -> `_gold_report`, with no special casing.

    The join's only job is to put the column where the existing gold path already
    looks for it. If this needed a second scoring route, the design would be wrong.
    """
    gold = pd.read_csv(ISSUE30_SIDECAR, dtype=str, keep_default_na=False).head(4)
    batch = pd.DataFrame(
        {
            "file": list(gold["file"]),
            "page_num": [int(x) for x in gold["page_num"]],
            "line_num": [int(x) for x in gold["line_num"]],
            "text": ["x"] * 4,
            "categ": ["Clear"] * 4,
        }
    )

    joined = attach_gold_sidecar(batch, ISSUE30_SIDECAR, verbose=False)
    perfect = joined.copy()
    perfect["categ"] = list(gold["gold_categ"])

    report = _gold_report(joined, perfect, GOLD_COLUMN_DEFAULT)

    assert report["n"] == 4
    assert report["delta_macro_f1"] > 0, "a perfect re-score must beat an all-Clear incumbent"


# ---------------------------------------------------------------------------
# Partial annotation. THE case every gold test missed, and the one that matters
# once gold arrives as a sidecar joined onto a much larger corpus.
# ---------------------------------------------------------------------------


def _partially_annotated_frame():
    """A sample frame with 2 of 15 rows annotated, as a sidecar join leaves it."""
    df = load_csvs(SAMPLE_DIR)
    df[GOLD_COLUMN_DEFAULT] = ""
    df.loc[df.index[0], GOLD_COLUMN_DEFAULT] = "Clear"
    df.loc[df.index[1], GOLD_COLUMN_DEFAULT] = "Trash"
    return df


def test_unannotated_rows_are_not_scored_as_a_sixth_category():
    """The defect this test exists for, stated as an assertion.

    `normalize_category` maps a blank cell to `""`. Unmasked, that becomes a
    class in the confusion matrix with support equal to the unannotated rows, it
    drags every real class's precision down, and `macro_f1` -- the sweep's
    default objective -- turns into a monotone function of the annotation RATE.
    Measured before the fix on exactly this frame: `line_count` 15 against 2
    annotated, `macro_f1` 0.0278, `flip_rate` 0.9333, and a `''` class of
    support 13.

    Every other gold test in this file uses a fully-annotated frame, which is
    why none of them could see it.
    """
    df = _partially_annotated_frame()
    n_annotated = int(annotated_mask(df, GOLD_COLUMN_DEFAULT).sum())
    assert n_annotated == 2, "premise: only two rows carry a label"

    metrics = evaluate_dataframe(df, gold_category_column=GOLD_COLUMN_DEFAULT)

    assert metrics["line_count"] == n_annotated, "line_count must count annotated rows, not frame rows"
    assert "" not in metrics["confusion"], "blank gold leaked in as a category"
    assert "" not in metrics["per_class_f1"]
    assert 0.0 <= metrics["flip_rate"] <= 1.0


def test_the_two_gold_paths_agree_on_what_annotated_means():
    """`evaluate_dataframe` and `_gold_report` used to disagree, and it mattered.

    `_gold_report` masked; `evaluate_dataframe` -- the path every sweep,
    ablation and A/B trial goes through -- did not. Both now route through
    `annotated_mask`, and this pins that they agree on the count.
    """
    df = _partially_annotated_frame()
    rescored = df.copy()
    report = _gold_report(df, rescored, GOLD_COLUMN_DEFAULT)
    metrics = evaluate_dataframe(df, gold_category_column=GOLD_COLUMN_DEFAULT)
    assert report["n"] == metrics["line_count"]


def test_gold_weights_change_the_score_and_are_opt_in():
    """Sampling weights, which the metrics had no notion of at all.

    Absent a `gold_weight` column nothing changes, so every existing caller is
    untouched. Present, it reweights -- which is the point: this gold set is
    stratified and its strata are inverted relative to the population, worth
    about 10 points of headline agreement.
    """
    df = _partially_annotated_frame()
    unweighted = evaluate_dataframe(df, gold_category_column=GOLD_COLUMN_DEFAULT)

    df["gold_weight"] = "1.0"
    df.loc[df.index[0], "gold_weight"] = "9.0"
    weighted = evaluate_dataframe(df, gold_category_column=GOLD_COLUMN_DEFAULT)

    assert weighted["line_count"] == unweighted["line_count"], "weights must not change the row count"
    assert weighted["flip_rate"] != pytest.approx(unweighted["flip_rate"]), "weights were ignored"


def test_a_malformed_gold_weight_raises_rather_than_being_dropped():
    """A silently-dropped weight is a silently-reweighted objective."""
    df = _partially_annotated_frame()
    df["gold_weight"] = "1.0"
    df.loc[df.index[0], "gold_weight"] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        evaluate_dataframe(df, gold_category_column=GOLD_COLUMN_DEFAULT)


@pytest.mark.parametrize(
    "file_id, expected",
    [
        ("CTX192400709", "1920s"),
        ("MTX194500261", "1940s"),
        ("CTX201600123", "2010s"),
        ("MtX202100614", "2020s"),
        ("P009_00014", "unknown"),
        ("", "unknown"),
    ],
)
def test_document_decade_parses_the_archive_naming(file_id, expected):
    assert document_decade(file_id) == expected


def test_the_gold_report_always_shows_its_composition():
    """A single agreement figure over a stratified sample hides a weighting choice.

    The issue-#30 sample over-represents the 1920s by ~95x and under-represents
    the 2010s by ~10x; unweighted agreement over it is 62.8% against 72.7%
    reweighted by decade. The report must name the strata so the headline cannot
    be read as a population estimate.
    """
    gold = pd.read_csv(ISSUE30_SIDECAR, dtype=str, keep_default_na=False)
    sample = pd.concat(
        [gold[gold.file.str.startswith("CTX192")].head(4), gold[gold.file.str.startswith("CTX201")].head(4)]
    )
    old = sample.copy()
    old["categ"] = "Clear"
    new = old.copy()
    new["categ"] = old[GOLD_COLUMN_DEFAULT]

    report = _gold_report(old, new, GOLD_COLUMN_DEFAULT)

    assert "strata" in report
    assert "gold_source" in report["strata"], "the two annotation rounds are different populations"
    assert "decade" in report["strata"], "decade is the stratification that is inverted vs the corpus"
    assert {"1920s", "2010s"} <= set(report["strata"]["decade"])
    for group in report["strata"]["decade"].values():
        assert group["n"] > 0 and 0.0 <= group["agreement"] <= 1.0
