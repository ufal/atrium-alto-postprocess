"""
Tests for aggregate_STAT.py pure-logic helpers.
"""

import pandas as pd
import pytest

from aggregate_STAT import _sum_metrics, load_config

DEFAULT_CONFIG = "setup/config.txt"
config = load_config(DEFAULT_CONFIG)
STANDARD_COLS = frozenset(config.get("standard_cols", "Clear,Noisy,Trash,Non-text,Empty").split(","))


def test_sum_metrics_basic_with_new_columns():
    # Mock a dataframe representing a DOC_LINE_CATEG CSV containing new columns
    df = pd.DataFrame(
        {
            "file": ["doc1", "doc1", "doc1"],
            "page_num": [1, 1, 1],
            "categ": ["Clear", "Noisy", "Trash"],
            "word_count": [10, 5, 2],
            "char_count": [50, 20, 5],
            "quality_score": [0.9, 0.6, 0.2],
            "word_weird": [0.0, 0.1, 0.8],
            "lang_score": [0.95, 0.8, 0.4],
            "perplex": [150, 400, 2000],
            "symbol": [0, 1, 5],
            "vowel_ratio": [0.4, 0.3, 0.0],
            "rot_ratio": [0.0, 0.0, 0.1],
            "lang": ["ces", "ces", "deu"],
            "caps_header": [False, False, False],
            # --- New columns added to verify survival/no-crash ---
            "original_text": ["a", "b", "c"],
            "original_lang": ["ces", "ces", "deu"],
            "orig_lang_score": [0.95, 0.8, 0.4],
            "weird_wx": [0, 0, 1],
            "pp_dedup": [False, False, False],
            "pp_surrounded_trash": [False, False, False],
            "pp_inverted_run": [False, False, False],
        }
    )

    res = _sum_metrics(df, STANDARD_COLS)

    assert len(res) == 1
    # Only "Clear" and "Noisy" lines are aggregated for word/char counts and averages
    assert res.iloc[0]["Clear"] == 1
    assert res.iloc[0]["Noisy"] == 1
    assert res.iloc[0]["Trash"] == 1
    assert res.iloc[0]["num_lines"] == 3  # Clear + Noisy + Trash

    # 10 (Clear) + 5 (Noisy) = 15 words. The 2 Trash words are ignored.
    assert res.iloc[0]["total_word_count"] == 15
    assert res.iloc[0]["main_lang"] == "ces"


def test_sum_metrics_empty():
    df = pd.DataFrame()
    res = _sum_metrics(df, STANDARD_COLS)
    assert res.empty


def test_load_config_reads_standard_cols(tmp_path):
    """(#7 Phase 0) [AGGREGATE] STANDARD_COLS must round-trip from the config
    file instead of always falling back to the hardcoded default."""
    cfg = tmp_path / "config.txt"
    cfg.write_text("[AGGREGATE]\nSTANDARD_COLS = Clear,Noisy\n", encoding="utf-8")
    loaded = load_config(str(cfg))
    assert loaded["standard_cols"] == "Clear,Noisy"

    # Shipped config carries the full five-category set.
    shipped = load_config(DEFAULT_CONFIG)
    assert frozenset(shipped["standard_cols"].split(",")) == frozenset({"Clear", "Noisy", "Trash", "Non-text", "Empty"})

    # Missing-file fallback must include the key too (main() relies on it).
    missing = load_config(str(tmp_path / "nope.txt"))
    assert "standard_cols" in missing


# ── (#31 Phase 4) document ids stay text, the final order stays the same ─────


def test_natural_file_key_orders_numeric_ids_numerically():
    from aggregate_STAT import _natural_file_key

    assert sorted(["10", "9", "0001", "CTX2", "CTX10"], key=_natural_file_key) == ["0001", "9", "10", "CTX10", "CTX2"]


def test_sort_page_stats_accepts_mixed_ids():
    from aggregate_STAT import sort_page_stats

    df = pd.DataFrame({"file": ["report", "0001", "12", "0001"], "page_num": [1, 2, 1, 1]})
    out = sort_page_stats(df)
    assert list(zip(out["file"], out["page_num"], strict=True)) == [("0001", 1), ("0001", 2), ("12", 1), ("report", 1)]


def test_process_csv_file_keeps_leading_zero_and_na_ids(tmp_path):
    from aggregate_STAT import process_csv_file

    head = (
        "file,page_num,categ,word_count,char_count,quality_score,word_weird,lang_score,perplex,vowel_ratio,"
        "rot_ratio,lang,caps_header"
    )
    row = "1,Clear,1,5,0.9,0,0.9,10,0.5,0,ces,False"  # everything after the file id
    (tmp_path / "0001.csv").write_text(f"{head}\n0001,{row}\n", encoding="utf-8")
    (tmp_path / "NA.csv").write_text(f"{head}\nNA,{row}\n", encoding="utf-8")
    assert list(process_csv_file(tmp_path / "0001.csv", STANDARD_COLS)["file"]) == ["0001"]
    assert list(process_csv_file(tmp_path / "NA.csv", STANDARD_COLS)["file"]) == ["NA"]


def test_the_final_sort_is_byte_identical_to_the_old_one_on_the_samples():
    """Numeric ids read the old way (int) and the new way (str) sort into the same CSV
    text, and so do the committed CTX samples."""
    from pathlib import Path

    from aggregate_STAT import process_csv_file, sort_page_stats

    frames = [process_csv_file(p, STANDARD_COLS) for p in sorted(Path("data_samples/DOC_LINE_CATEG").glob("*.csv"))]
    frames = [f for f in frames if isinstance(f, pd.DataFrame)]
    if not frames:
        pytest.skip("no categorized samples")
    samples = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=7)
    assert sort_page_stats(samples).to_csv(index=False) == samples.sort_values(by=["file", "page_num"]).to_csv(
        index=False
    )
    numeric = pd.DataFrame({"file": [10, 9, 100, 9], "page_num": [1, 2, 1, 1], "v": [1, 2, 3, 4]})
    as_text = numeric.assign(file=numeric["file"].astype(str))
    assert sort_page_stats(as_text).to_csv(index=False) == numeric.sort_values(by=["file", "page_num"]).to_csv(
        index=False
    )
