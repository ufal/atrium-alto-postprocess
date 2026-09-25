from unittest.mock import patch

import pandas as pd
import pytest

from classify_TEXT import apply_document_postprocessing


def test_apply_document_postprocessing_empty():
    """Ensure postprocessing handles empty DataFrames gracefully."""
    df = pd.DataFrame()
    res = apply_document_postprocessing(df)
    assert res.empty


def test_apply_document_postprocessing_dedup():
    """Test the header/footer deduplication logic.
    Identical text appearing multiple times should be harmonized to its modal category.
    """
    df = pd.DataFrame(
        {
            "page_num": [1, 1, 1],
            "line_num": [1, 2, 3],
            "text": ["Header Text", "Header Text", "Header Text"],
            "categ": ["Clear", "Trash", "Clear"],
            "quality_score": [0.95, 0.15, 0.92],
            "lang_score": [0.9, 0.1, 0.9],
            "original_lang": ["eng_Latn", "eng_Latn", "eng_Latn"],
            "perplex": [10.0, 10.0, 10.0],
            "rot_ratio": [0.0, 0.0, 0.0],
            "word_weird": [0.0, 0.0, 0.0],
        }
    )

    res = apply_document_postprocessing(df)

    # 'Clear' is the majority mode, so the 'Trash' row should be upgraded
    assert (res["categ"] == "Clear").all()
    # Ensure the postprocessing flag was recorded
    assert res.loc[1, "pp_dedup"]


@patch("classify_TEXT.pd.read_csv")
@patch("classify_TEXT.configparser.ConfigParser")
def test_main_graceful_exit(mock_cfg_class, mock_read_csv):
    """Test that main initializes but fails safely if the input CSV is missing."""
    mock_cfg = mock_cfg_class.return_value
    mock_cfg.get.return_value = "dummy_value"
    mock_cfg.getint.return_value = 1
    mock_cfg.getfloat.return_value = 1.0

    # Simulate the input CSV missing during the main loop bootup
    mock_read_csv.side_effect = FileNotFoundError("Missing INPUT_CSV")

    from classify_TEXT import main

    with pytest.raises(FileNotFoundError, match="Missing INPUT_CSV"):
        main([])


def test_load_page_index_keeps_file_ids_as_strings(tmp_path):
    """(#31) `0001` must not become 1 and `NA` must not become NaN — the page files of
    such a document were never found and it was skipped in silence."""
    from classify_TEXT import load_page_index

    csv_path = tmp_path / "stats.csv"
    csv_path.write_text("file,page,path\n0001,1,a\nNA,2,b\nx-y,10,c\nCTX000000001,,d\n", encoding="utf-8")
    df = load_page_index(csv_path)
    assert list(df["file"]) == ["0001", "NA", "x-y", "CTX000000001"]
    assert list(df["page"][:3]) == [1, 2, 10]
    assert pd.isna(df["page"].iloc[3])  # empty cells stay NaN, as before


# ── (#31 Phase 4) classify's re-read of its own per-document CSV ──────────────


def test_read_doc_line_csv_keeps_the_document_id_and_the_text_na_semantics(tmp_path):
    from classify_TEXT import read_doc_line_csv

    path = tmp_path / "0001.csv"
    path.write_text("file,page_num,line_num,text,categ\n0001,1,1,NA,Clear\n0001,1,2,Ahoj,Clear\n", encoding="utf-8")
    df = read_doc_line_csv(path, "0001")
    assert list(df["file"]) == ["0001", "0001"]
    assert pd.isna(df["text"][0]) and df["text"][1] == "Ahoj"  # text "NA" is still NaN, exactly as before
    na = tmp_path / "NA.csv"
    na.write_text("file,page_num,line_num,text\nNA,1,1,x\n", encoding="utf-8")
    assert list(read_doc_line_csv(na, "NA")["file"]) == ["NA"]


@pytest.mark.parametrize("folder", ["data_samples/DOC_LINE_CATEG", "data_samples/DOC_LINE_CATEG_gpt"])
def test_read_doc_line_csv_is_byte_identical_on_the_samples(folder):
    """The committed categorized samples round-trip exactly as the old inline read did."""
    from pathlib import Path

    from classify_TEXT import read_doc_line_csv

    old_dtypes = {"text": str, "original_text": str, "split_ws": str, "split_we": str, "lang": str,
                  "original_lang": str, "categ": str}  # fmt: skip
    paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        pytest.skip(f"no samples in {folder}")
    for path in paths:
        old = pd.read_csv(path, dtype=old_dtypes).to_csv(index=False)
        assert read_doc_line_csv(path, path.stem).to_csv(index=False) == old
