from unittest.mock import patch

import pandas as pd
import pytest

from langID_classify import apply_document_postprocessing


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


@patch("langID_classify.pd.read_csv")
@patch("langID_classify.configparser.ConfigParser")
def test_main_graceful_exit(mock_cfg_class, mock_read_csv):
    """Test that main initializes but fails safely if the input CSV is missing."""
    mock_cfg = mock_cfg_class.return_value
    mock_cfg.get.return_value = "dummy_value"
    mock_cfg.getint.return_value = 1
    mock_cfg.getfloat.return_value = 1.0

    # Simulate the input CSV missing during the main loop bootup
    mock_read_csv.side_effect = FileNotFoundError("Missing INPUT_CSV")

    from langID_classify import main

    with pytest.raises(FileNotFoundError, match="Missing INPUT_CSV"):
        main()
