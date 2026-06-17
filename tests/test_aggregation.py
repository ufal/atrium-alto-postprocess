"""
Tests for langID_aggregate_STAT.py pure-logic helpers.
"""
import pandas as pd
import numpy as np
from langID_aggregate_STAT import _sum_metrics


def test_sum_metrics_basic():
    # Mock a dataframe representing a DOC_LINE_CATEG CSV
    df = pd.DataFrame({
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
        "caps_header": [False, False, False]
    })

    res = _sum_metrics(df)

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
    res = _sum_metrics(df)
    assert res.empty