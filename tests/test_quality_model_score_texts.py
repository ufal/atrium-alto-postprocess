"""
tests/test_quality_model_score_texts.py
========================================
Fast, model-free tests for ``build_line_record`` — the faithful mirror of the
production per-line scoring orchestration (issue #23, Phase 0).

``build_line_record`` takes the FastText prediction and perplexity as plain
arguments, so it exercises the real ``text_util_langID`` engine end-to-end without
loading FastText or Qwen. The heavy model-loading helpers in ``score_texts`` are
NOT imported here (they lazy-import torch / transformers / fasttext).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_QM = _ROOT / "tools" / "quality_model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

import score_texts as S  # noqa: E402

EXPECTED = ["ces", "deu", "eng"]
TRUSTED = ["deu", "eng", "fra", "pol", "ita", "slk"]


def _score(text: str, lang: str = "ces_Latn", lang_score: float = 0.99, ppl: float = 80.0) -> dict:
    return S.build_line_record(text, lang, lang_score, ppl, EXPECTED, TRUSTED)


def _band_invariant_holds(rec: dict) -> bool:
    """The clamped score must sit inside its category band (categorize_line contract)."""
    categ, s = rec["categ"], rec["score_clamped"]
    if categ == "Trash":
        return s < 0.55
    if categ == "Noisy":
        return 0.55 <= s < 0.80
    if categ == "Clear":
        return s >= 0.80
    return s == 0.0  # Empty / Non-text


# ── Basic contract ─────────────────────────────────────────────────────────


def test_empty_line_is_empty_category_zero_score():
    rec = _score("")
    assert rec["categ"] == "Empty"
    assert rec["score_raw"] == 0.0
    assert rec["score_clamped"] == 0.0


def test_whitespace_only_line_is_empty():
    rec = _score("    ")
    assert rec["categ"] == "Empty"


def test_score_raw_is_bounded_unit_interval():
    rec = _score("Mocnost kulturní vrstvy činila 40 cm.")
    assert 0.0 <= rec["score_raw"] <= 1.0
    assert 0.0 <= rec["score_clamped"] <= 1.0


def test_band_invariant_across_examples():
    samples = [
        "Mocnost kulturní vrstvy činila 40 cm.",
        "Eva Procházková dokumentovala nálezy.",
        "rnn1 ww0rd vv_~~ qpqb dbqp",
        "",
        "A123/2024",
    ]
    for text in samples:
        rec = _score(text)
        assert _band_invariant_holds(rec), (text, rec["categ"], rec["score_clamped"])


def test_returns_all_declared_columns():
    rec = _score("Popis nálezu v sondě.")
    for col in S.SCORE_COLUMNS:
        assert col in rec


# ── Signal sanity ──────────────────────────────────────────────────────────


def test_clean_prose_outscores_garbage():
    clean = _score("Mocnost kulturní vrstvy činila 40 centimetrů.")
    garbage = _score("rnn1 ww0rd vv_~~ qpqb dbqp uunn")
    assert clean["score_raw"] > garbage["score_raw"]


def test_short_line_perplexity_is_capped():
    # <=2 words with a huge perplexity must be capped at SHORT_PPL_CAP.
    import text_util_langID as tu

    rec = S.build_line_record("Literatura", "ces_Latn", 0.9, 99999.0, EXPECTED, TRUSTED)
    assert rec["perplex"] <= tu.SHORT_PPL_CAP


def test_build_line_record_is_pure_function():
    args = ("Mocnost kulturní vrstvy činila 40 cm.", "ces_Latn", 0.99, 80.0, EXPECTED, TRUSTED)
    assert S.build_line_record(*args) == S.build_line_record(*args)


# ── Cross-module: corrupting a clean line lowers its raw score ──────────────


def test_corruption_lowers_raw_score_at_fixed_perplexity():
    """With FastText/perplexity held fixed, heavier text corruption must not raise
    the raw quality score — ties the corruption engine to the scorer without models."""
    import random

    import corrupt as C

    src = "Mocnost kulturní vrstvy činila čtyřicet centimetrů v sondě."
    clean = S.build_line_record(src, "ces_Latn", 0.99, 80.0, EXPECTED, TRUSTED)

    heavy = C.corrupt_line(
        src,
        random.Random(11),
        band="heavy",
        ops=["symbol_injection", "char_confusion", "vowel_strip"],
    )
    corrupted = S.build_line_record(heavy.text, "ces_Latn", 0.99, 80.0, EXPECTED, TRUSTED)

    assert corrupted["score_raw"] <= clean["score_raw"]
