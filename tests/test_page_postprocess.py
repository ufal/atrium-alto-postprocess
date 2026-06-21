"""
tests/test_page_postprocess.py
==============================
Unit tests for langID_classify.apply_document_postprocessing — the pure, GPU-free
document-level smoothing helper extracted in #3 (A3). The heavy GPU/ML imports
(``torch``, ``transformers``, ``fasttext``) are stubbed so this suite runs in the
fast lane without the model stack, exactly as the orchestrator's other hermetic
tests do.

It covers the part of the calibration that the per-line smoke path CANNOT reach:
multi-token / interspersed inverted garbage that only the page-level sweep
reclassifies, plus the run-based fallback and the no-op guarantees on clean pages.
"""

import sys
import types

# --- stub the GPU/ML stack BEFORE importing langID_classify -------------------
# Note: atrium_paradata was removed from this stub list. It has no ML dependencies
# and can be imported safely without polluting the test environment for test_paradata.py.
for _name in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

import pandas as pd  # noqa: E402

from langID_classify import (  # noqa: E402
    CSV_HEADER,
    INVERTED_PAGE_MAJORITY,
    INVERTED_RUN_MIN,
    apply_document_postprocessing,
)
from text_util_langID import LANG_SCORE_ROUGH  # noqa: E402

_LOW = LANG_SCORE_ROUGH - 0.10  # below the rough gate -> low_lang True
_HIGH = 0.95


def _row(page, line, categ, text, lang_score=_HIGH, rot=0.0, ppl=30.0, qs=0.80, word_weird=0.0):
    """Build a full CSV-schema row dict with sensible clean defaults."""
    d = {c: "" for c in CSV_HEADER}
    d.update(
        {
            "categ": categ,
            "quality_score": f"{qs:.4f}",
            "file": "DOC",
            "page_num": page,
            "line_num": line,
            "text": text,
            "original_text": text,
            "split_ws": "",
            "split_we": "",
            "lang": "ces_Latn",
            "lang_score": f"{lang_score:.4f}",
            "original_lang": "ces_Latn",
            "orig_lang_score": f"{lang_score:.4f}",
            "perplex": f"{ppl:.2f}",
            "word_count": len(text.split()),
            "char_count": len(text),
            "garbage_density": "0.0",
            "upper": 0,
            "repeated": 0,
            "ldl_fuses": 0,
            "fused_words": 0,
            "gibberish": 0,
            "weird_wx": 0,
            "word_weird": f"{float(word_weird):.4f}",
            "vowel_ratio": "0.4",
            "rot_ratio": f"{rot:.4f}",
            "caps_header": False,
            "allcaps_novowel": False,
            "lowppl_clear": False,
            "cleanprose_clear": False,
            "trash_threshold": False,
            "noisy_threshold": False,
            "clear_threshold": False,
            "pp_dedup": False,
            "pp_surrounded_trash": False,
            "pp_inverted_run": False,
        }
    )
    return d


def _df(rows):
    return pd.DataFrame(rows)


class TestPageMajoritySweep:
    """(#3 A3) page-MAJORITY arm catches interspersed garbage with no long run."""

    def test_interspersed_garbage_trashed_by_majority(self):
        # 6 scoreable lines, 4 suspicious (66% >= 60%), but broken up by
        # Empty/Non-text so the longest contiguous suspicious run is only 2 ->
        # the run rule alone would NOT fire; the majority arm must.
        rows = [
            _row("1", 1, "Noisy", "wL-U kyuto", lang_score=_LOW),
            _row("1", 2, "Noisy", "Cona JaaVH", lang_score=_LOW),
            _row("1", 3, "Empty", ""),
            _row("1", 4, "Noisy", "e ao u xqz", lang_score=_LOW),
            _row("1", 5, "Non-text", "---"),
            _row("1", 6, "Noisy", "zzx qwp lm", lang_score=_LOW),
            _row("1", 7, "Clear", "Náčrt sondy", lang_score=_HIGH),
            _row("1", 8, "Clear", "Praha mesto", lang_score=_HIGH),
        ]
        out = apply_document_postprocessing(_df(rows))
        cats = dict(zip(out["text"], out["categ"], strict=False))
        flags = dict(zip(out["text"], out["pp_inverted_run"], strict=False))
        for g in ("wL-U kyuto", "Cona JaaVH", "e ao u xqz", "zzx qwp lm"):
            assert cats[g] == "Trash", f"{g!r} should be majority-Trashed"
            assert bool(flags[g]) is True
        # Clean lines must survive: the diacritic line and the high-lang line.
        assert cats["Náčrt sondy"] == "Clear"
        assert cats["Praha mesto"] == "Clear"

    def test_high_lang_line_not_suspicious(self):
        # A no-diacritics line is NOT suspicious if FastText is confident (high
        # stored lang score) — only low_lang + no_diacs together qualify.
        rows = [_row("1", i, "Clear", f"slovo bez diakritiky {i}", lang_score=_HIGH) for i in range(1, 6)]
        out = apply_document_postprocessing(_df(rows))
        assert (out["categ"] == "Clear").all()
        assert not out["pp_inverted_run"].any()


class TestRunBasedSweep:
    """The contiguous-run fallback still fires on mixed pages below majority."""

    def test_contiguous_run_trashed_below_majority(self):
        # 12 scoreable lines, only 4 suspicious and contiguous -> 33% < 60%,
        # so the majority arm must NOT fire but the run >= 4 rule must.
        rows = [_row("2", i, "Clear", f"cisty cesky radek {i}", lang_score=_HIGH) for i in range(1, 9)]
        rows += [_row("2", 9 + j, "Noisy", f"qzxw{j} plmk vbnm", lang_score=_LOW) for j in range(INVERTED_RUN_MIN)]
        out = apply_document_postprocessing(_df(rows))
        suspicious = out[out["text"].str.startswith("qzxw")]
        assert (suspicious["categ"] == "Trash").all()
        assert suspicious["pp_inverted_run"].all()
        # Clean majority untouched.
        assert (out[out["text"].str.startswith("cisty")]["categ"] == "Clear").all()

    def test_short_run_below_threshold_survives(self):
        # Only 3 contiguous suspicious lines (< INVERTED_RUN_MIN) on an otherwise
        # clean, large page -> neither arm fires.
        rows = [_row("3", i, "Clear", f"radek cislo {i} text", lang_score=_HIGH) for i in range(1, 9)]
        rows += [_row("3", 9 + j, "Noisy", f"zzz{j} qqq www", lang_score=_LOW) for j in range(INVERTED_RUN_MIN - 1)]
        out = apply_document_postprocessing(_df(rows))
        survivors = out[out["text"].str.startswith("zzz")]
        assert (survivors["categ"] == "Noisy").all()
        assert not survivors["pp_inverted_run"].any()


class TestRotationArm:
    """Ensures the rotation arm of the page-level sweep requires corroborating signals."""

    def test_genuine_inversion_trashed(self):
        # High rot, high ppl, weirdness > 0, low/mid lang score -> forms a valid run
        rows = [_row("7", i, "Clear", f"cisty cesky radek {i}", lang_score=_HIGH) for i in range(1, 5)]
        rows += [
            _row("7", 4 + j, "Noisy", f"uonn qpqb dbqp {j}", lang_score=_LOW, rot=0.60, ppl=250.0, word_weird=0.50)
            for j in range(INVERTED_RUN_MIN)
        ]
        out = apply_document_postprocessing(_df(rows))
        suspicious = out[out["text"].str.startswith("uonn")]
        assert (suspicious["categ"] == "Trash").all()
        assert suspicious["pp_inverted_run"].all()

    def test_clean_prose_survives_rotation_false_positive(self):
        # Clean prose (mimicking the lines 68, 73, 75 false positives)
        # They have high rot & ppl, but are saved by missing weirdness or having high lang score
        rows = [_row("8", i, "Clear", f"jasny text cislo {i}", lang_score=_HIGH) for i in range(1, 5)]
        rows += [
            _row(
                "8",
                5,
                "Clear",
                "všech obcích vyvéšeny jsou.",
                lang_score=1.0000,
                rot=0.60,
                ppl=250.0,
                qs=0.90,
                word_weird=0.0,
            ),
            _row(
                "8",
                6,
                "Clear",
                ". až 50 korun; v pádu nedobytnosti vězením.",
                lang_score=0.9814,
                rot=0.60,
                ppl=250.0,
                qs=0.83,
                word_weird=0.02,
            ),
            _row(
                "8",
                7,
                "Clear",
                "eni - trestá so pokutou penéžitou",
                lang_score=0.6824,
                rot=0.60,
                ppl=250.0,
                qs=0.74,
                word_weird=0.0,
            ),
            _row(
                "8",
                8,
                "Clear",
                "dalsi normalni radek s rotacnimi znaky",
                lang_score=0.9500,
                rot=0.60,
                ppl=250.0,
                qs=0.85,
                word_weird=0.0,
            ),
        ]
        out = apply_document_postprocessing(_df(rows))
        fp_lines = out[out["line_num"].isin([5, 6, 7, 8])]

        assert (fp_lines["categ"] == "Clear").all()
        assert not fp_lines["pp_inverted_run"].any()


class TestDedupAndNoOp:
    def test_identical_text_harmonised_to_modal(self):
        rows = [
            _row("4", 1, "Clear", "OPAKUJÍCÍ ZÁHLAVÍ"),
            _row("4", 2, "Clear", "OPAKUJÍCÍ ZÁHLAVÍ"),
            _row("4", 3, "Noisy", "OPAKUJÍCÍ ZÁHLAVÍ"),
            _row("4", 4, "Clear", "jiný řádek"),
        ]
        out = apply_document_postprocessing(_df(rows))
        header = out[out["text"] == "OPAKUJÍCÍ ZÁHLAVÍ"]
        assert (header["categ"] == "Clear").all()
        assert header["pp_dedup"].sum() == 1  # only the flipped row flagged

    def test_clean_page_untouched(self):
        rows = [_row("5", i, "Clear", f"jasný český text číslo {i}", lang_score=_HIGH) for i in range(1, 7)]
        out = apply_document_postprocessing(_df(rows))
        assert (out["categ"] == "Clear").all()
        assert not out["pp_inverted_run"].any()
        assert not out["pp_dedup"].any()
        assert not out["pp_surrounded_trash"].any()

    def test_empty_frame_safe(self):
        out = apply_document_postprocessing(pd.DataFrame(columns=CSV_HEADER))
        assert out.empty

    def test_output_keeps_canonical_columns(self):
        rows = [_row("6", 1, "Clear", "text"), _row("6", 2, "Clear", "text2")]
        out = apply_document_postprocessing(_df(rows))
        for col in ("pp_dedup", "pp_surrounded_trash", "pp_inverted_run"):
            assert col in out.columns


class TestRotArmWeirdless:
    """(#3 Problem 1) the rot arm catches no-diacritics, structurally rotatable,
    LM-lost lines that the WEIRD arm misses because word_weird == 0 — the exact
    gap that let noxer/uuysod through. Isolated here with a MID lang score
    (between LANG_SCORE_ROUGH and ROT_HIGH_LANG_CONF) so the diacritic-absence
    arm does NOT fire: only the rot arm can be responsible."""

    _MID = 0.50  # > LANG_SCORE_ROUGH (0.45), < ROT_HIGH_LANG_CONF (0.90)

    def test_weirdless_rot_run_trashed_by_rot_arm(self):
        # 4 clean + 4 suspicious = 50% < majority, so the page-MAJORITY arm does
        # NOT fire; the contiguous run >= INVERTED_RUN_MIN does, via the rot arm.
        # word_weird=0 disqualifies the weird arm; _MID lang disqualifies the
        # diacritic-absence arm -> only (no_diacs & high_rot & high_ppl &
        # ~high_lang_conf) remains.
        rows = [_row("9", i, "Clear", f"cisty cesky radek {i}", lang_score=_HIGH) for i in range(1, 5)]
        rows += [
            _row("9", 4 + j, "Noisy", f"uonn qpqb dbqp {j}", lang_score=self._MID, rot=0.60, ppl=250.0, word_weird=0.0)
            for j in range(INVERTED_RUN_MIN)
        ]
        out = apply_document_postprocessing(_df(rows))
        suspicious = out[out["text"].str.startswith("uonn")]
        assert (suspicious["categ"] == "Trash").all()
        assert suspicious["pp_inverted_run"].all()
        assert (out[out["text"].str.startswith("cisty")]["categ"] == "Clear").all()

    def test_high_lang_conf_rotatable_line_not_trashed(self):
        # A no-diacritics, high-rot, high-ppl line is SPARED when FastText is
        # confident (lang_score >= ROT_HIGH_LANG_CONF): ~high_lang_conf is False
        # so the rot arm cannot fire (guards confident upright ASCII Czech such
        # as 'dalsi normalni radek').
        rows = [_row("10", i, "Clear", f"jasny cesky radek {i}", lang_score=_HIGH) for i in range(1, 5)]
        rows += [
            _row("10", 4 + j, "Clear", f"uonn qpqb dbqp znak {j}", lang_score=0.95, rot=0.60, ppl=250.0, word_weird=0.0)
            for j in range(INVERTED_RUN_MIN)
        ]
        out = apply_document_postprocessing(_df(rows))
        assert (out["categ"] == "Clear").all()
        assert not out["pp_inverted_run"].any()


def test_module_constants_sane():
    assert INVERTED_RUN_MIN >= 2
    assert 0.0 < INVERTED_PAGE_MAJORITY <= 1.0
