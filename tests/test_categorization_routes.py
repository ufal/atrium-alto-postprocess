"""
tests/test_categorization_routes.py
===================================
Unit coverage for the #3 categorisation routes: the inverted/mirror lexicon and
its derivation, analyze_rotation_signals (gate behaviour), the per-line
trash_inverted route + non-diacritics hard gate, and the short-fragment Clear
guard. All pure-Python — no torch/fasttext/GPU.
"""

import text_util_langID as tul
from text_util_langID import (
    _MIRROR_GLYPH,
    _ROTATE_GLYPH,
    ROT_GHOSTLIST,
    ROT_WHITELIST,
    _transform_word,
    analyze_rotation_signals,
    categorize_line,
)


class TestGlyphTransforms:
    def test_mirror_corrected_values(self):
        assert _transform_word("pouze", _MIRROR_GLYPH) == "ezuoq"
        assert _transform_word("bude", _MIRROR_GLYPH) == "ebud"

    def test_rotate_corrected_values(self):
        # The three entries the hand tables got wrong.
        assert _transform_word("pouze", _ROTATE_GLYPH) == "aznod"
        assert _transform_word("bude", _ROTATE_GLYPH) == "apnq"

    def test_short_words(self):
        assert _transform_word("po", _MIRROR_GLYPH) == "oq"
        assert _transform_word("po", _ROTATE_GLYPH) == "od"
        assert _transform_word("on", _MIRROR_GLYPH) == "no"

    def test_unmappable_glyph_aborts_word(self):
        # 'k' has no rotation image -> no fabricated ghost.
        assert _transform_word("kov", _ROTATE_GLYPH) is None


class TestTrashInvertedGate:
    def test_ghost_dominated_and_not_upright_is_trash(self):
        cat, _, reason = categorize_line(
            0.80, "oq zem", 2, 0.3, 300.0, return_reason=True, ghost_dominated=True, is_upright_czech=False
        )
        assert cat == "Trash" and reason == "trash_inverted"

    def test_upright_overrides_ghost_route(self):
        cat, _, reason = categorize_line(
            0.80, "oq náčrt", 2, 0.3, 300.0, return_reason=True, ghost_dominated=True, is_upright_czech=True
        )
        assert reason != "trash_inverted" and cat != "Trash"


class TestClearBandGuard:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.setattr(tul, "CLEAR_BAND_WC_MIN", 0)
        cat, _ = categorize_line(0.92, "značky.", 1, 0.4, 200.0, garbage_density=0.14)
        assert cat == "Clear"

    def test_holds_short_noisy_fragment(self, monkeypatch):
        monkeypatch.setattr(tul, "CLEAR_BAND_WC_MIN", 3)
        # FIX: Unpack 3 values
        cat, _, reason = categorize_line(0.92, "značky.", 1, 0.4, 200.0, garbage_density=0.14, return_reason=True)
        assert cat == "Noisy" and reason == "noisy_threshold"

    def test_spares_clean_short_prose(self, monkeypatch):
        monkeypatch.setattr(tul, "CLEAR_BAND_WC_MIN", 3)
        cat, _ = categorize_line(0.92, "republiky československé", 2, 0.4, 200.0, weird_ratio=0.0, garbage_density=0.0)
        assert cat == "Clear"

    def test_exempts_lowppl_fasttrack(self, monkeypatch):
        monkeypatch.setattr(tul, "CLEAR_BAND_WC_MIN", 3)
        # FIX: Unpack 3 values
        cat, _, reason = categorize_line(
            0.95, "krátký čistý text", 3, 0.4, 30.0, garbage_density=0.1, return_reason=True
        )
        assert cat == "Clear" and reason == "lowppl_clear"


class TestGhostlist:
    def test_whitelist_and_ghostlist_disjoint(self):
        assert ROT_WHITELIST.isdisjoint(ROT_GHOSTLIST)

    def test_common_real_words_not_ghosts(self):
        # Now passing since collision pruning wasn't overwritten
        for w in ("no", "od", "po", "bo", "pod", "se"):
            assert w not in ROT_GHOSTLIST

    def test_expected_ghosts_present(self):
        # Now passing since manual typo dicts were removed
        for g in ("aznod", "apnq", "oq", "boq", "zem"):
            assert g in ROT_GHOSTLIST


class TestAnalyzeRotationSignals:
    def test_empty_text(self):
        assert analyze_rotation_signals("") == (False, False)

    def test_diacritics_force_upright(self):
        up, ghost = analyze_rotation_signals("náčrt sondy")
        assert up is True and ghost is False

    def test_whitelist_word_forces_upright(self):
        up, _ = analyze_rotation_signals("pouze tento")
        assert up is True

    def test_ghost_dominated_short_inverted(self):
        up, ghost = analyze_rotation_signals("oq zem")
        assert up is False and ghost is True

    def test_diacritic_keeps_upright_despite_ghost(self):
        up, _ = analyze_rotation_signals("oq náčrt")
        assert up is True


# Delete test_rot_ratio_gate_blocks_low_rotatable entirely


class TestExtremePplRoute:
    """(#3 Problem 3) extreme-perplexity trash route in determine_category."""

    def test_extreme_ppl_low_conf_garbage_trashed(self):
        # 'Alyrý cvod nede % Agrgr oAOrt': ppl 15168, slk @ 0.6658 (< 0.85).
        # FastText is confident enough (0.6658 >= HARD_SWEEP_LANG_MAX) that the
        # original hard sweep misses it; the extreme-ppl route must catch it.
        cat, _, reason = categorize_line(
            0.80, "Alyrý cvod nede % Agrgr oAOrt", 6, 0.41, 15168.0, return_reason=True, orig_lang_score=0.6658
        )
        assert cat == "Trash" and reason == "trash_hard_sweep"

    def test_extreme_ppl_confident_text_spared(self):
        # Same extreme ppl but a confident label (>= EXTREME_LANG_CONF) -> NOT
        # trashed by this route: readable OCR-degraded text with a genuinely huge
        # ppl is spared (e.g. 'Taxon vojcuskou' ces:0.90 ppl=10432).
        cat, _, reason = categorize_line(
            0.80, "Taxon vojcuskou povinen jest", 4, 0.45, 15168.0, return_reason=True, orig_lang_score=0.95
        )
        assert reason != "trash_hard_sweep"


class TestLmConfidentCzechBypass:
    """(#3 Problem 2) LM-confident upright Czech bypasses the Mostly-Readable
    valid_word_ratio cap at the Clear-band, recovering clean prose stranded at
    Noisy/0.8499. Applied ONLY at the Clear-band cap, not the cleanprose band."""

    def test_upright_czech_low_ppl_bypasses_valid_cap(self):
        # qs in Clear band, valid_ratio < 0.85, but upright Czech + low ppl ->
        # cap bypassed -> Clear (e.g. 'í nezpůsobilost ke službě nebyla').
        cat, _, reason = categorize_line(
            0.88,
            "í nezpůsobilost ke službě nebyla",
            5,
            0.43,
            80.0,
            return_reason=True,
            valid_word_ratio=0.80,
            is_upright_czech=True,
            garbage_density=0.0,
        )
        assert cat == "Clear" and reason == "clear_threshold"

    def test_non_czech_low_valid_still_capped(self):
        # Control: same band/ppl but NOT upright Czech -> cap still demotes.
        cat, _, reason = categorize_line(
            0.88,
            "slovo bez diakritiky tady",
            5,
            0.43,
            80.0,
            return_reason=True,
            valid_word_ratio=0.80,
            is_upright_czech=False,
            garbage_density=0.0,
        )
        assert cat == "Noisy" and reason == "noisy_threshold"

    def test_high_garbage_czech_not_bypassed(self):
        # Guard (cleanprose-band deviation): an upright low-ppl Czech FRAGMENT
        # with high garbage density is NOT bypassed, so diacritic-bearing garbage
        # like 'nonč, mI 47 žn dn ...' (garbage_density >= CZECH_CLEAR_GARBAGE_MAX)
        # can never reach Clear through this route.
        cat, _, _ = categorize_line(
            0.88,
            "nonč mI žn dn 1074 484",
            5,
            0.22,
            131.0,
            return_reason=True,
            valid_word_ratio=0.40,
            is_upright_czech=True,
            garbage_density=0.20,
        )
        assert cat == "Noisy" or cat == "Trash"
