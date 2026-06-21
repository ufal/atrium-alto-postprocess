"""
tests/test_smoke.py
===================
End-to-end smoke tests for the pipeline categorization logic with mocked model inferences.
"""

from text_util_langID import (
    analyze_rotation_signals,
    categorize_line,
    compute_garbage_density,
    compute_quality_score,
    compute_valid_ratio,
    compute_vowel_ratio,
    compute_word_weird_ratio,
    detect_fused_words,
    detect_gibberish_words,
    detect_wx_words,
    pre_filter_line,
    score_words_in_line,
)


class TestFullPipelineSmoke:
    def _process_mocked_line(self, line_text, mock_ppl, mock_lang_score):
        """Simulates the CPU/GPU orchestrator logic for a single line."""
        cat, clean_text = pre_filter_line(line_text)
        if cat != "Process":
            return cat

        wc = len(clean_text.split())
        cc = len(clean_text)
        original_text = line_text

        vowel_ratio = compute_vowel_ratio(original_text)
        g_density = compute_garbage_density(original_text)
        # rot_ratio = compute_rotatable_ratio(clean_text)
        weird_ratio = compute_word_weird_ratio(score_words_in_line(clean_text))
        valid_ratio = compute_valid_ratio(clean_text)

        gibb_count = detect_gibberish_words(clean_text)
        wx_count = detect_wx_words(clean_text)
        fused_words = detect_fused_words(clean_text)

        is_upright_czech, ghost_dominated = analyze_rotation_signals(clean_text)

        qs = compute_quality_score(
            valid_word_ratio=valid_ratio,
            perplexity=mock_ppl,
            text_length=cc,
            weird_ratio=weird_ratio,
            vowel_ratio=vowel_ratio,
            garbage_density=g_density,
            lang_score=mock_lang_score,
            gibberish_ratio=(gibb_count + wx_count) / max(wc, 1),
            fused_ratio=fused_words / max(wc, 1),
            is_upright_czech=is_upright_czech,
        )

        capped_lang_score = min(mock_lang_score, 0.75)
        final_cat, _ = categorize_line(
            qs,
            clean_text,
            wc,
            vowel_ratio,
            mock_ppl,
            weird_ratio=weird_ratio,
            valid_word_ratio=valid_ratio,
            lang_score=capped_lang_score,
            gibberish_present=(gibb_count + wx_count) > 0,
            is_upright_czech=is_upright_czech,
            ghost_dominated=ghost_dominated,
        )
        return final_cat

    def test_clean_czech_prose_is_clear_or_noisy(self):
        prose_lines = [
            "Poučení o povinnosti ku taxe vojenské.",
            "Tento nález byl učiněn v hloubce 30 cm pod povrchem.",
            "Keramické zlomky s vlnovkou.",
        ]
        for line in prose_lines:
            cat = self._process_mocked_line(line, mock_ppl=150.0, mock_lang_score=0.95)
            assert cat in ("Clear", "Noisy"), f"Clean text '{line}' misclassified as {cat}"

    def test_garbage_and_mirror_is_trash_or_nontext(self):
        garbage_lines = [
            "TYRSOVA5===aras",
            "WVL e##xon w!wx",  # Added symbols so weirdness tanks the QS
            "pbqdnuwmoxszeyv!!",  # Added punctuation so weirdness > 0, triggering rot_penalty
            "AAMMNAbSSOAO###",  # Spurious caps + symbols
            "123 456 789",  # Pure digits -> Non-text
        ]
        for line in garbage_lines:
            cat = self._process_mocked_line(line, mock_ppl=3000.0, mock_lang_score=0.15)
            assert cat in ("Trash", "Non-text"), f"Garbage text '{line}' misclassified as {cat}"

    # ────────────────────────────────────────────────────────────────────────
    # (#3) Real-data calibration fixtures.
    #
    # IMPORTANT boundary: only garbage that the PER-LINE path can route on its
    # own belongs here. Multi-token / interspersed inverted garbage (e.g.
    # "NU -", "e.ao u", "wL-U kyuto Cona JaaVHUoaAL") scores as Noisy in
    # isolation and is only reclassified by the page-level inverted-scan sweep —
    # those cases live in tests/test_page_postprocess.py, which exercises
    # apply_document_postprocessing. Asserting Trash for them here would be
    # dishonest about where the fix actually lives.
    # ────────────────────────────────────────────────────────────────────────
    def test_real_short_garbage_is_trash_per_line(self):
        # 'olie' -> short-garbage route; '° 47' -> plain quality-score Trash.
        for line in ["olie", "° 47"]:
            cat = self._process_mocked_line(line, mock_ppl=300.0, mock_lang_score=0.40)
            assert cat == "Trash", f"Short garbage '{line}' misclassified as {cat}"

    def test_real_clean_prose_promotes_to_clear(self):
        # Diacritic-rich Czech prose dense in short function words must reach Clear
        # now that compute_valid_ratio counts those short words (#3 C).
        clear_lines = [
            "svým jménem, nýbrž i lidovým podáním,které tvrdí,že v místech těchto stávala",
            "Pátral Jsem v první řadě po stříbrných penězích .které prý",
        ]
        for line in clear_lines:
            cat = self._process_mocked_line(line, mock_ppl=40.0, mock_lang_score=0.97)
            assert cat == "Clear", f"Clean prose '{line[:30]}…' misclassified as {cat}"

    def test_clean_czech_never_demoted_to_trash(self):
        # Regression guard for the short-garbage route: genuine short Czech must
        # never be Trashed.
        for line in ["Náčrt sondy.", "Praha", "kostra hrob náramek"]:
            cat = self._process_mocked_line(line, mock_ppl=200.0, mock_lang_score=0.97)
            assert cat != "Trash", f"Clean Czech '{line}' wrongly Trashed ({cat})"
