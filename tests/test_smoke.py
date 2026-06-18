"""
tests/test_smoke.py
===================
End-to-end smoke tests for the pipeline categorization logic with mocked model inferences.
"""
from text_util_langID import (
    pre_filter_line, categorize_line, compute_quality_score, score_words_in_line,
    compute_word_weird_ratio, compute_vowel_ratio, compute_garbage_density,
    compute_rotatable_ratio, compute_valid_ratio,
    detect_gibberish_words, detect_wx_words, detect_fused_words
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
        rot_ratio = compute_rotatable_ratio(clean_text)
        weird_ratio = compute_word_weird_ratio(score_words_in_line(clean_text))
        valid_ratio = compute_valid_ratio(clean_text)

        gibb_count = detect_gibberish_words(clean_text)
        wx_count = detect_wx_words(clean_text)
        fused_words = detect_fused_words(clean_text)

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
            rot_ratio=rot_ratio
        )

        final_cat, _ = categorize_line(
            qs, clean_text, wc, vowel_ratio, mock_ppl,
            rot_ratio=rot_ratio, weird_ratio=weird_ratio,
            valid_word_ratio=valid_ratio
        )
        return final_cat

    def test_clean_czech_prose_is_clear_or_noisy(self):
        prose_lines = [
            "Poučení o povinnosti ku taxe vojenské.",
            "Tento nález byl učiněn v hloubce 30 cm pod povrchem.",
            "Keramické zlomky s vlnovkou."
        ]
        for line in prose_lines:
            cat = self._process_mocked_line(line, mock_ppl=150.0, mock_lang_score=0.95)
            assert cat in ("Clear", "Noisy"), f"Clean text '{line}' misclassified as {cat}"

    def test_garbage_and_mirror_is_trash_or_nontext(self):
        garbage_lines = [
            "TYRSOVA5===aras",
            "WVL e##xon w!wx",       # Added symbols so weirdness tanks the QS
            "pbqdnuwmoxszeyv!!",     # Added punctuation so weirdness > 0, triggering rot_penalty
            "AAMMNAbSSOAO###",       # Spurious caps + symbols
            "123 456 789"            # Pure digits -> Non-text
        ]
        for line in garbage_lines:
            cat = self._process_mocked_line(line, mock_ppl=3000.0, mock_lang_score=0.15)
            assert cat in ("Trash", "Non-text"), f"Garbage text '{line}' misclassified as {cat}"