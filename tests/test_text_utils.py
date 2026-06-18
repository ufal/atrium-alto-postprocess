"""
tests/test_text_util.py
=======================
Unit tests for text_util_langID.py  —  all pure-Python, zero ML dependencies.
"""

import re
import random
import pytest

from text_util_langID import (
    compute_garbage_density, compute_symbol_ratio, compute_digit_ratio,
    compute_vowel_ratio, compute_rotatable_ratio, compute_valid_ratio,
    detect_strange_symbols, detect_repeated_chars, detect_gibberish_words,
    detect_letter_digit_letter, detect_mid_uppercase, detect_fused_words,
    detect_wx_words, is_all_caps_line, is_non_text, infer_lang_from_diacritics,
    pre_filter_line, parse_line_splits, score_word, score_words_in_line,
    compute_word_weird_ratio, compute_quality_score, determine_category,
    categorize_line, remap_lang,
    CATEG_TRASH_SCORE_MAX, CATEG_NOISY_SCORE_MAX, CLEAN_PROSE_MIN_SCORE,
    CLEAN_PROSE_WEIRD_MAX, CLEAN_PROSE_PPL_MAX, CLEAN_PROSE_WC_MIN,
)


from langID_classify import CSV_HEADER, _fast_track_row, _row_from_dict

def test_csv_header_and_fast_track_row_arity():
    """Asserts that the fast-track row builder exactly matches the global CSV_HEADER length."""
    row = _fast_track_row(
        file_id="CTX000001", page_id="1", line_num=1,
        clean_text="", original_text="",
        split_ws="", split_we="", categ="Empty"
    )
    assert len(row) == len(CSV_HEADER), \
        f"Fast-track row length ({len(row)}) does not match CSV_HEADER length ({len(CSV_HEADER)})."

def test_row_from_dict_covers_header_exactly():
    """Asserts that _row_from_dict enforces the exact column sequence and arity."""
    dummy_dict = {col: "test_val" for col in CSV_HEADER}
    try:
        main_row = _row_from_dict(dummy_dict)
        assert len(main_row) == len(CSV_HEADER)
        assert main_row[0] == dummy_dict[CSV_HEADER[0]] # Check sequence preservation
    except KeyError as e:
        pytest.fail(f"_row_from_dict failed due to missing key matching CSV_HEADER: {e}")



# To this (remove 'self'):
def test_score_word_respects_exemptions():
    # Academic titles should not receive a mid-uppercase penalty (score should be 0.0)
    assert score_word("PhDr.") == 0.0
    assert score_word("MUDr") == 0.0

    # Valid measurements should not receive an LDL penalty
    assert score_word("30cm") == 0.0
    assert score_word("90,9g") == 0.0

    # Backward letter-digit adjacency (OCR error) should be penalized
    assert score_word("vyt1") > 0.0

# Find this test in class TestDetectFusedWords:
def test_three_consecutive_vowels_triggers(self):
    # Change "krásnoočko" to an OCR fused string with 3+ vowels like "krásnoooučko"
    assert detect_fused_words("krásnoooučko") >= 1

# ════════════════════════════════════════════════════════════════════════════
# Densities and Ratios
# ════════════════════════════════════════════════════════════════════════════
class TestComputeGarbageDensity:
    def test_empty_string_returns_zero(self):
        assert compute_garbage_density("") == 0.0

    def test_space_only_string_returns_zero(self):
        assert compute_garbage_density("   ") == 0.0

    def test_clean_alphanumeric_text_returns_zero(self):
        assert compute_garbage_density("hello world 123") == 0.0

    def test_common_punctuation_not_counted_as_noise(self):
        assert compute_garbage_density("hello, world! (test) 1/2 a-b") == 0.0

    def test_hash_characters_counted_as_noise(self):
        assert compute_garbage_density("he##lo") > 0.0

    def test_dots_are_kept_chars_so_ellipsis_is_zero(self):
        assert compute_garbage_density("konec...") == 0.0


class TestComputeVowelRatio:
    def test_empty_returns_zero(self):
        assert compute_vowel_ratio("") == 0.0

    def test_no_alpha_returns_zero(self):
        assert compute_vowel_ratio("123!!!") == 0.0

    def test_pure_vowels_returns_one(self):
        assert compute_vowel_ratio("aeiou") == 1.0

    def test_pure_consonants_returns_zero(self):
        assert compute_vowel_ratio("bcdfg") == 0.0

    def test_symbols_included_in_denominator(self):
        # 'a' is vowel (1). '!' is symbol (1). Total denominator = 2.
        assert compute_vowel_ratio("a!") == 0.5

    def test_digits_excluded_from_denominator(self):
        # 'a' is vowel (1). '1' is digit (excluded). Total denominator = 1.
        assert compute_vowel_ratio("a1") == 1.0


class TestComputeRotatableRatio:
    def test_all_rotatable_returns_one(self):
        assert compute_rotatable_ratio("pbqd") == 1.0

    def test_no_rotatables_returns_zero(self):
        assert compute_rotatable_ratio("fghjkl") == 0.0


# ════════════════════════════════════════════════════════════════════════════
# Structural Detectors
# ════════════════════════════════════════════════════════════════════════════
class TestDetectStrangeSymbols:
    def test_clean_text_returns_zero(self):
        assert detect_strange_symbols("hello world") == 0

    def test_two_strange_chars_in_word_counted_each(self):
        assert detect_strange_symbols("he##lo") == 2


class TestDetectRepeatedChars:
    def test_clean_word_returns_zero(self):
        assert detect_repeated_chars("hello world") == 0

    def test_triple_consonant_repeat_triggers(self):
        assert detect_repeated_chars("ssset") >= 1

    def test_double_consonant_does_not_trigger_by_default(self):
        # "panna" has one double 'n' (2 total) - below guarded REPEATED_DOUBLE_MIN=3
        assert detect_repeated_chars("panna") == 0

    def test_digit_repeat_not_counted(self):
        assert detect_repeated_chars("1111") == 0


class TestDetectGibberishWords:
    def test_normal_word_returns_zero(self):
        assert detect_gibberish_words("hello world") == 0

    def test_no_vowels_does_not_trigger(self):
        # The low-vowel arm was removed; pure consonants are skipped
        assert detect_gibberish_words("bcdfg") == 0

    def test_all_caps_word_skipped(self):
        # 100% vowels but all-caps
        assert detect_gibberish_words("AAAAAAA") == 0

    def test_all_vowels_triggers_high_vowel_ratio(self):
        assert detect_gibberish_words("aaaaaaa") >= 1


class TestDetectLetterDigitLetter:
    def test_simple_ldl_pattern_detected(self):
        assert detect_letter_digit_letter("a1b") >= 1

    def test_measurement_units_not_ldl(self):
        assert detect_letter_digit_letter("30cm") == 0
        assert detect_letter_digit_letter("5mm") == 0
        assert detect_letter_digit_letter("90,9g") == 0

    def test_ocr_digit_insertion_catches_5x(self):
        assert detect_letter_digit_letter("5x") >= 1


class TestDetectMidUppercase:
    def test_initial_capital_not_mid_uppercase(self):
        assert detect_mid_uppercase("Praha") == 0

    def test_academic_titles_skipped(self):
        assert detect_mid_uppercase("PhDr.") == 0
        assert detect_mid_uppercase("MUDr") == 0

    def test_caps_prefix_lowercase_detected(self):
        assert detect_mid_uppercase("AAaaaa") >= 1


class TestDetectFusedWords:
    def test_token_longer_than_14_chars_triggers(self):
        assert detect_fused_words("aaaaaaaaaaaaaaaaaa") >= 1

    def test_three_consecutive_vowels_triggers(self):
        assert detect_fused_words("krásnoočko") >= 1

    def test_subtoken_split_prevents_hiding(self):
        # Internal dots split the token so consonant runs are measured safely
        assert detect_fused_words("str.nk") == 0


class TestDetectWxWords:
    def test_empty_returns_zero(self):
        assert detect_wx_words("") == 0

    def test_clean_returns_zero(self):
        assert detect_wx_words("hello") == 0

    def test_high_w_x_density_triggers(self):
        assert detect_wx_words("exxon") >= 1
        assert detect_wx_words("wwx") >= 1


class TestLangRemapFloor:
    def test_known_base_preserved(self):
        lbl, sc = remap_lang("deu_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "deu_Latn"
        assert sc == 0.4

    def test_unknown_remapped_and_floored(self):
        lbl, sc = remap_lang("fra_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.75

    def test_slk_relabelled_but_score_preserved(self):
        lbl, sc = remap_lang("slk_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.4


# ════════════════════════════════════════════════════════════════════════════
# Pre-filtering
# ════════════════════════════════════════════════════════════════════════════
class TestPreFilterLine:
    def test_empty_string_gives_empty(self):
        cat, text = pre_filter_line("")
        assert cat == "Empty"

    def test_pure_digits_gives_non_text(self):
        cat, _ = pre_filter_line("12345")
        assert cat == "Non-text"

    def test_symbol_letter_digit_gives_non_text(self):
        cat, _ = pre_filter_line("TYRSOVA5===")
        assert cat == "Non-text"

    def test_isolated_chars_gives_non_text(self):
        cat, _ = pre_filter_line("r n n 1")
        assert cat == "Non-text"

    def test_normal_czech_text_gives_process(self):
        cat, text = pre_filter_line("Tento řádek je normálně psaný text.")
        assert cat == "Process"


# ════════════════════════════════════════════════════════════════════════════
# Scoring and Category
# ════════════════════════════════════════════════════════════════════════════
class TestScoreWord:
    def test_single_common_letter_scores_zero(self):
        assert score_word("a") == 0.0

    def test_single_unknown_alpha_scores_high(self):
        # 'O' is not in SINGLE_CHAR_ALLOWED by default
        assert score_word("O") == 0.85
        assert score_word("o") == 0.85


class TestWordWeirdRatio:
    def test_clean_line_gives_zero_ratio(self):
        pairs = score_words_in_line("hello world text")
        assert compute_word_weird_ratio(pairs) == 0.0


class TestComputeValidRatio:
    def test_clean_czech_words_all_valid(self):
        assert compute_valid_ratio("kostra hrob náramek") == 1.0


class TestComputeQualityScore:
    def test_output_in_zero_one_range(self):
        q = compute_quality_score(
            valid_word_ratio=0.8, symbol_ratio=0.05,
            perplexity=200.0, text_length=50, weird_ratio=0.0,
        )
        assert 0.0 <= q <= 1.0


class TestCategorizeLineReason:
    def test_return_reason_gives_three_tuple(self):
        result = categorize_line(0.7, "some text here", 3, 0.4, 300.0, return_reason=True)
        assert len(result) == 3

    def test_clear_threshold_with_clamped_score(self):
        qs = CATEG_NOISY_SCORE_MAX + 0.02
        cat, score, reason = categorize_line(qs, "čistý text", 2, 0.4, 200.0, return_reason=True)
        assert cat == "Clear" and reason == "clear_threshold" and score >= CATEG_NOISY_SCORE_MAX