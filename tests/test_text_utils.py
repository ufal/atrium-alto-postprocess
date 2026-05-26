"""
tests/test_text_util.py
=======================
Unit tests for text_util_langID.py  —  all pure-Python, zero ML dependencies.

Design notes
------------
* No ML models, no GPU, no network required.
* Weights and thresholds are read back from the module itself (not hardcoded
  in assertions) so the suite stays valid if someone adjusts config_langID.txt.
* Tests are ordered from atomic helpers up to the top-level categoriser.
"""

import re
import random
import pytest

from text_util_langID import (
    # ── ratio / density helpers ────────────────────────────────────────────
    compute_garbage_density,
    compute_symbol_ratio,
    compute_digit_ratio,
    compute_vowel_ratio,
    compute_rotatable_ratio,
    compute_valid_ratio,
    # ── structural detectors ───────────────────────────────────────────────
    detect_strange_symbols,
    detect_repeated_chars,
    detect_gibberish_words,
    detect_letter_digit_letter,
    detect_mid_uppercase,
    detect_fused_words,
    is_all_caps_line,
    is_non_text,
    infer_lang_from_diacritics,
    # ── pre-processing ─────────────────────────────────────────────────────
    pre_filter_line,
    parse_line_splits,
    # ── per-word scoring ───────────────────────────────────────────────────
    score_word,
    score_words_in_line,
    compute_word_weird_ratio,
    # ── composite score + categoriser ─────────────────────────────────────
    compute_quality_score,
    categorize_line,
    # ── exported thresholds (read back so tests survive config changes) ────
    CATEG_TRASH_SCORE_MAX,
    CATEG_NOISY_SCORE_MAX,
    CLEAN_PROSE_MIN_SCORE,
    CLEAN_PROSE_WEIRD_MAX,
    CLEAN_PROSE_PPL_MAX,
    CLEAN_PROSE_WC_MIN,
)


# ════════════════════════════════════════════════════════════════════════════
# compute_garbage_density
# ════════════════════════════════════════════════════════════════════════════
class TestComputeGarbageDensity:

    def test_empty_string_returns_zero(self):
        assert compute_garbage_density("") == 0.0

    def test_space_only_string_returns_zero(self):
        # spaces are in the excluded set for noise counting → density = 0
        assert compute_garbage_density("   ") == 0.0

    def test_clean_alphanumeric_text_returns_zero(self):
        assert compute_garbage_density("hello world 123") == 0.0

    def test_common_punctuation_not_counted_as_noise(self):
        # The noise-excluded set is ' ,.?!()/-'; all of these are clean
        assert compute_garbage_density("hello, world! (test) 1/2 a-b") == 0.0

    def test_hash_characters_counted_as_noise(self):
        # '#' is not in the excluded set
        density = compute_garbage_density("he##lo")
        assert density > 0.0

    def test_tilde_counted_as_noise(self):
        density = compute_garbage_density("normal~text")
        assert density > 0.0

    def test_ellipsis_stripped_before_calculation(self):
        # The function removes "..." runs first; remaining text is clean
        assert compute_garbage_density("konec...") == 0.0

    def test_density_strictly_bounded(self):
        for text in ["abc", "###", "a#b#c#", "", "hello world"]:
            d = compute_garbage_density(text)
            assert 0.0 <= d <= 1.0, f"out of range for {text!r}: {d}"

    def test_more_noise_produces_higher_density(self):
        low  = compute_garbage_density("hello world text")
        high = compute_garbage_density("h#l#o w#r#d t#x#")
        assert high > low


# ════════════════════════════════════════════════════════════════════════════
# compute_symbol_ratio
# ════════════════════════════════════════════════════════════════════════════
class TestComputeSymbolRatio:

    def test_empty_returns_zero(self):
        assert compute_symbol_ratio("") == 0.0

    def test_pure_alpha_returns_zero(self):
        assert compute_symbol_ratio("hello") == 0.0

    def test_non_space_symbol_contributes(self):
        # The implementation excludes spaces (`not c.isspace()`), so a trailing
        # space does NOT count.  A '!' does count: 1 symbol out of 6 chars.
        assert compute_symbol_ratio("hello ") == 0.0     # space excluded
        assert compute_symbol_ratio("hello!") > 0.0      # ! is non-alnum, non-space

    def test_all_symbols_returns_one(self):
        assert compute_symbol_ratio("###") == 1.0

    def test_ratio_bounded(self):
        for text in ["!", "abc", "1.2,3", "", "###"]:
            assert 0.0 <= compute_symbol_ratio(text) <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# compute_digit_ratio
# ════════════════════════════════════════════════════════════════════════════
class TestComputeDigitRatio:

    def test_empty_returns_zero(self):
        assert compute_digit_ratio("") == 0.0

    def test_no_digits_returns_zero(self):
        assert compute_digit_ratio("hello world") == 0.0

    def test_all_digits_returns_one(self):
        assert compute_digit_ratio("1234") == 1.0

    def test_half_digits_returns_half(self):
        assert abs(compute_digit_ratio("ab12") - 0.5) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# compute_vowel_ratio
# ════════════════════════════════════════════════════════════════════════════
class TestComputeVowelRatio:

    def test_empty_returns_zero(self):
        assert compute_vowel_ratio("") == 0.0

    def test_no_alpha_returns_zero(self):
        assert compute_vowel_ratio("123!!!") == 0.0

    def test_pure_vowels_returns_one(self):
        assert compute_vowel_ratio("aeiou") == 1.0

    def test_pure_consonants_returns_zero(self):
        assert compute_vowel_ratio("bcdfg") == 0.0

    def test_czech_diacritic_á_counts_as_vowel(self):
        assert compute_vowel_ratio("á") == 1.0

    def test_czech_diacritic_ě_counts_as_vowel(self):
        assert compute_vowel_ratio("ě") == 1.0

    def test_mixed_word_in_range(self):
        ratio = compute_vowel_ratio("hello")  # e, o → 2 of 5
        assert 0.0 < ratio < 1.0

    def test_ratio_bounded(self):
        for text in ["hello", "bcdfg", "aeiou", "", "123"]:
            r = compute_vowel_ratio(text)
            assert 0.0 <= r <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# compute_rotatable_ratio
# ════════════════════════════════════════════════════════════════════════════
class TestComputeRotatableRatio:

    # Rotatable set: "pbqdnuwmoxszeyv"

    def test_empty_returns_zero(self):
        assert compute_rotatable_ratio("") == 0.0

    def test_no_alpha_returns_zero(self):
        assert compute_rotatable_ratio("123!!!") == 0.0

    def test_all_rotatable_returns_one(self):
        # p, b, q, d are all in the rotatable set
        assert compute_rotatable_ratio("pbqd") == 1.0

    def test_no_rotatables_returns_zero(self):
        # f, g, h, j, k, l are NOT in the rotatable set
        assert compute_rotatable_ratio("fghjkl") == 0.0

    def test_mixed_gives_partial_ratio(self):
        # "pb" are rotatable; "ff" are not
        ratio = compute_rotatable_ratio("pbff")
        assert 0.0 < ratio < 1.0

    def test_ratio_bounded(self):
        for text in ["pbqd", "fghjkl", "hello", "", "123"]:
            assert 0.0 <= compute_rotatable_ratio(text) <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# detect_strange_symbols
# ════════════════════════════════════════════════════════════════════════════
class TestDetectStrangeSymbols:

    def test_clean_text_returns_zero(self):
        assert detect_strange_symbols("hello world") == 0

    def test_empty_returns_zero(self):
        assert detect_strange_symbols("") == 0

    def test_allowed_percent_not_counted(self):
        # '%' is in ALLOWED_INTERNAL
        assert detect_strange_symbols("100%") == 0

    def test_allowed_colon_not_counted(self):
        assert detect_strange_symbols("DAT: 1820.") == 0

    def test_trailing_exclamation_stripped_from_edge(self):
        # '!' is in _STRIP_CHARS so it's stripped before checking; core = "hello"
        assert detect_strange_symbols("hello!") == 0

    def test_hash_inside_word_is_strange(self):
        assert detect_strange_symbols("he##lo") == 1

    def test_tilde_inside_word_is_strange(self):
        assert detect_strange_symbols("hel~lo") >= 1

    def test_two_bad_words_counted_separately(self):
        assert detect_strange_symbols("he# wo#ld") == 2

    def test_czech_text_with_standard_punctuation_returns_zero(self):
        assert detect_strange_symbols("kostra hrob, pece. ULOŽ: 7.") == 0


# ════════════════════════════════════════════════════════════════════════════
# detect_repeated_chars
# ════════════════════════════════════════════════════════════════════════════
class TestDetectRepeatedChars:

    def test_empty_returns_zero(self):
        assert detect_repeated_chars("") == 0

    def test_clean_word_returns_zero(self):
        assert detect_repeated_chars("hello world") == 0

    def test_triple_consonant_repeat_triggers(self):
        # "ssset" → 's' × 3 triggers the triple-repeat check
        assert detect_repeated_chars("ssset") >= 1

    def test_vowel_double_not_counted(self):
        # The vowel exclusion only applies to the 2nd and 3rd conditions, NOT to
        # the first triple-repeat check (ch * 3 in core).  "aaabcd" contains "aaa"
        # so it DOES trigger.  Use "aabcd" (only 2 a's) to verify no false positive:
        # 2nd check requires count >= 3 (fails); 3rd check excludes vowels (fails).
        assert detect_repeated_chars("aabcd") == 0

    def test_word_shorter_than_4_chars_skipped(self):
        # The detector skips len(core) < 4
        assert detect_repeated_chars("sss") == 0

    def test_high_ratio_consonant_repeat_triggers(self):
        # 'r' appears 5/8 = 62.5% of "rrrrabcd" → ≥ 0.30 threshold and count ≥ 3
        assert detect_repeated_chars("rrrrabcd") >= 1

    def test_consecutive_doubles_of_consonant_triggers(self):
        # "rr" × 3 occurrences in longer word
        assert detect_repeated_chars("rrrrword") >= 1


# ════════════════════════════════════════════════════════════════════════════
# detect_gibberish_words
# ════════════════════════════════════════════════════════════════════════════
class TestDetectGibberishWords:

    def test_empty_returns_zero(self):
        assert detect_gibberish_words("") == 0

    def test_normal_word_returns_zero(self):
        assert detect_gibberish_words("hello world") == 0

    def test_word_shorter_than_4_skipped(self):
        assert detect_gibberish_words("bcd") == 0

    def test_no_vowels_in_long_word_triggers(self):
        # "bcdfg" — 5 chars, zero vowels
        assert detect_gibberish_words("bcdfg") >= 1

    def test_mostly_numeric_word_excluded(self):
        # "90,9g" — > 60% digits+separators → excluded from gibberish check
        assert detect_gibberish_words("90,9g") == 0

    def test_all_vowels_triggers_high_vowel_ratio(self):
        # "aaaaaaa" — 100% vowels, well above the 80% cap threshold
        assert detect_gibberish_words("aaaaaaa") >= 1

    def test_czech_word_not_gibberish(self):
        # "náramek" has multiple vowels in normal proportion
        assert detect_gibberish_words("náramek") == 0

    def test_czech_archaeological_terms_not_gibberish(self):
        assert detect_gibberish_words("kostra hrob náramek keramika") == 0


# ════════════════════════════════════════════════════════════════════════════
# detect_letter_digit_letter
# ════════════════════════════════════════════════════════════════════════════
class TestDetectLetterDigitLetter:

    def test_empty_returns_zero(self):
        assert detect_letter_digit_letter("") == 0

    def test_clean_word_returns_zero(self):
        assert detect_letter_digit_letter("hello world") == 0

    def test_simple_ldl_pattern_detected(self):
        # "a1b" → alpha → digit → alpha
        assert detect_letter_digit_letter("a1b") >= 1

    def test_trailing_digits_not_ldl(self):
        # "abc123" → digits at end, no letter after
        assert detect_letter_digit_letter("abc123") == 0

    def test_leading_digits_not_ldl(self):
        # "123abc" → no alpha before the digit sequence
        assert detect_letter_digit_letter("123abc") == 0

    def test_ocr_digit_insertion_mid_word(self):
        # Classic OCR error: '1' inserted inside "vytačená"
        assert detect_letter_digit_letter("vyt1ačená") >= 1

    def test_two_ldl_words_counted_separately(self):
        assert detect_letter_digit_letter("a1b c2d") == 2


# ════════════════════════════════════════════════════════════════════════════
# detect_mid_uppercase
# ════════════════════════════════════════════════════════════════════════════
class TestDetectMidUppercase:

    def test_empty_returns_zero(self):
        assert detect_mid_uppercase("") == 0

    def test_lowercase_word_returns_zero(self):
        assert detect_mid_uppercase("hello world") == 0

    def test_initial_capital_not_mid_uppercase(self):
        # "Praha" — capital P at start, no lowercase→uppercase transition inside
        assert detect_mid_uppercase("Praha") == 0

    def test_all_caps_word_skipped(self):
        # core.isupper() → skip entirely
        assert detect_mid_uppercase("HELLO") == 0

    def test_single_char_skipped(self):
        # len(core) < 2 → skip
        assert detect_mid_uppercase("A") == 0

    def test_mid_uppercase_detected(self):
        # "dalSÍ" → 'l' (lower) followed by 'S' (upper)
        assert detect_mid_uppercase("dalSÍ") >= 1

    def test_camelcase_transition_detected(self):
        # "PrAha" → 'r' (lower) followed by 'A' (upper)
        assert detect_mid_uppercase("PrAha") >= 1

    def test_multiple_bad_words_counted(self):
        assert detect_mid_uppercase("dalSÍ prAha") >= 2


# ════════════════════════════════════════════════════════════════════════════
# detect_fused_words
# ════════════════════════════════════════════════════════════════════════════
class TestDetectFusedWords:

    def test_empty_returns_zero(self):
        assert detect_fused_words("") == 0

    def test_normal_words_return_zero(self):
        # "hello" has consonant runs h(1), ll(2) — both < 5
        assert detect_fused_words("hello world") == 0

    def test_token_longer_than_14_chars_triggers(self):
        # 18 'a' chars → len > 14 → fused
        assert detect_fused_words("aaaaaaaaaaaaaaaaaa") >= 1

    def test_five_consecutive_consonants_triggers(self):
        # "strnk" = s,t,r,n,k — all consonants in the regex set, run of 5
        assert detect_fused_words("strnk") >= 1

    def test_four_consecutive_consonants_does_not_trigger(self):
        # "strn" = 4 consonants — below the {5,} threshold
        assert detect_fused_words("strn") == 0

    def test_token_with_only_digits_skipped(self):
        # No alpha chars → skipped entirely
        assert detect_fused_words("12345678901234567890") == 0

    def test_eight_consonant_run_triggers(self):
        assert detect_fused_words("strntkkf") >= 1


# ════════════════════════════════════════════════════════════════════════════
# is_all_caps_line
# ════════════════════════════════════════════════════════════════════════════
class TestIsAllCapsLine:

    def test_all_caps_returns_true(self):
        assert is_all_caps_line("HELLO WORLD") is True

    def test_mixed_case_returns_false(self):
        assert is_all_caps_line("Hello World") is False

    def test_lowercase_returns_false(self):
        assert is_all_caps_line("hello world") is False

    def test_empty_string_returns_false(self):
        assert is_all_caps_line("") is False

    def test_digits_only_no_alpha_returns_false(self):
        # No alphabetical words → all-caps check always returns False
        assert is_all_caps_line("123 456") is False

    def test_single_caps_word_returns_true(self):
        assert is_all_caps_line("ÚSTAV") is True

    def test_caps_word_mixed_with_digits_returns_true(self):
        # "ROK 1922" — "ROK" is upper; "1922" has no alpha so not considered
        assert is_all_caps_line("ROK 1922") is True

    def test_caps_with_czech_diacritics_returns_true(self):
        assert is_all_caps_line("STÁTNÍ ARCHAEOLOGICKÝ ÚSTAV") is True


# ════════════════════════════════════════════════════════════════════════════
# is_non_text
# ════════════════════════════════════════════════════════════════════════════
class TestIsNonText:

    def test_empty_returns_false(self):
        assert is_non_text("") is False

    def test_pure_digits_returns_true(self):
        assert is_non_text("12345") is True

    def test_date_returns_true(self):
        # RE_NON_TEXT matches strings of digits / spaces / slashes / dots etc.
        assert is_non_text("1.3.1922") is True

    def test_normal_czech_sentence_returns_false(self):
        assert is_non_text("Tento řádek je normální text.") is False

    def test_short_high_digit_ratio_returns_true(self):
        # len("1234a") = 5 < 15; digit ratio = 4/5 = 0.8 > 0.5 → True
        assert is_non_text("1234a") is True

    def test_return_type_is_bool(self):
        for text in ["", "XIV", "123 45", "text here"]:
            assert isinstance(is_non_text(text), bool)


# ════════════════════════════════════════════════════════════════════════════
# infer_lang_from_diacritics
# ════════════════════════════════════════════════════════════════════════════
class TestInferLangFromDiacritics:

    def test_czech_heavy_text_inferred_as_ces(self):
        # ř, š, č, ž are distinctive Czech diacritics
        result = infer_lang_from_diacritics(
            "čistý česky psaný text", expected_bases=frozenset(["ces", "deu"])
        )
        assert result == "ces"

    def test_no_diacritics_returns_none(self):
        result = infer_lang_from_diacritics(
            "hello world", expected_bases=frozenset(["ces", "deu"])
        )
        assert result is None

    def test_empty_string_returns_none(self):
        result = infer_lang_from_diacritics("", expected_bases=frozenset(["ces"]))
        assert result is None

    def test_no_alpha_chars_returns_none(self):
        result = infer_lang_from_diacritics("123!!!", expected_bases=frozenset(["ces"]))
        assert result is None

    def test_lang_not_in_expected_bases_not_returned(self):
        # German diacritics present but "deu" is not in expected_bases
        result = infer_lang_from_diacritics(
            "schön und gemütlich", expected_bases=frozenset(["ces"])
        )
        assert result is None


# ════════════════════════════════════════════════════════════════════════════
# pre_filter_line
# ════════════════════════════════════════════════════════════════════════════
class TestPreFilterLine:
    """
    pre_filter_line returns (category, cleaned_text).
    category is one of "Empty", "Non-text", "Process".
    """

    VALID_CATEGORIES = {"Empty", "Non-text", "Process"}

    # ── Contract: return shape and valid values ───────────────────────────

    def test_always_returns_two_tuple(self):
        for line in ["", "   ", "hello text", "XIV.", "12345"]:
            result = pre_filter_line(line)
            assert isinstance(result, tuple) and len(result) == 2

    def test_category_always_valid(self):
        for line in ["", "XIV.", "aa", "12345", "normální text je zde"]:
            cat, _ = pre_filter_line(line)
            assert cat in self.VALID_CATEGORIES, f"bad category {cat!r} for {line!r}"

    # ── Empty ─────────────────────────────────────────────────────────────

    def test_empty_string_gives_empty(self):
        cat, text = pre_filter_line("")
        assert cat == "Empty" and text == ""

    def test_whitespace_only_gives_empty(self):
        cat, _ = pre_filter_line("   \t  ")
        assert cat == "Empty"

    # ── Non-text ──────────────────────────────────────────────────────────

    def test_pure_digits_gives_non_text(self):
        cat, _ = pre_filter_line("12345")
        assert cat == "Non-text"

    def test_roman_numeral_gives_non_text(self):
        cat, _ = pre_filter_line("XIV.")
        assert cat == "Non-text"

    def test_two_char_string_gives_non_text(self):
        # n_chars < 4 → Non-text
        cat, _ = pre_filter_line("ab")
        assert cat == "Non-text"

    def test_low_unique_symbols_gives_non_text(self):
        # "aaaa" → only 1 unique non-space char, below the threshold of 3
        cat, _ = pre_filter_line("aaaa")
        assert cat == "Non-text"

    def test_low_letter_ratio_gives_non_text(self):
        # "1234a" → 1 letter / 5 chars = 0.20 < 0.30
        cat, _ = pre_filter_line("1234a")
        assert cat == "Non-text"

    # ── Metadata bypass → Process ─────────────────────────────────────────

    def test_neg_marker_forces_process(self):
        cat, _ = pre_filter_line("č.neg")
        assert cat == "Process"

    def test_datum_marker_forces_process(self):
        cat, _ = pre_filter_line("Datum 14.5.1922")
        assert cat == "Process"

    def test_str_marker_forces_process(self):
        cat, _ = pre_filter_line("str. 24")
        assert cat == "Process"

    # ── High digit ratio → Process ────────────────────────────────────────

    def test_digit_ratio_above_40pct_gives_process(self):
        # "abc12345" looks like it would hit the digit bypass, but is_non_text() runs
        # first and classifies it as a Non-text archive code (RE_ARCHIVE_CODE matches
        # 1-3 letters followed by 3+ digits).
        # "vzorek 1234567890": the 6-letter prefix is longer than RE_ARCHIVE_CODE's
        # {1,3} and longer than RE_ARCHIVE_REF_SPACED's {1,5}, and len=17 ≥ 15 so
        # the short-high-digit-ratio guard in is_non_text also does not fire.
        # Digit ratio = 10/17 ≈ 59 % > 40 % → pre_filter returns "Process".
        cat, _ = pre_filter_line("vzorek 1234567890")
        assert cat == "Process"

    # ── Normal text → Process ─────────────────────────────────────────────

    def test_normal_czech_text_gives_process(self):
        cat, text = pre_filter_line("Tento řádek je normálně psaný text.")
        assert cat == "Process"
        assert len(text) > 0

    def test_archive_line_gives_process(self):
        cat, _ = pre_filter_line("504 ONDREJOV, okr. Praha-východ")
        assert cat == "Process"

    def test_returned_text_is_stripped(self):
        _, text = pre_filter_line("  hello world  ")
        assert not text.startswith(" ")
        assert not text.endswith(" ")

    # ── Digit-substitution repair ─────────────────────────────────────────

    def test_digit_1_between_letters_repaired_to_l(self):
        # "he1lo" → '1' between alpha letters should be repaired to 'l'
        _, text = pre_filter_line("he1lo world")
        # The repair converts the isolated '1' sandwiched by letters to 'l'
        assert "hel" in text


# ════════════════════════════════════════════════════════════════════════════
# parse_line_splits
# ════════════════════════════════════════════════════════════════════════════
class TestParseLineSplits:

    def test_no_split_returns_unchanged_text(self):
        text, pre, suf = parse_line_splits("normal line without split")
        assert text == "normal line without split"
        assert pre == ""
        assert suf == ""

    def test_empty_string_returns_triple_empty(self):
        text, pre, suf = parse_line_splits("")
        assert (text, pre, suf) == ("", "", "")

    def test_hyphen_split_merged_to_full_word(self):
        # "za- {započne}" → the split "za" + continuation is reconstructed
        text, pre, suf = parse_line_splits("za- {započne}")
        assert "započne" in text
        assert pre == "za"

    def test_prefix_plus_suffix_equals_full_word(self):
        # pre="za", suf="počne", full word = "započne"
        # Verified: "za" + "počne" == "započne" (same 7-char Czech string)
        _, pre, suf = parse_line_splits("za- {započne}")
        assert pre + suf == "započne"

    def test_split_inside_longer_line(self):
        line = "přijel naleziště shlednouti; k to- {tomu}"
        text, pre, suf = parse_line_splits(line)
        assert "tomu" in text
        assert pre == "to"
        # "to" + "mu" == "tomu"
        assert pre + suf == "tomu"

    def test_no_curly_braces_is_not_a_split(self):
        # A bare hyphen without the {…} annotation is not a recorded split
        text, pre, suf = parse_line_splits("word- another")
        assert pre == ""
        assert suf == ""


# ════════════════════════════════════════════════════════════════════════════
# score_word
# ════════════════════════════════════════════════════════════════════════════
class TestScoreWord:

    def test_empty_string_scores_zero(self):
        assert score_word("") == 0.0

    def test_clean_alpha_word_scores_zero(self):
        assert score_word("hello") == 0.0

    def test_clean_czech_word_scores_zero(self):
        assert score_word("náramek") == 0.0

    def test_single_common_letter_scores_zero(self):
        # 'a', 'A', 'i', 'v' etc. are in the single-letter safe list
        for ch in list("aAiIoOuUvV"):
            assert score_word(ch) == 0.0, f"Expected 0.0 for single letter {ch!r}"

    def test_single_digit_scores_low(self):
        assert score_word("5") == 0.25

    def test_single_unknown_alpha_scores_high(self):
        # 'X' is NOT in the single-letter safe list and is alpha → 0.85
        assert score_word("X") == 0.85

    def test_mid_uppercase_raises_score(self):
        # "heLLo" has a lowercase→uppercase transition inside the word
        assert score_word("heLLo") > 0.0

    def test_triple_consonant_repeat_raises_score(self):
        # "sssword" — 's' repeated 3+ times (non-vowel)
        assert score_word("sssword") > 0.0

    def test_ldl_fusion_raises_score(self):
        # "a1b" — letter-digit-letter adds to weirdness
        assert score_word("a1b") > 0.0

    def test_all_caps_prefix_followed_by_lowercase_raises_score(self):
        # "AAMNAbSSOAO" — long uppercase prefix before a lowercase fragment
        assert score_word("AAMNAbSSOAO") > 0.0

    def test_score_always_bounded(self):
        for word in ["hello", "heLLo", "sssword", "a1b", "BCDFG",
                     "", ".", "X", "5", "náramek", "AAMNAbSSOAO"]:
            s = score_word(word)
            assert 0.0 <= s <= 1.0, f"score {s} out of [0,1] for {word!r}"


# ════════════════════════════════════════════════════════════════════════════
# score_words_in_line / compute_word_weird_ratio
# ════════════════════════════════════════════════════════════════════════════
class TestWordWeirdRatio:

    def test_clean_line_gives_zero_ratio(self):
        pairs = score_words_in_line("hello world text")
        assert compute_word_weird_ratio(pairs) == 0.0

    def test_empty_list_gives_zero(self):
        assert compute_word_weird_ratio([]) == 0.0

    def test_corrupted_words_give_positive_ratio(self):
        pairs = score_words_in_line("heLLo sssword a1b")
        assert compute_word_weird_ratio(pairs) > 0.0

    def test_ratio_always_bounded(self):
        for text in ["hello world", "heLLo sssword a1b", "", "X Y Z"]:
            pairs = score_words_in_line(text)
            r = compute_word_weird_ratio(pairs)
            assert 0.0 <= r <= 1.0

    def test_score_words_returns_word_score_pairs(self):
        pairs = score_words_in_line("hello world")
        assert len(pairs) == 2
        for word, s in pairs:
            assert isinstance(word, str)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_clean_czech_line_gives_zero(self):
        pairs = score_words_in_line("kostra hrob náramek Bubníkova usedlost")
        assert compute_word_weird_ratio(pairs) == 0.0


# ════════════════════════════════════════════════════════════════════════════
# compute_valid_ratio
# ════════════════════════════════════════════════════════════════════════════
class TestComputeValidRatio:

    def test_empty_returns_zero(self):
        assert compute_valid_ratio("") == 0.0

    def test_all_numeric_tokens_give_zero(self):
        # Tokens with < 70% alpha chars don't qualify as valid words
        assert compute_valid_ratio("123 456 789") == 0.0

    def test_clean_czech_words_all_valid(self):
        assert compute_valid_ratio("kostra hrob náramek") == 1.0

    def test_word_with_strange_symbol_not_valid(self):
        # "he##lo" contains '#' which is outside ALLOWED_INTERNAL → not valid
        assert compute_valid_ratio("he##lo") == 0.0

    def test_normal_text_gives_high_ratio(self):
        ratio = compute_valid_ratio("normal czech text here")
        assert ratio > 0.5

    def test_ratio_bounded(self):
        for text in ["hello world", "123 #$%", "čistý text", ""]:
            assert 0.0 <= compute_valid_ratio(text) <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# compute_quality_score
# ════════════════════════════════════════════════════════════════════════════
class TestComputeQualityScore:
    """
    Tests focus on the *contract* (range, monotonicity, specific boundary effects)
    rather than exact numeric values, which depend on configurable weights.
    """

    def test_output_in_zero_one_range(self):
        q = compute_quality_score(
            valid_word_ratio=0.8, symbol_ratio=0.05,
            perplexity=200.0, text_length=50, weird_ratio=0.0,
        )
        assert 0.0 <= q <= 1.0

    def test_ideal_inputs_give_high_score(self):
        q = compute_quality_score(
            valid_word_ratio=1.0, symbol_ratio=0.0,
            perplexity=0.0, text_length=80, weird_ratio=0.0,
            vowel_ratio=0.40, garbage_density=0.0,
            lang_score=0.99, gibberish_ratio=0.0, fused_ratio=0.0,
        )
        assert q > 0.7

    def test_worst_inputs_give_low_score(self):
        q = compute_quality_score(
            valid_word_ratio=0.0, symbol_ratio=1.0,
            perplexity=9999.0, text_length=0, weird_ratio=1.0,
            vowel_ratio=0.0, garbage_density=1.0,
            lang_score=0.0, gibberish_ratio=1.0, fused_ratio=1.0,
        )
        assert q < 0.5

    def test_higher_valid_word_ratio_improves_score(self):
        base = dict(symbol_ratio=0.0, perplexity=200.0,
                    text_length=50, weird_ratio=0.0)
        q_low  = compute_quality_score(valid_word_ratio=0.0, **base)
        q_high = compute_quality_score(valid_word_ratio=1.0, **base)
        assert q_high > q_low

    def test_higher_perplexity_lowers_score(self):
        base = dict(valid_word_ratio=0.8, symbol_ratio=0.0,
                    text_length=50, weird_ratio=0.0)
        q_low_ppl  = compute_quality_score(perplexity=10.0,   **base)
        q_high_ppl = compute_quality_score(perplexity=5000.0, **base)
        assert q_low_ppl > q_high_ppl

    def test_rotation_penalty_applied_when_weird_ratio_high(self):
        # rot_ratio triggers the penalty gate only when weird_ratio is also elevated
        base = dict(valid_word_ratio=0.5, symbol_ratio=0.1,
                    perplexity=500.0, text_length=40, weird_ratio=0.5)
        q_no_rot   = compute_quality_score(rot_ratio=0.0, **base)
        q_high_rot = compute_quality_score(rot_ratio=0.9, **base)
        assert q_no_rot > q_high_rot

    def test_rotation_penalty_not_applied_when_weird_ratio_zero(self):
        # weird_ratio=0.0 closes the gate; rot_ratio has no effect
        base = dict(valid_word_ratio=0.8, symbol_ratio=0.0,
                    perplexity=100.0, text_length=60, weird_ratio=0.0)
        q_no_rot   = compute_quality_score(rot_ratio=0.0, **base)
        q_high_rot = compute_quality_score(rot_ratio=0.9, **base)
        assert q_no_rot == q_high_rot

    def test_score_always_bounded_over_random_inputs(self):
        rng = random.Random(42)
        for _ in range(40):
            q = compute_quality_score(
                valid_word_ratio=rng.random(),
                symbol_ratio=rng.random(),
                perplexity=rng.uniform(0, 10_000),
                text_length=rng.randint(0, 200),
                weird_ratio=rng.random(),
                vowel_ratio=rng.random(),
                garbage_density=rng.random(),
                lang_score=rng.random(),
                gibberish_ratio=rng.random(),
                fused_ratio=rng.random(),
                rot_ratio=rng.random(),
            )
            assert 0.0 <= q <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# categorize_line
# ════════════════════════════════════════════════════════════════════════════
class TestCategorizeLine:
    """
    categorize_line(qs, txt, wc, vowel_ratio, perplexity, rot_ratio, weird_ratio)
    → (category_str, aligned_score)

    The aligned_score is clamped into the bucket of the assigned category;
    we verify (category, score) consistency using the thresholds read from
    the module itself.
    """

    # ── Return type contract ──────────────────────────────────────────────

    def test_returns_tuple_of_str_and_float(self):
        result = categorize_line(0.7, "some text here", 3, 0.4, 300.0)
        assert isinstance(result, tuple) and len(result) == 2
        cat, score = result
        assert isinstance(cat, str)
        assert isinstance(score, float)

    # ── Empty ─────────────────────────────────────────────────────────────

    def test_empty_text_gives_empty(self):
        cat, _ = categorize_line(0.8, "", 0, 0.4, 100.0)
        assert cat == "Empty"

    def test_zero_word_count_gives_empty(self):
        cat, _ = categorize_line(0.8, "   ", 0, 0.4, 100.0)
        assert cat == "Empty"

    # ── High-confidence LM override → Clear ──────────────────────────────

    def test_low_ppl_and_sufficient_words_gives_clear(self):
        # ppl < 50 and wc >= 3 → immediate Clear regardless of qs value
        cat, score = categorize_line(0.1, "čistý text v češtině", 4, 0.4, 30.0)
        assert cat == "Clear"
        assert score >= CATEG_NOISY_SCORE_MAX

    def test_low_ppl_override_requires_at_least_three_words(self):
        # ppl < 50 but wc == 2 → override does not fire; uses qs threshold instead
        cat, _ = categorize_line(0.2, "dva slova", 2, 0.4, 30.0)
        assert cat != "Clear"   # qs=0.2 < CATEG_TRASH_SCORE_MAX → Trash

    # ── All-caps with no vowels → Trash ──────────────────────────────────

    def test_all_caps_with_no_vowels_gives_trash(self):
        # is_all_caps_line=True and vowel_ratio < 0.10 → immediate Trash
        cat, _ = categorize_line(0.8, "BCDFGHJKL", 1, 0.01, 2000.0)
        assert cat == "Trash"

    def test_all_caps_with_vowels_does_not_trigger_trash_override(self):
        # vowel_ratio = 0.40 ≥ 0.10 → override does not fire
        cat, _ = categorize_line(0.95, "HELLO WORLD", 2, 0.40, 2000.0)
        assert cat != "Trash"

    # ── Threshold-based routing ───────────────────────────────────────────

    def test_qs_below_trash_threshold_gives_trash(self):
        qs = CATEG_TRASH_SCORE_MAX * 0.5       # clearly below threshold
        cat, score = categorize_line(qs, "noisy text here now", 4, 0.4, 800.0)
        assert cat == "Trash"
        assert score < CATEG_TRASH_SCORE_MAX

    def test_qs_in_noisy_band_gives_noisy_when_promotion_blocked(self):
        # qs is in the Noisy band; high weird_ratio blocks clean-prose promotion
        qs = (CATEG_TRASH_SCORE_MAX + CATEG_NOISY_SCORE_MAX) / 2
        cat, score = categorize_line(qs, "some noisy text x", 2, 0.4, 1000.0,
                                     weird_ratio=CLEAN_PROSE_WEIRD_MAX + 0.2)
        assert cat == "Noisy"
        assert CATEG_TRASH_SCORE_MAX <= score < CATEG_NOISY_SCORE_MAX

    def test_qs_above_noisy_threshold_gives_clear(self):
        qs = CATEG_NOISY_SCORE_MAX + 0.01
        cat, score = categorize_line(qs, "čistý text", 2, 0.4, 200.0)
        assert cat == "Clear"
        assert score >= CATEG_NOISY_SCORE_MAX

    # ── Clean-prose promotion ─────────────────────────────────────────────

    def test_all_promotion_conditions_met_gives_clear(self):
        # qs ≥ CLEAN_PROSE_MIN_SCORE, wc ≥ CLEAN_PROSE_WC_MIN,
        # weird < CLEAN_PROSE_WEIRD_MAX, ppl < CLEAN_PROSE_PPL_MAX
        qs  = CLEAN_PROSE_MIN_SCORE + 0.01
        ppl = CLEAN_PROSE_PPL_MAX   - 10.0
        wc  = CLEAN_PROSE_WC_MIN
        cat, _ = categorize_line(
            qs, "normální česky psaný archeologický text", wc, 0.4, ppl,
            weird_ratio=0.0,
        )
        assert cat == "Clear"

    def test_high_ppl_blocks_clean_prose_promotion(self):
        qs  = CLEAN_PROSE_MIN_SCORE + 0.01
        ppl = CLEAN_PROSE_PPL_MAX   + 100.0    # too high
        wc  = CLEAN_PROSE_WC_MIN
        cat, _ = categorize_line(
            qs, "normální česky psaný text zde", wc, 0.4, ppl,
            weird_ratio=0.0,
        )
        assert cat == "Noisy"

    def test_high_weird_ratio_blocks_clean_prose_promotion(self):
        qs  = CLEAN_PROSE_MIN_SCORE + 0.01
        ppl = CLEAN_PROSE_PPL_MAX   - 10.0
        wc  = CLEAN_PROSE_WC_MIN
        cat, _ = categorize_line(
            qs, "normální text zde je", wc, 0.4, ppl,
            weird_ratio=CLEAN_PROSE_WEIRD_MAX + 0.1,
        )
        assert cat == "Noisy"

    def test_too_few_words_blocks_clean_prose_promotion(self):
        qs  = CLEAN_PROSE_MIN_SCORE + 0.01
        ppl = CLEAN_PROSE_PPL_MAX   - 10.0
        wc  = CLEAN_PROSE_WC_MIN    - 1       # one word short of minimum
        cat, _ = categorize_line(
            qs, "text zde", wc, 0.4, ppl,
            weird_ratio=0.0,
        )
        assert cat == "Noisy"

    def test_qs_below_prose_min_score_blocks_promotion(self):
        qs  = CLEAN_PROSE_MIN_SCORE - 0.01    # just below the floor
        ppl = CLEAN_PROSE_PPL_MAX   - 10.0
        wc  = CLEAN_PROSE_WC_MIN
        cat, _ = categorize_line(
            qs, "normální text zde je píše", wc, 0.4, ppl,
            weird_ratio=0.0,
        )
        assert cat == "Noisy"

    # ── Aligned-score clamping consistency ───────────────────────────────

    def test_trash_aligned_score_is_below_trash_threshold(self):
        qs = CATEG_TRASH_SCORE_MAX * 0.3
        cat, score = categorize_line(qs, "noisy garbage text here", 4, 0.4, 3000.0)
        assert cat == "Trash"
        assert score < CATEG_TRASH_SCORE_MAX

    def test_noisy_aligned_score_is_in_noisy_band(self):
        qs = (CATEG_TRASH_SCORE_MAX + CATEG_NOISY_SCORE_MAX) / 2
        cat, score = categorize_line(qs, "some degraded text x", 2, 0.4, 1000.0,
                                     weird_ratio=CLEAN_PROSE_WEIRD_MAX + 0.2)
        assert cat == "Noisy"
        assert CATEG_TRASH_SCORE_MAX <= score < CATEG_NOISY_SCORE_MAX

    def test_clear_aligned_score_is_above_noisy_threshold(self):
        qs = CATEG_NOISY_SCORE_MAX + 0.02
        cat, score = categorize_line(qs, "čistý text", 2, 0.4, 200.0)
        assert cat == "Clear"
        assert score >= CATEG_NOISY_SCORE_MAX


# ════════════════════════════════════════════════════════════════════════════
# Integration smoke-tests: full pipeline on representative lines
# ════════════════════════════════════════════════════════════════════════════
class TestFullPipelineSmoke:
    """
    Exercises the pre_filter → score → compute_quality_score → categorize_line
    chain on a representative set of inputs drawn from the real ATRIUM corpus,
    replacing only the language-model perplexity call with a fixed mock value.
    No ML models required.
    """

    # (raw_line, expected_prefilter_cat, description)
    PREFILTER_CASES = [
        ("",                                             "Empty",    "blank line"),
        ("   ",                                          "Empty",    "whitespace only"),
        ("XIV.",                                         "Non-text", "Roman numeral"),
        ("1820.",                                        "Non-text", "year dot"),
        ("12345",                                        "Non-text", "pure digits"),
        ("504 ONDREJOV, okr. Praha-východ",              "Process",  "archive header"),
        ("Bubníkova usedlost NALEZ: kostr. hrob.",       "Process",  "archive line"),
        ("Při stavbě základů nalezena kostra.",          "Process",  "clear prose"),
        ("č.neg",                                        "Process",  "metadata marker"),
    ]

    @pytest.mark.parametrize("line,expected,desc", PREFILTER_CASES)
    def test_prefilter_category(self, line, expected, desc):
        cat, _ = pre_filter_line(line)
        assert cat == expected, f"[{desc}] expected {expected!r}, got {cat!r}"

    def test_process_lines_produce_valid_quality_scores(self):
        process_lines = [
            "504 ONDREJOV, okr. Praha-východ",
            "Bubníkova usedlost NALEZ: kostr. hrob.",
            "Při stavbě základů nalezena kostra.",
            "V srpnu jsme pracovali v sondách 14ch a 14i.",
        ]
        for line in process_lines:
            cat, clean = pre_filter_line(line)
            assert cat == "Process", f"Expected Process for {line!r}, got {cat!r}"
            qs = compute_quality_score(
                valid_word_ratio=compute_valid_ratio(clean),
                symbol_ratio=compute_symbol_ratio(clean),
                perplexity=200.0,           # mocked LM call
                text_length=len(clean),
                weird_ratio=compute_word_weird_ratio(score_words_in_line(clean)),
                vowel_ratio=compute_vowel_ratio(clean),
                garbage_density=compute_garbage_density(clean),
                lang_score=0.85,            # mocked FastText confidence
            )
            assert 0.0 <= qs <= 1.0, f"score out of range for {line!r}: {qs}"

    def test_end_to_end_pipeline_produces_valid_category_and_score(self):
        """
        Simulates exactly what langID_classify.py does per line, replacing the
        two GPU calls (LM perplexity + FastText) with fixed mock values.
        """
        VALID_CATEGORIES = {"Clear", "Noisy", "Trash", "Empty", "Non-text"}

        test_cases = [
            # (raw_line, mock_perplexity, mock_lang_score)
            ("Při kopání základů hrnčíř pece nalezena kostra.",  200.0, 0.92),
            ("za- {započne} merge reconstruction test",          300.0, 0.88),
            ("BCDFGHJ BCDFGHJ BCDFGHJ BCDFGHJ",                 9000.0, 0.10),
            ("AAMMNAbSSOAO ###@~@ vyt1ačená ZZZZZ",             8500.0, 0.12),
            ("",                                                    0.0, 0.00),
            ("XIV.",                                                0.0, 0.00),
        ]

        for raw_line, mock_ppl, mock_lang_score in test_cases:
            prefilter_cat, clean = pre_filter_line(raw_line)

            # Lines caught by pre_filter don't reach the GPU pipeline
            if prefilter_cat != "Process":
                assert prefilter_cat in VALID_CATEGORIES
                continue

            wc    = len(clean.split())
            cc    = len(clean)
            vr    = compute_vowel_ratio(clean)
            rr    = compute_rotatable_ratio(clean)
            ws    = score_words_in_line(clean)
            weird = compute_word_weird_ratio(ws)
            gib   = detect_gibberish_words(clean)
            fused = detect_fused_words(clean)

            qs = compute_quality_score(
                valid_word_ratio=compute_valid_ratio(clean),
                symbol_ratio=compute_symbol_ratio(clean),
                perplexity=mock_ppl,
                text_length=cc,
                weird_ratio=weird,
                vowel_ratio=vr,
                garbage_density=compute_garbage_density(clean),
                lang_score=mock_lang_score,
                gibberish_ratio=gib / max(wc, 1),
                fused_ratio=fused / max(wc, 1),
                rot_ratio=rr,
            )
            assert 0.0 <= qs <= 1.0, f"qs out of range for {raw_line!r}"

            cat, aligned_qs = categorize_line(
                qs, clean, wc, vr, mock_ppl,
                rot_ratio=rr, weird_ratio=weird,
            )
            assert cat in VALID_CATEGORIES, f"bad category {cat!r} for {raw_line!r}"
            assert 0.0 <= aligned_qs <= 1.0

    def test_clean_prose_line_not_categorised_as_trash(self):
        """A well-formed Czech sentence must never land in Trash."""
        line = "Nalezl jsem rozklady na katastru obce s pohřebními nálezy."
        _, clean = pre_filter_line(line)
        vr    = compute_vowel_ratio(clean)
        rr    = compute_rotatable_ratio(clean)
        ws    = score_words_in_line(clean)
        weird = compute_word_weird_ratio(ws)
        wc    = len(clean.split())
        qs = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(clean),
            symbol_ratio=compute_symbol_ratio(clean),
            perplexity=180.0, text_length=len(clean),
            weird_ratio=weird, vowel_ratio=vr,
            garbage_density=compute_garbage_density(clean),
            lang_score=0.93,
        )
        cat, _ = categorize_line(qs, clean, wc, vr, 180.0,
                                 rot_ratio=rr, weird_ratio=weird)
        assert cat != "Trash"

    def test_heavily_corrupted_line_not_categorised_as_clear(self):
        """A heavily OCR-corrupted line must not be classified as Clear."""
        line = "##@#!~~## vyt1ačená AAMMNAbSSOAO ###@~@"
        _, clean = pre_filter_line(line)
        if clean == "":   # pre-filtered out entirely → acceptable
            return
        vr    = compute_vowel_ratio(clean)
        rr    = compute_rotatable_ratio(clean)
        ws    = score_words_in_line(clean)
        weird = compute_word_weird_ratio(ws)
        wc    = len(clean.split())
        qs = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(clean),
            symbol_ratio=compute_symbol_ratio(clean),
            perplexity=9000.0, text_length=len(clean),
            weird_ratio=weird, vowel_ratio=vr,
            garbage_density=compute_garbage_density(clean),
            lang_score=0.10,
        )
        cat, _ = categorize_line(qs, clean, wc, vr, 9000.0,
                                 rot_ratio=rr, weird_ratio=weird)
        assert cat != "Clear"