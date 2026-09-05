"""
tests/test_text_util.py
=======================
Unit tests for text_util.py  —  all pure-Python, zero ML dependencies.
"""

import random

import pytest

from classify_TEXT import CSV_HEADER, _fast_track_row, _row_from_dict
from text_util import (
    CATEG_NOISY_SCORE_MAX,
    LANG_REMAP_ALWAYS,
    categorize_line,
    compute_garbage_density,
    compute_quality_score,
    compute_rotatable_ratio,
    compute_valid_ratio,
    compute_vowel_ratio,
    compute_word_weird_ratio,
    detect_fused_words,
    detect_gibberish_words,
    detect_letter_digit_letter,
    detect_mid_uppercase,
    detect_repeated_chars,
    detect_strange_symbols,
    detect_wx_words,
    is_domain_notation,
    is_structured_line,
    override_constants,
    pre_filter_line,
    remap_lang,
    score_word,
    score_words_in_line,
)


class TestComputeGarbageDensity:
    def test_clean_alphanumeric_text_returns_zero(self):
        assert compute_garbage_density("hello world 123") == 0.0

    def test_common_punctuation_now_counted_as_noise(self):
        assert compute_garbage_density("hello, world! (test) 1/2 a-b") > 0.0

    def test_dots_are_now_counted_as_noise(self):
        assert compute_garbage_density("konec...") > 0.0


class TestDetectRepeatedChars:
    def test_clean_word_returns_zero(self):
        assert detect_repeated_chars("ahoj svete") == 0

    def test_double_consonant_now_triggers(self):
        assert detect_repeated_chars("panna") >= 1


class TestShortGarbageRoute:
    """(#3 A2/B) structural short-garbage route in determine_category."""

    def test_short_gibberish_token_routed_to_trash(self):
        # 'olie': 1 token, capped lang score, gibberish present.
        cat, score, reason = categorize_line(
            0.895, "olie", 1, 0.75, 1.0, return_reason=True, lang_score=0.75, gibberish_present=True
        )
        assert cat == "Trash" and reason == "trash_threshold"

    def test_clean_diacritic_fragment_not_trashed(self):
        # A short clean Czech fragment with diacritics must survive the route.
        cat, _, reason = categorize_line(
            0.70, "Náčrt sondy.", 2, 0.4, 200.0, return_reason=True, lang_score=0.99, gibberish_present=False
        )
        assert cat != "Trash"


def test_csv_header_and_fast_track_row_arity():
    """Asserts that the fast-track row builder exactly matches the global CSV_HEADER length."""
    row = _fast_track_row(
        file_id="CTX000001",
        page_id="1",
        line_num=1,
        clean_text="",
        original_text="",
        split_ws="",
        split_we="",
        categ="Empty",
    )
    assert len(row) == len(CSV_HEADER)


def test_row_from_dict_covers_header_exactly():
    """Asserts that _row_from_dict enforces the exact column sequence and arity."""
    dummy_dict = {col: "test_val" for col in CSV_HEADER}
    main_row = _row_from_dict(dummy_dict)
    assert len(main_row) == len(CSV_HEADER)
    assert main_row[0] == dummy_dict[CSV_HEADER[0]]


class TestIsDomainNotation:
    """Regression lock for `is_domain_notation()` (rule_domain_notation).

    The predicate exists because every one of the seven predicates behind
    `is_structured_line()` is inert on archaeological notation: `II/C`,
    `KK-XIII`, `Reg.Bez.Aussig.` and `1 ks` all return False from all seven, so
    they reach `rule_short_garbage` and are trashed. `_RE_SIGLUM` in particular
    caps abbreviation segments at four characters, which is why
    `Reg.Bez.Aussig.` fails on "Aussig".

    Two boundaries are load-bearing and each has its own test below: the
    predicate must recognise notation WITHOUT recognising vocabulary (a lexicon
    problem, deliberately left alone), and it must not re-open the spaced bare
    metre that `_looks_like_measurement` deliberately refuses.
    """

    # Shapes the predicate must recover — all of them reach rule_short_garbage
    # on an unpatched tree.
    NOTATION = [
        "II/C",
        "I-VIII-c",
        "KK-XIII",
        "A/1",
        "XIV-2b",
        "Lokalisace: MM-III",
        "sonda: III",
        "1 ks",
        "2 ks",
        "Reg.Bez.Aussig.",
        "radius prox.sin.",
    ]

    # Garbage and probe noise that must stay unmatched. The first four are the
    # same lines pinned by
    # test_recategorize_parity.py::test_obvious_garbage_is_not_structured; the
    # rest are drawn from the issue #30 thread and from the single-letter-unit
    # and spaced-metre tripwires.
    NEGATIVES = [
        "olie",
        "oueussd",
        "pbqdnuwmoxszeyv!!",
        "NINNNIC",
        "3 m",
        "o 5 m",
        "cuxoaid v. 12",
        "clouCelRa pr. 4",
        "2,10 m",
        "pr. 4",
        "ab c9 xz",
        "vansasaasasa",
        "Tthts I",
        "rragment",
        "vfetennl k.",
        "Slaot-o hiezazzt",
        "IDIDIDIDIDIDUOID",
        "zcv7",
        "/7suuk",
        "sektlll",
        "edelite",
        "Ch. i6dn.283/54",
        "p.nA o.",
        "cuxoaid ,",
    ]

    @pytest.mark.parametrize("text", NOTATION)
    def test_notation_is_recognised(self, text):
        assert is_domain_notation(text) is True

    @pytest.mark.parametrize("text", NEGATIVES)
    def test_garbage_is_not_notation(self, text):
        assert is_domain_notation(text) is False

    @pytest.mark.parametrize("text", ["malakofauna", "Equus caballus", "diapozitiv", "Ossa tarsi"])
    def test_vocabulary_is_out_of_scope(self, text):
        """Words, not shapes.

        Separating `malakofauna` from `oueussd` needs a lexicon; no regex does
        it. Matching these here would be the predicate quietly claiming to solve
        the harder half of issue #30, so they stay with rule_short_garbage.
        """
        assert is_domain_notation(text) is False

    @pytest.mark.parametrize("text", ["12,5 cm", "145-167mm", "0,46m"])
    def test_dimensions_are_left_to_looks_like_measurement(self, text):
        """Dimensions already satisfy `is_structured_line()`, so they never
        reach rule_short_garbage and need no second implementation here. A copy
        would only be able to drift from the original."""
        assert is_structured_line(text) is True

    def test_spaced_bare_metre_stays_closed(self):
        """Mirror of
        test_categorization_routes.py::test_not_yet_wired_into_looks_like_measurement.

        `_RE_UNIT_CANDIDATE` is dormant on purpose and wiring it in is a separate,
        deliberate decision. This predicate must not re-open it through the back
        door."""
        assert is_domain_notation("2,10 m") is False
        assert is_domain_notation("0,2-0,4 m") is False

    def test_single_letter_units_stay_closed(self):
        """`3 m` / `o 5 m` are pinned as non-measurements elsewhere; the count
        pattern accepts only multi-character unit words (`ks`)."""
        assert is_domain_notation("3 m") is False
        assert is_domain_notation("1 ks") is True


class TestNotationIsNotExemptFromHardSweep:
    """Why `rule_hard_sweep` stays armed for notation — issue #30, @david-spacil.

    He observed that with `SHORT_PPL_CAP` lifted, `II/C`, `1 ks` and
    `Reg.Bez.Aussig.` still route to `trash_hard_sweep`, so exempting notation
    from the perplexity routes changed nothing for the uncapping experiment.
    That is correct, and it is deliberate rather than an oversight.

    `rule_extreme_ppl` and `rule_absolute_ppl` convict on perplexity ALONE, and
    perplexity is meaningless on this population — it runs backwards, demoting
    real notation around 3,000 while `oueussd` survives to 30,000. Notation is
    exempt from those two.

    `rule_hard_sweep` additionally requires `orig_lang_score <
    HARD_SWEEP_LANG_MAX`, an independent second witness: FastText also failed to
    place the line. Notation is NOT exempt from it, because the predicate is not
    strong enough to be trusted alone — see the measured hole below.
    """

    _GARBAGE_VOCAB = [
        "oueussd",
        "vansasaasasa",
        "NINNNIC",
        "rragment",
        "sektlll",
        "edelite",
        "zcv7",
        "kfjs",
        "qmwx",
        "zzpl",
        "vvbn",
        "nupoy",
        "yoysqu",
        "olie",
        "Slaot",
        "hiezazzt",
        "cuxoaid",
        "vfetennl",
        "Tthts",
    ]

    def _capitalised_dot_chains(self, n=400, seed=20260905):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            k = rng.choice([2, 3])
            out.append(
                ".".join(rng.choice(self._GARBAGE_VOCAB).capitalize()[: rng.choice([3, 4, 5, 6, 7])] for _ in range(k))
                + "."
            )
        return out

    def test_abbreviation_pattern_has_a_large_hole_for_capitalised_dot_chains(self):
        """The measurement that justifies keeping hard sweep armed.

        `_RE_NOTATION_ABBR` requires a capital initial per segment, which closed
        the *lowercase* dot-chain hole completely. It does not close the
        capitalised one: OCR garbage is frequently capitalised, and
        `Vvbn.Slaot.Vansas.` is structurally indistinguishable from
        `Reg.Bez.Aussig.`

        Two narrower rules were measured and rejected. Requiring a vowel per
        segment cuts acceptance to ~40% but discards legitimate consonant
        abbreviations (`Kr.Hr.`, `St.Pol.`). Requiring a majority of segments to
        be <= 4 characters keeps every real chain but only reaches ~64%. Neither
        earns its complexity, and both confirm that shape alone cannot separate
        these — the same wall the vocabulary half of issue #30 runs into.

        This test asserts the hole is STILL THERE. If a future change closes it,
        this goes red, and exempting notation from `rule_hard_sweep` becomes
        worth reconsidering.
        """
        chains = self._capitalised_dot_chains()
        accepted = [c for c in chains if is_domain_notation(c)]
        share = len(accepted) / len(chains)

        assert share > 0.5, (
            f"only {share:.1%} of capitalised dot-chained garbage is accepted by "
            "is_domain_notation() — the hole may have been closed, in which case "
            "exempting notation from rule_hard_sweep is worth re-measuring"
        )

    def test_lowercase_dot_chains_stay_closed(self):
        """The half the capital-initial rule DID fix. Pinned so it stays fixed."""
        rng = random.Random(20260905)
        chains = [
            ".".join(
                rng.choice(self._GARBAGE_VOCAB).lower()[: rng.choice([3, 4, 5, 6])] for _ in range(rng.choice([2, 3]))
            )
            + "."
            for _ in range(400)
        ]
        accepted = [c for c in chains if is_domain_notation(c)]
        assert len(accepted) / len(chains) < 0.05

    def test_real_administrative_chains_are_still_recognised(self):
        """Closing the hole must not come at the cost of the real thing."""
        for chain in ("Reg.Bez.Aussig.", "Kr.Hr.", "St.Pol.", "Bez.Leitm.", "Reg.Bez.Eger."):
            assert is_domain_notation(chain) is True, chain


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
        assert compute_vowel_ratio("a!") == 0.5

    def test_digits_excluded_from_denominator(self):
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


class TestDetectGibberishWords:
    def test_normal_word_returns_zero(self):
        assert detect_gibberish_words("hello world") == 0

    def test_no_vowels_does_not_trigger(self):
        assert detect_gibberish_words("bcdfg") == 0

    def test_all_caps_word_skipped(self):
        assert detect_gibberish_words("AAAAAAA") == 0

    def test_all_vowels_triggers_high_vowel_ratio(self):
        assert detect_gibberish_words("aaaaaaa") >= 1

    def test_real_high_vowel_fragment_flagged(self):
        # (#3) 'olie' (3 vowels / 4 letters) is the canonical short-garbage token.
        assert detect_gibberish_words("olie") >= 1


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
        assert detect_fused_words("krásnoooučko") >= 1

    def test_subtoken_split_prevents_hiding(self):
        assert detect_fused_words("str.nk") == 0


class TestDetectWxWords:
    def test_empty_returns_zero(self):
        assert detect_wx_words("") == 0

    def test_clean_returns_zero(self):
        assert detect_wx_words("hello") == 0

    def test_high_w_x_density_triggers(self):
        assert detect_wx_words("exxon") >= 1
        assert detect_wx_words("wwx") >= 1


class TestLangRemap:
    """(#3 2026-07-02 calibration) remap_lang's fixed 0.75/0.50 assignment is
    gated by the LANG_REMAP_ALWAYS config toggle. true (default) — DanaKriv:
    "the original lang score should not matter." false — restores the prior
    #3 A1 "cap, don't inflate a weak guess" behaviour without a code change,
    since this exact call has already flipped once before in the thread."""

    def test_known_base_preserved(self):
        lbl, sc = remap_lang("deu_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "deu_Latn"
        assert sc == 0.4

    def test_slk_relabelled_but_score_preserved(self):
        lbl, sc = remap_lang("slk_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.4

    # -- LANG_REMAP_ALWAYS = true (default): the original score never matters --

    def test_default_is_always_on(self):
        assert LANG_REMAP_ALWAYS is True

    def test_unknown_latin_weak_score_set_to_remap(self):
        # The original score must NOT matter: a weak guess is still set to 0.75.
        lbl, sc = remap_lang("fra_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.75

    def test_unknown_latin_confident_score_set_to_remap(self):
        # A confident foreign guess on Czech data is also fixed at LANG_SCORE_REMAP.
        lbl, sc = remap_lang("dan_Latn", 0.96, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.75

    def test_non_latin_set_to_remap_far(self):
        # Non-Latin scripts are fixed at LANG_SCORE_REMAP_FAR (0.50), regardless
        # of the original confidence.
        lbl, sc = remap_lang("kor_Hang", 0.90, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Hang"
        assert sc == 0.50

        lbl, sc = remap_lang("kor_Hang", 0.05, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Hang"
        assert sc == 0.50

    # -- LANG_REMAP_ALWAYS = false: cap, don't inflate (the #3 A1 behaviour) --

    def test_toggle_off_weak_latin_score_left_alone(self):
        with override_constants({"LANG_REMAP_ALWAYS": False}):
            lbl, sc = remap_lang("fra_Latn", 0.4, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.4

    def test_toggle_off_confident_latin_score_still_capped(self):
        with override_constants({"LANG_REMAP_ALWAYS": False}):
            lbl, sc = remap_lang("dan_Latn", 0.96, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Latn"
        assert sc == 0.75

    def test_toggle_off_weak_non_latin_score_left_alone(self):
        with override_constants({"LANG_REMAP_ALWAYS": False}):
            lbl, sc = remap_lang("kor_Hang", 0.05, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Hang"
        assert sc == 0.05

    def test_toggle_off_confident_non_latin_score_still_capped(self):
        with override_constants({"LANG_REMAP_ALWAYS": False}):
            lbl, sc = remap_lang("kor_Hang", 0.90, frozenset(["deu", "eng"]), "ces")
        assert lbl == "ces_Hang"
        assert sc == 0.50


# ════════════════════════════════════════════════════════════════════════════
# Pre-filtering
# ════════════════════════════════════════════════════════════════════════════
class TestPreFilterLine:
    def test_empty_string_gives_empty(self):
        cat, _ = pre_filter_line("")
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
        cat, _ = pre_filter_line("Tento řádek je normálně psaný text.")
        assert cat == "Process"


# ════════════════════════════════════════════════════════════════════════════
# Scoring and Category
# ════════════════════════════════════════════════════════════════════════════
class TestScoreWord:
    def test_single_common_letter_scores_zero(self):
        assert score_word("a") == 0.0

    def test_single_unknown_alpha_scores_high(self):
        assert score_word("O") == 0.85
        assert score_word("o") == 0.85

    def test_score_word_respects_exemptions(self):
        assert score_word("PhDr.") == 0.0
        assert score_word("MUDr") == 0.0
        assert score_word("30cm") == 0.0
        assert score_word("90,9g") == 0.0
        assert score_word("vyt1") > 0.0


class TestWordWeirdRatio:
    def test_clean_line_gives_zero_ratio(self):
        pairs = score_words_in_line("tento text je fajn")
        assert compute_word_weird_ratio(pairs) <= 0.1


class TestComputeValidRatio:
    def test_clean_czech_words_all_valid(self):
        assert compute_valid_ratio("kostra hrob náramek") == 1.0

    def test_short_function_words_count_as_valid(self):
        # (#3 C) prose dense in 1-2 letter prepositions/clitics must not be
        # under-counted: every token here is valid -> ratio 1.0 (>= 0.85 gate).
        assert compute_valid_ratio("v první řadě po stříbrných penězích") >= 0.85

    def test_single_preposition_line_valid(self):
        # "z a v" are all allowed single chars / short words.
        assert compute_valid_ratio("z a v") == 1.0

    def test_garbage_short_tokens_still_invalid(self):
        # Short non-words must NOT be promoted by the short-word allowance.
        assert compute_valid_ratio("qx zp") == 0.0


class TestComputeQualityScore:
    def test_output_in_zero_one_range(self):
        q = compute_quality_score(
            valid_word_ratio=0.8,
            perplexity=200.0,
            text_length=50,
            weird_ratio=0.0,
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
