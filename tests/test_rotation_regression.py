import pytest

from tests.calibration_fixtures import CLEAR, ROT_FALSE_POSITIVE_GUARDS, TRASH_INVERTED
from text_util_langID import (
    LANG_SCORE_REMAP,
    analyze_rotation_signals,
    categorize_line,
    compute_garbage_density,
    compute_quality_score,
    compute_valid_ratio,
    compute_vowel_ratio,
    compute_word_weird_ratio,
    detect_fused_words,
    detect_gibberish_words,
    override_constants,
    pre_filter_line,
    score_words_in_line,
)
from tools.const_importance_sweep import SEARCH_SPACE


def _process_mocked_line(text: str, ppl: float, orig_lang_score: float) -> str:
    """
    Mirrors the production pipeline flow (pre_filter -> compute_quality_score -> categorize_line)
    without requiring FastText or GPU models.
    """
    action, clean_text = pre_filter_line(text)
    if action != "Process":
        return action

    wc = len(clean_text.split())
    vowel_ratio = compute_vowel_ratio(clean_text)
    garbage_density = compute_garbage_density(clean_text)

    is_upright, ghost_dom = analyze_rotation_signals(clean_text)
    word_scores = score_words_in_line(clean_text)
    weird_ratio = compute_word_weird_ratio(word_scores)
    valid_ratio = compute_valid_ratio(clean_text)

    gibb_count = detect_gibberish_words(clean_text)
    fused_count = detect_fused_words(clean_text)

    # Simulate remap_lang() which caps unknown/foreign language predictions
    # In production, an inverted string like 'oueussd' without diacritics is predicted as foreign/Latn
    lang_score = min(orig_lang_score, LANG_SCORE_REMAP) if not is_upright else orig_lang_score

    qs = compute_quality_score(
        valid_word_ratio=valid_ratio,
        perplexity=ppl,
        text_length=len(clean_text),
        weird_ratio=weird_ratio,
        vowel_ratio=vowel_ratio,
        garbage_density=garbage_density,
        lang_score=lang_score,
        gibberish_ratio=(gibb_count / max(wc, 1)),
        fused_ratio=(fused_count / max(wc, 1)),
        is_upright_czech=is_upright,
    )

    categ, _ = categorize_line(
        qs=qs,
        txt=clean_text,
        wc=wc,
        vowel_ratio=vowel_ratio,
        perplexity=ppl,
        weird_ratio=weird_ratio,
        return_reason=False,
        valid_word_ratio=valid_ratio,
        lang_score=lang_score,
        orig_lang_score=orig_lang_score,
        gibberish_present=(gibb_count > 0),
        garbage_density=garbage_density,
        is_upright_czech=is_upright,
        ghost_dominated=ghost_dom,
    )
    return categ


@pytest.mark.parametrize("text, ppl, lang_score, expected, note", ROT_FALSE_POSITIVE_GUARDS + CLEAR)
def test_clean_czech_never_demoted_to_trash_default_config(text, ppl, lang_score, expected, note):
    """Invariant at default config: Clean and highly-rotated valid Czech stays out of Trash."""
    categ = _process_mocked_line(text, ppl, lang_score)
    assert categ != "Trash", f"False positive demotion at default config: {note}"


@pytest.mark.parametrize("text, ppl, lang_score, expected, note", TRASH_INVERTED)
def test_inverted_trash_stays_trash_at_default(text, ppl, lang_score, expected, note):
    """Ensure the rotation guard remains strictly effective at default configurations."""
    categ = _process_mocked_line(text, ppl, lang_score)
    assert categ == "Trash", f"Failed to catch inverted trash at default config: {note}"


@pytest.mark.parametrize("text, ppl, lang_score, expected, note", ROT_FALSE_POSITIVE_GUARDS + CLEAR)
def test_clean_czech_tuning_robustness_swept_bounds(text, ppl, lang_score, expected, note):
    """
    Tuning-robustness gate: sweep critical garbage/rotation constants across their allowed
    const_importance_sweep search space boundaries to ensure parameter tuning won't
    accidentally demote valid Czech text.
    """
    params_to_sweep = [
        "ROT_RATIO_INVERTED_MIN",
        "WEIRD_RATIO_INVERTED_MIN",
        "PPL_INVERTED_MIN",
        "SUSPICIOUS_ROT_RATIO",
        "CATEG_GARBAGE_DENSITY_HIGH",
        "SUSPICIOUS_WQX_RATIO",
        "INVERTED_WEIRD_PENALTY",
    ]

    for param in params_to_sweep:
        for bound in ("low", "high"):
            val = SEARCH_SPACE[param][bound]
            with override_constants({param: val}):
                categ = _process_mocked_line(text, ppl, lang_score)
                assert categ != "Trash", f"Regression with {param}={val} ({bound}): {note}"
