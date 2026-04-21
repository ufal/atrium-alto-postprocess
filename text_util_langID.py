#!/usr/bin/env python3
"""
text_util_langID.py

Purpose:
Provides the core text-processing utilities for the ALTO OCR post-processing pipeline.
This includes functions for detecting OCR noise, calculating character/symbol density,
scoring word "weirdness", and running text chunks through a GPU-accelerated Perplexity model.

Categories Outputted:
  - Empty     : A blank line.
  - Non-text  : Lines that are too short, lack letters, or are purely numbers/symbols.
  - Trash     : Severe OCR corruption, high symbol density, gibberish, or failed language ID.
  - Noisy     : Partially degraded text (e.g., isolated strange symbols, mid-word uppercase).
  - Clear     : Structurally sound text with low perplexity.
"""

import sys
import re
import torch
from torch import nn
import itertools
import configparser
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration & Regular Expressions
# ---------------------------------------------------------------------------

# Use RawConfigParser to prevent string interpolation errors on characters like '%'
_config = configparser.RawConfigParser()
_config_path = Path("config_langID.txt")
if _config_path.exists():
    _config.read(_config_path)


def _get_float(section, key, default):
    return _config.getfloat(section, key, fallback=default) if _config.has_section(section) else default


def _get_str(section, key, default):
    return _config.get(section, key, fallback=default) if _config.has_section(section) else default


# Default languages deemed standard for this pipeline. Short lines outside this set
# face stricter quality thresholds to prevent noise from being tagged as exotic languages.
COMMON_LANGS = ["ces", "deu", "eng"]
if _config.has_section("CLASSIFY") and _config.has_option("CLASSIFY", "EXPECTED_LANGS"):
    COMMON_LANGS = [lang.strip() for lang in _config.get("CLASSIFY", "EXPECTED_LANGS").split(",") if lang.strip()]

# Languages that FastText reliably identifies in Czech archival documents.
# All other detected languages are remapped to Czech (ces) since FastText
# frequently misidentifies short/noisy Czech text as exotic languages.
# Slovak (slk) is intentionally excluded — it is treated as Czech in this corpus.
_TRUSTED_FOREIGN_LANG_BASES: frozenset = frozenset(
    lang.strip()
    for lang in _get_str("CLASSIFY", "TRUSTED_FOREIGN_LANGS", "deu,eng,fra,pol,ita").split(",")
    if lang.strip()
)


def _lang_base(lang_code: str) -> str:
    """Strip FastText script suffix: 'ces_Latn' → 'ces', 'eng_Latn' → 'eng'."""
    return lang_code.split("_")[0]


_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(l) for l in COMMON_LANGS)

# Perplexity thresholds (used with the distilgpt2 model)
# Higher perplexity = the model finds the text more "surprising" (likely gibberish).
PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 5000.0)
PERPLEXITY_THRESHOLD_MIN = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MIN", 1500.0)

# Minimum confidence scores required from the FastText language ID model.
LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)
LANG_SCORE_CLEAR = _get_float("TEXT_UTILS", "LANG_SCORE_CLEAR", 0.75)

# Characters allowed inside words without triggering the "strange symbol" penalty.
ALLOWED_INTERNAL: frozenset = frozenset(_get_str("TEXT_UTILS", "ALLOWED_INTERNAL", '.-,+()"\'/_—–:%'))

# Characters stripped from the edges of words before evaluation.
_STRIP_CHARS: str = _get_str("TEXT_UTILS", "STRIP_CHARS", '.,;:!?()[]"\'/\\')

# Standard regex for isolating specific structural errors.
RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL: re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')  # e.g., "a1b"
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')

# Regex to catch common OCR mathematical/garbage clusters
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r'[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]')

# Diacritics that are strongly diagnostic of specific target languages.
_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": frozenset("áčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ"),
    "deu": frozenset("äöüßÄÖÜ"),
}


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------

def infer_lang_from_diacritics(text: str, expected_bases: frozenset, threshold: float = 0.07) -> str | None:
    """
    If FastText is unsure (low confidence or unexpected language), look at whether
    the line's character profile matches a target language's diagnostic diacritics.
    Returns the inferred language base code, or None if inconclusive.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return None
    for lang_code, diacs in _LANG_DIACRITICS.items():
        if lang_code not in expected_bases:
            continue
        ratio = sum(1 for c in alpha if c in diacs) / len(alpha)
        if ratio >= threshold:
            return lang_code
    return None


def compute_garbage_density(text: str) -> float:
    if not text:
        return 0.0
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in ' ,.?!()')
    return noise_chars / len(text)


def detect_strange_symbols(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if not core: continue
        for ch in core:
            if not ch.isalnum() and ch not in ALLOWED_INTERNAL:
                count += 1
                break
    return count


def detect_repeated_chars(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if len(core) < 3: continue
        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL: continue
            if core.count(ch) / len(core) >= 0.40:
                count += 1
                break
    return count


def detect_gibberish_words(text: str) -> int:
    words = text.split()
    if not words:
        return 0

    caps_ratio = sum(1 for w in words if w.strip(_STRIP_CHARS).isupper()) / len(words)
    is_caps_header = caps_ratio >= 0.6

    count = 0
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    for word in words:
        core = word.strip(_STRIP_CHARS)
        if len(core) < 7:
            continue

        # NEW: skip numeric/date ranges — digits, hyphens, periods, slashes
        if len(core) > 0:
            numeric_chars = sum(1 for c in core if c.isdigit() or c in '-./,')
            if numeric_chars / len(core) >= 0.6:
                continue

        if core.isupper() and not is_caps_header:
            count += 1
            continue

        vowel_count = sum(1 for c in core if c in vowels)
        if vowel_count == 0:
            count += 1
            continue

        v_ratio = vowel_count / len(core)
        if v_ratio < 0.15 or v_ratio > 0.80:
            count += 1

    return count


def detect_letter_digit_letter(text: str) -> int:
    count = 0
    for word in text.split():
        prev2, prev1 = None, None
        for ch in word:
            if (prev2 is not None and prev2.isalpha() and prev1 is not None and prev1.isdigit() and ch.isalpha()):
                count += 1
                break
            prev2, prev1 = prev1, ch
    return count


def detect_mid_uppercase(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip('.,;:!?()[]"\'-/')
        if len(core) < 4 or core.isupper(): continue

        flagged = False
        lower_run = 0
        for ch in core:
            if ch.islower():
                lower_run += 1
            elif ch.isupper():
                if lower_run >= 2:
                    flagged = True
                    break
                lower_run = 0
            else:
                lower_run = 0

        # Look for 3+ character OCR uppercase prefixes
        if not flagged and len(core) >= 5:
            upper_start = 0
            for ch in core:
                if ch.isupper():
                    upper_start += 1
                else:
                    break
            if (upper_start >= 3 and upper_start < len(core) and core[upper_start].islower()):
                flagged = True

        if flagged: count += 1
    return count


# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    clean_text = line.strip()
    if not clean_text: return "Empty", ""

    if clean_text.startswith('"') and not clean_text.endswith('"'):
        clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'):
        clean_text = '"' + clean_text

    n_chars = len(clean_text)
    unique_symbols = set(c for c in clean_text if not c.isspace())

    if n_chars < 4 or len(unique_symbols) < 3:
        return "Non-text", clean_text

    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3:
        return "Non-text", clean_text

    if is_non_text(clean_text):
        return "Non-text", clean_text

    return "Process", clean_text


def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    clean_line = line_text.strip()
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"
    matches = list(re.finditer(pattern, clean_line))
    if not matches: return clean_line, "", ""

    last_prefix = ""
    last_suffix = ""

    def replace_match(match):
        nonlocal last_prefix, last_suffix
        prefix = match.group(1)
        content = match.group(2)
        last_prefix = prefix
        last_suffix = content[len(prefix):] if content.startswith(prefix) else ""
        return content

    merged_text = re.sub(pattern, replace_match, clean_line)
    return merged_text, last_prefix, last_suffix


# ---------------------------------------------------------------------------
# Per-Word Weirdness Scoring
# ---------------------------------------------------------------------------

def score_word(word: str) -> float:
    core = word.strip(_STRIP_CHARS)
    if len(core) < 2: return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)

    has_rep = False
    if len(core) >= 3:
        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL: continue
            if core.count(ch) / len(core) >= 0.40:
                has_rep = True
                break

    has_ldl = False
    prev2, prev1 = None, None
    for ch in core:
        if prev2 is not None and prev2.isalpha() and prev1 is not None and prev1.isdigit() and ch.isalpha():
            has_ldl = True
            break
        prev2, prev1 = prev1, ch

    has_uppercase = False
    if len(core) >= 4 and not core.isupper():
        lower_run = 0
        for ch in core:
            if ch.islower():
                lower_run += 1
            elif ch.isupper():
                if lower_run >= 2:
                    has_uppercase = True
                    break
                lower_run = 0
            else:
                lower_run = 0

    return min(1.0, 0.40 * has_strange + 0.35 * has_rep + 0.15 * has_ldl + 0.10 * has_uppercase)


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    return [(w, score_word(w)) for w in text.split()]


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    if not word_scores: return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)


# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------

def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    if not texts: return []
    try:
        max_length = model.config.max_position_embeddings
        tokenizer.pad_token = tokenizer.eos_token

        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(target_ids.size(0), -1)

            non_masked = shift_labels != -100
            seq_loss = (loss * non_masked).sum(dim=1)
            num_tokens = non_masked.sum(dim=1).clamp(min=1)

            ppl = torch.exp(seq_loss / num_tokens)
            return ppl.tolist()

    except Exception as e:
        print(f"[Error] Batch PPL: {e}", file=sys.stderr)
        return [0.0] * len(texts)


# ---------------------------------------------------------------------------
# Categorisation (REFINED PENALTY SYSTEM)
# ---------------------------------------------------------------------------

def categorize_line(
    ppl: float,
    text_source: str,
    lang: str,
    lang_score: float,
    weird_ratio: float,
    expected_langs: list[str] | None = None,
    quality_score: float | None = None,
) -> str:
    """
    Assign a quality category to a classified text line.

    Language handling (revised):
      FastText regularly misidentifies short Czech text as non-European languages.
      Only languages listed in TRUSTED_FOREIGN_LANGS are taken at face value.
      All others are remapped to 'ces' (Czech) before applying any language-based
      penalty, so structurally clean text is not unfairly penalised.

    Quality score (optional modifier):
      If provided, quality_score acts as a weak secondary signal:
        - Very low (< 0.35) with existing penalties → push toward Trash
        - Very high (> 0.88) with zero structural issues → protect from Noisy
    """
    if expected_langs is None:
        expected_langs = COMMON_LANGS

    expected_bases = frozenset(_lang_base(l) for l in expected_langs)
    lang_base = _lang_base(lang)

    # ------------------------------------------------------------------
    # Language resolution: remap non-trusted languages to Czech
    # ------------------------------------------------------------------
    # FastText often labels short/noisy Czech text as Turkish, Lithuanian,
    # Maltese, Afrikaans, etc.  If the detected language is not in our
    # trusted-foreign set, assume the identification is wrong and treat
    # the line as Czech for all penalty purposes.
    if lang_base not in _TRUSTED_FOREIGN_LANG_BASES:
        effective_lang_base = "ces"
    else:
        effective_lang_base = lang_base

    in_expected = effective_lang_base in expected_bases

    # Also run the diacritic fallback for the *original* lang_base when confidence
    # is low — this catches cases where FastText picked the right script family
    # but the wrong specific language.
    if not in_expected and lang_score < 0.55:
        inferred = infer_lang_from_diacritics(text_source, expected_bases)
        if inferred is not None:
            effective_lang_base = inferred
            in_expected = True

    words = text_source.split()
    wc = len(words)

    if wc == 0:
        return "Empty"

    # ------------------------------------------------------------------
    # Immediate Trash overrides (structural — language-independent)
    # ------------------------------------------------------------------
    g_density = compute_garbage_density(text_source)
    if g_density > 0.35 or (wc <= 3 and g_density > 0.20):
        return "Trash"

    if ppl > 500.0 and weird_ratio > 0.4:
        return "Trash"

    # ------------------------------------------------------------------
    # Structural penalty accumulation
    # ------------------------------------------------------------------
    struct_penalties = 0.0

    sym_count = detect_strange_symbols(text_source)
    struct_penalties += sym_count * 0.4
    if sym_count >= 2:
        struct_penalties += 0.5

    upper_count = detect_mid_uppercase(text_source)
    upper_weight = 0.35 if (wc <= 2 and upper_count >= 1) else 0.2
    struct_penalties += upper_count * upper_weight

    struct_penalties += detect_letter_digit_letter(text_source) * 0.3
    struct_penalties += detect_repeated_chars(text_source) * 0.4
    struct_penalties += detect_gibberish_words(text_source) * 0.5

    penalties = struct_penalties

    # ------------------------------------------------------------------
    # Perplexity penalty
    # Skipped for clean short phrases in expected/remapped-to-expected language.
    # NOTE: effective_lang_base is used here, so remapped-Czech lines benefit
    # from the same forgiveness as lines FastText correctly identified as Czech.
    # ------------------------------------------------------------------
    is_forgiven_short_phrase = (wc < 5) and in_expected and (struct_penalties == 0.0)

    if not is_forgiven_short_phrase:
        adjusted_min = PERPLEXITY_THRESHOLD_MIN * (1.5 if wc < 5 else 1.0)
        if ppl > adjusted_min:
            penalties += 0.5
        if ppl > PERPLEXITY_THRESHOLD_MAX:
            penalties += 1.0

    # ------------------------------------------------------------------
    # Language confidence penalty (now only for genuinely trusted-foreign langs
    # that are NOT in the expected set, or for expected langs with very low confidence)
    # ------------------------------------------------------------------
    if not in_expected:
        # Only reached when effective_lang_base is a trusted-foreign lang
        # that's not in expected_langs (e.g. French in a Czech-only config).
        if lang_score < 0.60:
            penalties += 0.8
    else:
        # In expected (or remapped to Czech): mild penalty for very low confidence
        if lang_score < 0.30:
            penalties += 0.5

    # ------------------------------------------------------------------
    # Final classification via normalized penalty
    # ------------------------------------------------------------------
    normalized_penalty = penalties / max(1.0, wc / 5.0)

    # Quality score as a weak secondary modifier
    # This does not replace the penalty system; it nudges genuinely ambiguous cases.
    if quality_score is not None:
        # Very poor quality AND already penalised → confirm Trash
        if quality_score < 0.35 and normalized_penalty >= 0.15:
            return "Trash"
        # Excellent quality AND structurally clean → protect from Noisy
        if quality_score >= 0.88 and struct_penalties == 0.0 and normalized_penalty < 0.5:
            return "Clear"

    if normalized_penalty >= 1.2:
        return "Trash"
    if normalized_penalty >= 0.3:
        return "Noisy"
    return "Clear"
# ---------------------------------------------------------------------------
# Simple Ratio & General Helpers
# ---------------------------------------------------------------------------

def compute_symbol_ratio(text: str) -> float:
    if not text: return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / len(text)


def compute_digit_ratio(text: str) -> float:
    if not text: return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def compute_valid_ratio(text: str, word_set: set | None = None) -> float:
    words = text.split()
    if not words: return 0.0
    valid = 0
    for word in words:
        core = word.strip(_STRIP_CHARS)
        if not core: continue
        if word_set is not None:
            if core.lower() in word_set: valid += 1
        else:
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                valid += 1
    return valid / len(words)


def is_non_text(text: str) -> bool:
    if not text: return False
    if RE_NON_TEXT.match(text.strip()): return True
    if len(text) < 15 and compute_digit_ratio(text) > 0.4: return True
    return False


def compute_quality_score(valid_word_ratio: float, symbol_ratio: float, perplexity: float, text_length: int, *,
                          ppl_max: float = PERPLEXITY_THRESHOLD_MAX, length_max: int = 100) -> float:
    norm_symbol = min(symbol_ratio, 1.0)
    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)
    return 0.4 * valid_word_ratio + 0.3 * (1.0 - norm_symbol) + 0.2 * norm_ppl + 0.1 * norm_len