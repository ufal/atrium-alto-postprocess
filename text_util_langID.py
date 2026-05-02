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


# Default languages deemed standard for this pipeline.
COMMON_LANGS = ["ces", "deu", "eng"]
if _config.has_section("CLASSIFY") and _config.has_option("CLASSIFY", "EXPECTED_LANGS"):
    COMMON_LANGS = [lang.strip() for lang in _config.get("CLASSIFY", "EXPECTED_LANGS").split(",") if lang.strip()]

_TRUSTED_FOREIGN_LANG_BASES: frozenset = frozenset(
    lang.strip()
    for lang in _get_str("CLASSIFY", "TRUSTED_FOREIGN_LANGS", "deu,eng,fra,pol,ita").split(",")
    if lang.strip()
)


def _lang_base(lang_code: str) -> str:
    """Strip FastText script suffix: 'ces_Latn' → 'ces', 'eng_Latn' → 'eng'."""
    return lang_code.split("_")[0]


_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(l) for l in COMMON_LANGS)

PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)

# Perplexity cut-offs used inside categorize_line.
# Qwen2.5-0.5B scores clean text far lower than distilgpt2 did, so these are
# calibrated to its range.  Override in config_langID.txt if needed.
CATEG_PPL_SHORT_MAX = _get_float("TEXT_UTILS", "CATEG_PPL_SHORT_MAX", 700.0)   # was hardcoded 2000.0
CATEG_PPL_WEIRD_MAX = _get_float("TEXT_UTILS", "CATEG_PPL_WEIRD_MAX", 400.0)   # was hardcoded 1000.0

LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)
LANG_SCORE_CLEAR = _get_float("TEXT_UTILS", "LANG_SCORE_CLEAR", 0.75)

# add these six lines after the existing _get_float calls for the thresholds
QS_WEIGHT_VALID_WORD = _get_float("TEXT_UTILS", "QS_WEIGHT_VALID_WORD", 0.3)
QS_WEIGHT_SYMBOL     = _get_float("TEXT_UTILS", "QS_WEIGHT_SYMBOL",     0.2)
QS_WEIGHT_WEIRD      = _get_float("TEXT_UTILS", "QS_WEIGHT_WEIRD",      0.2)
QS_WEIGHT_PERPLEXITY = _get_float("TEXT_UTILS", "QS_WEIGHT_PERPLEXITY", 0.2)
QS_WEIGHT_LENGTH     = _get_float("TEXT_UTILS", "QS_WEIGHT_LENGTH",     0.1)
QS_LENGTH_MAX        = _get_float("TEXT_UTILS", "QS_LENGTH_MAX",        100.0)

CATEG_GARBAGE_DENSITY_HIGH  = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_HIGH",  0.35)
CATEG_GARBAGE_DENSITY_SHORT = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_SHORT", 0.20)
CATEG_GARBAGE_SHORT_WC      = _config.getint("TEXT_UTILS", "CATEG_GARBAGE_SHORT_WC", fallback=3)
CATEG_TRASH_SCORE_MAX       = _get_float("TEXT_UTILS", "CATEG_TRASH_SCORE_MAX",       0.40)
CATEG_NOISY_SCORE_MAX       = _get_float("TEXT_UTILS", "CATEG_NOISY_SCORE_MAX",       0.70)

# Characters allowed inside words without triggering the "strange symbol" penalty.
ALLOWED_INTERNAL: frozenset = frozenset(_get_str("TEXT_UTILS", "ALLOWED_INTERNAL", '.-,+()"\'/_—–:%;?!/'))

# Characters stripped from the edges of words before evaluation.
_STRIP_CHARS: str = _get_str("TEXT_UTILS", "STRIP_CHARS", '.,;:!?()[]"\'/\\')

# Standard regex for isolating specific structural errors.
RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL: re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')  # e.g., "a1b"
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r'[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]')

_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": frozenset("áčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ"),
    "deu": frozenset("äöüßÄÖÜ"),
}


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------

def infer_lang_from_diacritics(text: str, expected_bases: frozenset, threshold: float = 0.07) -> str | None:
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
    # Exclude / and - from being counted as garbage
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in ' ,.?!()/-')
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
        if len(core) < 4: continue
        for ch in set(core):
            # Trigger if character makes up 40% of the word AND appears at least 3 times
            if core.count(ch) / len(core) >= 0.40 and core.count(ch) >= 3:
                count += 1
                break
    return count


def compute_vowel_ratio(text: str) -> float:
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars: return 0.0
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    return sum(1 for c in alpha_chars if c in vowels) / len(alpha_chars)


def detect_gibberish_words(text: str) -> int:
    words = text.split()
    if not words:
        return 0

    count = 0
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    for word in words:
        core = word.strip(_STRIP_CHARS)
        if len(core) < 4:
            continue

        if len(core) > 0:
            numeric_chars = sum(1 for c in core if c.isdigit() or c in '-./,;:')
            if numeric_chars / len(core) >= 0.6:
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
        if len(core) < 2 or core.isupper(): continue

        flagged = False
        lower_run = 0
        for ch in core:
            if ch.islower():
                lower_run += 1
            elif ch.isupper() and lower_run >= 1:
                flagged = True
                break

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


def is_all_caps_line(text: str) -> bool:
    alpha_words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not alpha_words: return False
    return all(w.isupper() for w in alpha_words)


# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    clean_text = line.strip()
    if not clean_text: return "Empty", ""

    # Expanded metadata markers
    metadata_markers = [
        "Tb.", "č.neg", "neg.", "obr.", "obr ", "neg ", "Tb ", "č. neg",
        "č neg", "č.neg.", "neg.", "neg ", "Tb.", "Tb ", "č.neg.",
        "č. neg.", "č neg.", "č.", "str.", "Datum"
    ]
    if any(marker.lower() in clean_text.lower() for marker in metadata_markers):
        return "Process", clean_text

    if clean_text.startswith('"') and not clean_text.endswith('"'):
        clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'):
        clean_text = '"' + clean_text

    n_chars = len(clean_text)

    # NEW: Allow highly numeric lines (dates, measurements) to survive
    if sum(c.isdigit() for c in clean_text) / n_chars > 0.4:
        return "Process", clean_text

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

    # Penalise isolated characters that are not common single-letter words.
    # Common grammatical single letters across EN / CS / DE are whitelisted (0 weirdness).
    # Everything else (e.g. stray 'C', 's', 'W') receives a high weirdness score so
    # fragmented lines can no longer hide behind a low average.
    # Penalise isolated characters that are not common single-letter words.
    if len(core) == 1:
        # Expanded whitelist: includes prepositions + common archival initials (p. = pan, d. = doktor/den, etc.)
        if core in "aAiIoOuUvVzZkKsSpPbBjJdDrRnNmMtT" or '.' in word:
            return 0.0
        if core.isdigit():
            return 0.25  # Tolerable penalty for isolated numbers/measurements
        if not core.isalpha():
            return 0.0  # Forgive surviving punctuation separators like '-'
        return 0.85  # Severe weirdness for random isolated letters

    if len(core) < 2:
        return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)

    has_rep = False
    if len(core) >= 4:
        for ch in set(core):
            if core.count(ch) / len(core) >= 0.40 and core.count(ch) >= 3:
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
    if len(core) >= 2 and not core.isupper():
        lower_run = 0
        for ch in core:
            if ch.islower():
                lower_run += 1
            elif ch.isupper() and lower_run >= 1:
                has_uppercase = True
                break

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
# Categorisation
# ---------------------------------------------------------------------------

def categorize_line(
        quality_score: float,
        text_source: str,
        wc: int,
        weird_ratio: float,
        vowel_ratio: float,
        perplexity: float
) -> str:
    if wc == 0 or not text_source.strip():
        return "Empty"

    g_density = compute_garbage_density(text_source)
    if g_density > CATEG_GARBAGE_DENSITY_HIGH or (
            wc <= CATEG_GARBAGE_SHORT_WC and g_density > CATEG_GARBAGE_DENSITY_SHORT):
        return "Trash"

    # FIXED: Relax the fragmentation check to prevent valid measurement lines (like "145 mm, pr...")
    # from being marked as Trash. Add a weird_ratio requirement.
    avg_word_len = sum(len(w.strip(_STRIP_CHARS)) for w in text_source.split()) / wc if wc > 0 else 0
    if wc >= 5 and avg_word_len < 2.0 and weird_ratio > 0.1:
        return "Trash"

    # Catch 1: High perplexity on short lines (e.g., "z.6Z. 1369/o")
    if perplexity > CATEG_PPL_SHORT_MAX and wc < 5:
        # Bypass perplexity trap for lines that are purely Roman numerals or basic punctuation
        if not all(c in "IVXLCDMivxlcdm.-, " for c in text_source):
            if g_density < 0.1 and weird_ratio < 0.20:
                return "Noisy"
            return "Trash"

    # Catch 7: Single-character fragmentation (spaced out text)
    # e.g., "C A s 8." - Punishes lines where 50%+ of words are isolated characters
    single_char_ratio = sum(1 for w in text_source.split() if len(w.strip(_STRIP_CHARS)) <= 1) / wc if wc > 0 else 0
    if wc >= 3 and single_char_ratio >= 0.50 and weird_ratio > 0.15:
            return "Trash"

    # Catch 2: Extremely skewed vowel ratios indicating random consonants/vowels (e.g., "FAXAPOOXAXXXX")
    if len(text_source) > 5 and (vowel_ratio < 0.1 or vowel_ratio > 0.9):
        return "Trash"

    # Catch 3: High overall word weirdness (e.g., "0YM2aAS2AMOSA2CXs")
    if weird_ratio >= 0.25:
        return "Trash"

    # Catch 4: Moderately high weirdness combined with high perplexity
    if weird_ratio > 0.15 and perplexity > CATEG_PPL_WEIRD_MAX:
        return "Trash"

    if quality_score < CATEG_TRASH_SCORE_MAX:
        return "Trash"
    if quality_score < CATEG_NOISY_SCORE_MAX:
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


def compute_quality_score(valid_word_ratio: float, symbol_ratio: float, perplexity: float, text_length: int,
                          weird_ratio: float, ppl_max: float = PERPLEXITY_THRESHOLD_MAX,
                          length_max: float = QS_LENGTH_MAX) -> float:
    norm_symbol = min(symbol_ratio, 1.0)
    norm_ppl    = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len    = min(text_length / length_max, 1.0)
    norm_weird  = 1.0 - min(weird_ratio, 1.0)

    return (
        QS_WEIGHT_VALID_WORD * valid_word_ratio
        + QS_WEIGHT_SYMBOL   * (1.0 - norm_symbol)
        + QS_WEIGHT_WEIRD    * norm_weird
        + QS_WEIGHT_PERPLEXITY * norm_ppl
        + QS_WEIGHT_LENGTH   * norm_len
    )