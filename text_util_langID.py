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

_config = configparser.RawConfigParser()
_config_path = Path("config_langID.txt")
if _config_path.exists():
    _config.read(_config_path)


def _get_float(section, key, default):
    return _config.getfloat(section, key, fallback=default) if _config.has_section(section) else default


def _get_str(section, key, default):
    return _config.get(section, key, fallback=default) if _config.has_section(section) else default


COMMON_LANGS = ["ces", "deu", "eng"]
if _config.has_section("CLASSIFY") and _config.has_option("CLASSIFY", "EXPECTED_LANGS"):
    COMMON_LANGS = [lang.strip() for lang in _config.get("CLASSIFY", "EXPECTED_LANGS").split(",") if lang.strip()]

_TRUSTED_FOREIGN_LANG_BASES: frozenset = frozenset(
    lang.strip()
    for lang in _get_str("CLASSIFY", "TRUSTED_FOREIGN_LANGS", "deu,eng,fra,pol,ita").split(",")
    if lang.strip()
)


def _lang_base(lang_code: str) -> str:
    return lang_code.split("_")[0]


_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(l) for l in COMMON_LANGS)

PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)

CATEG_PPL_SHORT_MAX = _get_float("TEXT_UTILS", "CATEG_PPL_SHORT_MAX", 700.0)
CATEG_PPL_WEIRD_MAX = _get_float("TEXT_UTILS", "CATEG_PPL_WEIRD_MAX", 400.0)
SHORT_PPL_CAP = _get_float("TEXT_UTILS", "SHORT_PPL_CAP", 850.0)

LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)
LANG_SCORE_CLEAR = _get_float("TEXT_UTILS", "LANG_SCORE_CLEAR", 0.75)

# Core signal weights — must sum to 1.0 across all ten components.
QS_WEIGHT_VALID_WORD = _get_float("TEXT_UTILS", "QS_WEIGHT_VALID_WORD", 0.25)
QS_WEIGHT_SYMBOL     = _get_float("TEXT_UTILS", "QS_WEIGHT_SYMBOL",     0.13)
QS_WEIGHT_WEIRD      = _get_float("TEXT_UTILS", "QS_WEIGHT_WEIRD",      0.13)
QS_WEIGHT_PERPLEXITY = _get_float("TEXT_UTILS", "QS_WEIGHT_PERPLEXITY", 0.15)
QS_WEIGHT_LENGTH     = _get_float("TEXT_UTILS", "QS_WEIGHT_LENGTH",     0.05)
# Extended signal weights (previously checked ad-hoc inside _determine_category)
QS_WEIGHT_GARBAGE    = _get_float("TEXT_UTILS", "QS_WEIGHT_GARBAGE",    0.10)
QS_WEIGHT_VOWEL      = _get_float("TEXT_UTILS", "QS_WEIGHT_VOWEL",      0.07)
QS_WEIGHT_LANG       = _get_float("TEXT_UTILS", "QS_WEIGHT_LANG",       0.05)
QS_WEIGHT_GIBBERISH  = _get_float("TEXT_UTILS", "QS_WEIGHT_GIBBERISH",  0.04)
QS_WEIGHT_FUSED      = _get_float("TEXT_UTILS", "QS_WEIGHT_FUSED",      0.03)
QS_LENGTH_MAX        = _get_float("TEXT_UTILS", "QS_LENGTH_MAX",        100.0)

CATEG_GARBAGE_DENSITY_HIGH = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_HIGH", 0.35)
CATEG_GARBAGE_DENSITY_SHORT = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_SHORT", 0.20)
CATEG_GARBAGE_SHORT_WC = _config.getint("TEXT_UTILS", "CATEG_GARBAGE_SHORT_WC", fallback=3)

# Boundary Thresholds
CATEG_TRASH_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_TRASH_SCORE_MAX", 0.40)
CATEG_NOISY_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_NOISY_SCORE_MAX", 0.70)

ALLOWED_INTERNAL: frozenset = frozenset(_get_str("TEXT_UTILS", "ALLOWED_INTERNAL", '.-,+()"\'/—–:%;?!/'))
_STRIP_CHARS: str = _get_str("TEXT_UTILS", "STRIP_CHARS", '.,;:!?()[]"\'/\\')

RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL: re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r'[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]')
RE_ROMAN_NUMERAL: re.Pattern = re.compile(r'^[IVXLCDMivxlcdm]+\.?$')
RE_STAMP: re.Pattern = re.compile(r'^(?:[A-Za-z]+)?[\W_]*\d{2,4}\s*/\s*\d{2,4}[\W_]*$')

# Alphanumeric archive/inventory codes: letter prefix + digits, e.g. A678/2015, A1737, AG802045
RE_ARCHIVE_CODE: re.Pattern = re.compile(r'^[A-Za-z]{1,3}\d{3,}(?:/\d+)?$')
# Mixed-case alphanumeric tokens with digits, or all-caps tokens with 'X' placeholders e.g. VX5P3SosAX, FAXAPOOXAXXXX
RE_ALPHANUM_TOKEN: re.Pattern = re.compile(r'^[A-Za-z0-9]{5,}$')
# Multi-token archive/inventory references with spaces, e.g. "ČP. 10", "BZU 1982-1983 4", "P2N7-", "z.6Z. 1369/0"
RE_ARCHIVE_REF_SPACED: re.Pattern = re.compile(r'^[A-Za-záčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]{1,5}[\s.\-]+\d{1,}')

# ---------------------------------------------------------------------------
# Module-level regexes hoisted from inner functions for compile-once efficiency
# ---------------------------------------------------------------------------

# Prostrkávání / spaced-letter pattern: 4+ consecutive single uppercase letters
# separated by exactly one space, e.g. "S K U H R O V" → "Skuhrov"
_RE_SPACED_CAPS: re.Pattern = re.compile(
    r'(?<!\S)'
    r'([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ] ){3,}'
    r'[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]'
    r'(?!\S)'
)


def _collapse_spaced_caps(m: re.Match) -> str:
    letters = m.group(0).replace(' ', '')
    return letters[0].upper() + letters[1:].lower()


# Mid-word capitalisation artifact: any lowercase immediately followed by uppercase
# inside a word that is not all-caps.
_RE_MID_UPPER: re.Pattern = re.compile(r'[a-záčďéěíňóřšťůúýžäöü][A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽÄÖÜ]')

_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": frozenset("áčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ"),
    "deu": frozenset("äöüßÄÖÜ"),
}


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------

def infer_lang_from_diacritics(text: str, expected_bases: frozenset, threshold: float = 0.07) -> str | None:
    alpha = [c for c in text if c.isalpha()]
    if not alpha: return None
    for lang_code, diacs in _LANG_DIACRITICS.items():
        if lang_code not in expected_bases: continue
        ratio = sum(1 for c in alpha if c in diacs) / len(alpha)
        if ratio >= threshold: return lang_code
    return None


def compute_garbage_density(text: str) -> float:
    if not text: return 0.0

    # Pre-clean: Remove leader dots and ellipses (3 or more consecutive periods)
    # as these are structural formatting, not OCR noise.
    clean_text = re.sub(r'\.{3,}', '', text)
    if not clean_text: return 0.0

    # Calculate noise against the cleaned text length
    noise_chars = sum(1 for c in clean_text if not c.isalnum() and c not in ' ,.?!()/-')
    return noise_chars / len(clean_text)


def compute_rotatable_ratio(text: str) -> float:
    alpha_chars = [c.lower() for c in text if c.isalpha()]
    if not alpha_chars: return 0.0
    rotatable_set = frozenset("pbqdnuwmoxszeyv")
    rotatable_count = sum(1 for c in alpha_chars if c in rotatable_set)
    return rotatable_count / len(alpha_chars)


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
            # Flag 1: True OCR stutter (3 consecutive identical chars, e.g., 'hrobbb')
            if ch * 3 in core:
                count += 1
                break
            # Flag 2: Abnormal distribution, explicitly ignoring common Czech vowels
            if ch not in "aeiouyáéíóúýěůäöü" and (core.count(ch) / len(core) >= 0.40) and core.count(ch) >= 3:
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
    if not words: return 0
    count = 0
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    for word in words:
        core = word.strip(_STRIP_CHARS)
        if len(core) < 4: continue
        if len(core) > 0:
            numeric_chars = sum(1 for c in core if c.isdigit() or c in '-./,;:')
            if numeric_chars / len(core) >= 0.6: continue
        vowel_count = sum(1 for c in core if c in vowels)
        if vowel_count == 0:
            count += 1
            continue
        v_ratio = vowel_count / len(core)
        if v_ratio < 0.20 or v_ratio > 0.80:
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
    # Strict regex: any lowercase letter immediately followed by an uppercase letter
    # inside a word body is a reliable OCR mid-capitalisation artifact.
    # The word must not be all-caps (e.g. acronyms) and must be at least 2 chars.
    count = 0
    for word in text.split():
        core = word.strip('.,;:!?()[]"\'-/')
        if len(core) < 2 or core.isupper():
            continue
        if _RE_MID_UPPER.search(core):
            count += 1
    return count


def is_all_caps_line(text: str) -> bool:
    alpha_words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not alpha_words: return False
    return all(w.isupper() for w in alpha_words)


# Czech vowel-consonant alternation limit — a run of 5+ consonants or 4+ vowels
# in a row without a space is a reliable fused-word indicator.
_RE_FUSED_CONSONANT_RUN: re.Pattern = re.compile(
    r'[bcčdfghjklmnpqrřsštvwxzž]{5,}', re.IGNORECASE
)
_RE_FUSED_VOWEL_RUN: re.Pattern = re.compile(
    r'[aeiouyáéíóúýěůäöü]{4,}', re.IGNORECASE
)


def detect_fused_words(text: str) -> int:
    """
    Count tokens that are likely two Czech words merged without a space.
    Heuristics used:
      1. Token length > 14 characters (very rare in clean Czech)
      2. Consonant run of 5+ without a vowel interruption
      3. Vowel run of 4+ (rarer indicator)
    Returns the count of suspected fused tokens.
    """
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if not core or not any(c.isalpha() for c in core):
            continue
        if len(core) > 14:
            count += 1
        elif _RE_FUSED_CONSONANT_RUN.search(core):
            count += 1
        elif _RE_FUSED_VOWEL_RUN.search(core):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    clean_text = line.strip()
    if not clean_text: return "Empty", ""

    # ------------------------------------------------------------------
    # Phase 1 – OCR Normalisation (applied before any quality routing)
    # ------------------------------------------------------------------

    # 1a. Digit-letter substitution repair
    # Classic Type-1 OCR swap: isolated digit 1 inside a word → 'l',
    # and leading/mid digit 2 that forms a word-initial consonant → 'z'.
    # Only fires when the surrounding characters are alphabetic so we
    # don't corrupt genuine numbers.
    clean_text = re.sub(r'(?<=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])1(?=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])', 'l',
                        clean_text)
    clean_text = re.sub(r'\b2(?=[a-záčďéěíňóřšťůúýž])', 'z', clean_text)

    # 1b. Prostrkávání / spaced-letter repair
    # Headers OCR'd with one space between every letter, e.g.:
    #   "S K U H R O V N A D B Ě L O U"  →  "Skuhrov nad Bělou"
    # Detection: 4+ consecutive single uppercase/diacritic letters each
    # separated by exactly one space.
    clean_text = _RE_SPACED_CAPS.sub(_collapse_spaced_caps, clean_text)

    # 1c. OCR word-split repair for lone inserted characters
    # Handles cases like "Fotogra f ie" → "Fotografie" where OCR places a
    # single letter as its own token inside a word.  Conservative: only
    # collapses a lone single character that is flanked by word-body fragments
    # of at least 3 letters each on both sides.
    clean_text = re.sub(
        r'([a-záčďéěíňóřšťůúýž]{3,})\s([a-záčďéěíňóřšťůúýž])\s(?=[a-záčďéěíňóřšťůúýž]{2,})',
        lambda m: m.group(1) + m.group(2),
        clean_text,
    )

    # ------------------------------------------------------------------

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

    if is_non_text(clean_text): return "Non-text", clean_text
    if RE_ROMAN_NUMERAL.match(clean_text.strip()): return "Non-text", clean_text
    if RE_STAMP.search(clean_text) or "IVerc" in clean_text: return "Non-text", clean_text

    if sum(c.isdigit() for c in clean_text) / n_chars > 0.4: return "Process", clean_text

    unique_symbols = set(c for c in clean_text if not c.isspace())
    if n_chars < 4 or len(unique_symbols) < 3: return "Non-text", clean_text

    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3: return "Non-text", clean_text

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
    if len(core) == 1:
        if core in "aAiIoOuUvVzZkKsSpPbBjJdDrRnNmMtT" or '.' in word: return 0.0
        if core.isdigit(): return 0.25
        if not core.isalpha(): return 0.0
        return 0.85
    if len(core) < 2: return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)
    has_rep = False
    if len(core) >= 4:
        for ch in set(core):
            # Flag 1: True OCR stutter (3 consecutive identical chars, e.g., 'hrobbb')
            if ch * 3 in core:
                has_rep = True
                break
            # Flag 2: Abnormal distribution, explicitly ignoring common Czech vowels
            if ch not in "aeiouyáéíóúýěůäöü" and (core.count(ch) / len(core) >= 0.40) and core.count(ch) >= 3:
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

    # Detect leading all-caps OCR prefix on a mixed-case word (e.g. 'XXWžkumu' for 'výzkumu')
    has_caps_prefix = False
    if len(core) >= 4 and not core.isupper():
        # Count consecutive uppercase letters at the start of the token
        caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
        if caps_run >= 2 and any(c.islower() for c in core[caps_run:]):
            has_caps_prefix = True

    return min(1.0,
               0.40 * has_strange + 0.35 * has_rep + 0.15 * has_ldl + 0.10 * has_uppercase + 0.20 * has_caps_prefix)


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
        max_length = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 1024))
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        target_ids = input_ids.clone()
        target_ids[attention_mask == 0] = -100

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
# Categorisation & Clamping
# ---------------------------------------------------------------------------

def categorize_line(
        qs: float,
        txt: str,
        wc: int,
        vowel_ratio: float,
        perplexity: float,
) -> tuple[str, float]:
    """
    Assign a quality category to a processed line and return the (category, aligned_score) pair.

    Category decision depends on a single parameter — the pre-computed quality score *qs*,
    which already encodes every relevant signal (valid-word ratio, symbol density, weirdness,
    perplexity, text length, garbage density, vowel quality, language confidence, gibberish
    fraction, and fused-word fraction).

    Only three absolute structural overrides bypass the score:
      1. Empty / zero-word lines → "Empty"   (no meaningful score exists)
      2. All-caps line with near-zero vowels → "Trash"  (definitively unreadable)
      3. Ultra-low perplexity (< 50, wc ≥ 3) → "Clear"  (model is highly confident)

    All other decisions are pure threshold routing:
      qs < CATEG_TRASH_SCORE_MAX  → Trash
      qs < CATEG_NOISY_SCORE_MAX  → Noisy
      otherwise                   → Clear

    After categorisation the score is aligned to lie strictly within the numerical
    bounds of its category so downstream aggregation can rely on monotonic ordering.
    """

    def _determine_category(quality_score: float, text_source: str, word_count: int,
                            vr: float, ppl: float) -> str:
        # Override 1: empty line
        if word_count == 0 or not text_source.strip():
            return "Empty"
        # Override 2: all-caps with negligible vowels is definitively unreadable
        if is_all_caps_line(text_source) and vr < 0.10:
            return "Trash"
        # Override 3: language model is highly confident — trust it over heuristics
        if ppl < 50.0 and word_count >= 3:
            return "Clear"
        # Pure score-based routing — all other signals live inside quality_score
        if quality_score < CATEG_TRASH_SCORE_MAX:
            return "Trash"
        if quality_score < CATEG_NOISY_SCORE_MAX:
            return "Noisy"
        return "Clear"

    categ = _determine_category(qs, txt, wc, vowel_ratio, perplexity)

    # Enforce interconnected boundaries without modifying the logical routing
    if categ == "Trash":
        aligned_score = min(qs, CATEG_TRASH_SCORE_MAX - 0.0001)
    elif categ == "Noisy":
        aligned_score = max(qs, CATEG_TRASH_SCORE_MAX)
        aligned_score = min(aligned_score, CATEG_NOISY_SCORE_MAX - 0.0001)
    elif categ == "Clear":
        aligned_score = max(qs, CATEG_NOISY_SCORE_MAX)
    else:
        aligned_score = qs  # "Empty" or "Non-text"

    return categ, aligned_score


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
    # Czech postal code + city: e.g. "625 00 Brno" or "118 01 Praha 1 – Malá Strana"
    # These are readable geographic metadata and must not be discarded.
    if re.match(r'^\d{3}\s\d{2}\s+[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]', text.strip()):
        return False
    if RE_NON_TEXT.match(text.strip()): return True

    # Single-token identifiers: archive references and alphanumeric codes
    stripped = text.strip()
    if ' ' not in stripped:
        if RE_ARCHIVE_CODE.match(stripped):
            return True
        if RE_ALPHANUM_TOKEN.match(stripped):
            # Must contain a digit (e.g. VX5P3SosAX) OR be a weirdly long uppercase string with placeholders (e.g. FAXAPOOXAXXXX)
            if any(c.isdigit() for c in stripped) or (stripped.isupper() and ('X' in stripped or len(stripped) >= 10)):
                return True
    else:
        # Multi-token archive/inventory references: letter prefix(es) + digits possibly separated
        # by spaces, dots, hyphens — e.g. "ČP. 10", "BZU 1982-1983 4", "z.6Z. 1369/0", "P2N7-"
        # Guard: must be short (≤ 20 chars) and contain at least one digit to avoid
        # catching genuine two-word phrases that happen to start with a short word.
        if len(stripped) <= 20 and any(c.isdigit() for c in stripped):
            if RE_ARCHIVE_REF_SPACED.match(stripped):
                return True

    # Relaxed from 0.4 → 0.5: short lines with addresses or codes shouldn't
    # be caught by the digit ratio alone if they have real letters too.
    if len(text) < 15 and compute_digit_ratio(text) > 0.5: return True
    return False


def compute_quality_score(
        valid_word_ratio: float,
        symbol_ratio: float,
        perplexity: float,
        text_length: int,
        weird_ratio: float,
        vowel_ratio: float = 0.40,
        garbage_density: float = 0.0,
        lang_score: float | None = None,
        gibberish_ratio: float = 0.0,
        fused_ratio: float = 0.0,
        ppl_max: float = PERPLEXITY_THRESHOLD_MAX,
        length_max: float = QS_LENGTH_MAX,
) -> float:
    """
    Compute a single quality score in [0, 1] that encodes every meaningful
    signal available for an OCR text line.  This score is the *sole* input
    to the category routing in categorize_line(); no raw signal is checked
    there independently.

    Components
    ----------
    valid_word_ratio  — fraction of words that look like real words
    symbol_ratio      — fraction of non-alphanumeric, non-space characters
    weird_ratio       — mean per-word weirdness score (OCR artefact indicator)
    perplexity        — LM perplexity; low = confident valid text
    text_length       — character count; longer lines carry more evidence
    vowel_ratio       — fraction of alphabetic chars that are vowels;
                        ideal range [0.20, 0.75], penalised outside that
    garbage_density   — non-alnum density after removing leader dots;
                        normalised against CATEG_GARBAGE_DENSITY_HIGH
    lang_score        — FastText confidence for the detected language
                        (defaults to 0.5 when unavailable)
    gibberish_ratio   — fraction of words with no recognisable vowel pattern
    fused_ratio       — fraction of tokens that appear to be fused words
    """
    norm_symbol  = 1.0 - min(symbol_ratio, 1.0)
    norm_ppl     = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len     = min(text_length / length_max, 1.0)
    norm_weird   = 1.0 - min(weird_ratio, 1.0)
    norm_garbage = 1.0 - min(garbage_density / max(CATEG_GARBAGE_DENSITY_HIGH, 1e-9), 1.0)

    # Vowel quality: score 1.0 inside the ideal range [0.20, 0.75],
    # ramps down linearly to 0.0 at the extremes (0.0 and 1.0).
    vr = vowel_ratio
    if vr < 0.20:
        norm_vowel = vr / 0.20
    elif vr > 0.75:
        norm_vowel = max(0.0, 1.0 - (vr - 0.75) / 0.25)
    else:
        norm_vowel = 1.0

    norm_lang   = lang_score if lang_score is not None else 0.5
    norm_gibb   = 1.0 - min(gibberish_ratio, 1.0)
    norm_fused  = 1.0 - min(fused_ratio, 1.0)

    return (
            QS_WEIGHT_VALID_WORD * valid_word_ratio
            + QS_WEIGHT_SYMBOL   * norm_symbol
            + QS_WEIGHT_WEIRD    * norm_weird
            + QS_WEIGHT_PERPLEXITY * norm_ppl
            + QS_WEIGHT_LENGTH   * norm_len
            + QS_WEIGHT_GARBAGE  * norm_garbage
            + QS_WEIGHT_VOWEL    * norm_vowel
            + QS_WEIGHT_LANG     * norm_lang
            + QS_WEIGHT_GIBBERISH * norm_gibb
            + QS_WEIGHT_FUSED    * norm_fused
    )