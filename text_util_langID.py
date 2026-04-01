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


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------

def compute_garbage_density(text: str) -> float:
    """
    Calculates the ratio of non-alphanumeric noise characters relative to the total line length.

    Args:
        text (str): The raw text of the line to be evaluated.

    Returns:
        float: A ratio between 0.0 and 1.0 representing noise density.
               (e.g., 0.5 means half the characters are garbage).
    """
    if not text:
        return 0.0
    # Count anything that is NOT a letter/number AND NOT a standard punctuation mark
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in ' ,.?!()')
    return noise_chars / len(text)


def detect_strange_symbols(text: str) -> int:
    """
    Counts the number of *words* in a line that contain unallowed internal symbols.

    Args:
        text (str): The text line to evaluate.

    Returns:
        int: The count of corrupted words.
    """
    count = 0
    for word in text.split():
        # Remove normal edge punctuation to evaluate the core word
        core = word.strip(_STRIP_CHARS)
        if not core: continue

        # Flag the word if any internal character is illegal
        for ch in core:
            if not ch.isalnum() and ch not in ALLOWED_INTERNAL:
                count += 1
                break
    return count


def detect_repeated_chars(text: str) -> int:
    """
    Identifies words containing an abnormally high density of a single repeated character.
    Useful for catching OCR stutter (e.g., "bxxxoxx").

    Args:
        text (str): The text line to evaluate.

    Returns:
        int: The count of words failing this repeated character check.
    """
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if len(core) < 3: continue

        # Check the frequency of each unique non-alphanumeric character in the word
        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL: continue

            # If a single non-standard char makes up >= 40% of the word, flag it.
            if core.count(ch) / len(core) >= 0.40:
                count += 1
                break
    return count


def detect_gibberish_words(text: str) -> int:
    """
    Detects likely gibberish words based on extreme consonant/vowel ratios or pure uppercase formatting.

    Args:
        text (str): The text line to evaluate.

    Returns:
        int: The count of detected gibberish words.
    """
    count = 0
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if len(core) >= 7:
            # Check 1: All uppercase formatting in long words usually indicates OCR structural failure
            if core.isupper():
                count += 1
                continue

            # Check 2: The word lacks vowels entirely
            vowel_count = sum(1 for c in core if c in vowels)
            if vowel_count == 0:
                count += 1
                continue

            # Check 3: Extreme Consonant/Vowel ratio (e.g., "FAXAPOOXAXXXX")
            v_ratio = vowel_count / len(core)
            if v_ratio < 0.15 or v_ratio > 0.80:
                count += 1

    return count


def detect_letter_digit_letter(text: str) -> int:
    """
    Detects OCR fusing errors where a digit is sandwiched by letters (e.g., 'w0rd').

    Args:
        text (str): The text line to evaluate.

    Returns:
        int: The count of words containing this specific fusing error.
    """
    count = 0
    for word in text.split():
        prev2, prev1 = None, None
        for ch in word:
            # Look for the pattern: Letter -> Digit -> Letter
            if (prev2 is not None and prev2.isalpha() and prev1 is not None and prev1.isdigit() and ch.isalpha()):
                count += 1
                break
            prev2, prev1 = prev1, ch
    return count


def detect_mid_uppercase(text: str) -> int:
    """
    Detects unexpected uppercase letters sitting inside otherwise lowercase words (e.g., 'thEre').

    Args:
        text (str): The text line to evaluate.

    Returns:
        int: The count of words containing mid-word uppercase errors.
    """
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
                # Trigger if an uppercase appears after at least 2 lowercase letters
                if lower_run >= 2:
                    flagged = True
                    break
                lower_run = 0
            else:
                lower_run = 0

        # Alternate check: Look for CamelCase anomalies at the start of the word
        if not flagged and len(core) >= 5:
            upper_start = 0
            for ch in core:
                if ch.isupper():
                    upper_start += 1
                else:
                    break
            if (upper_start >= 2 and upper_start < len(core) and core[upper_start].islower()):
                flagged = True

        if flagged: count += 1
    return count


# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    """
    A fast CPU-bound filter to drop empty or pure-number/symbol lines before
    they reach the expensive GPU/model pipeline.

    Args:
        line (str): The raw input line.

    Returns:
        tuple[str, str]: A tuple containing the Category ("Empty", "Non-text", or "Process")
                         and the cleaned text string.
    """
    clean_text = line.strip()
    if not clean_text: return "Empty", ""

    # Fix dangling quotes
    if clean_text.startswith('"') and not clean_text.endswith('"'):
        clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'):
        clean_text = '"' + clean_text

    n_chars = len(clean_text)
    unique_symbols = set(c for c in clean_text if not c.isspace())

    # If the line is extremely short or lacks variation, skip processing
    if n_chars < 4 or len(unique_symbols) < 3:
        return "Non-text", clean_text

    # If less than 30% of the line consists of actual letters, skip processing
    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3:
        return "Non-text", clean_text

    # Fallback to pure regex checking for numbers/symbols
    if is_non_text(clean_text):
        return "Non-text", clean_text

    return "Process", clean_text


def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    """
    Parses hyphenated word splits originating from previous layout processing stages.
    Resolves the pattern: `prefix- {full_word}` into cohesive text.

    Args:
        line_text (str): The raw line containing potential split-word markers.

    Returns:
        tuple[str, str, str]: (merged_text, outgoing_prefix, outgoing_suffix)
    """
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

    # Replace the split markers with the resolved whole word
    merged_text = re.sub(pattern, replace_match, clean_line)
    return merged_text, last_prefix, last_suffix


# ---------------------------------------------------------------------------
# Per-Word Weirdness Scoring
# ---------------------------------------------------------------------------

def score_word(word: str) -> float:
    """
    Calculates a 'weirdness' score [0.0 - 1.0] for a single word based on
    a weighted combination of structural OCR flaws.

    Args:
        word (str): A single word token.

    Returns:
        float: A weirdness penalty score (0.0 = perfect word, 1.0 = total noise).
    """
    core = word.strip(_STRIP_CHARS)
    if len(core) < 2: return 0.0

    # Flag evaluations
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

    # Linearly combine the boolean flags into a final severity score
    return min(1.0, 0.40 * has_strange + 0.35 * has_rep + 0.15 * has_ldl + 0.10 * has_uppercase)


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    """Maps score_word across an entire line."""
    return [(w, score_word(w)) for w in text.split()]


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    """
    Calculates the mean weirdness score for all words in the line.

    Args:
        word_scores (list[tuple]): List of tuples containing (word, score).

    Returns:
        float: The average weirdness ratio.
    """
    if not word_scores: return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)


# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------

def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    """
    Calculates language perplexity using a HuggingFace CausalLM (distilgpt2).
    A high perplexity means the model struggles to predict the sequence,
    strongly indicating gibberish or severe OCR noise.

    Args:
        texts (list[str]): A list of string lines to evaluate.
        model: The loaded HuggingFace model.
        tokenizer: The tokenizer mapped to the model.
        device (str): Compute device ('cuda' or 'cpu').

    Returns:
        list[float]: A list of perplexity scores corresponding to the input lines.
    """
    if not texts: return []
    try:
        max_length = model.config.max_position_embeddings
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenize and pad the batch for parallel processing
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        # Clone inputs to act as labels (for next-token prediction)
        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100  # Ignore padding in loss

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits

            # Shift logits/labels to align prediction with the target
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(target_ids.size(0), -1)

            non_masked = shift_labels != -100
            seq_loss = (loss * non_masked).sum(dim=1)
            num_tokens = non_masked.sum(dim=1).clamp(min=1)

            # Perplexity = e ^ (cross_entropy_loss / num_tokens)
            ppl = torch.exp(seq_loss / num_tokens)
            return ppl.tolist()

    except Exception as e:
        print(f"[Error] Batch PPL: {e}", file=sys.stderr)
        return [0.0] * len(texts)


# ---------------------------------------------------------------------------
# Categorisation (REFINED PENALTY SYSTEM)
# ---------------------------------------------------------------------------

def categorize_line(ppl: float, text_source: str, lang: str, lang_score: float, expected_langs: list[str] = None) -> str:
    """
    The main decision tree determining the final category of a line based on a
    unified weighted penalty score. It dynamically evaluates structural noise alongside
    ML-derived perplexity and language confidence.

    Args:
        ppl (float): The calculated text perplexity from DistilGPT2.
        text_source (str): Raw text string to evaluate.
        lang (str): Predicted language code from FastText (e.g., "ces").
        lang_score (float): FastText language confidence score (0.0 to 1.0).
        expected_langs (list[str]): The dataset's explicit allowlist of languages. Defaults to COMMON_LANGS.

    Returns:
        str: "Trash", "Noisy", "Clear", or "Empty".
    """
    if expected_langs is None:
        expected_langs = COMMON_LANGS

    words = text_source.split()
    wc = len(words)

    if wc == 0:
        return "Empty"

    # 1. Immediate Trashing for severe baseline failure
    g_density = compute_garbage_density(text_source)
    if g_density > 0.35 or (wc <= 3 and g_density > 0.20):
        return "Trash"

    # 2. Cumulative Structural Penalty Calculation
    struct_penalties = 0.0

    sym_count = detect_strange_symbols(text_source)
    struct_penalties += sym_count * 0.4

    # Add a flat non-linear penalty for multiple strange symbols so word-count normalization doesn't wash it out.
    if sym_count >= 2:
        struct_penalties += 0.5

    struct_penalties += detect_letter_digit_letter(text_source) * 0.3
    struct_penalties += detect_mid_uppercase(text_source) * 0.2
    struct_penalties += detect_repeated_chars(text_source) * 0.4
    struct_penalties += detect_gibberish_words(text_source) * 0.5

    penalties = struct_penalties

    # 3. Dynamic Perplexity Thresholding (WITH SHORT-PHRASE GATE)
    # DistilGPT2 is English-centric. Valid short phrases often yield PPL > 5000.
    # If the text is structurally clean, short, and in a known ALLOWLISTED language, we trust the structure over PPL.
    is_forgiven_short_phrase = (wc < 5) and (lang in expected_langs) and (struct_penalties == 0.0)

    if not is_forgiven_short_phrase:
        adjusted_ppl_threshold = PERPLEXITY_THRESHOLD_MIN * (1.5 if wc < 5 else 1.0)
        if ppl > adjusted_ppl_threshold:
            penalties += 0.5
        if ppl > PERPLEXITY_THRESHOLD_MAX:
            penalties += 1.0

    # 4. Language Confidence Penalties
    if lang not in expected_langs:
        # Heavily penalize 'exotic' language guesses (not in the configurable allowlist)
        if lang_score < 0.60:
            penalties += 0.8
    elif lang_score < 0.30:
        # Standard languages get leeway, but severe unconfidence is penalized
        penalties += 0.5

    # 5. Final Categorization based on Cumulative Penalty
    # Normalize the penalty by word count to prevent long lines from being trashed by minor sum errors
    normalized_penalty = penalties / max(1.0, wc / 5.0)

    if normalized_penalty >= 1.2:
        return "Trash"
    if normalized_penalty >= 0.3:
        return "Noisy"
    return "Clear"


# ---------------------------------------------------------------------------
# Simple Ratio & General Helpers
# ---------------------------------------------------------------------------

def compute_symbol_ratio(text: str) -> float:
    """Calculates the ratio of symbols to total characters."""
    if not text: return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / len(text)


def compute_digit_ratio(text: str) -> float:
    """Calculates the ratio of digits to total characters."""
    if not text: return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def compute_valid_ratio(text: str, word_set: set | None = None) -> float:
    """Estimates the percentage of structurally 'valid' words in a line."""
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
    """Quickly checks if a line is structurally non-linguistic."""
    if not text: return False
    if RE_NON_TEXT.match(text.strip()): return True
    if len(text) < 15 and compute_digit_ratio(text) > 0.4: return True
    return False


def compute_quality_score(valid_word_ratio: float, symbol_ratio: float, perplexity: float, text_length: int, *,
                          ppl_max: float = PERPLEXITY_THRESHOLD_MAX, length_max: int = 100) -> float:
    """
    Calculates a unified, linear-weighted continuous confidence score (0.0 to 1.0)
    Useful for metadata logging and downstream analytics.
    """
    norm_symbol = min(symbol_ratio, 1.0)
    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)
    return 0.4 * valid_word_ratio + 0.3 * (1.0 - norm_symbol) + 0.2 * norm_ppl + 0.1 * norm_len