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

# ---------------------------------------------------------------------------
# Configuration & Regular Expressions
# ---------------------------------------------------------------------------

# Languages deemed standard for this pipeline. If a short line is detected
# outside this set, it is subjected to stricter quality thresholds.
COMMON_LANGS = ["ces", "deu", "eng"]

# Perplexity thresholds (used with the distilgpt2 model)
# Higher perplexity means the model finds the text more "surprising" (likely gibberish).
PERPLEXITY_THRESHOLD_MAX = 5000
PERPLEXITY_THRESHOLD_MIN = 1500

# Minimum confidence scores required from the FastText language ID model.
LANG_SCORE_ROUGH = 0.45
LANG_SCORE_CLEAR = 0.75

# Characters allowed inside words without triggering the "strange symbol" penalty.
ALLOWED_INTERNAL: frozenset = frozenset('.-,+()"\'/_—–')

# Characters stripped from the edges of words before evaluation.
_STRIP_CHARS: str = '.,;:!?()[]"\'/\\'

# Standard regex for isolating specific structural errors.
RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL: re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')  # e.g., "a1b"
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')

# Regex to catch common OCR mathematical/garbage clusters (e.g., "~=", "A=B", heavy accents)
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r'[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]')


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------

def compute_garbage_density(text: str) -> float:
    """
    Calculates the ratio of non-alphanumeric noise characters relative to line length.

    Args:
        text (str): The raw text of the line.

    Returns:
        float: A ratio between 0.0 and 1.0 representing noise density.
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
        text (str): The text line.

    Returns:
        int: The count of corrupted words.
    """
    count = 0
    for word in text.split():
        # Remove normal edge punctuation
        core = word.strip(_STRIP_CHARS)
        if not core: continue

        # If any character inside the stripped word is illegal, flag the word
        for ch in core:
            if not ch.isalnum() and ch not in ALLOWED_INTERNAL:
                count += 1
                break
    return count


def detect_repeated_chars(text: str) -> int:
    """
    Identifies words containing an abnormally high density of a single repeated character.
    (e.g., a word that is 40%+ composed of the letter 'x').

    Args:
        text (str): The text line.

    Returns:
        int: The count of words failing this check.
    """
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if len(core) < 3: continue

        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL: continue

            # If a single non-standard char makes up >= 40% of the word, flag it.
            if core.count(ch) / len(core) >= 0.40:
                count += 1
                break
    return count


def detect_gibberish_words(text: str) -> int:
    """
    Counts words (>= 7 chars) that are entirely uppercase or completely lack vowels.
    This effectively catches garbled strings of consonants produced by OCR failure.

    Args:
        text (str): The text line.

    Returns:
        int: The number of gibberish words found.
    """
    count = 0
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if len(core) >= 7:
            if core.isupper() or not any(c in vowels for c in core):
                count += 1
    return count


def pre_filter_line(line: str) -> tuple[str, str]:
    """
    A fast CPU-bound filter to drop empty or pure-number/symbol lines before
    they reach the expensive GPU/model pipeline.

    Args:
        line (str): The raw input line.

    Returns:
        tuple[str, str]: A tuple containing the Category ("Empty", "Non-text", or "Process")
                         and the cleaned string.
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

    # If the line is extremely short or lacks variation, skip it
    if n_chars < 4 or len(unique_symbols) < 3:
        return "Non-text", clean_text

    # If less than 30% of the line consists of actual letters, skip it
    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3:
        return "Non-text", clean_text

    # Fallback to regex check
    if is_non_text(clean_text):
        return "Non-text", clean_text

    return "Process", clean_text


def detect_letter_digit_letter(text: str) -> int:
    """Detects OCR fusing errors like 'w0rd' where a digit is sandwiched by letters."""
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
    """Detects unexpected uppercase letters sitting inside otherwise lowercase words."""
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

        # Alternate check: Check for CamelCase anomalies at the start
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
# Per-Word Weirdness Scoring
# ---------------------------------------------------------------------------

def score_word(word: str) -> float:
    """
    Calculates a 'weirdness' score [0.0 - 1.0] for a single word based on
    a weighted combination of structural OCR flaws.
    """
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

    # Linearly combine the boolean flags into a final severity score
    return min(1.0, 0.40 * has_strange + 0.35 * has_rep + 0.15 * has_ldl + 0.10 * has_uppercase)


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    return [(w, score_word(w)) for w in text.split()]


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    """Returns the mean weirdness score for all words in the line."""
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
        texts: A list of string lines to evaluate.
        model: The loaded HuggingFace model.
        tokenizer: The tokenizer mapped to the model.
        device: 'cuda' or 'cpu'.

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
# Categorisation
# ---------------------------------------------------------------------------

def categorize_line(ppl: float, text_source: str, sym_count: int, upper_count: int, lang: str = "ces",
                    lang_score: float = 1.0) -> str:
    """
    The main decision tree determining the final category of a line based on all extracted metrics.

    Args:
        ppl: The calculated perplexity.
        text_source: Raw text string.
        sym_count: Amount of words with strange symbols.
        upper_count: Amount of words with mid-word uppercase errors.
        lang: Predicted language code (e.g., "ces").
        lang_score: FastText confidence score.

    Returns:
        str: "Trash", "Noisy", or "Clear".
    """
    wc = len(text_source.split())

    # 1. Density and Garbage Cluster Checks
    g_density = compute_garbage_density(text_source)
    has_clusters = bool(RE_GARBAGE_CLUSTERS.search(text_source))

    # Immediate Trash for high symbol density (catches single-token math garbage like N=W=NM)
    if g_density > 0.35 or (wc <= 3 and g_density > 0.20):
        return "Trash"

    # Immediate Trash for math/OCR clusters combined with unexpected languages
    if has_clusters and (lang not in COMMON_LANGS or lang_score < 0.6):
        return "Trash"

    # 2. Extract standard structural counts
    rep_count = detect_repeated_chars(text_source)
    fuse_count = detect_letter_digit_letter(text_source)
    gibberish_count = detect_gibberish_words(text_source)

    sym_ratio = sym_count / wc if wc > 0 else 0.0
    fuse_ratio = fuse_count / wc if wc > 0 else 0.0

    # 3. Short Line Strict Language Check
    if wc < 7:
        if lang_score < LANG_SCORE_ROUGH or lang not in COMMON_LANGS:
            return "Trash"

    # 4. Long Line Rescue
    # If a line is long, has 2 symbol flaws, but extremely low perplexity, downgrade penalty to Noisy
    if sym_count == 2 and wc >= 8 and ppl < PERPLEXITY_THRESHOLD_MIN:
        return "Noisy"

    # 5. Standard Trash Thresholds
    if (sym_count >= 2 or sym_ratio >= 0.5 or (sym_count == 1 and rep_count > 0) or
            (sym_count >= 1 and upper_count >= 1) or fuse_count >= 2 or fuse_ratio >= 0.5 or
            (sym_count >= 1 and fuse_count >= 1) or gibberish_count > 0):
        return "Trash"

    # Adjust perplexity tolerance for short lines
    coef = 1.0 if wc >= 7 else 99999

    # 6. Standard Noisy Thresholds
    if sym_count == 1 or fuse_count >= 1 or upper_count > 0 or ppl >= PERPLEXITY_THRESHOLD_MIN * coef:
        return "Noisy"

    # 7. Default
    return "Clear"


def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    """
    Parses hyphenated word splits originating from previous processing stages.
    Looks for the pattern: `prefix- {full_word}`.
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

    merged_text = re.sub(pattern, replace_match, clean_line)
    return merged_text, last_prefix, last_suffix


# ---------------------------------------------------------------------------
# Simple Ratio & Score helpers
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
    """Calculates a unified, linear-weighted confidence score (0.0 to 1.0)."""
    norm_symbol = min(symbol_ratio, 1.0)
    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)
    return 0.4 * valid_word_ratio + 0.3 * (1.0 - norm_symbol) + 0.2 * norm_ppl + 0.1 * norm_len


def classify_by_score(score: float) -> str:
    """Fallback mechanism if rule-based classification disagrees with pipeline classification."""
    if score > 0.75: return "Clear"
    if score >= 0.45: return "Noisy"
    return "Trash"


def classify_pipeline(text: str, word_set: set | None = None) -> str:
    if not text or not text.strip(): return "Empty"
    if is_non_text(text): return "Non-text"
    symbol_ratio = compute_symbol_ratio(text)
    valid_ratio = compute_valid_ratio(text, word_set)
    if symbol_ratio > 0.5 and valid_ratio < 0.2: return "Trash"
    if valid_ratio > 0.75 and symbol_ratio < 0.04: return "Clear"
    if valid_ratio > 0.4: return "Noisy"
    return "Trash"