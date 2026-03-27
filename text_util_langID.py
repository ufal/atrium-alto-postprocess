#!/usr/bin/env python3
"""
text_util_langID.py

Purpose:
This module provides utility functions for the ALTO post-processing pipeline.

Core functionalities:
- Calculating text "perplexity" using a transformer model (distilgpt2).
- Structural text-quality detection via regex-based flag functions.
- Classifying text lines into quality categories (language-independent).
- Handling word-split reconstruction from annotated text files.

Category scheme:
  Empty     – blank line
  Non-text  – too short, too few letters, mostly symbols/numbers
  Trash     – structurally corrupt OCR: multiple tokens with strange symbols,
              high proportion of corrupted tokens, gibberish strings, OR failed
              language ID on short lines.
  Noisy     – degraded but recoverable: single strange-symbol token, mid-word
              uppercase artefacts, or elevated perplexity on longer lines
  Clear     – passes all checks
"""

import sys
import re
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMMON_LANGS = ["ces", "deu", "eng"]

# Perplexity thresholds (distilgpt2 scores)
PERPLEXITY_THRESHOLD_MAX = 5000
PERPLEXITY_THRESHOLD_MIN = 1500  # above this on lines with wc >= 7 → Noisy

# Legacy language-score thresholds
LANG_SCORE_ROUGH = 0.45
LANG_SCORE_CLEAR = 0.75

ALLOWED_INTERNAL: frozenset = frozenset('.-,+()"\'/_—–')
_STRIP_CHARS: str = '.,;:!?()[]"\'/\\'

RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL: re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')


# ---------------------------------------------------------------------------
# Structural text-quality detectors
# ---------------------------------------------------------------------------

def detect_strange_symbols(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if not core:
            continue
        for ch in core:
            if not ch.isalnum() and ch not in ALLOWED_INTERNAL:
                count += 1
                break
    return count


def detect_repeated_chars(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if len(core) < 3:
            continue
        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL:
                continue
            if core.count(ch) / len(core) >= 0.40:
                count += 1
                break
    return count


def detect_gibberish_words(text: str) -> int:
    """
    Count tokens that are >= 7 characters long and are either strictly uppercase
    or lack vowels entirely. This catches garbled alphabetic strings that bypass
    symbol-based checks.
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
    clean_text = line.strip()
    if not clean_text:
        return "Empty", ""

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


def detect_letter_digit_letter(text: str) -> int:
    count = 0
    for word in text.split():
        prev2, prev1 = None, None
        for ch in word:
            if (prev2 is not None and prev2.isalpha()
                    and prev1 is not None and prev1.isdigit()
                    and ch.isalpha()):
                count += 1
                break
            prev2, prev1 = prev1, ch
    return count


def detect_mid_uppercase(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip('.,;:!?()[]"\'-/')
        if len(core) < 4 or core.isupper():
            continue

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

        if not flagged and len(core) >= 5:
            upper_start = 0
            for ch in core:
                if ch.isupper():
                    upper_start += 1
                else:
                    break
            if (upper_start >= 2 and upper_start < len(core) and core[upper_start].islower()):
                flagged = True

        if not flagged and len(core) >= 6:
            prev_lower = False
            for ch in core:
                if ch.islower():
                    prev_lower = True
                elif ch.isupper():
                    if prev_lower:
                        flagged = True
                        break
                    prev_lower = False
                else:
                    prev_lower = False

        if flagged:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Per-word weirdness scoring
# ---------------------------------------------------------------------------

def score_word(word: str) -> float:
    core = word.strip(_STRIP_CHARS)
    if len(core) < 2:
        return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)

    has_rep = False
    if len(core) >= 3:
        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL:
                continue
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

        if not has_uppercase and len(core) >= 5:
            upper_start = sum(1 for _ in __import__('itertools').takewhile(str.isupper, core))
            if upper_start >= 2 and upper_start < len(core) and core[upper_start].islower():
                has_uppercase = True

        if not has_uppercase and len(core) >= 6:
            prev_lower = False
            for ch in core:
                if ch.islower():
                    prev_lower = True
                elif ch.isupper():
                    if prev_lower:
                        has_uppercase = True
                        break
                    prev_lower = False
                else:
                    prev_lower = False

    return min(1.0, 0.40 * has_strange + 0.35 * has_rep + 0.15 * has_ldl + 0.10 * has_uppercase)


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    return [(w, score_word(w)) for w in text.split()]


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    if not word_scores:
        return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)


# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------

def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    if not texts:
        return []

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

def categorize_line(ppl: float, text_source: str, sym_count: int, upper_count: int, lang: str = "ces",
                    lang_score: float = 1.0) -> str:
    rep_count = detect_repeated_chars(text_source)
    fuse_count = detect_letter_digit_letter(text_source)
    gibberish_count = detect_gibberish_words(text_source)
    wc = len(text_source.split())

    sym_ratio = sym_count / wc if wc > 0 else 0.0
    fuse_ratio = fuse_count / wc if wc > 0 else 0.0

    # --- Gibberish / Language ID Rescue for Short Lines ---
    if wc < 7:
        if lang_score < LANG_SCORE_ROUGH or lang not in COMMON_LANGS:
            return "Trash"

    # --- Rescue: sym == 2 on a long, low-perplexity line ---
    if sym_count == 2 and wc >= 8 and ppl < PERPLEXITY_THRESHOLD_MIN:
        return "Noisy"

    # --- Trash ---
    if (sym_count >= 2
            or sym_ratio >= 0.5
            or (sym_count == 1 and rep_count > 0)
            or (sym_count >= 1 and upper_count >= 1)
            or fuse_count >= 2
            or fuse_ratio >= 0.5
            or (sym_count >= 1 and fuse_count >= 1)
            or gibberish_count > 0):
        return "Trash"

    coef = 1.0 if wc >= 7 else 99999

    # --- Noisy ---
    if sym_count == 1 or fuse_count >= 1 or upper_count > 0 or ppl >= PERPLEXITY_THRESHOLD_MIN * coef:
        return "Noisy"

    return "Clear"


# ---------------------------------------------------------------------------
# Word-split parser
# ---------------------------------------------------------------------------

def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    clean_line = line_text.strip()
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"
    matches = list(re.finditer(pattern, clean_line))
    if not matches:
        return clean_line, "", ""

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
# Ratio-based quality metrics
# ---------------------------------------------------------------------------

def compute_symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / len(text)


def compute_digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def compute_valid_ratio(text: str, word_set: set | None = None) -> float:
    words = text.split()
    if not words:
        return 0.0

    valid = 0
    for word in words:
        core = word.strip(_STRIP_CHARS)
        if not core:
            continue
        if word_set is not None:
            if core.lower() in word_set:
                valid += 1
        else:
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                valid += 1

    return valid / len(words)


def is_non_text(text: str) -> bool:
    if not text:
        return False
    if RE_NON_TEXT.match(text.strip()):
        return True
    if len(text) < 15 and compute_digit_ratio(text) > 0.4:
        return True
    return False


# ---------------------------------------------------------------------------
# Score-based classification
# ---------------------------------------------------------------------------

def compute_quality_score(
        valid_word_ratio: float,
        symbol_ratio: float,
        perplexity: float,
        text_length: int,
        *,
        ppl_max: float = PERPLEXITY_THRESHOLD_MAX,
        length_max: int = 100,
) -> float:
    norm_symbol = min(symbol_ratio, 1.0)
    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)

    return (
            0.4 * valid_word_ratio
            + 0.3 * (1.0 - norm_symbol)
            + 0.2 * norm_ppl
            + 0.1 * norm_len
    )


def classify_by_score(score: float) -> str:
    if score > 0.75:
        return "Clear"
    if score >= 0.45:
        return "Noisy"
    return "Trash"


# ---------------------------------------------------------------------------
# Rule-based classification pipeline
# ---------------------------------------------------------------------------

def classify_pipeline(text: str, word_set: set | None = None) -> str:
    if not text or not text.strip():
        return "Empty"

    if is_non_text(text):
        return "Non-text"

    symbol_ratio = compute_symbol_ratio(text)
    valid_ratio = compute_valid_ratio(text, word_set)

    if symbol_ratio > 0.5 and valid_ratio < 0.2:
        return "Trash"

    if valid_ratio > 0.75 and symbol_ratio < 0.04:
        return "Clear"

    if valid_ratio > 0.4:
        return "Noisy"

    return "Trash"