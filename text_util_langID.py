#!/usr/bin/env python3
"""
text_util.py

Purpose:
This module provides a collection of utility functions for the ALTO
post-processing pipeline.

Core functionalities include:
- Extracting text lines from ALTO XML files using 'alto-tools'.
- Calculating text "perplexity" using a transformer model (distilgpt2).
- Classifying text lines into quality categories.
- Handling word-split reconstruction from annotated text files.
"""

import sys
import re
import torch
from torch import nn

# --- Configuration ---
COMMON_LANGS = ["ces", "deu", "eng"]

# Perplexity Thresholds
PERPLEXITY_THRESHOLD_MAX = 5000  # higher is trash
PERPLEXITY_THRESHOLD_MIN = 1500  # lower is clear

# Language Score Thresholds
LANG_SCORE_ROUGH = 0.45  # lower is trash
LANG_SCORE_CLEAR = 0.75  # higher is clear



def pre_filter_line(line: str) -> tuple[str, str]:
    """Fast CPU heuristic to discard garbage before it hits the GPU."""
    clean_text = line.strip()
    if not clean_text:
        return "Empty", ""

    # Complete opening and ending " marks in the text string
    if clean_text.startswith('"') and not clean_text.endswith('"'):
        clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'):
        clean_text = '"' + clean_text

    n_chars = len(clean_text)
    unique_symbols = set(c for c in clean_text if not c.isspace())

    # Fast check for very short or non-text content
    if n_chars < 4 or len(unique_symbols) < 3:
        return "Non-text", clean_text

    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3:  # Mostly symbols/numbers
        return "Non-text", clean_text

    return "Process", clean_text


def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    """Optimized batch perplexity calculation."""
    if not texts:
        return []

    try:
        max_length = model.config.max_position_embeddings
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenize batch
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits

            # Vectorized loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(target_ids.size(0), -1)

            non_masked = (shift_labels != -100)
            seq_loss = (loss * non_masked).sum(dim=1)
            num_tokens = non_masked.sum(dim=1).clamp(min=1)

            ppl = torch.exp(seq_loss / num_tokens)
            return ppl.tolist()

    except Exception as e:
        print(f"[Error] Batch PPL: {e}", file=sys.stderr)
        return [0.0] * len(texts)


def categorize_line(lang_code, score, ppl, text_source):
    """Pure logic function to determine category."""
    is_common = any(lang_code.startswith(cl) for cl in COMMON_LANGS)

    text_lenght = len(text_source)
    words_count = len(text_source.split())
    short_line_coef = 2.0 if text_lenght < 20 or words_count < 4 else 1.0

    if score > LANG_SCORE_CLEAR and is_common:
        return "Clear"

    if (ppl >= PERPLEXITY_THRESHOLD_MAX * short_line_coef or score <= LANG_SCORE_ROUGH) and not is_common:
        return "Trash"
    if ppl >= PERPLEXITY_THRESHOLD_MIN * short_line_coef or score <= LANG_SCORE_CLEAR or not is_common:
        return "Noisy"

    return "Clear"


def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    """
    Parses a line to detect and merge split words annotated with {}.

    Example Input: "the words were divi- {divided}"
    Returns:
       merged_text: "the words were divided"
       prefix: "divi"
       suffix: "ded"
    """
    clean_line = line_text.strip()

    # Pattern finds "prefix" + hyphen + " {content}"
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"

    matches = list(re.finditer(pattern, clean_line))

    if not matches:
        return clean_line, "", ""

    last_prefix = ""
    last_suffix = ""

    # We iterate to replace ALL occurrences in the line (text cleanup),
    # but we capture the LAST match for the split state (outgoing to next line).
    def replace_match(match):
        nonlocal last_prefix, last_suffix
        prefix = match.group(1)  # e.g., "divi"
        content = match.group(2)  # e.g., "divided"

        # Calculate suffix (what remains of content after removing prefix)
        if content.startswith(prefix):
            suffix = content[len(prefix):]
        else:
            # Fallback if annotation doesn't match prefix exactly
            suffix = ""

        last_prefix = prefix
        last_suffix = suffix
        return content

    merged_text = re.sub(pattern, replace_match, clean_line)

    return merged_text, last_prefix, last_suffix