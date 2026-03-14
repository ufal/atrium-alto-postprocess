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
              OR a single strange-symbol token combined with heavy symbol
              repetition within that token
  Noisy     – degraded but recoverable: single strange-symbol token, mid-word
              uppercase artefacts, or elevated perplexity on longer lines
  Clear     – passes all checks

NOTE on digit–letter fusions (e.g. "Ma1", "vyt1ačená", "nalez2í"):
  Pure letter+digit adjacencies with no other strange characters are NOT
  detected by the current symbol-based approach because all involved
  characters are alphanumeric.  This is a known limitation of switching
  from adjacency-based to symbol-based detection.  distilgpt2 perplexity
  on short Czech strings is too unreliable to serve as a fallback.
"""

import sys
import re
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMMON_LANGS = ["ces", "deu", "eng"]   # kept for reference / downstream use

# Perplexity thresholds (distilgpt2 scores)
PERPLEXITY_THRESHOLD_MAX = 5000   # kept as constant; no longer used for Trash
PERPLEXITY_THRESHOLD_MIN = 1500   # above this on lines with wc >= 7 → Noisy

# Legacy language-score thresholds (no longer used for categorisation;
# retained so callers that import them do not break)
LANG_SCORE_ROUGH = 0.45
LANG_SCORE_CLEAR = 0.75

# ---------------------------------------------------------------------------
# Symbol-detection configuration
# ---------------------------------------------------------------------------

# Characters that may legitimately appear inside a word token and should NOT
# be treated as strange symbols:
#   '.'  – abbreviations, decimal separator  (e.g. "r.1954", "26.IX.1957")
#   '-'  – hyphens in ranges and compounds   (e.g. "1956-1959", "80-90cm")
#   ','  – Czech decimal separator           (e.g. "90,9g", "186,1 m")
#   '+'  – archival list / additive notation (e.g. "+ 1 zl.", "atypické + 1")
#
# Everything else that is neither alphanumeric nor in this set is considered
# a "strange symbol" and flagged by detect_strange_symbols().
#
# To tune sensitivity: add characters here to suppress false positives,
# or remove characters to make detection stricter.
ALLOWED_INTERNAL: frozenset = frozenset('.-,+()"\'')

# Characters stripped from the leading and trailing edges of each token
# before inspecting its interior.  These are standard sentence-level
# punctuation marks that naturally occur at word boundaries and do not
# indicate corruption when peripheral.
_STRIP_CHARS: str = '.,;:!?()[]"\'/\\'


# ---------------------------------------------------------------------------
# Structural text-quality detectors
# ---------------------------------------------------------------------------

def detect_strange_symbols(text: str) -> int:
    """
    Count whitespace-delimited tokens that contain at least one character
    which is neither alphanumeric nor in ALLOWED_INTERNAL.

    Each token is inspected after stripping leading/trailing boundary
    punctuation (_STRIP_CHARS) so that e.g. trailing colons, closing
    parentheses, or leading quotes do not inflate the count.  Only the
    interior of each token matters.

    Examples with ALLOWED_INTERNAL = {'.', '-', ',', '+', '(', ')', '"', '\''}
      "90,9g"             → ',' is allowed                            → 0
      "80-90cm"           → '-' is allowed                            → 0
      "TYRSOVA5===aras"   → '=' is not allowed                        → 1
      "LOKALITA:"         → ':' stripped at edge → "LOKALITA" → clean  → 0
      "KONĚPRUS,PCI8TT._" → '_' inside is not allowed                → 1
      "~0c,A.A4-)"        → '~' inside is not allowed                 → 1
      "kez/.e"            → '/' inside is not allowed                 → 1
      "T>r«l"             → '>' and '«' are not allowed               → 1

    Known limitation: pure letter–digit adjacencies (e.g. "Ma1", "kost1")
    contain only alphanumeric characters and are therefore not flagged.
    """
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
    """
    Count tokens where a single non-allowed, non-alphanumeric character
    accounts for >= 40 % of the token's stripped length.

    This catches heavy symbol contamination used in combination with
    detect_strange_symbols for the Trash escalation rule
    (sym == 1 AND rep > 0 → Trash).

    The same ALLOWED_INTERNAL exception set applies: '.', '-', ',', '+'
    are not counted even if they repeat (e.g. decimal numbers, ranges).

    Tokens shorter than 3 characters after stripping are skipped.

    Examples:
      "==="         → '=' is 3/3 = 100 % → 1
      "-----"       → '-' is allowed → 0
      "90,9g"       → ',' is allowed → 0
      "TYRSOVA5===" → '=' is 3/15 = 20 % → below 40 % threshold → 0
    """
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


def detect_letter_digit_letter(text: str) -> int:
    """
    Count whitespace-delimited tokens that contain at least one
    letter–digit–letter sandwich: a Unicode letter immediately followed by
    a digit immediately followed by another Unicode letter, all within the
    same token, with no separator between them.

    This pattern is the structural fingerprint of OCR digit–letter fusions
    where a digit has been erroneously inserted into the middle of a word,
    or two adjacent words have been merged with a digit between them:
      "vyt1ačená" → 't' digit 'a' → 1
      "nalez2í"   → 'z' digit 'í' → 1
      "by1a"      → 'y' digit 'a' → 1
      "Poten3te"  → 'n' digit 't' → 1

    Terminal fusions (letter–digit at end, e.g. "kost1") and initial
    fusions (digit–letter at start, e.g. "2jiStěna") are NOT caught here —
    they lack the bounding letter on the exposed side.

    No stripping is performed: the full raw token is scanned so that
    sandwiched digits adjacent to the token boundary are still caught.

    Legitimate patterns that must NOT trigger (and do not):
      "26.IX.1957" → 'IX' and '1957' are separate sub-tokens split by '.'
                     (but '.' is inside the token — prev2 would be 'X',
                      prev1 '.', ch '1' → prev1 is not a digit → no hit)
      "90,9g"      → prev2=',', prev1='9', ch='g' → prev2 is not alpha → no hit
      "80-90cm"    → prev2='0', prev1='-' (not digit) → no hit on crossing '-'
      "1956-1959"  → all digits, no alpha → no hit
    """
    count = 0
    for word in text.split():
        prev2, prev1 = None, None
        for ch in word:
            if (prev2 is not None
                    and prev2.isalpha()
                    and prev1 is not None
                    and prev1.isdigit()
                    and ch.isalpha()):
                count += 1
                break          # one hit per token is sufficient
            prev2, prev1 = prev1, ch
    return count


def detect_mid_uppercase(text: str) -> int:
    """
    Returns the number of words in the line that contain an unexpected
    uppercase letter — one that cannot be explained by normal capitalisation.
    Each qualifying word contributes at most 1 to the count (per-word, not
    per-occurrence within a word).

    Two OCR artifact patterns are detected:

    Pattern 1 – lowercase run → uppercase mid-word  (e.g. "dalSÍ", "obkLADem")
        Requires >= 2 consecutive lowercase letters immediately before the
        uppercase.  This prevents false positives on Czech academic titles:
        "PhDr", "MUDr", "RNDr", "CSc" each have only 1 lowercase before an
        internal uppercase.

    Pattern 2 – word-initial uppercase run → lowercase  (e.g. "XXWžkumu")
        The word must start with >= 2 uppercase letters immediately followed by
        a lowercase letter, AND be >= 5 characters long (excludes short acronyms
        like "ČR", "AÚ").

    Entirely uppercase words (acronyms, headings) and words shorter than 4
    characters are always skipped.
    """
    count = 0
    for word in text.split():
        core = word.strip('.,;:!?()[]"\'-/')
        if len(core) < 4 or core.isupper():
            continue

        flagged = False

        # --- Pattern 1: lower{2+} → UPPER ---
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

        # --- Pattern 2: UPPER{2+} → lower at word start (long enough word) ---
        if not flagged and len(core) >= 5:
            upper_start = 0
            for ch in core:
                if ch.isupper():
                    upper_start += 1
                else:
                    break
            if (upper_start >= 2
                    and upper_start < len(core)
                    and core[upper_start].islower()):
                flagged = True

        if flagged:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Pre-filter (fast CPU heuristic before GPU inference)
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    """
    Quick CPU-side triage.  Returns (category, cleaned_text).

    Returns "Process" for lines that should proceed to model-based scoring.
    """
    clean_text = line.strip()
    if not clean_text:
        return "Empty", ""

    # Balance lone quotation marks so the language model sees well-formed input
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

    return "Process", clean_text


# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------

def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    """Vectorised perplexity over a batch of strings using a causal LM."""
    if not texts:
        return []

    try:
        max_length = model.config.max_position_embeddings
        tokenizer.pad_token = tokenizer.eos_token

        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
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
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
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
# Categorisation  (language-independent)
# ---------------------------------------------------------------------------

def categorize_line(ppl: float, text_source: str, sym_count: int, upper_count: int) -> str:
    """
    Assign a quality category to a transcribed text line.

    Decision logic (evaluated in priority order):

      TRASH — structurally corrupt, not worth processing:
        • sym_count >= 2            multiple tokens carry strange symbols
        • sym_count == 1 AND rep_count > 0
                                    single strange-symbol token with heavy
                                    symbol repetition (e.g. "TYRSOVA5===aras")
        • fuse_count >= 2           multiple tokens contain letter–digit–letter
                                    fusions (e.g. "vyt1ačená", "by1a" in same line)
        • sym_count >= 1 AND fuse_count >= 1
                                    combination of symbol contamination and a
                                    digit-fused token in the same line

      NOISY — degraded but potentially recoverable:
        • sym_count == 1            exactly one token with a strange symbol
        • fuse_count == 1           exactly one token with a letter–digit–letter
                                    fusion (single OCR digit inserted mid-word)
        • upper_count > 0           mid-word capitalisation artefact
        • ppl >= PERPLEXITY_THRESHOLD_MIN, but ONLY when wc >= 7.
          For shorter lines (place names, single words, postal codes,
          abbreviations, form-field labels) distilgpt2 produces unreliable
          scores on Czech text and the PPL gate is disabled.

      CLEAR — passes all checks.

    Perplexity is intentionally NOT used to determine Trash.  distilgpt2 is
    an English model and assigns very high PPL to legitimate short Czech
    strings, making it an unreliable Trash signal.

    Remaining known limitation: terminal fusions (digit at end of word,
    e.g. "kost1") and initial fusions (digit at start, e.g. "2jiStěna") are
    not detected because the letter–digit–letter pattern requires a bounding
    letter on both sides of the digit.

    Args:
        ppl:         Perplexity from calculate_perplexity_batch.
        text_source: Raw text of the line.
        sym_count:   Tokens with strange symbols (from detect_strange_symbols).
        upper_count: Words with mid-word uppercase (from detect_mid_uppercase).

    Returns:
        One of: "Trash", "Noisy", "Clear".
    """
    rep_count  = detect_repeated_chars(text_source)
    fuse_count = detect_letter_digit_letter(text_source)
    wc = len(text_source.split())

    # --- Trash ---
    if (sym_count >= 2
            or (sym_count == 1 and rep_count > 0)
            or fuse_count >= 2
            or (sym_count >= 1 and fuse_count >= 1)):
        return "Trash"

    # PPL gate: only applied on longer lines where the LM score is meaningful
    coef = 1.0 if wc >= 7 else 99999

    # --- Noisy ---
    if sym_count == 1 or fuse_count >= 1 or upper_count > 0 or ppl >= PERPLEXITY_THRESHOLD_MIN * coef:
        return "Noisy"

    return "Clear"


# ---------------------------------------------------------------------------
# Word-split parser
# ---------------------------------------------------------------------------

def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    """
    Parses a line to detect and merge split words annotated with {}.

    Example input:  "the words were divi- {divided}"
    Returns:
        merged_text : "the words were divided"
        prefix      : "divi"     (the partial word at the end of this line)
        suffix      : "ded"      (the continuation expected at the start of the next line)
    """
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