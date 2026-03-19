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
              repetition within that token, OR co-occurrence of a strange-symbol
              token with mid-word uppercase artefacts in the same line
  Noisy     – degraded but recoverable: single strange-symbol token, mid-word
              uppercase artefacts, or elevated perplexity on longer lines
  Clear     – passes all checks

NOTE on known limitations:
  - Very short corrupted words (≤4 chars, e.g. "LÁzE", "Nn") are not detected
    because they are indistinguishable from abbreviations at this length.
  - Czech-phonology impossibilities (e.g. "eý", "veý") are not checked.
  - Broken initial characters (e.g. "I lzeň" for "Plzeň") require a word list.
  - Digit–letter fusions at word boundaries (e.g. "kost1", "2jiStěna") are not
    caught by detect_letter_digit_letter which requires a bounding letter on
    both sides of the digit.
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
#
#   '.'  – abbreviations, decimal separator  (e.g. "r.1954", "26.IX.1957")
#   '-'  – hyphens in ranges and compounds   (e.g. "1956-1959", "80-90cm")
#   ','  – Czech decimal separator           (e.g. "90,9g", "186,1 m")
#   '+'  – archival list / additive notation (e.g. "+ 1 zl.", "atypické + 1")
#   '/'  – archival separators and date fractions (e.g. "1/56", "14./15.",
#            "A678/2015").  Genuinely corrupted tokens that contain '/' always
#            carry at least one other strange character (~, «, &, ■, …) so
#            suppressing '/' alone does not hide corruption.
#   '_'  – typewriter underline used as a field-separator in older documents
#   '—'  – Czech em dash (U+2014), standard editorial punctuation
#   '–'  – Czech en dash (U+2013), standard editorial punctuation
#   NOTE: '•' (U+2022 bullet) is intentionally NOT included — in this
#   corpus it acts as a list separator and is a valid Noisy indicator.
#
# Everything else that is neither alphanumeric nor in this set is considered
# a "strange symbol" and flagged by detect_strange_symbols().
#
# To tune sensitivity: add characters here to suppress false positives,
# or remove characters to make detection stricter.
ALLOWED_INTERNAL: frozenset = frozenset('.-,+()"\'/_—–')

# Characters stripped from the leading and trailing edges of each token
# before inspecting its interior.  These are standard sentence-level
# punctuation marks that naturally occur at word boundaries and do not
# indicate corruption when peripheral.
_STRIP_CHARS: str = '.,;:!?()[]"\'/\\'

# Trash indicators — structural fingerprints of heavily corrupted text.
#   RE_TRASH_MULTI_SYMBOL : two or more consecutive non-word characters
#                           (e.g. "==", "~«", "##!") that are unlikely to
#                           appear in legitimate text.
#   RE_TRASH_LDL          : letter → non-alpha/non-space run → letter within a
#                           single token (e.g. "vyt1ačená", "T>r«l", "k~Ua").
#                           Complements detect_letter_digit_letter by also
#                           catching symbol-based fusions.
RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL:          re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')

# Non-text indicator — entire string consists of digits, whitespace, and
# common separator/punctuation characters with no alphabetic content.
# Examples: "1956-1959", "80-90 cm", "14./15.", "100 %"
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')

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

    Examples (with current ALLOWED_INTERNAL):
      "90,9g"             → ',' is allowed                            → 0
      "80-90cm"           → '-' is allowed                            → 0
      "1/56"              → '/' is allowed                            → 0
      "14./15."           → '/' and '.' both allowed                  → 0
      "A678/2015"         → '/' allowed                               → 0
      "TYRSOVA5===aras"   → '=' is not allowed                        → 1
      "KONĚPRUS,PCI8TT._" → '_' is now allowed; no other strange char → 0
      "~0c,A.A4-)"        → '~' inside is not allowed                 → 1
      "kez/.e"            → '/' and '.' both allowed                  → 0
      "T>r«l"             → '>' and '«' are not allowed               → 1
      "—dtto"             → '—' is now allowed                        → 0
      "•"                 → bullet is NOT allowed; still flagged        → 1
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



# ---------------------------------------------------------------------------
# Pre-filter (fast CPU heuristic before GPU inference)
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    """
    Quick CPU-side triage.  Returns (category, cleaned_text).

    Returns "Process" for lines that should proceed to model-based scoring.

    Non-text detection uses both the original heuristics and the new
    is_non_text() check (RE_NON_TEXT regex + digit-ratio), making it more
    robust against numeric-only strings that pass the letter-ratio threshold
    because of a trailing unit like "cm", "kg", or "m²".
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

    # Additional check: numeric / separator-only content (e.g. date ranges,
    # measurements) that passes the letter-ratio gate because it contains a
    # trailing alphabetic unit like "cm", "kg", "m²".
    if is_non_text(clean_text):
        return "Non-text", clean_text

    return "Process", clean_text


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

    Three OCR artifact patterns are detected:

    Pattern 1 – lowercase run → uppercase mid-word  (e.g. "dalSÍ", "obkLADem")
        Requires >= 2 consecutive lowercase letters immediately before the
        uppercase.  This prevents false positives on Czech academic titles:
        "PhDr", "MUDr", "RNDr", "CSc" each have only 1 lowercase before an
        internal uppercase.

    Pattern 2 – word-initial uppercase run → lowercase  (e.g. "XXWžkumu")
        The word must start with >= 2 uppercase letters immediately followed by
        a lowercase letter, AND be >= 5 characters long (excludes short
        acronyms like "ČR", "AÚ").

    Pattern 3 – single lowercase → uppercase in a long word  (e.g. "PřUohy",
        "rShraní", "DrAMou", "nD-lou")
        The word must be >= 6 characters long.  This lower threshold (vs.
        Pattern 1's >= 2) would create false positives for common 4-char
        Czech academic abbreviations (PhDr, MUDr, DrSc, RNDr, CSc.) — all of
        which have <= 5 characters after stripping boundary punctuation.
        A non-alpha character (digit, symbol) resets the "prev lower" flag so
        that a symbol-separated uppercase does NOT trigger (e.g. "k«Uurn"
        does not fire here; the «  interrupts the chain).

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

        # --- Pattern 3: single lower → UPPER in a long word ---
        # Catches "PřUohy" (Přílohy), "rShraní", "DrAMou", "nD-lou" type
        # OCR errors where exactly one lowercase precedes a mid-word uppercase.
        # Minimum length 6 excludes Czech academic abbreviations (≤ 5 chars).
        # A non-alpha character (digit, symbol, punctuation) resets the
        # prev_lower flag: "k«Uurn" must NOT trigger here — the «  breaks
        # the letter chain and the uppercase U starts a new segment.
        if not flagged and len(core) >= 6:
            prev_lower = False
            for ch in core:
                if ch.islower():
                    prev_lower = True
                elif ch.isupper():
                    if prev_lower:
                        flagged = True
                        break
                    prev_lower = False   # uppercase with no preceding lower: reset
                else:
                    prev_lower = False   # digit, symbol, punctuation: break chain

        if flagged:
            count += 1

    return count

# ---------------------------------------------------------------------------
# Per-word weirdness scoring
# ---------------------------------------------------------------------------

def score_word(word: str) -> float:
    """
    Return a weirdness score in [0.0, 1.0] for a single whitespace-delimited
    token by combining the four structural detectors into one number.

    Signals and weights
    -------------------
    has_strange   (0.40) – token interior contains a character outside
                           ALLOWED_INTERNAL and not alphanumeric.
                           Strongest single indicator of corruption.
    has_rep       (0.35) – a non-allowed, non-alnum character accounts for
                           >= 40 % of the stripped token length.
                           Indicates symbol-dominated noise ("===", "~~~~").
    has_ldl       (0.15) – letter→digit→letter sandwich inside the token
                           (e.g. "vyt1ačená").  Strong OCR fusion signal.
    has_uppercase (0.10) – mid-word uppercase artefact (Patterns 1-3 from
                           detect_mid_uppercase).  Weaker, more ambiguous.

    Tokens shorter than 2 characters after boundary stripping are skipped
    and returned as 0.0 (indistinguishable from abbreviations or initials).

    The four signals are binary (0 or 1) and their weighted sum is the score.
    Because at most all four fire simultaneously, the theoretical maximum is
    0.40+0.35+0.15+0.10 = 1.00, so no clamping is needed in practice.
    We still apply min(1.0, …) as a safety net.

    Examples
    --------
    "divided"    → no signals → 0.00
    "vyt1ačená"  → has_ldl   → 0.15
    "obkLADem"   → has_upper → 0.10
    "T>r«l"      → has_strange → 0.40
    "TYRS==="    → has_strange + has_rep → 0.75
    "v^UlíLa"   → has_strange + has_upper → 0.50
    """
    core = word.strip(_STRIP_CHARS)
    if len(core) < 2:
        return 0.0

    # --- Signal 1: strange symbol ---
    has_strange = any(
        not ch.isalnum() and ch not in ALLOWED_INTERNAL
        for ch in core
    )

    # --- Signal 2: repeated symbol domination ---
    has_rep = False
    if len(core) >= 3:
        for ch in set(core):
            if ch.isalnum() or ch in ALLOWED_INTERNAL:
                continue
            if core.count(ch) / len(core) >= 0.40:
                has_rep = True
                break

    # --- Signal 3: letter–digit–letter fusion ---
    has_ldl = False
    prev2, prev1 = None, None
    for ch in core:
        if (prev2 is not None and prev2.isalpha()
                and prev1 is not None and prev1.isdigit()
                and ch.isalpha()):
            has_ldl = True
            break
        prev2, prev1 = prev1, ch

    # --- Signal 4: mid-word uppercase artefact ---
    has_uppercase = False
    if len(core) >= 4 and not core.isupper():
        # Pattern 1: lower{2+} → UPPER
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

        # Pattern 2: UPPER{2+} → lower at word start (len >= 5)
        if not has_uppercase and len(core) >= 5:
            upper_start = sum(
                1 for _ in __import__('itertools').takewhile(str.isupper, core)
            )
            if (upper_start >= 2
                    and upper_start < len(core)
                    and core[upper_start].islower()):
                has_uppercase = True

        # Pattern 3: single lower → UPPER in a long word (len >= 6)
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

    return min(
        1.0,
        0.40 * has_strange
        + 0.35 * has_rep
        + 0.15 * has_ldl
        + 0.10 * has_uppercase,
    )


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    """
    Score every whitespace-delimited token in *text* for weirdness.

    Returns a list of (word, score) pairs in the same order as the tokens.
    Each score is in [0.0, 1.0] as defined by score_word().

    Returns an empty list for blank input.

    Usage
    -----
    word_scores = score_words_in_line(text)
    for word, s in word_scores:
        if s > 0:
            print(f"  weird token: {word!r}  score={s:.2f}")
    """
    return [(w, score_word(w)) for w in text.split()]


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    """
    Aggregate per-word scores from score_words_in_line() to a single
    line-level weirdness ratio in [0.0, 1.0].

    Formula: arithmetic mean of all per-word scores.

    A value of 0.0 means every token scored clean; 1.0 (unreachable in
    practice) would mean every token simultaneously triggered all four
    detectors at maximum weight.

    Typical ranges in this corpus
    ------------------------------
    0.00        – fully clean line
    0.00–0.05   – one borderline token in a long line
    0.05–0.20   – one clearly corrupted token
    0.20–0.50   – several corrupted tokens (likely Noisy → Trash boundary)
    > 0.50      – majority of tokens corrupted → Trash

    Returns 0.0 for empty input (no tokens).

    Integration pattern (in process_and_write_batch)
    ------------------------------------------------
    word_scores  = score_words_in_line(text)
    weird_ratio  = compute_word_weird_ratio(word_scores)
    q_score      = compute_quality_score(
                       valid_word_ratio = 1.0 - weird_ratio,
                       symbol_ratio     = compute_symbol_ratio(text),
                       perplexity       = ppl_val,
                       text_length      = len(text),
                   )
    """
    if not word_scores:
        return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)


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

      RESCUE (evaluated before Trash) — sym == 2 on a long, coherent line:
        • sym_count == 2 AND wc >= 8 AND ppl < PERPLEXITY_THRESHOLD_MIN
          The line has enough tokens that the 2 corrupted ones are a minority,
          and distilgpt2 assigns a low-enough score to suggest meaningful
          surrounding content.  Classified as Noisy rather than Trash.
          Note: this rescue does NOT fire when ppl is high (garbled throughout)
          or when the line is short (corruption dominates).

      TRASH — structurally corrupt, not worth processing:
        • sym_count >= 2            multiple tokens carry strange symbols
        • sym_count == 1 AND rep_count > 0
                                    single strange-symbol token with heavy
                                    symbol repetition (e.g. "TYRSOVA5===aras")
        • sym_count >= 1 AND upper_count >= 1
                                    symbol corruption and mid-word uppercase
                                    co-occur in the same line — the combination
                                    of two independent corruption signals is a
                                    strong indicator of full-line garbling
                                    (e.g. "v^UlíLa uq (AAuu Aud. AnMlut")
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

    Remaining known limitations:
      - Very short corrupted words (e.g. "LÁzE", "Nn") slip through because
        they are indistinguishable from abbreviations at ≤ 4 characters.
      - Lines with a single sym token + single uppercase artefact are now
        escalated to Trash.  If a future corpus shows such combinations are
        common in legitimate text, move upper_count out of the Trash escalation.

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

    # --- Rescue: sym == 2 on a long, low-perplexity line ---
    # When exactly two tokens carry strange symbols but the surrounding words
    # are coherent enough that distilgpt2 scores the line well below the Noisy
    # PPL ceiling, classify as Noisy rather than escalating to Trash.
    # wc >= 8 ensures the two bad tokens are genuinely a minority; the PPL
    # guard ensures the remaining content is linguistically plausible.
    # Example rescued: long lines ending in clear Czech prose where one or two
    # corrupted tokens appear in the middle.
    if sym_count == 2 and wc >= 8 and ppl < PERPLEXITY_THRESHOLD_MIN:
        return "Noisy"

    # --- Trash ---
    if (sym_count >= 2
            or (sym_count == 1 and rep_count > 0)
            or (sym_count >= 1 and upper_count >= 1)
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



# ---------------------------------------------------------------------------
# Ratio-based quality metrics  (sections 3–4 of the analysis document)
# ---------------------------------------------------------------------------

def compute_symbol_ratio(text: str) -> float:
    """
    Fraction of characters in *text* that are non-alphanumeric and
    non-whitespace (i.e. punctuation, symbols, or other noise characters).

    This is the primary classification signal (section 3.1):
      low  → Clear
      mid  → Noisy
      high → Trash

    Returns 0.0 for empty strings.

    Examples:
      "clear text here"   → 0.0
      "noisy, text: here" → 2/18 ≈ 0.11
      "T>r«l ==="         → 4/10 = 0.4
    """
    if not text:
        return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / len(text)


def compute_digit_ratio(text: str) -> float:
    """
    Fraction of characters in *text* that are ASCII digits.

    Used together with length to detect Non-text strings (section 3.3).
    Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def compute_valid_ratio(text: str, word_set: set | None = None) -> float:
    """
    Fraction of whitespace-delimited tokens that are 'valid' words
    (section 3.2).

    If *word_set* is provided, a token is valid when its lowercased,
    boundary-stripped form appears in the set.  Supplying a domain
    dictionary (Czech, German, English …) gives the most accurate signal.

    If *word_set* is None a lightweight proxy heuristic is used:
      a token is considered valid when it is >= 3 characters long,
      consists predominantly of alphabetic characters (>= 70 %), and
      contains no strange symbols outside ALLOWED_INTERNAL.
    This proxy is consistent with the rest of the module's design and
    requires no external resources, but is less accurate than a real
    dictionary — accuracy note is reflected in the module docstring.

    Returns 0.0 for empty or whitespace-only strings.

    Thresholds from the analysis document (section 4):
      > 0.75  → Clear
      0.4–0.75 → Noisy
      < 0.4   → Trash
    """
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
            # Proxy: long enough, mostly alphabetic, no strange symbols
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(
                not c.isalnum() and c not in ALLOWED_INTERNAL for c in core
            )
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                valid += 1

    return valid / len(words)


def is_non_text(text: str) -> bool:
    """
    Return True when *text* contains no meaningful alphabetic content.

    Combines two complementary checks (section 3.3):
      1. Regex (RE_NON_TEXT): the entire string consists only of digits,
         whitespace, and common separator/punctuation characters.
         Examples: "1956-1959", "80-90 cm", "14./15.", "100 %"
      2. Heuristic: the string is short (< 15 chars) and > 40 % of its
         characters are digits.  Catches short numeric codes and
         measurements that slip through the regex (e.g. "4B", "2x3").

    Returns False for empty strings (those are handled as "Empty").
    """
    if not text:
        return False
    if RE_NON_TEXT.match(text.strip()):
        return True
    if len(text) < 15 and compute_digit_ratio(text) > 0.4:
        return True
    return False



# ---------------------------------------------------------------------------
# Score-based classification  (section 6 of the analysis document)
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
    """
    Compute a composite quality score in [0, 1].

    Formula (section 6):

        score = (
            0.4 * valid_word_ratio
          + 0.3 * (1 - normalized_symbol_ratio)
          + 0.2 * normalized_perplexity
          + 0.1 * length_score
        )

    Component definitions:
      normalized_symbol_ratio = min(symbol_ratio, 1.0)
      normalized_perplexity   = 1 - min(perplexity / ppl_max, 1.0)
          High perplexity → low contribution (distilgpt2-based; use as
          supporting signal only, not primary Trash indicator).
      length_score            = min(text_length / length_max, 1.0)

    Classification thresholds (use classify_by_score()):
      > 0.75   → "Clear"
      0.45–0.75 → "Noisy"
      < 0.45   → "Trash"

    Args:
        valid_word_ratio: Fraction of tokens that are valid words [0, 1].
                          Obtain via compute_valid_ratio().
        symbol_ratio:     Fraction of characters that are non-alnum, non-space
                          [0, 1].  Obtain via compute_symbol_ratio().
        perplexity:       Raw perplexity score from calculate_perplexity_batch().
        text_length:      Character length of the text.
        ppl_max:          Perplexity ceiling for normalisation.
                          Defaults to PERPLEXITY_THRESHOLD_MAX.
        length_max:       Character count mapped to length_score = 1.0.
                          Defaults to 100 characters.

    Returns:
        Composite quality score in [0.0, 1.0].
    """
    norm_symbol = min(symbol_ratio, 1.0)
    norm_ppl    = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len    = min(text_length / length_max, 1.0)

    return (
        0.4 * valid_word_ratio
        + 0.3 * (1.0 - norm_symbol)
        + 0.2 * norm_ppl
        + 0.1 * norm_len
    )


def classify_by_score(score: float) -> str:
    """
    Map a compute_quality_score() value to a category label.

    Thresholds (section 6 of the analysis document):
      > 0.75   → "Clear"
      0.45–0.75 → "Noisy"
      < 0.45   → "Trash"

    "Empty" and "Non-text" are handled upstream (classify_pipeline or
    pre_filter_line) and are never returned here.

    Args:
        score: Value in [0, 1] from compute_quality_score().

    Returns:
        One of: "Clear", "Noisy", "Trash".
    """
    if score > 0.75:
        return "Clear"
    if score >= 0.45:
        return "Noisy"
    return "Trash"


# ---------------------------------------------------------------------------
# Rule-based classification pipeline  (section 5 of the analysis document)
# ---------------------------------------------------------------------------

def classify_pipeline(
    text: str,
    word_set: set | None = None,
) -> str:
    """
    Recommended rule-based classification pipeline (section 5), implemented
    as a standalone function that is additive to the existing categorize_line()
    and pre_filter_line() approach.

    Decision order (evaluated top-to-bottom):
      1. Empty     – blank / whitespace-only string.
      2. Non-text  – purely numeric / separator content (is_non_text).
      3. Trash     – symbol_ratio > 0.5 AND valid_ratio < 0.2.
      4. Clear     – valid_ratio > 0.75 AND symbol_ratio < 0.04.
      5. Noisy     – valid_ratio > 0.4.
      6. Trash     – fallback when none of the above fire.

    This function intentionally does not call categorize_line() so it can
    be used in isolation (e.g. without a GPU, without perplexity scores) or
    combined with categorize_line() for a hybrid decision.

    Integration example (hybrid, score-weighted tiebreak):

        struct_cat = categorize_line(ppl, text, sym, upper)
        pipe_cat   = classify_pipeline(text, word_set)
        if struct_cat == pipe_cat:
            final = struct_cat           # unanimous → confident
        else:
            score = compute_quality_score(
                compute_valid_ratio(text, word_set),
                compute_symbol_ratio(text),
                ppl,
                len(text),
            )
            final = classify_by_score(score)  # score breaks tie

    Args:
        text:     The line text to classify.
        word_set: Optional set of known valid word forms for
                  compute_valid_ratio().  When None the built-in proxy
                  heuristic is used (see compute_valid_ratio docstring).

    Returns:
        One of: "Empty", "Non-text", "Trash", "Noisy", "Clear".
    """
    if not text or not text.strip():
        return "Empty"

    if is_non_text(text):
        return "Non-text"

    symbol_ratio = compute_symbol_ratio(text)
    valid_ratio  = compute_valid_ratio(text, word_set)

    # Fast Trash escalation: heavy noise + very few valid words
    if symbol_ratio > 0.5 and valid_ratio < 0.2:
        return "Trash"

    # Clear: high valid-word density, very low symbol noise
    if valid_ratio > 0.75 and symbol_ratio < 0.04:
        return "Clear"

    # Noisy: at least a moderate fraction of valid words
    if valid_ratio > 0.4:
        return "Noisy"

    return "Trash"