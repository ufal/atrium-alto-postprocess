#!/usr/bin/env python3
"""
text_util.py

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

import configparser
import itertools
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Ablation Kill-Switch (Part B)
# Inject rule names here via override_constants to disable them for ablation sweeps.
# ---------------------------------------------------------------------------
DISABLED_RULES: frozenset = frozenset()

# ---------------------------------------------------------------------------
# Rule-Fire Coverage Instrumentation (Increment B5)
# When RULE_FIRE_COUNTS is not None (i.e. inside a rule_fire_capture() block),
# every _fire(name) call increments the counter for that rule. Outside a capture
# block _fire() is a no-op so there is zero overhead during normal production runs.
# ---------------------------------------------------------------------------
RULE_FIRE_COUNTS: dict | None = None


def _fire(name: str) -> None:
    """Register a single rule execution against the active capture context."""
    if RULE_FIRE_COUNTS is not None:
        RULE_FIRE_COUNTS[name] = RULE_FIRE_COUNTS.get(name, 0) + 1


@contextmanager
def rule_fire_capture():
    """Context manager that enables rule-fire counting for the enclosed block.

    Yields the live counts dict so callers can inspect it after (or during)
    the run.
    """
    global RULE_FIRE_COUNTS
    prev, RULE_FIRE_COUNTS = RULE_FIRE_COUNTS, {}
    try:
        yield RULE_FIRE_COUNTS
    finally:
        RULE_FIRE_COUNTS = prev


# ---------------------------------------------------------------------------
# Configuration & Regular Expressions
# ---------------------------------------------------------------------------

_config = configparser.RawConfigParser()
_config_path = Path(os.getenv("LANGID_CONFIG", "setup/config.txt"))
if _config_path.exists():
    _config.read(_config_path)


def _get_float(section, key, default):
    return _config.getfloat(section, key, fallback=default) if _config.has_section(section) else default


def _get_str(section, key, default):
    return _config.get(section, key, fallback=default) if _config.has_section(section) else default


def _get_int(section, key, default):
    return _config.getint(section, key, fallback=default) if _config.has_section(section) else default


def _get_csv_set(section, key, default):
    """Parse a comma-separated config value into a frozenset of stripped tokens."""
    raw = _get_str(section, key, default)
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


COMMON_LANGS = ["ces", "deu", "eng"]
if _config.has_section("CLASSIFY") and _config.has_option("CLASSIFY", "EXPECTED_LANGS"):
    COMMON_LANGS = [lang.strip() for lang in _config.get("CLASSIFY", "EXPECTED_LANGS").split(",") if lang.strip()]

_TRUSTED_FOREIGN_LANG_BASES: frozenset = frozenset(
    lang.strip()
    for lang in _get_str("CLASSIFY", "TRUSTED_FOREIGN_LANGS", "deu,eng,fra,pol,ita,slk").split(",")
    if lang.strip()
)


def _lang_base(lang_code: str) -> str:
    return lang_code.split("_")[0]


CZ_DIACS = frozenset(_get_str("TEXT_UTILS", "CZ_DIACS", "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"))

METADATA_MARKERS = frozenset(
    _get_str("TEXT_UTILS", "METADATA_MARKERS", "Tb.,č.neg,neg.,obr.,obr ,neg ,Tb ,č. neg,č.neg.,č.,str.,Datum").split(
        ","
    )
)

VOWEL_CHARS = frozenset(_get_str("TEXT_UTILS", "VOWEL_CHARS", "aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ"))
ROTATABLE_CHARS = frozenset(_get_str("TEXT_UTILS", "ROTATABLE_CHARS", "pbqdnuwmoxszeyv"))
WQX_CHARS = frozenset(_get_str("TEXT_UTILS", "WQX_CHARS", "wqxWQX"))
NONTEXT_MARKERS = _get_csv_set("TEXT_UTILS", "NONTEXT_MARKERS", "IVerc")
REMAP_KEEP_SCORE_LANGS = _get_csv_set("CLASSIFY", "REMAP_KEEP_SCORE_LANGS", "slk")


def has_cz_diacs(text: str) -> bool:
    """True if *text* contains at least one Czech diacritic glyph."""
    return any(ch in CZ_DIACS for ch in text)


_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(lng) for lng in COMMON_LANGS)

PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)
SHORT_PPL_CAP = _get_float("TEXT_UTILS", "SHORT_PPL_CAP", 850.0)

LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)

# Core signal weights
QS_WEIGHT_VOWEL = _get_float("TEXT_UTILS", "QS_WEIGHT_VOWEL", 0.07)
QS_WEIGHT_LANG = _get_float("TEXT_UTILS", "QS_WEIGHT_LANG", 0.05)
QS_WEIGHT_GIBBERISH = _get_float("TEXT_UTILS", "QS_WEIGHT_GIBBERISH", 0.04)
QS_WEIGHT_FUSED = _get_float("TEXT_UTILS", "QS_WEIGHT_FUSED", 0.03)
QS_LENGTH_MAX = _get_float("TEXT_UTILS", "QS_LENGTH_MAX", 100.0)
QS_WEIGHT_VALID_WORD = _get_float("TEXT_UTILS", "QS_WEIGHT_VALID_WORD", 0.35)
QS_WEIGHT_WEIRD = _get_float("TEXT_UTILS", "QS_WEIGHT_WEIRD", 0.18)
QS_WEIGHT_PERPLEXITY = _get_float("TEXT_UTILS", "QS_WEIGHT_PERPLEXITY", 0.08)
QS_WEIGHT_LENGTH = _get_float("TEXT_UTILS", "QS_WEIGHT_LENGTH", 0.02)
QS_WEIGHT_GARBAGE = _get_float("TEXT_UTILS", "QS_WEIGHT_GARBAGE", 0.18)

CATEG_TRASH_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_TRASH_SCORE_MAX", 0.55)
CATEG_NOISY_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_NOISY_SCORE_MAX", 0.80)
CATEG_GARBAGE_DENSITY_HIGH = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_HIGH", 0.35)
QS_GARBAGE_NORM_MAX = _get_float("TEXT_UTILS", "QS_GARBAGE_NORM_MAX", 0.35)

# Inverted / 180°-rotated scan detection
ROT_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "ROT_RATIO_INVERTED_MIN", 0.55)
WEIRD_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "WEIRD_RATIO_INVERTED_MIN", 0.35)
PPL_INVERTED_MIN = _get_float("TEXT_UTILS", "PPL_INVERTED_MIN", 200.0)
ROT_HIGH_LANG_CONF = _get_float("TEXT_UTILS", "ROT_HIGH_LANG_CONF", 0.90)

LOWPPL_CLEAR_MAX = _get_float("TEXT_UTILS", "LOWPPL_CLEAR_MAX", 50.0)
HARD_SWEEP_LANG_MAX = _get_float("TEXT_UTILS", "HARD_SWEEP_LANG_MAX", 0.45)
HARD_SWEEP_PPL_MIN = _get_float("TEXT_UTILS", "HARD_SWEEP_PPL_MIN", 1000.0)
GHOST_DOMINATED_MIN_RATIO = _get_float("TEXT_UTILS", "GHOST_DOMINATED_MIN_RATIO", 0.5)
WORD_W_PENALTY = _get_float("TEXT_UTILS", "WORD_W_PENALTY", 0.20)

INVERTED_RUN_MIN = _get_int("TEXT_UTILS", "INVERTED_RUN_MIN", 4)
INVERTED_PAGE_MAJORITY = _get_float("TEXT_UTILS", "INVERTED_PAGE_MAJORITY", 0.60)

SURROUNDED_TRASH_QS_MARGIN = _get_float("TEXT_UTILS", "SURROUNDED_TRASH_QS_MARGIN", 0.15)
PAGE_GARBAGE_CLEAR_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_CLEAR_MAX", 0.05)
PAGE_GARBAGE_LANG_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_LANG_MAX", 0.50)
PAGE_GARBAGE_MEDIAN_QS_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_MEDIAN_QS_MAX", 0.55)
PAGE_GARBAGE_NOISY_QS_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_NOISY_QS_MAX", 0.80)
PAGE_CLEAN_CLEAR_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_CLEAR_MIN", 0.60)
PAGE_CLEAN_MEDIAN_QS_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_MEDIAN_QS_MIN", 0.80)
PAGE_CLEAN_RECOVER_QS_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_RECOVER_QS_MIN", 0.45)

TRASH_REASONS = frozenset({"trash_threshold", "trash_hard_sweep", "trash_inverted"})

MOSTLY_READABLE_VALID_MIN = _get_float("TEXT_UTILS", "MOSTLY_READABLE_VALID_MIN", 0.85)
SHORT_NOISY_QS_PENALTY = _get_float("TEXT_UTILS", "SHORT_NOISY_QS_PENALTY", 0.20)

LANG_SCORE_REMAP = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP", 0.75)
LANG_SCORE_REMAP_FAR = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP_FAR", 0.50)
LANG_REMAP_ALWAYS = _get_str("TEXT_UTILS", "LANG_REMAP_ALWAYS", "true").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)
SINGLE_CHAR_ALLOWED = _get_str("TEXT_UTILS", "SINGLE_CHAR_ALLOWED", "aAiIuUvVzZkKsS")
# ── Page-relative perplexity blend (default OFF) ────────────────────────────
# SHORT_PPL_CAP flattens perplexity to a constant for wc <= 2, and the capped
# value is what is both scored AND stored -- the raw LM number is discarded and
# is unrecoverable from a delivered CSV. That leaves short lines with no usable
# perplexity at all: 850 is below HARD_SWEEP_PPL_MIN (1000), PPL_EXTREME_MIN
# (3000) and PPL_GARBAGE_ABSOLUTE (30000), so no perplexity rule can fire on a
# one- or two-token line.
#
# Simply uncapping does not fix it: both surviving perplexity routes gate on LOW
# language-ID confidence, while garbage tokens score high (oueussd at 0.9163)
# and real domain words score low (malakofauna at 0.56). Uncapped, real notation
# is demoted around perplexity 3000 while oueussd survives to 30000.
#
# The blend instead reads a short line's perplexity RELATIVE to the long lines on
# its own page, which is the comparison a human makes. It is a page-consistency
# prior, not a garbage detector: on a mixed page it pulls a garbage token toward
# its clean neighbours. Off by default until calibrated on a real ARUP/ARUB run.
PAGE_PPL_BLEND_ENABLE = _get_str("TEXT_UTILS", "PAGE_PPL_BLEND_ENABLE", "false").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)
# Geometric blend weight on the line's own perplexity. Log-space is where the
# average has to happen: perplexities span six orders of magnitude here (II/C
# measures ~6e7), so an arithmetic blend is dominated by the tail.
PAGE_PPL_BLEND_WEIGHT = _get_float("TEXT_UTILS", "PAGE_PPL_BLEND_WEIGHT", 0.5)
# Minimum word count for a line to count toward the page reference.
PAGE_PPL_LONG_MIN_WC = _get_int("TEXT_UTILS", "PAGE_PPL_LONG_MIN_WC", 4)
# Minimum number of such lines before a page reference is trusted at all.
PAGE_PPL_MIN_LONG_LINES = _get_int("TEXT_UTILS", "PAGE_PPL_MIN_LONG_LINES", 3)

SHORT_VALID_WORDS = _get_csv_set(
    "TEXT_UTILS",
    "SHORT_VALID_WORDS",
    "a,i,k,o,s,u,v,z,se,si,po,na,za,ze,do,od,ke,ku,ve,ní,mi,ti,by,je,to,co,ač,my,ty,on,ji,jí,už,až",
)

REPEAT_ALLOWED_CHARS = _get_str("TEXT_UTILS", "REPEAT_ALLOWED_CHARS", "oOuU")
REPEATED_DOUBLE_MIN = _get_int("TEXT_UTILS", "REPEATED_DOUBLE_MIN", 2)
VOWEL_RATIO_LOW = _get_float("TEXT_UTILS", "VOWEL_RATIO_LOW", 0.20)
VOWEL_RATIO_HIGH = _get_float("TEXT_UTILS", "VOWEL_RATIO_HIGH", 0.70)
ACADEMIC_TITLES = _get_csv_set(
    "TEXT_UTILS",
    "ACADEMIC_TITLES",
    "PhDr,MUDr,JUDr,MVDr,RNDr,PaedDr,CSc,DrSc,Ing,Mgr,Bc,PhD,DiS,prof,doc",
)

LDL_ALLOWED_FOLLOW = frozenset(_get_str("TEXT_UTILS", "LDL_ALLOWED_FOLLOW", ".,/:%-;?)="))
LDL_UNITS = _get_csv_set("TEXT_UTILS", "LDL_UNITS", "m,cm,mm,g,kg,km,ha,l,ml")

SHORT_EXCEPTION_TOKENS = _get_csv_set(
    "TEXT_UTILS",
    "SHORT_EXCEPTION_TOKENS",
    "mm,cm,m,g,kg,km,ha,l,ml,tb,neg,obr,str,č,čneg",
)
HEADLINE_MAX_WORDS = _get_int("TEXT_UTILS", "HEADLINE_MAX_WORDS", 8)
HEADLINE_MAX_DIGITS = _get_int("TEXT_UTILS", "HEADLINE_MAX_DIGITS", 2)

GARBAGE_KEEP_CHARS = frozenset(_get_str("TEXT_UTILS", "GARBAGE_KEEP_CHARS", "")) | {" "}
FUSED_VOWEL_RUN_MIN = _get_int("TEXT_UTILS", "FUSED_VOWEL_RUN_MIN", 3)
WX_REPEAT_MIN = _get_int("TEXT_UTILS", "WX_REPEAT_MIN", 2)

ISOLATED_CHAR_RATIO_MAX = _get_float("TEXT_UTILS", "ISOLATED_CHAR_RATIO_MAX", 0.40)
ISOLATED_CHAR_MIN_TOKENS = _get_int("TEXT_UTILS", "ISOLATED_CHAR_MIN_TOKENS", 3)
SYM_LET_DIG_NONTEXT = _get_str("TEXT_UTILS", "SYM_LET_DIG_NONTEXT", "true").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)

ALLOWED_INTERNAL: frozenset = frozenset(_get_str("TEXT_UTILS", "ALLOWED_INTERNAL", ".-,+()\"'/—–:%;?!/"))
_STRIP_CHARS: str = _get_str("TEXT_UTILS", "STRIP_CHARS", ".,;:!?()[]\"'/\\")

RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r"[^\w\s]{2,}")
RE_TRASH_LDL: re.Pattern = re.compile(r"[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]")
RE_NON_TEXT: re.Pattern = re.compile(r"^[\d\s\-\u2013\u2014/:.,()%]+$")
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r"[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]")
RE_ROMAN_NUMERAL: re.Pattern = re.compile(r"^[IVXLCDMivxlcdm]+\.?$")
RE_STAMP: re.Pattern = re.compile(r"^(?:[A-Za-z]+)?[\W_]*\d{2,4}\s*/\s*\d{2,4}[\W_]*$")
RE_ARCHIVE_CODE: re.Pattern = re.compile(r"^[A-Za-z]{1,3}\d{3,}(?:/\d+)?$")
RE_ALPHANUM_TOKEN: re.Pattern = re.compile(r"^[A-Za-z0-9]{5,}$")
RE_ARCHIVE_REF_SPACED: re.Pattern = re.compile(r"^[A-Za-záčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]{1,5}[\s.\-]+\d{1,}")

_RE_SPACED_CAPS: re.Pattern = re.compile(
    r"(?<!\S)"
    r"([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ] ){3,}"
    r"[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]"
    r"(?!\S)"
)


def _collapse_spaced_caps(m: re.Match) -> str:
    letters = m.group(0).replace(" ", "")
    return letters[0].upper() + letters[1:].lower()


DEU_DIACS = frozenset(_get_str("TEXT_UTILS", "DEU_DIACS", "äöüßÄÖÜ"))

_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": CZ_DIACS,
    "deu": DEU_DIACS,
}

DIACRITIC_INFER_THRESHOLD = _get_float("TEXT_UTILS", "DIACRITIC_INFER_THRESHOLD", 0.07)

PPL_EXTREME_MIN = _get_float("TEXT_UTILS", "PPL_EXTREME_MIN", 3000.0)
EXTREME_LANG_CONF = _get_float("TEXT_UTILS", "EXTREME_LANG_CONF", 0.85)
LOWPPL_CZECH_CLEAR_MAX = _get_float("TEXT_UTILS", "LOWPPL_CZECH_CLEAR_MAX", 180.0)
CZECH_CLEAR_GARBAGE_MAX = _get_float("TEXT_UTILS", "CZECH_CLEAR_GARBAGE_MAX", 0.15)

ANCHOR_MIN_WORDS = _get_int("TEXT_UTILS", "ANCHOR_MIN_WORDS", 2)
ANCHOR_WORD_LEN = _get_int("TEXT_UTILS", "ANCHOR_WORD_LEN", 3)
ANCHOR_VOWEL_RATIO = _get_float("TEXT_UTILS", "ANCHOR_VOWEL_RATIO", 0.10)

SUSPICIOUS_ROT_RATIO = _get_float("TEXT_UTILS", "SUSPICIOUS_ROT_RATIO", 0.65)
SUSPICIOUS_WQX_RATIO = _get_float("TEXT_UTILS", "SUSPICIOUS_WQX_RATIO", 0.15)
INVERTED_WEIRD_PENALTY = _get_float("TEXT_UTILS", "INVERTED_WEIRD_PENALTY", 0.45)

PPL_GARBAGE_ABSOLUTE = _get_float("TEXT_UTILS", "PPL_GARBAGE_ABSOLUTE", 30000.0)
GHOST_HITS_INVERTED_MIN = _get_int("TEXT_UTILS", "GHOST_HITS_INVERTED_MIN", 1)
TRAILING_FILL_CHARS = (
    _get_str("TEXT_UTILS", "TRAILING_FILL_CHARS", "\\x20._:-<\\u2013\\u2014")
    .encode("latin-1", "backslashreplace")
    .decode("unicode_escape")
)

# ---------------------------------------------------------------------------
# Lexicon Integration for Rotation/Inversion detection
# ---------------------------------------------------------------------------

_MIRROR_GLYPH = {
    "b": "d",
    "d": "b",
    "p": "q",
    "q": "p",
    "a": "a",
    "e": "e",
    "i": "i",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y": "y",
    "z": "z",
}
_ROTATE_GLYPH = {
    "b": "q",
    "q": "b",
    "d": "p",
    "p": "d",
    "h": "y",
    "n": "u",
    "u": "n",
    "m": "w",
    "w": "m",
    "y": "h",
    "a": "e",
    "e": "a",
    "i": "!",
    "l": "l",
    "o": "o",
    "s": "s",
    "x": "x",
    "z": "z",
}


def _transform_word(w: str, glyph_map: dict) -> str | None:
    out = []
    for ch in w:
        img = glyph_map.get(ch)
        if img is None:
            return None
        out.append(img)
    return "".join(reversed(out))


ROT_WHITELIST: frozenset = _get_csv_set(
    "TEXT_UTILS", "ROT_WHITELIST", "po,pod,do,od,on,ony,by,bez,ne,nebo,ven,den,zde,se,ve,mez,pouze,bude"
)
_GHOST_REAL_WORD_COLLISIONS: frozenset = _get_csv_set("TEXT_UTILS", "GHOST_WORD_COLLISIONS", "no,bo")


def _build_ghostlist() -> frozenset:
    ghosts = set()
    for w in ROT_WHITELIST:
        for img in (_transform_word(w, _MIRROR_GLYPH), _transform_word(w, _ROTATE_GLYPH)):
            if img:
                ghosts.add(img)
    return frozenset(ghosts - ROT_WHITELIST - _GHOST_REAL_WORD_COLLISIONS)


ROT_GHOSTLIST: frozenset = _build_ghostlist()


def analyze_rotation_signals(text: str) -> tuple[bool, bool]:
    words = [w.lower() for w in re.split(r"\W+", text) if w]
    if not words:
        return has_cz_diacs(text), False

    real_hits = sum(1 for w in words if w in ROT_WHITELIST)
    ghost_hits = sum(1 for w in words if w in ROT_GHOSTLIST)

    is_upright_czech = has_cz_diacs(text) or real_hits > 0

    ghost_share = ghost_hits / len(words)
    ghost_dominated = ghost_hits > 0 and ghost_share >= GHOST_DOMINATED_MIN_RATIO
    return is_upright_czech, ghost_dominated


def ghost_word_share(text: str) -> tuple[int, float]:
    words = [w.lower() for w in re.split(r"\W+", text) if w]
    if not words:
        return 0, 0.0
    ghost_hits = sum(1 for w in words if w in ROT_GHOSTLIST)
    return ghost_hits, ghost_hits / len(words)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_mid_uppercase(core: str) -> bool:
    if len(core) < 2 or core.isupper():
        return False
    if core.rstrip(".") in ACADEMIC_TITLES:
        return False

    caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
    if caps_run >= 2 and any(c.islower() for c in core[caps_run:]):
        return True

    for i in range(1, len(core)):
        if core[i].isupper() and core[i - 1].islower():
            return True

    return False


def _has_starting_uppercase(core: str) -> bool:
    if len(core) < 2 or core.isupper():
        return False
    if core.rstrip(".") in ACADEMIC_TITLES:
        return False
    return core[0].isupper() and core[1].isupper()


def _split_subtokens(word: str) -> list[str]:
    return [p for p in re.split(r"[.\-\u2013]", word) if p]


def remap_lang(
    label: str, score: float, known_bases: frozenset, default_lang: str, remap_floor: float = LANG_SCORE_REMAP
) -> tuple[str, float]:
    base = _lang_base(label)
    if base in known_bases:
        return label, score
    suffix = label[len(base) :]
    new_label = default_lang + suffix
    if base in REMAP_KEEP_SCORE_LANGS:
        return new_label, score
    cap = remap_floor if suffix == "_Latn" else LANG_SCORE_REMAP_FAR
    if LANG_REMAP_ALWAYS or score > cap:
        return new_label, cap
    return new_label, score


def compute_garbage_density(text: str) -> float:
    if not text:
        return 0.0
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in GARBAGE_KEEP_CHARS)
    return noise_chars / len(text)


def _has_repeated_run(core: str) -> bool:
    if len(core) < 4:
        return False
    for ch in set(core):
        if ch.isdigit():
            continue
        if ch * 3 in core:
            return True
        if ch in REPEAT_ALLOWED_CHARS:
            continue
        if ch * 2 in core and core.count(ch) >= REPEATED_DOUBLE_MIN:
            return True
        if (core.count(ch) / len(core) >= 0.30) and core.count(ch) >= 3:
            return True
    return False


def _trailing_alpha_run(token: str, start: int) -> str:
    j = start
    while j < len(token) and token[j].isalpha():
        j += 1
    return token[start:j]


def has_symbol_letter_digit(word: str) -> bool:
    has_letter = any(c.isalpha() for c in word)
    has_digit = any(c.isdigit() for c in word)
    has_symbol = any((not c.isalnum()) and not c.isspace() and c not in ALLOWED_INTERNAL for c in word)
    return has_letter and has_digit and has_symbol


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------


def infer_lang_from_diacritics(text: str, expected_bases: frozenset, threshold: float | None = None) -> str | None:
    if threshold is None:
        threshold = DIACRITIC_INFER_THRESHOLD
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


def compute_rotatable_ratio(text: str) -> float:
    alpha_chars = [c.lower() for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    rotatable_count = sum(1 for c in alpha_chars if c in ROTATABLE_CHARS)
    return rotatable_count / len(alpha_chars)


def detect_strange_symbols(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if not core:
            continue
        count += sum(1 for ch in core if not ch.isalnum() and ch not in ALLOWED_INTERNAL)
    return count


def detect_repeated_chars(text: str) -> int:
    count = 0
    for word in text.split():
        if any(_has_repeated_run(sub.strip(_STRIP_CHARS)) for sub in _split_subtokens(word)):
            count += 1
    return count


def compute_vowel_ratio(text: str) -> float:
    denom = [c for c in text if c.isalpha() or ((not c.isalnum()) and not c.isspace())]
    if not denom:
        return 0.0
    return sum(1 for c in denom if c in VOWEL_CHARS) / len(denom)


def detect_gibberish_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if len(core) < 4 or core.isupper():
                continue
            numeric_chars = sum(1 for c in core if c.isdigit() or c in "-./,;:")
            if numeric_chars / len(core) >= 0.6:
                continue
            letters = [c for c in core if c.isalpha()]
            if not letters:
                continue
            if sum(1 for c in letters if c in VOWEL_CHARS) / len(letters) > VOWEL_RATIO_HIGH:
                flagged = True
                break
        if flagged:
            count += 1
    return count


def _has_ldl(token: str) -> bool:
    n = len(token)
    for i, ch in enumerate(token):
        if not ch.isdigit():
            continue
        nxt = token[i + 1] if i + 1 < n else ""
        prev = token[i - 1] if i > 0 else ""
        if nxt and not nxt.isspace() and not nxt.isdigit() and nxt not in LDL_ALLOWED_FOLLOW:
            if nxt.isalpha():
                run = _trailing_alpha_run(token, i + 1)
                if run.lower() in LDL_UNITS:
                    continue
            return True
        if prev.isalpha():
            return True
    return False


def detect_letter_digit_letter(text: str) -> int:
    return sum(1 for word in text.split() if _has_ldl(word))


def detect_mid_uppercase(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(".,;:!?()[]\"'-/")
        if _is_mid_uppercase(core):
            count += 1
    return count


def detect_wx_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if not core:
                continue
            if sum(1 for c in core if c in "wW") >= WX_REPEAT_MIN or sum(1 for c in core if c in "xX") >= WX_REPEAT_MIN:
                flagged = True
                break
        if flagged:
            count += 1
    return count


def is_all_caps_line(text: str) -> bool:
    alpha_words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False
    return all(w.isupper() for w in alpha_words)


_RE_FUSED_CONSONANT_RUN: re.Pattern = re.compile(r"[bcčdfghjklmnpqrřsštvwxzž]{5,}", re.IGNORECASE)
_RE_FUSED_VOWEL_RUN: re.Pattern = re.compile(r"[aeiouyáéíóúýěůäöü]{%d,}" % FUSED_VOWEL_RUN_MIN, re.IGNORECASE)


def detect_fused_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if not core or not any(c.isalpha() for c in core):
                continue
            if len(core) > 14 or _RE_FUSED_CONSONANT_RUN.search(core) or _RE_FUSED_VOWEL_RUN.search(core):
                flagged = True
                break
        if flagged:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------


def pre_filter_line(line: str) -> tuple[str, str]:
    clean_text = line.strip()
    if not clean_text:
        return "Empty", ""

    clean_text = re.sub(
        r"(?<=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])1(?=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])", "l", clean_text
    )
    clean_text = re.sub(r"(?<![\d.,])\b2(?=[a-záčďéěíňóřšťůúýž])", "z", clean_text)
    clean_text = _RE_SPACED_CAPS.sub(_collapse_spaced_caps, clean_text)

    if any(marker.lower() in clean_text.lower() for marker in METADATA_MARKERS):
        return "Process", clean_text

    if is_forgiven_headline(clean_text, compute_garbage_density(clean_text)):
        return "Process", clean_text

    if clean_text.startswith('"') and not clean_text.endswith('"'):
        clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'):
        clean_text = '"' + clean_text

    n_chars = len(clean_text)

    if is_non_text(clean_text):
        return "Non-text", clean_text
    if RE_ROMAN_NUMERAL.match(clean_text.strip()):
        return "Non-text", clean_text
    if RE_STAMP.search(clean_text) or any(m in clean_text for m in NONTEXT_MARKERS):
        return "Non-text", clean_text

    tokens = clean_text.split()
    valid_long_words = sum(
        1
        for tok in tokens
        if len(tok.strip(_STRIP_CHARS)) >= ANCHOR_WORD_LEN
        and tok.strip(_STRIP_CHARS).isalpha()
        and compute_vowel_ratio(tok.strip(_STRIP_CHARS)) >= ANCHOR_VOWEL_RATIO
    )
    if valid_long_words >= ANCHOR_MIN_WORDS:
        return "Process", clean_text

    if sum(c.isdigit() for c in clean_text) / n_chars > 0.4:
        return "Process", clean_text

    unique_symbols = set(c for c in clean_text if not c.isspace())
    if n_chars < 4 or len(unique_symbols) < 3:
        return "Non-text", clean_text

    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3:
        return "Non-text", clean_text

    if SYM_LET_DIG_NONTEXT and len(tokens) == 1 and has_symbol_letter_digit(tokens[0]):
        return "Non-text", clean_text

    if len(tokens) >= ISOLATED_CHAR_MIN_TOKENS:
        alpha_tokens = [tok for tok in tokens if any(c.isalpha() for c in tok)]
        if alpha_tokens:
            valid_singles = frozenset(SINGLE_CHAR_ALLOWED)
            single_char_tokens = [
                tok for tok in alpha_tokens if len(tok.strip(_STRIP_CHARS)) == 1 and tok.strip(_STRIP_CHARS).isalpha()
            ]
            invalid_singles = [tok for tok in single_char_tokens if tok.strip(_STRIP_CHARS) not in valid_singles]

            is_pure_isolated = len(single_char_tokens) == len(alpha_tokens)
            high_isolated_ratio = (len(invalid_singles) / len(alpha_tokens)) >= ISOLATED_CHAR_RATIO_MAX

            if is_pure_isolated or high_isolated_ratio:
                run_length = 0
                collapsed_spans = []
                current_span = []

                for tok in tokens:
                    core = tok.strip(_STRIP_CHARS)
                    if len(core) == 1 and core.isalpha():
                        run_length += 1
                        current_span.append(core)
                    else:
                        if run_length >= 3:
                            collapsed_spans.append("".join(current_span))
                        run_length = 0
                        current_span = []
                if run_length >= 3:
                    collapsed_spans.append("".join(current_span))

                rescued = False
                for span in collapsed_spans:
                    if compute_vowel_ratio(span) > 0.15 and compute_garbage_density(span) < 0.20:
                        rescued = True
                        break

                if not rescued:
                    return "Non-text", clean_text

    return "Process", clean_text


def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    clean_line = line_text.strip()
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"
    matches = list(re.finditer(pattern, clean_line))
    if not matches:
        return clean_line, "", ""
    last_prefix = last_suffix = ""

    def replace_match(match):
        nonlocal last_prefix, last_suffix
        prefix = match.group(1)
        content = match.group(2)
        last_prefix = prefix
        last_suffix = content[len(prefix) :] if content.startswith(prefix) else ""
        return content

    merged_text = re.sub(pattern, replace_match, clean_line)
    return merged_text, last_prefix, last_suffix


# ---------------------------------------------------------------------------
# Per-Word Weirdness Scoring
# ---------------------------------------------------------------------------


def score_word(word: str) -> float:
    core = word.strip(_STRIP_CHARS)
    if len(core) == 1:
        if core in SINGLE_CHAR_ALLOWED or "." in word:
            return 0.0
        if core.isdigit():
            return 0.25
        if not core.isalpha():
            return 0.0
        return 0.85
    if len(core) < 2:
        return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)
    has_rep = _has_repeated_run(core)
    has_ldl = _has_ldl(core)
    has_uppercase = _is_mid_uppercase(core)
    has_wqx = any(c in WQX_CHARS for c in core)

    has_caps_prefix = False
    if len(core) >= 4 and not core.isupper() and core.rstrip(".") not in ACADEMIC_TITLES:
        caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
        if caps_run >= 2 and any(c.islower() for c in core[caps_run:]):
            has_caps_prefix = True

    alpha_chars = [c for c in core if c.isalpha()]
    is_vowelless_long = (
        len(alpha_chars) >= 3
        and not any(c in VOWEL_CHARS for c in alpha_chars)
        and core.rstrip(".") not in ACADEMIC_TITLES
    )

    return min(
        1.0,
        0.40 * has_strange
        + 0.35 * has_rep
        + 0.15 * has_ldl
        + 0.25 * has_uppercase
        + 0.20 * has_caps_prefix
        + WORD_W_PENALTY * has_wqx
        + 0.50 * is_vowelless_long,
    )


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    is_upright, ghost_dom = analyze_rotation_signals(text)
    rot_ratio = compute_rotatable_ratio(text)

    words = text.split()
    wqx_words = sum(1 for w in words if any(c in WQX_CHARS for c in w))
    wqx_ratio = wqx_words / len(words) if words else 0.0

    is_suspicious_rot = rot_ratio > SUSPICIOUS_ROT_RATIO and wqx_ratio >= SUSPICIOUS_WQX_RATIO and not is_upright

    frag_count = sum(1 for w in words if w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2)
    frag_ratio = frag_count / len(words) if words else 0.0

    is_highly_fragmented = frag_ratio > 0.60 and len(words) >= 4

    results = []
    for w in words:
        s = score_word(w)
        if (ghost_dom or is_suspicious_rot) and not is_upright:
            s = min(1.0, s + INVERTED_WEIRD_PENALTY)

        if is_highly_fragmented:
            core = w.strip(_STRIP_CHARS)
            if core.isdigit() or len(core) <= 2:
                s = min(1.0, s + 0.35)

        results.append((w, s))

    return results


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    if not word_scores:
        return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)


# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------


def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    import torch
    from torch import nn

    if not texts:
        return []
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
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(target_ids.size(0), -1)
            non_masked = shift_labels != -100
            seq_loss = (loss * non_masked).sum(dim=1)
            num_tokens = non_masked.sum(dim=1).clamp(min=1)
            ppl = torch.exp(seq_loss / num_tokens)
            return ppl.tolist()
    except Exception as e:
        print(f"[Error] Batch PPL ({len(texts)} lines) failed: {e}", file=sys.stderr, flush=True)
        return [99999.0] * len(texts)


# ---------------------------------------------------------------------------
# Categorisation & Clamping
# ---------------------------------------------------------------------------


def _lm_confident_czech(is_upright_czech, ppl, garbage_density):
    return is_upright_czech and ppl < LOWPPL_CZECH_CLEAR_MAX and garbage_density < CZECH_CLEAR_GARBAGE_MAX


def _trailing_fill_rescued(text_source: str, valid_word_ratio: float, word_count: int) -> bool:
    if valid_word_ratio <= 0.0:
        return False
    core = text_source.rstrip(TRAILING_FILL_CHARS)
    if not core or core == text_source:
        return False
    if compute_garbage_density(core) >= CATEG_GARBAGE_DENSITY_HIGH:
        return False
    return has_cz_diacs(core) or (word_count <= 4 and len(text_source) <= 25)


def is_forgiven_headline(text: str, garbage_density: float) -> bool:
    """(#3 2026-07-02 calibration) Recognise short numbered headlines/captions
    (``"2, Popis nálezu i - 3"``, ``"Plánek č. 1"``) and bare domain
    abbreviations (``mm``, ``Tb.``, ``č.neg.``) that would otherwise mis-route
    to Trash/Non-text purely because the digits/symbols around one or two real
    words drag ``valid_word_ratio`` down.

    Every token is classified as one of:
      * NUMBERING  — a pure digit (short numbering only, see
        ``HEADLINE_MAX_DIGITS``) or a roman numeral. Supplies *context*.
      * ABBREV     — a known unit/abbreviation (``SHORT_EXCEPTION_TOKENS``), an
        academic title, or a ``METADATA_MARKERS`` marker. Supplies both
        *content* and *context* (a bare ``mm`` line qualifies on its own).
      * FUNCTION   — a whitelisted short Czech word (``SHORT_VALID_WORDS`` /
        ``SINGLE_CHAR_ALLOWED``). Real *content*, but no context by itself.
      * CLEAN WORD — passes the same acceptance test as ``compute_valid_ratio``'s
        inner branch, plus a vowel-bearing check. Real *content*, no context.
        Multi-token lines only: a single bare "clean-looking" word is exactly the
        profile of an inverted-scan / short-garbage token (``oueussd``, ``olie``)
        that rule_inverted / rule_short_garbage exist to catch.
      * STRUCTURAL — pure punctuation: no information either way.
      * GARBAGE    — anything else, and disqualifies the whole line.

    A line is forgiven only when it carries BOTH real *content* (a clean word,
    abbreviation, or function word) AND genuine numbering/abbreviation *context*
    (a digit, roman numeral, or domain abbreviation). Requiring the context term
    is what keeps a bare short prose fragment (``"popel dřevo kůstky"``) — no
    numbering, no abbreviation — out of the forgiveness path; those must route on
    their own quality score, exactly as before this pass. Every DanaKriv example
    carries such context (``2, ...``, ``4. ...``, ``Plánek č. 1``, ``mm``).

    Deliberately tight: a single OCR-mangled token (``oAOrt``, ``vyt1ačená``)
    or an over-long digit run (an archive/stamp code, not a caption number)
    disqualifies the line, so genuine garbage is never rescued.
    """
    tokens = text.split()
    if not tokens or len(tokens) > HEADLINE_MAX_WORDS:
        return False
    if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
        return False

    multi_token = len(tokens) >= 2
    has_content = False  # a clean word, abbreviation, or function word
    has_context = False  # numbering (digit / roman) or a domain abbreviation
    for tok in tokens:
        core = tok.strip(_STRIP_CHARS)

        # STRUCTURAL — pure punctuation (no alnum at all) carries no
        # information either way.
        if not core or not any(c.isalnum() for c in core):
            continue
        # NUMBERING — short numbering only; longer digit runs are archive/stamp
        # codes, not caption numbers.
        if core.isdigit():
            if len(core) > HEADLINE_MAX_DIGITS:
                return False
            has_context = True
            continue

        normalized = core.lower().replace(".", "").replace(",", "")

        # ABBREV — a domain unit/marker/title supplies both content and context,
        # so a bare "mm" / "Tb." / "č.neg." line qualifies on its own.
        if (
            normalized in SHORT_EXCEPTION_TOKENS
            or core.rstrip(".") in ACADEMIC_TITLES
            or any(marker.lower() in tok.lower() for marker in METADATA_MARKERS)
        ):
            has_content = True
            has_context = True
            continue

        # NUMBERING — roman numeral (checked after ABBREV so real abbreviations
        # built only of I/V/X/L/C/D/M aren't misread as numbering). A lone
        # ambiguous glyph ("v", "i", "l", ...) is a Czech function word, not a
        # numeral, so genuine roman numbering needs at least two glyphs.
        if len(core.rstrip(".")) >= 2 and RE_ROMAN_NUMERAL.match(core):
            has_context = True
            continue

        # FUNCTION — a whitelisted short Czech word / single char is real
        # content, but is NOT numbering/abbreviation context on its own.
        if core.lower() in SHORT_VALID_WORDS or core in SINGLE_CHAR_ALLOWED:
            has_content = True
            continue

        # CLEAN WORD — multi-token lines only (see docstring).
        if multi_token:
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if (
                len(core) >= 3
                and alpha / len(core) >= 0.70
                and not has_strange
                and not _is_mid_uppercase(core)
                and compute_vowel_ratio(core) > 0.0
            ):
                has_content = True
                continue

        # GARBAGE
        return False

    return has_content and has_context


def inspect_short_line_telemetry(
    text_source: str,
    word_count: int,
    valid_word_ratio: float,
    lang_score: float,
    perplexity: float,
    weird_ratio: float = 0.0,
    garbage_density: float = 0.0,
    is_upright_czech: bool = False,
) -> dict:
    """
    Step 4 Telemetry Helper: Audit short lines (word_count <= 2) to log properties
    and identify potential false Clear/Noisy promotions.
    """
    structured = is_structured_line(text_source)
    damaged = count_damaged_tokens(text_source) > 0
    forgiven = is_forgiven_headline(text_source, garbage_density)

    category, score, reason = categorize_line(
        qs=compute_quality_score(
            valid_word_ratio=valid_word_ratio,
            perplexity=perplexity,
            text_length=len(text_source),
            weird_ratio=weird_ratio,
            garbage_density=garbage_density,
            lang_score=lang_score,
            is_upright_czech=is_upright_czech,
        ),
        txt=text_source,
        wc=word_count,
        vowel_ratio=compute_vowel_ratio(text_source),
        perplexity=perplexity,
        weird_ratio=weird_ratio,
        valid_word_ratio=valid_word_ratio,
        lang_score=lang_score,
        garbage_density=garbage_density,
        is_upright_czech=is_upright_czech,
        return_reason=True,
    )

    return {
        "text": text_source,
        "word_count": word_count,
        "valid_word_ratio": valid_word_ratio,
        "lang_score": lang_score,
        "perplexity": perplexity,
        "weird_ratio": weird_ratio,
        "garbage_density": garbage_density,
        "structured": structured,
        "damaged": damaged,
        "forgiven_headline": forgiven,
        "is_upright_czech": is_upright_czech,
        "final_category": category,
        "quality_score": score,
        "route_reason": reason,
    }


def determine_category(
    qs: float,
    text_source: str,
    word_count: int,
    vr: float,
    ppl: float,
    weird_ratio: float = 0.0,
    valid_word_ratio: float = 1.0,
    lang_score: float = 1.0,
    orig_lang_score: float = 1.0,
    gibberish_present: bool = False,
    garbage_density: float = 0.0,
    is_upright_czech: bool = False,
    ghost_dominated: bool = False,
) -> tuple[str, str]:
    if word_count == 0 or not text_source.strip():
        return "Empty", "empty"

    stripped = text_source.strip()
    rot_ratio = compute_rotatable_ratio(text_source)
    words = text_source.split()

    structured = is_structured_line(text_source)

    # ------------------------------------------------------------
    # 1. Hard sweep
    # ------------------------------------------------------------
    if "rule_hard_sweep" not in DISABLED_RULES:
        if orig_lang_score < HARD_SWEEP_LANG_MAX and ppl > HARD_SWEEP_PPL_MIN:
            _fire("rule_hard_sweep")
            return "Trash", "trash_hard_sweep"

    if "rule_extreme_ppl" not in DISABLED_RULES:
        if ppl >= PPL_EXTREME_MIN and orig_lang_score < EXTREME_LANG_CONF:
            _fire("rule_extreme_ppl")
            return "Trash", "trash_hard_sweep"

    if "rule_absolute_ppl" not in DISABLED_RULES:
        if ppl >= PPL_GARBAGE_ABSOLUTE and not is_upright_czech:
            _fire("rule_absolute_ppl")
            return "Trash", "trash_hard_sweep"

    # ------------------------------------------------------------
    # 2. Inverted / mirrored scan
    # ------------------------------------------------------------
    if "rule_inverted" not in DISABLED_RULES:
        if not is_upright_czech and (
            ghost_dominated
            or (
                not has_cz_diacs(text_source)
                and rot_ratio >= SUSPICIOUS_ROT_RATIO
                and ppl >= PPL_INVERTED_MIN
                and ghost_word_share(text_source)[0] >= GHOST_HITS_INVERTED_MIN
            )
        ):
            _fire("rule_inverted")
            return "Trash", "trash_inverted"

    # ------------------------------------------------------------
    # 3. All-caps vowel-less scramble
    # ------------------------------------------------------------
    if "rule_allcaps" not in DISABLED_RULES:
        if vr < 0.10 and is_all_caps_line(text_source) and not structured:
            _fire("rule_allcaps")
            return "Trash", "allcaps_novowel"

    # ------------------------------------------------------------
    # 4. Extreme garbage density
    # ------------------------------------------------------------
    if "rule_garbage_density" not in DISABLED_RULES:
        if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
            is_siglum = word_count <= 2 and _RE_SIGLUM.match(stripped)

            if structured or is_siglum:
                pass

            elif "rule_trailing_fill_rescue" not in DISABLED_RULES and _trailing_fill_rescued(
                text_source,
                valid_word_ratio,
                word_count,
            ):
                pass

            else:
                _fire("rule_garbage_density")
                return "Trash", "trash_threshold"

    # ------------------------------------------------------------
    # 5. Determine structural state
    # ------------------------------------------------------------
    forgiven = "rule_forgiven_headline" not in DISABLED_RULES and is_forgiven_headline(
        text_source,
        garbage_density,
    )

    damaged = "rule_damaged_token" not in DISABLED_RULES and word_count >= 3 and count_damaged_tokens(text_source) > 0

    # ------------------------------------------------------------
    # 5b. Zero-alphabetic content
    # ------------------------------------------------------------
    if (
        "rule_zero_alpha" not in DISABLED_RULES
        and not structured
        and not is_upright_czech
        and not any(c.isalpha() for c in text_source)
    ):
        _fire("rule_zero_alpha")

        if forgiven:
            return "Noisy", "noisy_threshold"

        return "Trash", "trash_threshold"

    # ------------------------------------------------------------
    # 6. Short-line garbage
    # ------------------------------------------------------------
    if (
        "rule_short_garbage" not in DISABLED_RULES
        and not forgiven
        and not structured
        and not ("rule_domain_notation" not in DISABLED_RULES and is_domain_notation(text_source))
    ):
        if (
            word_count <= ISOLATED_CHAR_MIN_TOKENS
            and not has_cz_diacs(text_source)
            and (lang_score <= LANG_SCORE_REMAP or rot_ratio >= SUSPICIOUS_ROT_RATIO)
            and (gibberish_present or weird_ratio > 0.0)
        ):
            _fire("rule_short_garbage")
            return "Trash", "trash_threshold"

    elif (
        "rule_short_garbage" not in DISABLED_RULES
        and not forgiven
        and not structured
        and "rule_domain_notation" not in DISABLED_RULES
        and is_domain_notation(text_source)
    ):
        # Reached only when the notation predicate is the DECIDING term — the
        # other three would have let rule_short_garbage run. Recorded so coverage
        # and ablation see the suppression rather than just its absence.
        _fire("rule_domain_notation")

    # ------------------------------------------------------------
    # 7. Short lines (1-2 words)
    # ------------------------------------------------------------
    if "rule_short_line" not in DISABLED_RULES and word_count <= 2:
        _fire("rule_short_line")

        if _RE_SIGLUM.match(stripped) and sum(c.isalpha() for c in stripped) >= 2:
            return "Clear", "clear_threshold"

        if word_count == 1:
            solitary = stripped.strip(_STRIP_CHARS)

            if len(solitary) == 1 and solitary.isalpha() and "." not in stripped:
                return "Trash", "trash_threshold"

        if any(_RE_BIGRAM_RUN.search(w.strip(_STRIP_CHARS)) for w in words):
            if _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return "Trash", "trash_threshold"

        structurally_clean = valid_word_ratio >= 1.0

        damage = (
            weird_ratio >= 0.40
            or (any(_RE_BIGRAM_RUN.search(w.strip(_STRIP_CHARS)) for w in words) and not structured)
            or ((gibberish_present or detect_fused_words(text_source) > 0) and not structurally_clean)
            or (
                garbage_density >= CATEG_GARBAGE_DENSITY_HIGH
                and not _trailing_fill_rescued(
                    text_source,
                    valid_word_ratio,
                    word_count,
                )
                and not structured
            )
        )

        if damage:
            return "Noisy", "noisy_threshold"

        if valid_word_ratio <= 0.0 and not is_upright_czech and not structured:
            if forgiven:
                return "Noisy", "noisy_threshold"

            return "Trash", "trash_threshold"

        if is_upright_czech or valid_word_ratio >= 1.0 or structured:
            if count_damaged_tokens(text_source) > 0 or (not is_upright_czech and weird_ratio >= 0.40):
                return "Noisy", "noisy_threshold"

            return "Clear", "clear_threshold"

        return "Noisy", "noisy_threshold"

    # ------------------------------------------------------------
    # 8. High-confidence LM override
    # ------------------------------------------------------------
    if "rule_lowppl_clear" not in DISABLED_RULES:
        if ppl < LOWPPL_CLEAR_MAX and word_count >= 3:
            if valid_word_ratio < MOSTLY_READABLE_VALID_MIN:
                _fire("rule_lowppl_clear")
                return "Noisy", "noisy_threshold"

            if damaged:
                _fire("rule_damaged_token")
                return "Noisy", "noisy_threshold"

            _fire("rule_lowppl_clear")
            return "Clear", "lowppl_clear"

    # ------------------------------------------------------------
    # 9. Explicit diagnostic hard gates [STEP 5]
    # ------------------------------------------------------------
    thresh_trash = CATEG_TRASH_SCORE_MAX + 0.35

    def check_rescues() -> tuple[str, str]:
        if "rule_trailing_fill_rescue" not in DISABLED_RULES and _trailing_fill_rescued(
            text_source,
            valid_word_ratio,
            word_count,
        ):
            _fire("rule_trailing_fill_rescue")
            return "Noisy", "noisy_threshold"

        if forgiven:
            _fire("rule_forgiven_headline")
            return "Noisy", "noisy_threshold"

        if "rule_reference_floor" not in DISABLED_RULES and is_clean_reference(text_source):
            _fire("rule_reference_floor")
            return "Noisy", "noisy_threshold"

        return "Trash", "trash_threshold"

    # 9a. WQX / rotation
    if "rule_wqx_rot" not in DISABLED_RULES:
        wqx_ratio = sum(1 for w in words if any(c in WQX_CHARS for c in w)) / max(word_count, 1)

        if (rot_ratio > 0.50 or wqx_ratio > 0.10) and orig_lang_score < 0.75 and not is_upright_czech:
            _fire("rule_wqx_rot")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9b. Vowelless / all-caps
    if "rule_vowelless" not in DISABLED_RULES:
        if word_count <= 3 and vr < 0.30 and not is_upright_czech and is_all_caps_line(text_source) and not structured:
            _fire("rule_vowelless")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9c. Ledger fragmentation
    if "rule_ledger_fragmentation" not in DISABLED_RULES:
        if len(words) >= 4:
            frag_count = sum(1 for w in words if (w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2))

            if (frag_count / len(words)) > 0.60:
                _fire("rule_ledger_fragmentation")

                if qs < thresh_trash and _has_strong_garbage_evidence(
                    text_source,
                    valid_word_ratio=valid_word_ratio,
                    lang_score=lang_score,
                    orig_lang_score=orig_lang_score,
                    gibberish_present=gibberish_present,
                    garbage_density=garbage_density,
                    weird_ratio=weird_ratio,
                    is_upright_czech=is_upright_czech,
                ):
                    return check_rescues()

    # 9d. Mid-uppercase
    if "rule_mid_uppercase" not in DISABLED_RULES:
        if word_count <= 2 and any(_is_mid_uppercase(w.strip(_STRIP_CHARS)) for w in words) and not structured:
            _fire("rule_mid_uppercase")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9e. Bigram run
    if "rule_bigram_run" not in DISABLED_RULES:
        has_bigram_run = any(_RE_BIGRAM_RUN.search(w.strip(_STRIP_CHARS)) for w in words)

        if has_bigram_run:
            _fire("rule_bigram_run")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9f. Fragment tokens
    if "rule_fragment_tokens" not in DISABLED_RULES:
        lengths = [
            len(core) + 1 if w.endswith(".") else len(core) for w in words for core in [w.strip(_STRIP_CHARS)] if core
        ]

        fragment_like = bool(lengths and (sum(lengths) / len(lengths)) < 2.0)

        if fragment_like:
            _fire("rule_fragment_tokens")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # ------------------------------------------------------------
    # 10. Quality-score band routing
    # ------------------------------------------------------------
    if qs < CATEG_TRASH_SCORE_MAX:
        return check_rescues()

    if "rule_mostly_readable_noisy" not in DISABLED_RULES:
        if valid_word_ratio < MOSTLY_READABLE_VALID_MIN and not _lm_confident_czech(
            is_upright_czech,
            ppl,
            garbage_density,
        ):
            _fire("rule_mostly_readable_noisy")
            return "Noisy", "noisy_threshold"

    # [STEP 2] Restored character-level damage invariant capping route to Noisy
    if damaged:
        _fire("rule_damaged_token")
        return "Noisy", "noisy_threshold"

    return "Clear", "clear_threshold"


def categorize_line(
    qs: float,
    txt: str,
    wc: int,
    vowel_ratio: float,
    perplexity: float,
    weird_ratio: float = 0.0,
    return_reason: bool = False,
    valid_word_ratio: float = 1.0,
    lang_score: float = 1.0,
    orig_lang_score: float = 1.0,
    gibberish_present: bool = False,
    garbage_density: float = 0.0,
    is_upright_czech: bool = False,
    ghost_dominated: bool = False,
) -> tuple[str, float] | tuple[str, float, str]:
    categ, reason = determine_category(
        qs,
        txt,
        wc,
        vowel_ratio,
        perplexity,
        weird_ratio,
        valid_word_ratio,
        lang_score,
        orig_lang_score,
        gibberish_present,
        garbage_density,
        is_upright_czech,
        ghost_dominated,
    )

    if categ == "Trash":
        aligned_score = min(qs, CATEG_TRASH_SCORE_MAX - 0.0001)
    elif categ == "Noisy":
        aligned_score = max(qs, CATEG_TRASH_SCORE_MAX)
        aligned_score = min(aligned_score, CATEG_NOISY_SCORE_MAX - 0.0001)
    elif categ == "Clear":
        aligned_score = max(qs, CATEG_NOISY_SCORE_MAX)
    else:
        aligned_score = qs

    if return_reason:
        return categ, aligned_score, reason
    return categ, aligned_score


def _has_strong_garbage_evidence(
    text_source: str,
    *,
    valid_word_ratio: float,
    lang_score: float,
    orig_lang_score: float,
    gibberish_present: bool,
    garbage_density: float,
    weird_ratio: float,
    is_upright_czech: bool,
) -> bool:
    if is_structured_line(text_source):
        return False

    if gibberish_present:
        return True

    if valid_word_ratio <= 0.20:
        return True

    if lang_score <= 0.20 and orig_lang_score <= 0.50:
        return True

    if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
        return True

    if weird_ratio >= 0.75:
        return True

    if not is_upright_czech and lang_score <= 0.40 and weird_ratio >= 0.40:
        return True

    return False


def _looks_like_measurement(text_source: str) -> bool:
    """
    Detect structured archaeological measurement lines.

    The goal is to protect lines that may be noisy OCR but still contain
    meaningful measurements, dimensions, quantities, or physical descriptions.

    This predicate is deliberately conservative. It requires either:
      - an explicit measurement keyword with numeric context, or
      - multi-character measurement units (mm, cm, km, kg, ml, ha), or a
        bare metre unit glued directly to its number (0,4m), or
      - explicit measurement separator structures (e.g. v - 112, pr.okraje - 145).

    It strictly avoids classifying arbitrary digit-containing OCR or weak
    single-letter substrings/units as structured.

    Keyword/descriptor entries ending in a literal "." (rozm., pr., hl.) are
    matched WITHOUT a trailing \\b. A "." is a non-word character, so a
    trailing \\b can only hold when the abbreviation happens to be glued to a
    following word/digit character ("pr.okraje") — it silently fails whenever
    OCR (correctly) leaves a space or line-end after the dot ("pr. okraje",
    "rozm. 12", "pr. dna - 7"). The leading \\b is unaffected and still keeps
    these from matching mid-word; a lone hit still isn't enough on its own
    ("clouCelRa pr. 4" stays unmatched — one descriptor hit is below the
    corroboration threshold below).
    """
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    lowered = stripped.lower()

    # [STEP 3] Token-bounded full Czech measurement keywords.
    # Word-form entries keep a trailing boundary (avoids matching inside a
    # longer word); dot-terminated abbreviations must NOT have one — see
    # the docstring note on why \b cannot follow a literal "." into
    # whitespace or line-end.
    full_measurement_keywords = r"\brozm\." r"|\b(?:rozměry?|výška|šířka|délka|hloubka|průměr)\b"
    has_full_keyword = bool(re.search(full_measurement_keywords, lowered))

    # Secondary measurement descriptors (require multiple hits or numeric context).
    # Same dotted/word split as above, same reason.
    descriptor_keywords = r"\b(?:pr|hl)\." r"|\b(?:dna|hrdla|okraje)\b"
    descriptor_hits = len(re.findall(descriptor_keywords, lowered))

    # [STEP 3] Multi-character measurement units, or a bare metre unit glued
    # directly to its number (0,4m / 145-167m). A *spaced* bare unit ("3 m",
    # "o 5 m") stays unmatched — too weak a signal alone — but a glued bare
    # "m" is this corpus's single most common unit (far ahead of "mm"), so
    # excluding it outright cost real protection on genuine depth/height
    # readings; only the multi-character units are excluded from the
    # space-tolerant branch.
    has_unit = bool(
        re.search(
            r"\b\d+(?:[.,]\d+)?(?:\s*(?:mm|cm|km|kg|ml|ha)\b|m\b)",
            lowered,
        )
    )

    # Common measurement notation with explicit separators:
    #   v - 112mm
    #   pr.okraje - 145
    #   pr. dna - 7
    #   v: 144 mm
    has_measurement_separator = bool(
        re.search(
            r"\b(?:v|š|s|d|hl|pr|prům|výš|šíř|dél|hloub)"
            r"\.?\s*[:=\-]\s*\d",
            lowered,
        )
    )

    has_digits = any(char.isdigit() for char in stripped)

    if has_full_keyword and has_digits:
        return True

    if descriptor_hits >= 2 and has_digits:
        return True

    if has_measurement_separator:
        numeric_count = sum(char.isdigit() for char in stripped)
        if numeric_count >= 2:
            return True

    if has_unit:
        return True

    return False


# ---------------------------------------------------------------------------
# DORMANT candidate refinement to `_looks_like_measurement`'s `has_unit` branch,
# for the metre-spacing gap observed in issue #30. NOT WIRED INTO THE LIVE
# DETECTOR -- it exists only as a probe, exposed through
# `probe_spaced_decimal_metre_candidate()` and measured by
# `tools/recategorize_from_csv.py --probe-metre-candidate`.
#
# `has_unit` currently accepts a bare "m" only when glued directly to its
# number (`0,46m`), not when spaced (`3 m`) -- the latter stayed unmatched
# because a bare spaced unit alone is too weak a signal (see
# test_single_letter_units_and_probe_noise_rejected: "3 m", "o 5 m").
# Measurement against the 273-doc corpus found that restriction is now
# costing real coverage: 1,155 spaced-metre lines are unrecovered, and the
# quoted examples ("2,10 m", "215,5 193,120 m", "Z /193,445 m/",
# "Rovina profilu Z-V 214 192,740 m") are levelling/elevation readings, not
# noise. All of them carry a decimal separator in the number itself, which
# the rejected probes never do -- so a spaced bare "m" can be allowed
# specifically when the number is decimal-marked, without reopening the
# bare-integer case the pinned test guards against.
#
# Verified here to preserve every existing _looks_like_measurement assertion
# (glued forms, both pinned "3 m"/"o 5 m" rejections) and to recover the four
# quoted corpus lines, but the 1,155 figure above describes the *unrecovered*
# count under the current, narrower regex -- how much of it is actually
# decimal-marked (vs. some other spaced shape) hasn't been measured. Promoting
# this means replacing `has_unit`'s regex with the one below and re-running
# `tools/recategorize_from_csv.py` against the full local corpus to confirm the
# recovery and check for new false positives.
# ---------------------------------------------------------------------------
_RE_UNIT_CANDIDATE = re.compile(
    r"\b\d+(?:[.,]\d+\s*m\b|\s*(?:mm|cm|km|kg|ml|ha)\b|m\b)",
)


def _has_unit_with_spaced_decimal_metre_candidate(lowered: str) -> bool:
    """Probe for the dormant metre-spacing candidate on already-lowercased text."""
    return bool(_RE_UNIT_CANDIDATE.search(lowered))


def probe_spaced_decimal_metre_candidate(text_source: str) -> bool:
    """Return True when `text_source` matches the dormant metre-spacing probe."""
    return _has_unit_with_spaced_decimal_metre_candidate(text_source.lower())


# ---------------------------------------------------------------------------
# Rule for issue #30's "sonda: XIV." finding.
#
# `641f936`'s leading-\b fix on `_looks_like_measurement` correctly closed
# an accidental match — a bare "v." token used to match inside the Roman
# numeral of lines like "Plocha: 3; sonda: XIV." purely because keyword
# matching had no leading word-boundary yet.
#
# This rule explicitly recovers those plot/probe headers. It is now wired
# into `_looks_like_catalogue_reference` to protect these headers.
# ---------------------------------------------------------------------------
_RE_PLOT_PROBE_HEADER = re.compile(
    r"^plocha\s*:\s*\d+\s*;\s*sonda\s*:\s*[a-z]?[ivxlcdm]+\.$",
    flags=re.IGNORECASE,
)


def _looks_like_plot_probe_header(text_source: str) -> bool:
    """Match archaeological plot/probe headers: "Plocha: <n>; sonda: <roman>."."""
    stripped = " ".join(text_source.split())
    if not stripped:
        return False
    return bool(_RE_PLOT_PROBE_HEADER.match(stripped))


def _looks_like_catalogue_reference(text_source: str) -> bool:
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    if is_clean_reference(stripped):
        return True

    if _RE_SIGLUM.match(stripped):
        return True

    if _looks_like_plot_probe_header(stripped):
        return True

    identifier_pattern = re.compile(
        r"^[A-ZČŠŽĚŘÁÉÍÓÚŮÝ]{1,5}"
        r"\d{1,6}"
        r"(?:[/-][A-Z0-9ČŠŽĚŘÁÉÍÓÚŮÝ]{1,12}){1,8}"
        r"[.]?$",
        flags=re.IGNORECASE,
    )

    if identifier_pattern.match(stripped):
        return True

    table_reference_pattern = re.compile(
        r"\b(?:tb|tab|tabul)[.]?\s*"
        r"[IVXLCDM0-9]+"
        r".{0,20}"
        r"\b(?:č|čís|neg|negativ)[.]?"
        r"\s*\d+",
        flags=re.IGNORECASE,
    )

    if table_reference_pattern.search(stripped):
        return True

    reference_pattern = re.compile(
        r"(?:č[.]?\s*j[.]?|"
        r"č[.]?\s*neg[.]?|"
        r"čís[.]?|"
        r"inv[.]?|"
        r"kat[.]?|"
        r"\bčp[.]?|"
        r"ref[.]?)"
        r"\s*[A-Z0-9/-]*\d",
        flags=re.IGNORECASE,
    )

    if reference_pattern.search(stripped):
        return True

    return False


def _looks_like_date_or_document_reference(text_source: str) -> bool:
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    lowered = stripped.lower()

    has_date = bool(
        re.search(
            r"\b\d{1,2}\s*[./-]\s*\d{1,2}"
            r"(?:\s*[./-]\s*\d{2,4})?\b",
            stripped,
        )
    )

    has_year = bool(
        re.search(
            r"\b(?:18|19|20)\d{2}\b",
            stripped,
        )
    )

    document_marker = bool(
        re.search(
            r"\b(?:datum|date|č[.]?\s*j[.]?|"
            r"podpis|sign|spis|číslo|cislo|ref[.]?)\b",
            lowered,
            flags=re.IGNORECASE,
        )
    )

    if has_date and has_year:
        return True

    if document_marker and any(char.isdigit() for char in stripped):
        return True

    return False


_RE_TOC_ENTRY = re.compile(r"^\d{1,2}[.,]\s+\S.*\s(?:\d{1,3}|\S{1,2}\s*[-–]\s*\d{1,3})$")

_DOCUMENT_HEADER_LABELS = frozenset(
    {
        "obsah",
        "úvod",
        "závěr",
        "literatura",
        "seznam",
        "přílohy",
        "příloha",
        "poznámka",
        "shrnutí",
        "resumé",
    }
)


def _looks_like_document_structure_label(text_source: str) -> bool:
    stripped = " ".join(text_source.split())
    if not stripped:
        return False
    core = stripped.rstrip(" :;.-–—")
    return core.lower() in _DOCUMENT_HEADER_LABELS


def is_domain_notation(text_source: str) -> bool:
    """Archaeological / administrative notation shapes that must escape
    `rule_short_garbage`.

    Consulted at exactly ONE site — the outer guard of `rule_short_garbage` in
    `determine_category()` — and confers no other exemption. This is the whole
    point of it being separate from `is_structured_line()`: that predicate is
    read at eleven places in the rule chain and is an absolute veto at the first
    statement of `_has_strong_garbage_evidence()`, so widening it to cover
    notation would also switch off `rule_allcaps`, `rule_garbage_density`,
    `rule_zero_alpha`, `rule_vowelless` and `rule_mid_uppercase`, and could
    promote lines directly to Clear. That is far more blast radius than
    recovering grid references needs.

    Recognises notation, never vocabulary. `II/C`, `Reg.Bez.Aussig.` and `1 ks`
    have a *shape*; `malakofauna` and `Equus caballus` do not, and separating
    those from `oueussd` needs a lexicon rather than a pattern. They stay with
    `rule_short_garbage`.
    """
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    # The grid pattern is the loose one — its single-letter segment would
    # otherwise accept all-lowercase OCR mush like `o-e`. Require a capital or a
    # digit there, so `sektlll` and `edelite` stay out. The other four shapes
    # are self-identifying (a colon label, a unit word, a dotted tail), so they
    # do not need the guard and must not be subject to it: `radius prox.sin.` is
    # legitimate notation with no capital in it.
    if _RE_NOTATION_GRID.match(stripped) and _RE_NOTATION_HAS_CODE_CHAR.search(stripped):
        return True

    return bool(
        _RE_NOTATION_LABELLED.match(stripped)
        or _RE_NOTATION_COUNT.match(stripped)
        or _RE_NOTATION_ABBR.match(stripped)
        or _RE_NOTATION_ABBR_SP.match(stripped)
    )


def is_structured_line(text_source: str) -> bool:
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    if is_clean_reference(stripped):
        return True

    if _RE_SIGLUM.match(stripped):
        return True

    if _looks_like_catalogue_reference(stripped):
        return True

    if _looks_like_measurement(stripped):
        return True

    if _looks_like_date_or_document_reference(stripped):
        return True

    if _RE_TOC_ENTRY.match(stripped):
        return True

    if _looks_like_document_structure_label(stripped):
        return True

    return False


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


_RE_INITIALS = re.compile(r"^([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]\.?){1,3}$")
_RE_DOTTED_ABBREV = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ſ]{1,4}(\.[A-Za-zÀ-ÖØ-öø-ſ]{1,4})*$")
_NEUTRAL_LEXICON = frozenset({"dr", "x", "mm", "cm", "dm", "km", "g", "dkg", "kg", "ha", "hl", "ks", "m", "l"})
_RE_ROMAN_TOKEN = re.compile(r"^[IVXLCDM]{1,7}$")
_RE_ABBREV_NUM = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ſ]{1,4}[.,]\d+[a-z]?$")


def _is_neutral_token(core: str, raw: str = "", next_core: str = "") -> bool:
    if not any(c.isalnum() for c in core):
        return True
    if sum(c.isdigit() for c in core) / len(core) >= 0.50:
        return True
    if _RE_INITIALS.match(core):
        return True
    if core.lower() in _NEUTRAL_LEXICON:
        return True
    if _RE_ABBREV_NUM.match(core):
        return True
    if raw.rstrip(",;:-–—/)").endswith(".") and _RE_DOTTED_ABBREV.match(core):
        alpha = sum(c.isalpha() for c in core)
        if 2 <= alpha <= 5:
            return True
        if alpha == 1 and next_core and (any(c.isdigit() for c in next_core) or _RE_ROMAN_TOKEN.match(next_core)):
            return True
        if alpha == 1 and raw.rstrip().endswith(":"):
            return True
    return False


_RE_SIGLUM = re.compile(r"^([A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{1,4}\.){1,4}$")

# ── rule_domain_notation ────────────────────────────────────────────────────
# Archaeological / administrative notation that `rule_short_garbage` must not
# trash. Deliberately NARROWER than `is_structured_line()`: consulted at exactly
# one site (the outer guard of rule_short_garbage) and conferring no other
# exemption, so it cannot silently disable rule_allcaps, rule_garbage_density,
# rule_zero_alpha, rule_vowelless or rule_mid_uppercase the way widening
# `is_structured_line()` would.
#
# Scope is NOTATION, not vocabulary. Grid/context refs, counts, abbreviation
# chains and labelled refs are shapes a regex can recognise. Latin binomials
# (`Equus caballus`, `Ossa tarsi`) and bare foreign words (`malakofauna`) are
# vocabulary — they need a lexicon, not a pattern, and are deliberately left to
# `rule_short_garbage` here.
#
# Dimensions (`12,5 cm`, `145-167mm`, `0,46m`) are NOT covered: they already
# match `_looks_like_measurement()`, so they never reach rule_short_garbage in
# the first place. A second copy here would only be able to drift from it.

# A grid/context segment: roman numeral, short all-caps run, a single letter, or
# digits with an optional letter suffix. Lowercase word fragments are excluded
# on purpose — that is what keeps `Slaot-o` and `sektlll` out.
_NOTATION_SEGMENT = r"(?:[IVXLCDM]{1,6}|[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]{1,4}|[A-Za-z]|\d{1,3}[a-z]?)"

# `II/C`, `I-VIII-c`, `KK-XIII`, `A/1`, `XIV-2b` — at least one separator.
_RE_NOTATION_GRID = re.compile(rf"^{_NOTATION_SEGMENT}(?:[/\-]{_NOTATION_SEGMENT}){{1,4}}$")

# `Lokalisace: MM-III`, `sonda: III` — a word label, then a grid reference.
_RE_NOTATION_LABELLED = re.compile(
    rf"^[A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{{3,20}}\s*:\s*"
    rf"{_NOTATION_SEGMENT}(?:[/\-]{_NOTATION_SEGMENT}){{0,4}}$"
)

# `1 ks`, `2 ks`. Multi-character count units only — a bare single-letter unit
# (`3 m`, `o 5 m`) stays out, matching `_looks_like_measurement`'s own refusal.
_RE_NOTATION_COUNT = re.compile(r"^\d{1,4}\s*(?:ks|kusy|kusů|ex)\.?$", re.IGNORECASE)

# `Reg.Bez.Aussig.` — the same shape as _RE_SIGLUM but with segments up to 8
# characters, so German administrative chains stop failing on "Aussig".
_RE_NOTATION_ABBR = re.compile(r"^(?:[A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{1,8}\.){2,6}$")

# `radius prox.sin.` — a word followed by a dotted tail. Requires >= 2 tail
# segments: one is not enough (`vfetennl k.` must stay unmatched).
_RE_NOTATION_ABBR_SP = re.compile(
    r"^[A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{2,20}\s+"
    r"(?:[A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{1,8}\.){2,4}$"
)

_RE_NOTATION_HAS_CODE_CHAR = re.compile(r"[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ0-9]")


_DMG_SYMBOLS = frozenset("^»«■□¤§~<>#*@$")
_RE_DMG_APOSTROPHE = re.compile(r"[^\W\d_][’‘][^\W\d_]")
_RE_DMG_DIGIT_IN_WORD = re.compile(r"[a-záčďéěíňóřšťúůýž]{2}\d|\d[a-záčďéěíňóřšťúůýž]{2}")
_RE_DMG_VOWELLESS = re.compile(r"[a-záčďéěíňóřšťúůýž]{3,}$")
_CZ_VOWELS = frozenset("aeiouyáéěíóúůý")
_DMG_SYMBOLS_V8 = frozenset("©®™।")
_RE_DMG_CASE_MIX = re.compile(r"[a-záčďéěíňóřšťúůýž][A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]")


def count_damaged_tokens(text: str) -> int:
    """Count tokens carrying character-level OCR damage."""
    count = 0
    for token in text.split():
        core = token.strip(_STRIP_CHARS)
        if not core:
            continue
        if (
            any(c in _DMG_SYMBOLS for c in core)
            or any(c in _DMG_SYMBOLS_V8 for c in core)
            or _RE_DMG_APOSTROPHE.search(core)
            or _RE_DMG_DIGIT_IN_WORD.search(core)
            or (len(core) >= 4 and not token.endswith(".") and _RE_DMG_CASE_MIX.search(core))
            or (
                not token.rstrip(",;:").endswith(".")
                and _RE_DMG_VOWELLESS.match(core)
                and not (_CZ_VOWELS & set(core))
                and "r" not in core
                and "l" not in core
            )
        ):
            count += 1
    return count


_RE_BIGRAM_RUN = re.compile(r"([^\W\d_]{2})\1\1")

_RE_REF_MARKER = re.compile(
    r"\binv\b|\binv\.|\bkont\b|\bkont\.|\bn[áa]l\b|\bn[áa]l\.|\bobr\b|\bobr\.|"
    r"\btab\b|\btab\.|\bneg\b|\bneg\.|\bmax\.|\bmin\.|"
    r"č\.\s?j|č\.\s?inv|č\.\s?pl|inv\.\s?č|s\.\s?j\b",
    re.IGNORECASE,
)
_RE_MEASUREMENT = re.compile(
    r"\d\s?[.,]?\s?(?:mm|cm|km|kg|ml|ha)\b",
    re.IGNORECASE,
)


def is_clean_reference(text: str) -> bool:
    """True for a catalogue/measurement/inventory line with no OCR damage."""
    if count_damaged_tokens(text) > 0:
        return False
    return bool(_RE_REF_MARKER.search(text) or _RE_MEASUREMENT.search(text))


def compute_valid_ratio(text: str, word_set: set | None = None) -> float:
    words = text.split()
    if not words:
        return 0.0
    valid = 0
    evaluable = 0
    for wi, word in enumerate(words):
        core = word.strip(_STRIP_CHARS)
        if not core:
            continue
        if word_set is not None:
            evaluable += 1
            if core.lower() in word_set:
                valid += 1
        else:
            next_core = words[wi + 1].strip(_STRIP_CHARS) if wi + 1 < len(words) else ""
            if _is_neutral_token(core, word, next_core):
                continue
            evaluable += 1
            if core.lower() in SHORT_VALID_WORDS or core in SINGLE_CHAR_ALLOWED:
                valid += 1
                continue
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                if _is_mid_uppercase(core):
                    continue
                valid += 1
    if evaluable == 0:
        return 1.0
    return valid / evaluable


def is_non_text(text: str) -> bool:
    if not text:
        return False
    if re.match(r"^\d{3}\s\d{2}\s+[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]", text.strip()):
        return False
    if RE_NON_TEXT.match(text.strip()):
        return True

    stripped = text.strip()
    if " " not in stripped:
        if RE_ARCHIVE_CODE.match(stripped):
            return True
        if RE_ALPHANUM_TOKEN.match(stripped):
            if any(c.isdigit() for c in stripped):
                return True
            if stripped.isupper():
                if "X" in stripped:
                    return True
                if len(stripped) >= 10 and compute_vowel_ratio(stripped) < VOWEL_RATIO_LOW:
                    return True
    else:
        if len(stripped) <= 20 and any(c.isdigit() for c in stripped):
            if RE_ARCHIVE_REF_SPACED.match(stripped):
                return True

    if len(text) < 15 and compute_digit_ratio(text) > 0.5:
        return True
    return False


@contextmanager
def override_constants(values, modules=None):
    if modules is None:
        modules = (sys.modules[__name__],)
    saved: list[tuple[object, str, object]] = []
    try:
        for mod in modules:
            for name, value in values.items():
                if hasattr(mod, name):
                    saved.append((mod, name, getattr(mod, name)))
                    setattr(mod, name, value)
        yield
    finally:
        for mod, name, old in reversed(saved):
            setattr(mod, name, old)


def compute_quality_score(
    valid_word_ratio: float,
    perplexity: float,
    text_length: int,
    weird_ratio: float,
    vowel_ratio: float = 0.40,
    garbage_density: float = 0.0,
    lang_score: float | None = None,
    gibberish_ratio: float = 0.0,
    fused_ratio: float = 0.0,
    ppl_max: float | None = None,
    length_max: float | None = None,
    is_upright_czech: bool = False,
) -> float:
    if ppl_max is None:
        ppl_max = PERPLEXITY_THRESHOLD_MAX
    if length_max is None:
        length_max = QS_LENGTH_MAX

    total_weight = (
        QS_WEIGHT_VALID_WORD
        + QS_WEIGHT_WEIRD
        + QS_WEIGHT_PERPLEXITY
        + QS_WEIGHT_LENGTH
        + QS_WEIGHT_GARBAGE
        + QS_WEIGHT_VOWEL
        + QS_WEIGHT_LANG
        + QS_WEIGHT_GIBBERISH
        + QS_WEIGHT_FUSED
    )

    if total_weight <= 0.0:
        total_weight = 1.0

    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)
    norm_weird = 1.0 - min(weird_ratio, 1.0)

    active_garbage_weight = QS_WEIGHT_GARBAGE
    if text_length <= 12 and weird_ratio == 0.0 and garbage_density < max(QS_GARBAGE_NORM_MAX, 1e-9):
        active_garbage_weight = active_garbage_weight / 2.0

    norm_garbage = 1.0 - min(garbage_density / max(QS_GARBAGE_NORM_MAX, 1e-9), 1.0)

    vr = vowel_ratio
    if vr < VOWEL_RATIO_LOW:
        norm_vowel = (vr / VOWEL_RATIO_LOW) if VOWEL_RATIO_LOW > 0 else 1.0
    elif vr > VOWEL_RATIO_HIGH:
        span = max(1.0 - VOWEL_RATIO_HIGH, 1e-9)
        norm_vowel = max(0.0, 1.0 - (vr - VOWEL_RATIO_HIGH) / span)
    else:
        norm_vowel = 1.0

    norm_lang = lang_score if lang_score is not None else 0.5
    norm_gibb = 1.0 - min(gibberish_ratio, 1.0)
    norm_fused = 1.0 - min(fused_ratio, 1.0)

    base_score = (
        QS_WEIGHT_VALID_WORD * valid_word_ratio
        + QS_WEIGHT_WEIRD * norm_weird
        + QS_WEIGHT_PERPLEXITY * norm_ppl
        + QS_WEIGHT_LENGTH * norm_len
        + active_garbage_weight * norm_garbage
        + QS_WEIGHT_VOWEL * norm_vowel
        + QS_WEIGHT_LANG * norm_lang
        + QS_WEIGHT_GIBBERISH * norm_gibb
        + QS_WEIGHT_FUSED * norm_fused
    )

    if active_garbage_weight != QS_WEIGHT_GARBAGE:
        base_score += QS_WEIGHT_GARBAGE - active_garbage_weight

    base_score = base_score / total_weight

    short_penalty = 0.0
    if text_length <= 12 and (weird_ratio > 0.0 or garbage_density >= QS_GARBAGE_NORM_MAX):
        short_penalty = SHORT_NOISY_QS_PENALTY

    final_score = max(0.0, base_score - short_penalty)
    return min(1.0, final_score)
