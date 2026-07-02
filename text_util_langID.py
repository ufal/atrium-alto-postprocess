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

import configparser
import itertools
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
# every _fire(name) call increments the counter for that rule.  Outside a capture
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
    the run.  Nested calls stack correctly: the outer context is restored on
    exit, so existing sweep harnesses that call rule_fire_capture() inside
    override_constants() are safe.

    Usage::

        with rule_fire_capture() as counts:
            recategorize_dataframe(df, ...)
        print(counts)  # {'rule_hard_sweep': 12, 'penalty_wqx_rot': 0, ...}
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
_config_path = Path("config_langID.txt")
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
    for lang in _get_str("CLASSIFY", "TRUSTED_FOREIGN_LANGS", "deu,eng,fra,pol,ita").split(",")
    if lang.strip()
)


def _lang_base(lang_code: str) -> str:
    return lang_code.split("_")[0]


# (#3) Czech-specific diacritic glyphs. Presence of even one is a strong signal
# that a line is genuine Czech text rather than inverted/foreign garbage OCR;
# the page-level inverted-scan sweep and the short-garbage route both use it.
CZ_DIACS = frozenset(_get_str("TEXT_UTILS", "CZ_DIACS", "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"))

METADATA_MARKERS = frozenset(_get_str("TEXT_UTILS", "METADATA_MARKERS", "©,®").split(","))

VOWEL_CHARS = frozenset(_get_str("TEXT_UTILS", "VOWEL_CHARS", "aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ"))

ROTATABLE_CHARS = frozenset(_get_str("TEXT_UTILS", "ROTATBLE_CHARS", "pbqdnuwmoxszeyv"))


def has_cz_diacs(text: str) -> bool:
    """True if *text* contains at least one Czech diacritic glyph."""
    return any(ch in CZ_DIACS for ch in text)


_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(lng) for lng in COMMON_LANGS)

PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)

SHORT_PPL_CAP = _get_float("TEXT_UTILS", "SHORT_PPL_CAP", 850.0)

LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)
LANG_SCORE_CLEAR = _get_float("TEXT_UTILS", "LANG_SCORE_CLEAR", 0.75)

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

# (B2) Separate scale for normalising garbage_density inside compute_quality_score.
# Previously the same constant (CATEG_GARBAGE_DENSITY_HIGH) was reused at both the
# hard gate (rule_garbage_density) and the three QS-normalisation sites, making
# the two effects inseparable in the importance sweep.
# Default 0.35 == CATEG_GARBAGE_DENSITY_HIGH → bit-identical output at default config.
QS_GARBAGE_NORM_MAX = _get_float("TEXT_UTILS", "QS_GARBAGE_NORM_MAX", 0.35)


# Inverted / 180°-rotated scan detection
ROT_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "ROT_RATIO_INVERTED_MIN", 0.55)
WEIRD_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "WEIRD_RATIO_INVERTED_MIN", 0.35)
PPL_INVERTED_MIN = _get_float("TEXT_UTILS", "PPL_INVERTED_MIN", 200.0)
ROT_HIGH_LANG_CONF = _get_float("TEXT_UTILS", "ROT_HIGH_LANG_CONF", 0.90)

# (#3 Phase 2) override + structural-route thresholds, now config-driven.
LOWPPL_CLEAR_MAX = _get_float("TEXT_UTILS", "LOWPPL_CLEAR_MAX", 50.0)
HARD_SWEEP_LANG_MAX = _get_float("TEXT_UTILS", "HARD_SWEEP_LANG_MAX", 0.45)
HARD_SWEEP_PPL_MIN = _get_float("TEXT_UTILS", "HARD_SWEEP_PPL_MIN", 1000.0)
GHOST_DOMINATED_MIN_RATIO = _get_float("TEXT_UTILS", "GHOST_DOMINATED_MIN_RATIO", 0.5)
WORD_W_PENALTY = _get_float("TEXT_UTILS", "WORD_W_PENALTY", 0.20)

# (#3 A3) Page-level inverted-scan sweep — defined here (config-driven) and
# re-exported via `from text_util_langID import *` so langID_classify and the
# tests share one tunable source of truth.
INVERTED_RUN_MIN = _get_int("TEXT_UTILS", "INVERTED_RUN_MIN", 4)
INVERTED_PAGE_MAJORITY = _get_float("TEXT_UTILS", "INVERTED_PAGE_MAJORITY", 0.60)

# (#5) Page-context smoothing thresholds — promoted from inline literals in
# apply_document_postprocessing so they are config-driven, parity-overridable,
# and visible to the importance sweep. Defaults equal the previous literals, so
# the categoriser's output is unchanged.
SURROUNDED_TRASH_QS_MARGIN = _get_float("TEXT_UTILS", "SURROUNDED_TRASH_QS_MARGIN", 0.15)
PAGE_GARBAGE_CLEAR_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_CLEAR_MAX", 0.05)
PAGE_GARBAGE_LANG_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_LANG_MAX", 0.50)
PAGE_GARBAGE_MEDIAN_QS_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_MEDIAN_QS_MAX", 0.55)
PAGE_GARBAGE_NOISY_QS_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_NOISY_QS_MAX", 0.80)
PAGE_CLEAN_CLEAR_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_CLEAR_MIN", 0.60)
PAGE_CLEAN_MEDIAN_QS_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_MEDIAN_QS_MIN", 0.80)
PAGE_CLEAN_RECOVER_QS_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_RECOVER_QS_MIN", 0.45)

# Trash routes inside determine_category that all fold to the single
# `trash_threshold` diagnostic boolean (keeps "exactly one categoriser flag True"
# while preserving granular reason strings for logging / the re-scorer).
TRASH_REASONS = frozenset({"trash_threshold", "trash_hard_sweep", "trash_inverted"})

# Phase 4
MOSTLY_READABLE_VALID_MIN = _get_float("TEXT_UTILS", "MOSTLY_READABLE_VALID_MIN", 0.85)
SHORT_NOISY_QS_PENALTY = _get_float("TEXT_UTILS", "SHORT_NOISY_QS_PENALTY", 0.20)

LANG_SCORE_REMAP = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP", 0.75)
LANG_SCORE_REMAP_FAR = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP_FAR", 0.50)
SINGLE_CHAR_ALLOWED = _get_str("TEXT_UTILS", "SINGLE_CHAR_ALLOWED", "aAiIuUvVzZkKsS")
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

# (#3 2026-07-02 calibration) is_forgiven_headline tunables — see that function
# for the full token-classification contract.
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

# ---------------------------------------------------------------------------
# Module-level regexes hoisted from inner functions
# ---------------------------------------------------------------------------

_RE_SPACED_CAPS: re.Pattern = re.compile(
    r"(?<!\S)"
    r"([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ] ){3,}"
    r"[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]"
    r"(?!\S)"
)


def _collapse_spaced_caps(m: re.Match) -> str:
    letters = m.group(0).replace(" ", "")
    return letters[0].upper() + letters[1:].lower()


_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": frozenset("áčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ"),
    "deu": frozenset("äöüßÄÖÜ"),
}

# (#3) Extreme-perplexity trash route + LM-confident upright-Czech recovery.
PPL_EXTREME_MIN = _get_float("TEXT_UTILS", "PPL_EXTREME_MIN", 3000.0)
EXTREME_LANG_CONF = _get_float("TEXT_UTILS", "EXTREME_LANG_CONF", 0.85)
LOWPPL_CZECH_CLEAR_MAX = _get_float("TEXT_UTILS", "LOWPPL_CZECH_CLEAR_MAX", 180.0)
CZECH_CLEAR_GARBAGE_MAX = _get_float("TEXT_UTILS", "CZECH_CLEAR_GARBAGE_MAX", 0.15)

# Fix 1: Linguistic Anchor Bypass Config
ANCHOR_MIN_WORDS = _get_int("TEXT_UTILS", "ANCHOR_MIN_WORDS", 2)
ANCHOR_WORD_LEN = _get_int("TEXT_UTILS", "ANCHOR_WORD_LEN", 3)
ANCHOR_VOWEL_RATIO = _get_float("TEXT_UTILS", "ANCHOR_VOWEL_RATIO", 0.10)

# Fix 2: Suspicious Rotation Config
SUSPICIOUS_ROT_RATIO = _get_float("TEXT_UTILS", "SUSPICIOUS_ROT_RATIO", 0.65)
SUSPICIOUS_WQX_RATIO = _get_float("TEXT_UTILS", "SUSPICIOUS_WQX_RATIO", 0.15)
INVERTED_WEIRD_PENALTY = _get_float("TEXT_UTILS", "INVERTED_WEIRD_PENALTY", 0.45)

PPL_GARBAGE_ABSOLUTE = _get_float("TEXT_UTILS", "PPL_GARBAGE_ABSOLUTE", 30000.0)
GHOST_HITS_INVERTED_MIN = _get_int("TEXT_UTILS", "GHOST_HITS_INVERTED_MIN", 1)
TRAILING_FILL_CHARS = " ._:-<\u2013\u2014"

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


ROT_WHITELIST: frozenset = frozenset(
    {
        "po",
        "pod",
        "do",
        "od",
        "on",
        "ony",
        "by",
        "bez",
        "ne",
        "nebo",
        "ven",
        "den",
        "zde",
        "se",
        "ve",
        "mez",
        "pouze",
        "bude",
    }
)
_GHOST_REAL_WORD_COLLISIONS: frozenset = frozenset({"no", "bo"})


def _build_ghostlist() -> frozenset:
    ghosts = set()
    for w in ROT_WHITELIST:
        for img in (_transform_word(w, _MIRROR_GLYPH), _transform_word(w, _ROTATE_GLYPH)):
            if img:
                ghosts.add(img)
    return frozenset(ghosts - ROT_WHITELIST - _GHOST_REAL_WORD_COLLISIONS)


ROT_GHOSTLIST: frozenset = _build_ghostlist()

MIR_PAIRS = {
    "po": "oq",
    "pod": "boq",
    "do": "ob",
    "od": "bo",
    "on": "no",
    "ony": "yno",
    "by": "yd",
    "bez": "zed",
    "ne": "en",
    "nebo": "oden",
    "ven": "nev",
    "den": "neb",
    "zde": "ebz",
    "se": "es",
    "ve": "ev",
    "mez": "zem",
    "pouze": "ezouq",
    "bude": "ebud",
}
ROT_PAIRS = {
    "po": "od",
    "pod": "pod",
    "do": "op",
    "od": "po",
    "on": "uo",
    "by": "hq",
    "bez": "zeq",
    "ne": "eu",
    "nebo": "oqeu",
    "den": "uep",
    "zde": "epz",
    "se": "es",
    "mez": "zew",
    "pouze": "ezond",
    "bude": "epuq",
}
ROT_WHITELIST = set(MIR_PAIRS.keys()).union(set(ROT_PAIRS.keys()))
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
    if base == "slk":
        return new_label, score
    cap = remap_floor if suffix == "_Latn" else LANG_SCORE_REMAP_FAR
    return new_label, min(score, cap)


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
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in GARBAGE_KEEP_CHARS)
    return noise_chars / len(text)


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
    clean_text = re.sub(r"\b2(?=[a-záčďéěíňóřšťůúýž])", "z", clean_text)
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
    if RE_STAMP.search(clean_text) or "IVerc" in clean_text:
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
    has_wqx = any(c in "wqxWQX" for c in core)

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
    wqx_words = sum(1 for w in words if any(c in "wqxWQX" for c in w))
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


def determine_category(
    quality_score: float,
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

    rot_ratio = compute_rotatable_ratio(text_source)
    words = text_source.split()

    thresh_trash = CATEG_TRASH_SCORE_MAX + 0.35

    # --- Strict thresholds replacing legacy cumulative penalties ---
    if "rule_wqx_rot" not in DISABLED_RULES:
        wqx_ratio = sum(1 for w in words if any(c in "wqxWQX" for c in w)) / max(word_count, 1)
        if (rot_ratio > 0.50 or wqx_ratio > 0.10) and orig_lang_score < 0.75 and not is_upright_czech:
            _fire("rule_wqx_rot")
            return ("Trash", "trash_threshold") if quality_score < thresh_trash else ("Noisy", "noisy_threshold")

    if "rule_vowelless" not in DISABLED_RULES:
        if word_count <= 3 and vr < 0.30 and not is_upright_czech:
            if is_all_caps_line(text_source):
                _fire("rule_vowelless")
                return ("Trash", "trash_threshold") if quality_score < thresh_trash else ("Noisy", "noisy_threshold")

    if "rule_ledger_fragmentation" not in DISABLED_RULES:
        if words and len(words) >= 4:
            frag_count = sum(1 for w in words if w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2)
            if (frag_count / len(words)) > 0.60:
                _fire("rule_ledger_fragmentation")
                return ("Trash", "trash_threshold") if quality_score < thresh_trash else ("Noisy", "noisy_threshold")

    if "rule_mid_uppercase" not in DISABLED_RULES:
        if word_count <= 2 and any(_is_mid_uppercase(w.strip(_STRIP_CHARS)) for w in words):
            _fire("rule_mid_uppercase")
            return ("Trash", "trash_threshold") if quality_score < thresh_trash else ("Noisy", "noisy_threshold")

    # 1. Hard sweep
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

    # 2. Inverted / mirrored scan
    if "rule_inverted" not in DISABLED_RULES:
        if not is_upright_czech and (
            ghost_dominated
            or (
                not has_cz_diacs(text_source)
                and compute_rotatable_ratio(text_source) >= SUSPICIOUS_ROT_RATIO
                and ppl >= PPL_INVERTED_MIN
                and ghost_word_share(text_source)[0] >= GHOST_HITS_INVERTED_MIN
            )
        ):
            _fire("rule_inverted")
            return "Trash", "trash_inverted"

    # 3. All-caps vowel-less scramble
    # Evaluating vr < 0.10 first fail-fast is cheaper than checking is_all_caps_line
    if "rule_allcaps" not in DISABLED_RULES:
        if vr < 0.10 and is_all_caps_line(text_source):
            _fire("rule_allcaps")
            return "Trash", "allcaps_novowel"

    # 4. Overwhelming non-alphanumeric density
    if "rule_garbage_density" not in DISABLED_RULES:
        if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
            if "rule_trailing_fill_rescue" not in DISABLED_RULES and _trailing_fill_rescued(
                text_source, valid_word_ratio, word_count
            ):
                pass  # Bypass this override and allow it to route naturally
            else:
                _fire("rule_garbage_density")
                return "Trash", "trash_threshold"

    # (#3 2026-07-02 calibration) computed once, after the hard-sweep /
    # inverted / all-caps / garbage-density overrides above, so genuine
    # garbage is untouched — it only ever lifts a line from Trash to Noisy.
    forgiven = "rule_forgiven_headline" not in DISABLED_RULES and is_forgiven_headline(text_source, garbage_density)

    # 5. Structural short-garbage route
    if "rule_short_garbage" not in DISABLED_RULES and not forgiven:
        if (
            word_count <= ISOLATED_CHAR_MIN_TOKENS
            and not has_cz_diacs(text_source)
            and lang_score <= LANG_SCORE_REMAP
            and (gibberish_present or weird_ratio > 0.0)
        ):
            _fire("rule_short_garbage")
            return "Trash", "trash_threshold"

    # 6. High-confidence LM override
    if "rule_lowppl_clear" not in DISABLED_RULES:
        if ppl < LOWPPL_CLEAR_MAX and word_count >= 3:
            if valid_word_ratio < MOSTLY_READABLE_VALID_MIN:
                _fire("rule_lowppl_clear")
                return "Noisy", "noisy_threshold"
            _fire("rule_lowppl_clear")
            return "Clear", "lowppl_clear"

    # --- Strict thresholds replacing legacy cumulative penalties ---
    # Moved down to immediately precede QS band routing. Rules 1-6 ignore QS
    # and must take precedence. This restores parity by mimicking the priority
    # of the legacy cumulative subtraction.
    thresh_trash = CATEG_TRASH_SCORE_MAX + 0.35

    def check_rescues():
        if "rule_trailing_fill_rescue" not in DISABLED_RULES and _trailing_fill_rescued(
            text_source, valid_word_ratio, word_count
        ):
            _fire("rule_trailing_fill_rescue")
            return "Noisy", "noisy_threshold"
        if forgiven:
            _fire("rule_forgiven_headline")
            return "Noisy", "noisy_threshold"
        return "Trash", "trash_threshold"

    if "rule_wqx_rot" not in DISABLED_RULES:
        wqx_ratio = sum(1 for w in words if any(c in "wqxWQX" for c in w)) / max(word_count, 1)
        if (rot_ratio > 0.50 or wqx_ratio > 0.10) and orig_lang_score < 0.75 and not is_upright_czech:
            _fire("rule_wqx_rot")
            if quality_score < thresh_trash:
                return check_rescues()

    if "rule_vowelless" not in DISABLED_RULES:
        if word_count <= 3 and vr < 0.30 and not is_upright_czech:
            if is_all_caps_line(text_source):
                _fire("rule_vowelless")
                if quality_score < thresh_trash:
                    return check_rescues()

    if "rule_ledger_fragmentation" not in DISABLED_RULES:
        if words and len(words) >= 4:
            frag_count = sum(1 for w in words if w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2)
            if (frag_count / len(words)) > 0.60:
                _fire("rule_ledger_fragmentation")
                if quality_score < thresh_trash:
                    return check_rescues()

    if "rule_mid_uppercase" not in DISABLED_RULES:
        if word_count <= 2 and any(_is_mid_uppercase(w.strip(_STRIP_CHARS)) for w in words):
            _fire("rule_mid_uppercase")
            if quality_score < thresh_trash:
                return check_rescues()

    # 7. Quality-score band routing
    if quality_score < CATEG_TRASH_SCORE_MAX:
        return check_rescues()

    if "rule_mostly_readable_noisy" not in DISABLED_RULES:
        if valid_word_ratio < MOSTLY_READABLE_VALID_MIN and not _lm_confident_czech(
            is_upright_czech, ppl, garbage_density
        ):
            _fire("rule_mostly_readable_noisy")
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
    # Delegate immediately to the strict thresholds rather than applying
    # cumulative subtraction modifiers to the quality score.
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


# def categorize_line(
#     qs: float,
#     txt: str,
#     wc: int,
#     vowel_ratio: float,
#     perplexity: float,
#     weird_ratio: float = 0.0,
#     return_reason: bool = False,
#     valid_word_ratio: float = 1.0,
#     lang_score: float = 1.0,
#     orig_lang_score: float = 1.0,
#     gibberish_present: bool = False,
#     garbage_density: float = 0.0,
#     is_upright_czech: bool = False,
#     ghost_dominated: bool = False,
# ) -> tuple[str, float] | tuple[str, float, str]:
#     rot_ratio = compute_rotatable_ratio(txt)
#     words = txt.split()
#
#     # --- FIX 1: Sneaky leaks (WQX & Rotation) ---
#     if "penalty_wqx_rot" not in DISABLED_RULES:
#         wqx_ratio = sum(1 for w in words if any(c in "wqxWQX" for c in w)) / max(wc, 1)
#         if (rot_ratio > 0.50 or wqx_ratio > 0.10) and orig_lang_score < 0.75 and not is_upright_czech:
#             _fire("penalty_wqx_rot")
#             qs = max(0.0, qs - 0.35)
#
#     # --- FIX 2: Vowelless/Acronym gibberish ("WVL A") ---
#     if "penalty_vowelless" not in DISABLED_RULES:
#         if wc <= 3 and vowel_ratio < 0.30 and not is_upright_czech:
#             if is_all_caps_line(txt):
#                 _fire("penalty_vowelless")
#                 qs = max(0.0, qs - 0.35)
#
#     # --- FIX 3: Ledger / Table Fragmentation Loophole ---
#     if "penalty_ledger_fragmentation" not in DISABLED_RULES:
#         if words and len(words) >= 4:
#             frag_count = sum(1 for w in words if w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2)
#             if (frag_count / len(words)) > 0.60:
#                 _fire("penalty_ledger_fragmentation")
#                 qs = max(0.0, qs - 0.35)
#
#     # --- FIX 4: Isolated Mid-Uppercase Fragments ("ClAŕ") ---
#     if "penalty_mid_uppercase" not in DISABLED_RULES:
#         if wc <= 2 and any(_is_mid_uppercase(w.strip(_STRIP_CHARS)) for w in words):
#             _fire("penalty_mid_uppercase")
#             qs = max(0.0, qs - 0.35)
#
#     categ, reason = determine_category(
#         qs,
#         txt,
#         wc,
#         vowel_ratio,
#         perplexity,
#         weird_ratio,
#         valid_word_ratio,
#         lang_score,
#         orig_lang_score,
#         gibberish_present,
#         garbage_density,
#         is_upright_czech,
#         ghost_dominated,
#     )
#
#     if categ == "Trash":
#         aligned_score = min(qs, CATEG_TRASH_SCORE_MAX - 0.0001)
#     elif categ == "Noisy":
#         aligned_score = max(qs, CATEG_TRASH_SCORE_MAX)
#         aligned_score = min(aligned_score, CATEG_NOISY_SCORE_MAX - 0.0001)
#     elif categ == "Clear":
#         aligned_score = max(qs, CATEG_NOISY_SCORE_MAX)
#     else:
#         aligned_score = qs
#
#     if return_reason:
#         return categ, aligned_score, reason
#     return categ, aligned_score


# ---------------------------------------------------------------------------
# Simple Ratio & General Helpers
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
            if core.lower() in SHORT_VALID_WORDS or core in SINGLE_CHAR_ALLOWED:
                valid += 1
                continue
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                if _is_mid_uppercase(core):
                    continue
                valid += 1
    return valid / len(words)


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
                # (#3 2026-07-02 calibration) a genuine all-caps headline word
                # (e.g. "LITERATURA") should be scored, not hard-routed here —
                # but vowel-starved all-caps codes/garbage, and anything with
                # "X" (the original garbage-code signal), still are.
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

    # ABLATION GUARD (Part A): Prevent zero division if all weights are artificially wiped out
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
