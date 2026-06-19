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
CZ_DIACS = frozenset("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")


def has_cz_diacs(text: str) -> bool:
    """True if *text* contains at least one Czech diacritic glyph."""
    return any(ch in CZ_DIACS for ch in text)

_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(l) for l in COMMON_LANGS)

PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)

SHORT_PPL_CAP = _get_float("TEXT_UTILS", "SHORT_PPL_CAP", 850.0)

LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)
LANG_SCORE_CLEAR = _get_float("TEXT_UTILS", "LANG_SCORE_CLEAR", 0.75)

# Core signal weights
QS_WEIGHT_VALID_WORD = _get_float("TEXT_UTILS", "QS_WEIGHT_VALID_WORD", 0.30)
QS_WEIGHT_WEIRD      = _get_float("TEXT_UTILS", "QS_WEIGHT_WEIRD",      0.16)
QS_WEIGHT_PERPLEXITY = _get_float("TEXT_UTILS", "QS_WEIGHT_PERPLEXITY", 0.15)
QS_WEIGHT_LENGTH     = _get_float("TEXT_UTILS", "QS_WEIGHT_LENGTH",     0.05)
QS_WEIGHT_GARBAGE    = _get_float("TEXT_UTILS", "QS_WEIGHT_GARBAGE",    0.15)
QS_WEIGHT_VOWEL      = _get_float("TEXT_UTILS", "QS_WEIGHT_VOWEL",      0.07)
QS_WEIGHT_LANG       = _get_float("TEXT_UTILS", "QS_WEIGHT_LANG",       0.05)
QS_WEIGHT_GIBBERISH  = _get_float("TEXT_UTILS", "QS_WEIGHT_GIBBERISH",  0.04)
QS_WEIGHT_FUSED      = _get_float("TEXT_UTILS", "QS_WEIGHT_FUSED",      0.03)
QS_LENGTH_MAX        = _get_float("TEXT_UTILS", "QS_LENGTH_MAX",        100.0)

CATEG_GARBAGE_DENSITY_HIGH = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_HIGH", 0.35)

# Boundary Thresholds
CATEG_TRASH_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_TRASH_SCORE_MAX", 0.50)
CATEG_NOISY_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_NOISY_SCORE_MAX", 0.90)

# Inverted / 180°-rotated scan detection
ROT_RATIO_INVERTED_MIN   = _get_float("TEXT_UTILS", "ROT_RATIO_INVERTED_MIN",   0.55)
WEIRD_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "WEIRD_RATIO_INVERTED_MIN", 0.35)
PPL_INVERTED_MIN         = _get_float("TEXT_UTILS", "PPL_INVERTED_MIN",         200.0)
ROT_HIGH_LANG_CONF       = _get_float("TEXT_UTILS", "ROT_HIGH_LANG_CONF",       0.90)

# Near-boundary promotion
CLEAN_PROSE_MIN_SCORE = _get_float("TEXT_UTILS", "CLEAN_PROSE_MIN_SCORE", 0.65)
CLEAN_PROSE_WEIRD_MAX = _get_float("TEXT_UTILS", "CLEAN_PROSE_WEIRD_MAX", 0.08)
CLEAN_PROSE_PPL_MAX   = _get_float("TEXT_UTILS", "CLEAN_PROSE_PPL_MAX",   400.0)
CLEAN_PROSE_WC_MIN    = _config.getint("TEXT_UTILS", "CLEAN_PROSE_WC_MIN", fallback=4)

# Phase 4
MOSTLY_READABLE_VALID_MIN = _get_float("TEXT_UTILS", "MOSTLY_READABLE_VALID_MIN", 0.85)
SHORT_NOISY_QS_PENALTY    = _get_float("TEXT_UTILS", "SHORT_NOISY_QS_PENALTY", 0.20)

LANG_SCORE_REMAP = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP", 0.75)
LANG_SCORE_REMAP_FAR = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP_FAR", 0.50)
SINGLE_CHAR_ALLOWED = _get_str("TEXT_UTILS", "SINGLE_CHAR_ALLOWED", "aAiIuUvVzZkKsS")
SHORT_VALID_WORDS = _get_csv_set(
    "TEXT_UTILS", "SHORT_VALID_WORDS",
    "a,i,k,o,s,u,v,z,se,si,po,na,za,ze,do,od,ke,ku,ve,ní,mi,ti,by,je,to,co,ač,my,ty,on,ji,jí,už,až",
)

REPEAT_ALLOWED_CHARS = _get_str("TEXT_UTILS", "REPEAT_ALLOWED_CHARS", "oOuU")
REPEATED_DOUBLE_MIN = _get_int("TEXT_UTILS", "REPEATED_DOUBLE_MIN", 2)
VOWEL_RATIO_LOW  = _get_float("TEXT_UTILS", "VOWEL_RATIO_LOW",  0.20)
VOWEL_RATIO_HIGH = _get_float("TEXT_UTILS", "VOWEL_RATIO_HIGH", 0.70)
ACADEMIC_TITLES = _get_csv_set(
    "TEXT_UTILS", "ACADEMIC_TITLES",
    "PhDr,MUDr,JUDr,MVDr,RNDr,PaedDr,CSc,DrSc,Ing,Mgr,Bc,PhD,DiS,prof,doc",
)

LDL_ALLOWED_FOLLOW = frozenset(_get_str("TEXT_UTILS", "LDL_ALLOWED_FOLLOW", ".,/:%-;?)="))
LDL_UNITS = _get_csv_set("TEXT_UTILS", "LDL_UNITS", "m,cm,mm,g,kg,km,ha,l,ml")
GARBAGE_KEEP_CHARS = frozenset(_get_str("TEXT_UTILS", "GARBAGE_KEEP_CHARS", "")) | {" "}
FUSED_VOWEL_RUN_MIN = _get_int("TEXT_UTILS", "FUSED_VOWEL_RUN_MIN", 3)
WX_REPEAT_MIN = _get_int("TEXT_UTILS", "WX_REPEAT_MIN", 2)

ISOLATED_CHAR_RATIO_MAX  = _get_float("TEXT_UTILS", "ISOLATED_CHAR_RATIO_MAX", 0.40)
ISOLATED_CHAR_MIN_TOKENS = _get_int("TEXT_UTILS", "ISOLATED_CHAR_MIN_TOKENS", 3)
SYM_LET_DIG_NONTEXT = _get_str(
    "TEXT_UTILS", "SYM_LET_DIG_NONTEXT", "true"
).strip().lower() in ("true", "1", "yes", "on")

ALLOWED_INTERNAL: frozenset = frozenset(_get_str("TEXT_UTILS", "ALLOWED_INTERNAL", '.-,+()"\'/—–:%;?!/'))
_STRIP_CHARS: str = _get_str("TEXT_UTILS", "STRIP_CHARS", '.,;:!?()[]"\'/\\')

RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r'[^\w\s]{2,}')
RE_TRASH_LDL: re.Pattern = re.compile(r'[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]')
RE_NON_TEXT: re.Pattern = re.compile(r'^[\d\s\-\u2013\u2014/:.,()%]+$')
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r'[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]')
RE_ROMAN_NUMERAL: re.Pattern = re.compile(r'^[IVXLCDMivxlcdm]+\.?$')
RE_STAMP: re.Pattern = re.compile(r'^(?:[A-Za-z]+)?[\W_]*\d{2,4}\s*/\s*\d{2,4}[\W_]*$')
RE_ARCHIVE_CODE: re.Pattern = re.compile(r'^[A-Za-z]{1,3}\d{3,}(?:/\d+)?$')
RE_ALPHANUM_TOKEN: re.Pattern = re.compile(r'^[A-Za-z0-9]{5,}$')
RE_ARCHIVE_REF_SPACED: re.Pattern = re.compile(r'^[A-Za-záčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]{1,5}[\s.\-]+\d{1,}')

# ---------------------------------------------------------------------------
# Module-level regexes hoisted from inner functions
# ---------------------------------------------------------------------------

_RE_SPACED_CAPS: re.Pattern = re.compile(
    r'(?<!\S)'
    r'([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ] ){3,}'
    r'[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]'
    r'(?!\S)'
)

def _collapse_spaced_caps(m: re.Match) -> str:
    letters = m.group(0).replace(' ', '')
    return letters[0].upper() + letters[1:].lower()

_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": frozenset("áčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ"),
    "deu": frozenset("äöüßÄÖÜ"),
}

# ---------------------------------------------------------------------------
# Lexicon Integration for Rotation/Inversion detection
# ---------------------------------------------------------------------------

# Left-right mirror mapping + reversal
MIR_PAIRS = {
    "po": "oq", "pod": "boq", "do": "ob", "od": "bo",
    "on": "no", "ony": "yno", "by": "yd", "bez": "zed",
    "ne": "en", "nebo": "oden", "ven": "nev", "den": "neb",
    "zde": "ebz", "se": "es", "ve": "ev", "mez": "zem",
    "pouze": "ezouq", "bude": "ebud"
}

# True 180-degree rotation + reversal
ROT_PAIRS = {
    "po": "od", "pod": "pod", "do": "op", "od": "po",
    "on": "uo", "by": "hq", "bez": "zeq",
    "ne": "eu", "nebo": "oqeu", "den": "uep",
    "zde": "epz", "se": "es", "mez": "zew",
    "pouze": "ezond", "bude": "epuq"
}

# Unified Whitelist: Any real word from either dictionary
ROT_WHITELIST = set(MIR_PAIRS.keys()).union(set(ROT_PAIRS.keys()))

# Unified Ghostlist: All mirrored/rotated outcomes.
# Collisions are strictly pruned so upright text isn't flagged as its own ghost.
ROT_GHOSTLIST = set(MIR_PAIRS.values()).union(set(ROT_PAIRS.values())) - ROT_WHITELIST

def analyze_rotation_signals(text: str, rot_ratio: float) -> tuple[bool, bool]:
    """
    Analyzes text for both mirrored and 180-rotated signals using a unified metric.
    Returns (is_upright_czech, ghost_dominated).
    """
    words = [w.lower() for w in re.split(r'\W+', text) if w]
    if not words:
        return has_cz_diacs(text), False

    # Count hits against the combined dictionaries
    real_hits = sum(1 for w in words if w in ROT_WHITELIST)
    ghost_hits = sum(1 for w in words if w in ROT_GHOSTLIST)

    # Upright confirmation: Spares any line with a real rotatable Czech word or diacritic
    is_upright_czech = has_cz_diacs(text) or (real_hits > 0)

    # Estimate susceptible tokens using rot_ratio
    rotatable_tokens_est = max(1, int(rot_ratio * len(words)))

    # Single unified metric: Is the line dominated by mirrored OR rotated ghosts?
    ghost_dominated = (ghost_hits > 0) and (ghost_hits / rotatable_tokens_est >= 0.5)

    return is_upright_czech, ghost_dominated

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_mid_uppercase(core: str) -> bool:
    if len(core) < 2 or core.isupper(): return False
    if core.rstrip('.') in ACADEMIC_TITLES: return False
    if _has_starting_uppercase(core): return True
    caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
    if caps_run >= 2 and any(c.islower() for c in core[caps_run:]): return True
    return False

def _has_starting_uppercase(core: str) -> bool:
    if len(core) < 2 or core.isupper(): return False
    if core.rstrip('.') in ACADEMIC_TITLES: return False
    return core[0].isupper() and core[1].isupper()

def _split_subtokens(word: str) -> list[str]:
    return [p for p in re.split(r"[.\-\u2013]", word) if p]

def remap_lang(label: str, score: float, known_bases: frozenset, default_lang: str, remap_floor: float = LANG_SCORE_REMAP) -> tuple[str, float]:
    base = _lang_base(label)
    if base in known_bases: return label, score
    suffix = label[len(base):]
    new_label = default_lang + suffix
    if base == "slk": return new_label, score
    cap = remap_floor if suffix == "_Latn" else LANG_SCORE_REMAP_FAR
    return new_label, min(score, cap)

def _has_repeated_run(core: str) -> bool:
    if len(core) < 4: return False
    for ch in set(core):
        if ch.isdigit(): continue
        if ch * 3 in core: return True
        if ch in REPEAT_ALLOWED_CHARS: continue
        if ch * 2 in core and core.count(ch) >= REPEATED_DOUBLE_MIN: return True
        if (core.count(ch) / len(core) >= 0.30) and core.count(ch) >= 3: return True
    return False

def _trailing_alpha_run(token: str, start: int) -> str:
    j = start
    while j < len(token) and token[j].isalpha(): j += 1
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
    if not alpha: return None
    for lang_code, diacs in _LANG_DIACRITICS.items():
        if lang_code not in expected_bases: continue
        ratio = sum(1 for c in alpha if c in diacs) / len(alpha)
        if ratio >= threshold: return lang_code
    return None

def compute_garbage_density(text: str) -> float:
    if not text: return 0.0
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in GARBAGE_KEEP_CHARS)
    return noise_chars / len(text)

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
        count += sum(1 for ch in core if not ch.isalnum() and ch not in ALLOWED_INTERNAL)
    return count

def detect_repeated_chars(text: str) -> int:
    count = 0
    for word in text.split():
        if any(_has_repeated_run(sub.strip(_STRIP_CHARS)) for sub in _split_subtokens(word)): count += 1
    return count

def compute_vowel_ratio(text: str) -> float:
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    denom = [c for c in text if c.isalpha() or ((not c.isalnum()) and not c.isspace())]
    if not denom: return 0.0
    return sum(1 for c in denom if c in vowels) / len(denom)

def detect_gibberish_words(text: str) -> int:
    vowels = frozenset("aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ")
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if len(core) < 4 or core.isupper(): continue
            numeric_chars = sum(1 for c in core if c.isdigit() or c in '-./,;:')
            if numeric_chars / len(core) >= 0.6: continue
            letters = [c for c in core if c.isalpha()]
            if not letters: continue
            if sum(1 for c in letters if c in vowels) / len(letters) > VOWEL_RATIO_HIGH:
                flagged = True
                break
        if flagged: count += 1
    return count

def _has_ldl(token: str) -> bool:
    n = len(token)
    for i, ch in enumerate(token):
        if not ch.isdigit(): continue
        nxt = token[i + 1] if i + 1 < n else ""
        prev = token[i - 1] if i > 0 else ""
        if nxt and not nxt.isspace() and not nxt.isdigit() and nxt not in LDL_ALLOWED_FOLLOW:
            if nxt.isalpha():
                run = _trailing_alpha_run(token, i + 1)
                if run.lower() in LDL_UNITS: continue
            return True
        if prev.isalpha(): return True
    return False

def detect_letter_digit_letter(text: str) -> int:
    return sum(1 for word in text.split() if _has_ldl(word))

def detect_mid_uppercase(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip('.,;:!?()[]"\'-/')
        if _is_mid_uppercase(core): count += 1
    return count

def detect_wx_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if not core: continue
            if sum(1 for c in core if c in "wW") >= WX_REPEAT_MIN or sum(1 for c in core if c in "xX") >= WX_REPEAT_MIN:
                flagged = True
                break
        if flagged: count += 1
    return count

def is_all_caps_line(text: str) -> bool:
    alpha_words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not alpha_words: return False
    return all(w.isupper() for w in alpha_words)

_RE_FUSED_CONSONANT_RUN: re.Pattern = re.compile(r'[bcčdfghjklmnpqrřsštvwxzž]{5,}', re.IGNORECASE)
_RE_FUSED_VOWEL_RUN: re.Pattern = re.compile(r'[aeiouyáéíóúýěůäöü]{%d,}' % FUSED_VOWEL_RUN_MIN, re.IGNORECASE)

def detect_fused_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if not core or not any(c.isalpha() for c in core): continue
            if len(core) > 14 or _RE_FUSED_CONSONANT_RUN.search(core) or _RE_FUSED_VOWEL_RUN.search(core):
                flagged = True
                break
        if flagged: count += 1
    return count

# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------

def pre_filter_line(line: str) -> tuple[str, str]:
    clean_text = line.strip()
    if not clean_text: return "Empty", ""

    clean_text = re.sub(r'(?<=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])1(?=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])', 'l', clean_text)
    clean_text = re.sub(r'\b2(?=[a-záčďéěíňóřšťůúýž])', 'z', clean_text)
    clean_text = _RE_SPACED_CAPS.sub(_collapse_spaced_caps, clean_text)

    metadata_markers = [
        "Tb.", "č.neg", "neg.", "obr.", "obr ", "neg ", "Tb ", "č. neg",
        "č neg", "č.neg.", "neg.", "neg ", "Tb.", "Tb ", "č.neg.",
        "č. neg.", "č neg.", "č.", "str.", "Datum"
    ]
    if any(marker.lower() in clean_text.lower() for marker in metadata_markers): return "Process", clean_text

    if clean_text.startswith('"') and not clean_text.endswith('"'): clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'): clean_text = '"' + clean_text

    n_chars = len(clean_text)

    if is_non_text(clean_text): return "Non-text", clean_text
    if RE_ROMAN_NUMERAL.match(clean_text.strip()): return "Non-text", clean_text
    if RE_STAMP.search(clean_text) or "IVerc" in clean_text: return "Non-text", clean_text

    if sum(c.isdigit() for c in clean_text) / n_chars > 0.4: return "Process", clean_text

    unique_symbols = set(c for c in clean_text if not c.isspace())
    if n_chars < 4 or len(unique_symbols) < 3: return "Non-text", clean_text

    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3: return "Non-text", clean_text

    tokens = clean_text.split()
    if SYM_LET_DIG_NONTEXT and len(tokens) == 1 and has_symbol_letter_digit(tokens[0]):
        return "Non-text", clean_text

    if len(tokens) >= ISOLATED_CHAR_MIN_TOKENS:
        isolated = sum(1 for tok in tokens if len(tok.strip(_STRIP_CHARS)) == 1 and tok.strip(_STRIP_CHARS).isalnum())
        if isolated / len(tokens) >= ISOLATED_CHAR_RATIO_MAX: return "Non-text", clean_text

    return "Process", clean_text

def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    clean_line = line_text.strip()
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"
    matches = list(re.finditer(pattern, clean_line))
    if not matches: return clean_line, "", ""
    last_prefix = last_suffix = ""

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
        if core in SINGLE_CHAR_ALLOWED or '.' in word: return 0.0
        if core.isdigit(): return 0.25
        if not core.isalpha(): return 0.0
        return 0.85
    if len(core) < 2: return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)
    has_rep = _has_repeated_run(core)
    has_ldl = _has_ldl(core)
    has_uppercase = _is_mid_uppercase(core)
    has_w = 'w' in core.lower()

    has_caps_prefix = False
    if len(core) >= 4 and not core.isupper() and core.rstrip('.') not in ACADEMIC_TITLES:
        caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
        if caps_run >= 2 and any(c.islower() for c in core[caps_run:]): has_caps_prefix = True

    return min(1.0, 0.40 * has_strange + 0.35 * has_rep + 0.15 * has_ldl + 0.10 * has_uppercase + 0.20 * has_caps_prefix + 0.20 * has_w)

def score_words_in_line(text: str) -> list[tuple[str, float]]:
    return [(w, score_word(w)) for w in text.split()]

def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    if not word_scores: return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)

# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------

def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    import torch
    from torch import nn
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

def determine_category(quality_score: float, text_source: str, word_count: int,
                       vr: float, ppl: float, weird_ratio: float = 0.0,
                       valid_word_ratio: float = 1.0,
                       lang_score: float = 1.0,
                       orig_lang_score: float = 1.0,
                       gibberish_present: bool = False,
                       garbage_density: float = 0.0,
                       is_upright_czech: bool = False,
                       ghost_dominated: bool = False) -> tuple[str, str]:
    if word_count == 0 or not text_source.strip():
        return "Empty", "empty"

    # 1. NEW: Hard sweep for true garbage leaking via capped language confidence
    if orig_lang_score < 0.45 and ppl > 1000.0:
        return "Trash", "trash_hard_sweep"

    # 2. NEW: Exact lexicon ghost matching sweep for inverted scans
    if ghost_dominated and not is_upright_czech:
        return "Trash", "trash_inverted"

    if is_all_caps_line(text_source) and vr < 0.10:
        return "Trash", "allcaps_novowel"

    if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
        return "Trash", "trash_threshold"

    if (word_count <= ISOLATED_CHAR_MIN_TOKENS
            and not has_cz_diacs(text_source)
            and lang_score <= LANG_SCORE_REMAP
            and (gibberish_present or weird_ratio > 0.0)):
        return "Trash", "trash_threshold"

    if ppl < 50.0 and word_count >= 3:
        if valid_word_ratio < MOSTLY_READABLE_VALID_MIN: return "Noisy", "noisy_threshold"
        return "Clear", "lowppl_clear"

    if quality_score < CATEG_TRASH_SCORE_MAX:
        return "Trash", "trash_threshold"

    if quality_score < CATEG_NOISY_SCORE_MAX:
        if (quality_score >= CLEAN_PROSE_MIN_SCORE
                and word_count >= CLEAN_PROSE_WC_MIN
                and weird_ratio < CLEAN_PROSE_WEIRD_MAX
                and ppl < CLEAN_PROSE_PPL_MAX):
            if valid_word_ratio < MOSTLY_READABLE_VALID_MIN: return "Noisy", "noisy_threshold"
            return "Clear", "cleanprose_clear"
        return "Noisy", "noisy_threshold"

    if valid_word_ratio < MOSTLY_READABLE_VALID_MIN:
        return "Noisy", "noisy_threshold"

    return "Clear", "clear_threshold"


def categorize_line(
        qs: float,
        txt: str,
        wc: int,
        vowel_ratio: float,
        perplexity: float,
        rot_ratio: float = 0.0,
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
        qs, txt, wc, vowel_ratio, perplexity, weird_ratio,
        valid_word_ratio, lang_score, orig_lang_score,
        gibberish_present, garbage_density,
        is_upright_czech, ghost_dominated
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
            if core.lower() in SHORT_VALID_WORDS or core in SINGLE_CHAR_ALLOWED:
                valid += 1
                continue
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                if _is_mid_uppercase(core): continue
                valid += 1
    return valid / len(words)

def is_non_text(text: str) -> bool:
    if not text: return False
    if re.match(r'^\d{3}\s\d{2}\s+[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]', text.strip()): return False
    if RE_NON_TEXT.match(text.strip()): return True

    stripped = text.strip()
    if ' ' not in stripped:
        if RE_ARCHIVE_CODE.match(stripped): return True
        if RE_ALPHANUM_TOKEN.match(stripped):
            if any(c.isdigit() for c in stripped) or (stripped.isupper() and ('X' in stripped or len(stripped) >= 10)): return True
    else:
        if len(stripped) <= 20 and any(c.isdigit() for c in stripped):
            if RE_ARCHIVE_REF_SPACED.match(stripped): return True

    if len(text) < 15 and compute_digit_ratio(text) > 0.5: return True
    return False

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
        ppl_max: float = PERPLEXITY_THRESHOLD_MAX,
        length_max: float = QS_LENGTH_MAX,
        rot_ratio: float = 0.0,
        is_upright_czech: bool = False,
) -> float:
    total_weight = (
            QS_WEIGHT_VALID_WORD + QS_WEIGHT_WEIRD + QS_WEIGHT_PERPLEXITY +
            QS_WEIGHT_LENGTH + QS_WEIGHT_GARBAGE + QS_WEIGHT_VOWEL +
            QS_WEIGHT_LANG + QS_WEIGHT_GIBBERISH + QS_WEIGHT_FUSED
    )

    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)
    norm_weird = 1.0 - min(weird_ratio, 1.0)

    active_garbage_weight = QS_WEIGHT_GARBAGE
    if text_length <= 12 and weird_ratio == 0.0 and garbage_density < max(CATEG_GARBAGE_DENSITY_HIGH, 1e-9):
        active_garbage_weight = active_garbage_weight / 2.0

    norm_garbage = 1.0 - min(garbage_density / max(CATEG_GARBAGE_DENSITY_HIGH, 1e-9), 1.0)

    vr = vowel_ratio
    if vr < VOWEL_RATIO_LOW: norm_vowel = (vr / VOWEL_RATIO_LOW) if VOWEL_RATIO_LOW > 0 else 1.0
    elif vr > VOWEL_RATIO_HIGH:
        span = max(1.0 - VOWEL_RATIO_HIGH, 1e-9)
        norm_vowel = max(0.0, 1.0 - (vr - VOWEL_RATIO_HIGH) / span)
    else: norm_vowel = 1.0

    norm_lang = lang_score if lang_score is not None else 0.5
    norm_gibb = 1.0 - min(gibberish_ratio, 1.0)
    norm_fused = 1.0 - min(fused_ratio, 1.0)

    base_score = (
            QS_WEIGHT_VALID_WORD * valid_word_ratio +
            QS_WEIGHT_WEIRD * norm_weird +
            QS_WEIGHT_PERPLEXITY * norm_ppl +
            QS_WEIGHT_LENGTH * norm_len +
            active_garbage_weight * norm_garbage +
            QS_WEIGHT_VOWEL * norm_vowel +
            QS_WEIGHT_LANG * norm_lang +
            QS_WEIGHT_GIBBERISH * norm_gibb +
            QS_WEIGHT_FUSED * norm_fused
    )

    if active_garbage_weight != QS_WEIGHT_GARBAGE:
        base_score += (QS_WEIGHT_GARBAGE - active_garbage_weight)

    base_score = base_score / total_weight

    rot_penalty = 0.0
    # BYPASS rotation penalty entirely for confirmed upright Czech
    if not is_upright_czech and rot_ratio >= ROT_RATIO_INVERTED_MIN:
        if weird_ratio >= WEIRD_RATIO_INVERTED_MIN:
            rot_penalty = (rot_ratio * weird_ratio) * 2.0
        elif perplexity >= PPL_INVERTED_MIN:
            rot_penalty = rot_ratio * 0.5

        if lang_score is not None and lang_score >= ROT_HIGH_LANG_CONF:
            rot_penalty *= 0.5

    short_penalty = 0.0
    if text_length <= 12 and (weird_ratio > 0.0 or garbage_density >= CATEG_GARBAGE_DENSITY_HIGH):
        short_penalty = SHORT_NOISY_QS_PENALTY

    final_score = max(0.0, base_score - rot_penalty - short_penalty)
    return min(1.0, final_score)