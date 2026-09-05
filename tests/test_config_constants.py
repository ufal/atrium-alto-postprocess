"""
(#7 Tier 1) Regression locks for the config-backed language/collection
constants. Every default must be bit-identical to the previous in-code
literal so the migration is behaviour-neutral at the shipped config
(tests/test_recategorize_parity.py remains the end-to-end gate).
"""

from __future__ import annotations

import configparser
import os
import re
import subprocess
import sys
from pathlib import Path

import text_util as tu
from classify_TEXT import FASTTEXT_MODEL, TRUST_TIER_TRUSTED, TRUST_TIER_UNKNOWN

_ROOT = Path(__file__).resolve().parent.parent


def test_tier1_defaults_match_previous_literals():
    """Shipped config values == the literals they replaced."""
    assert tu.WQX_CHARS == frozenset("wqxWQX")
    assert tu.DEU_DIACS == frozenset("äöüßÄÖÜ")
    assert tu._LANG_DIACRITICS["ces"] == tu.CZ_DIACS
    assert tu._LANG_DIACRITICS["deu"] == tu.DEU_DIACS
    assert tu.DIACRITIC_INFER_THRESHOLD == 0.07
    assert tu.NONTEXT_MARKERS == frozenset({"IVerc"})
    assert tu.REMAP_KEEP_SCORE_LANGS == frozenset({"slk"})
    assert tu._GHOST_REAL_WORD_COLLISIONS == frozenset({"no", "bo"})
    assert FASTTEXT_MODEL == "lid.176.bin"
    assert TRUST_TIER_TRUSTED == 0.85
    assert TRUST_TIER_UNKNOWN == 0.50


def test_rot_whitelist_matches_previous_effective_union():
    """The old effective ROT_WHITELIST was the union of MIR_PAIRS/ROT_PAIRS
    keys (the dicts' values were dead); the config default must equal it."""
    expected = frozenset(
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
    assert frozenset(tu.ROT_WHITELIST) == expected
    # Derived ghostlist stays disjoint from real words and collisions.
    assert not (tu.ROT_GHOSTLIST & frozenset(tu.ROT_WHITELIST))
    assert not (tu.ROT_GHOSTLIST & tu._GHOST_REAL_WORD_COLLISIONS)
    assert tu.ROT_GHOSTLIST == tu._build_ghostlist()


def test_trailing_fill_chars_escape_decoding():
    """The \\x20 escape convention must survive configparser's leading-
    whitespace stripping and decode back to the previous literal."""
    assert tu.TRAILING_FILL_CHARS == " ._:-<\u2013\u2014"
    assert tu.TRAILING_FILL_CHARS.startswith(" ")


def test_remap_lang_keep_score_langs():
    """slk keeps its original confidence through the remap (config-driven)."""
    label, score = tu.remap_lang("slk_Latn", 0.42, frozenset({"ces", "deu", "eng"}), "ces")
    assert label == "ces_Latn"
    assert score == 0.42


def test_nontext_marker_routes_prefilter():
    """A configured marker still forces the Non-text route."""
    categ, _ = tu.pre_filter_line("IVerc 123/45")
    assert categ == "Non-text"


def test_tier1_key_roundtrip_from_alternate_config(tmp_path):
    """A changed config value must actually reach the module constants
    (guards the LANGID_CONFIG path and the key spellings end-to-end)."""
    cfg = tmp_path / "alt_config.txt"
    cfg.write_text(
        "[CLASSIFY]\n"
        "FASTTEXT_MODEL = custom.bin\n"
        "REMAP_KEEP_SCORE_LANGS = slk,pol\n"
        "TRUST_TIER_TRUSTED = 0.9\n"
        "[TEXT_UTILS]\n"
        "WQX_CHARS = xyz\n"
        "ROT_WHITELIST = po,do\n"
        "TRAILING_FILL_CHARS = \\x20.:\n"
        "NONTEXT_MARKERS = FOO,BAR\n",
        encoding="utf-8",
    )
    code = (
        "import text_util as tu;"
        "import classify_TEXT as lc;"
        "assert tu.WQX_CHARS == frozenset('xyz'), tu.WQX_CHARS;"
        "assert frozenset(tu.ROT_WHITELIST) == frozenset({'po', 'do'});"
        "assert tu.TRAILING_FILL_CHARS == ' .:', repr(tu.TRAILING_FILL_CHARS);"
        "assert tu.NONTEXT_MARKERS == frozenset({'FOO', 'BAR'});"
        "assert tu.REMAP_KEEP_SCORE_LANGS == frozenset({'slk', 'pol'});"
        "assert lc.FASTTEXT_MODEL == 'custom.bin';"
        "assert lc.TRUST_TIER_TRUSTED == 0.9;"
    )
    env = dict(os.environ, LANGID_CONFIG=str(cfg))
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


# ---------------------------------------------------------------------------
# (12-factor III) Config <-> code agreement.
#
# Nothing used to check that the two sides matched in either direction, which is
# how eight TEXT_UTILS constants came to be declared in text_util.py, read
# through _get_float(), and advertised as tunable in tools/SWEEP_NOTES.md while
# having no key in setup/config.txt at all — they silently ran on their in-code
# defaults and no test could see it.
# ---------------------------------------------------------------------------

_RE_GET = re.compile(r'_get_(?:float|int|str|csv_set)\(\s*"([A-Z_]+)"\s*,\s*"([A-Z_0-9]+)"')


def _declared_text_utils_keys():
    """Every ("TEXT_UTILS", KEY) pair text_util.py actually reads."""
    source = (_ROOT / "text_util.py").read_text(encoding="utf-8")
    return {key for section, key in _RE_GET.findall(source) if section == "TEXT_UTILS"}


def _config_text_utils_keys():
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read(_ROOT / "setup" / "config.txt")
    return set(parser.options("TEXT_UTILS"))


def test_every_constant_read_by_code_has_a_config_key():
    """A constant with no key cannot be configured, only recompiled."""
    missing = sorted(_declared_text_utils_keys() - _config_text_utils_keys())
    assert not missing, (
        f"read by text_util.py but absent from setup/config.txt [TEXT_UTILS]: {missing}\n"
        "Add the key with its in-code default (behaviour-neutral), or stop reading it from config."
    )


def test_every_config_key_is_read_by_code():
    """An orphan key is a lie to whoever edits it."""
    orphans = sorted(_config_text_utils_keys() - _declared_text_utils_keys())
    assert not orphans, (
        f"present in setup/config.txt [TEXT_UTILS] but never read by text_util.py: {orphans}\n"
        "Remove the key, or wire it up."
    )


def test_env_override_beats_the_config_file():
    """(12-factor III) ATRIUM_<SECTION>_<KEY> must reach the module constant.

    The file used to be the only way to change a value — LANGID_CONFIG names a
    path, not a value — so a deploy could not move one threshold without editing
    a file inside its image.
    """
    code = "import text_util as tu; assert tu.SHORT_PPL_CAP == 1234.5, tu.SHORT_PPL_CAP"
    env = dict(os.environ, ATRIUM_TEXT_UTILS_SHORT_PPL_CAP="1234.5")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, cwd=str(_ROOT))


def test_unparseable_env_override_fails_loudly():
    """A typo must not silently fall through to the default."""
    env = dict(os.environ, ATRIUM_TEXT_UTILS_SHORT_PPL_CAP="not-a-number")
    done = subprocess.run(
        [sys.executable, "-c", "import text_util"],
        env=env,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert "is not a float" in done.stderr


def test_missing_explicit_config_path_fails_loudly():
    """A typo'd LANGID_CONFIG used to run the whole collection on defaults."""
    env = dict(os.environ, LANGID_CONFIG="/nonexistent/config.txt")
    done = subprocess.run(
        [sys.executable, "-c", "import text_util"],
        env=env,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert "does not exist" in done.stderr
