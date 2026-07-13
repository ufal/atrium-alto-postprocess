"""
(#7 Tier 1) Regression locks for the config-backed language/collection
constants. Every default must be bit-identical to the previous in-code
literal so the migration is behaviour-neutral at the shipped config
(tests/test_recategorize_parity.py remains the end-to-end gate).
"""

from __future__ import annotations

import os
import subprocess
import sys

import text_util_langID as tu
from langID_classify import FASTTEXT_MODEL, TRUST_TIER_TRUSTED, TRUST_TIER_UNKNOWN


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
        "import text_util_langID as tu;"
        "import langID_classify as lc;"
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
