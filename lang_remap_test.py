#!/usr/bin/env python3
"""Unit tests for the fixed language-remapping logic in langID_classify.

The fix lives inline in process_and_write_batch_cpu, so we replicate the exact
remap block here and assert behaviour against the README spec:
  rule 1: base code in EXPECTED or TRUSTED  -> keep code + score unchanged
  rule 2: base code NOT known               -> remap to EXPECTED[0], PRESERVE
          script suffix, score = max(orig, LANG_SCORE_CLEAR=0.75)
"""
from text_util_langID import _lang_base, LANG_SCORE_CLEAR

EXPECTED = ["ces", "deu", "eng"]
TRUSTED = ["deu", "eng", "fra", "pol", "ita"]
_known_bases = frozenset(_lang_base(l) for l in (TRUSTED + EXPECTED))


def remap(lang, score):
    """Exact copy of the fixed inline logic."""
    if _lang_base(lang) not in _known_bases:
        suffix = lang[len(_lang_base(lang)):]
        lang = EXPECTED[0] + suffix
        score = max(score, LANG_SCORE_CLEAR)
    return lang, score


def check(desc, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: got {got}  want {want}")
    return ok


def main():
    allok = True
    # rule 1: German kept (this is the exact bug the user reported)
    allok &= check("deu_Latn kept as-is", remap("deu_Latn", 0.93), ("deu_Latn", 0.93))
    # rule 1: Czech kept
    allok &= check("ces_Latn kept as-is", remap("ces_Latn", 0.88), ("ces_Latn", 0.88))
    # rule 1: trusted English kept with its real (low) score, NOT floored
    allok &= check("eng_Latn low score kept", remap("eng_Latn", 0.40), ("eng_Latn", 0.40))
    # rule 1: trusted French kept
    allok &= check("fra_Latn kept", remap("fra_Latn", 0.6), ("fra_Latn", 0.6))
    # rule 2: Slovak -> ces, suffix preserved, score floored to 0.75
    allok &= check("slk_Latn -> ces_Latn floored", remap("slk_Latn", 0.50),
                   ("ces_Latn", 0.75))
    # rule 2: Slovenian high score -> ces, suffix preserved, score kept (>0.75)
    allok &= check("slv_Latn high score -> ces_Latn", remap("slv_Latn", 0.97),
                   ("ces_Latn", 0.97))
    # rule 2: a Cyrillic prediction keeps its _Cyrl suffix on remap
    allok &= check("rus_Cyrl -> ces_Cyrl floored", remap("rus_Cyrl", 0.3),
                   ("ces_Cyrl", 0.75))
    # bare codes (no suffix) still work both ways
    allok &= check("bare deu kept", remap("deu", 0.9), ("deu", 0.9))
    allok &= check("bare slk -> ces floored", remap("slk", 0.2), ("ces", 0.75))

    print("\nALL PASS" if allok else "\nSOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())