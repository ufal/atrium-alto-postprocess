"""
tests/test_short_garbage_witness_wiring.py
==========================================
(#30 D15) Covers `_has_shape_garbage_evidence()` at its call site in gate 6.

The predicate itself is unit-tested in tests/test_text_utils.py
(`TestShapeGarbageWitness`) and the composed condition is pinned on production
signal vectors in tests/test_calibration.py. What was missing, and what this
module adds, is the WIRING: that the disjunct is inert while the flag is off,
that it reaches a site which actually returns `Trash`, and that turning it on
convicts the garbage without taking the vocabulary and notation down with it.

That last point is the whole trade. The plan records two mistakes that "passed
code review and failed on execution", both about placement rather than about the
predicate, so the placement is what is asserted here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENV_FLAG = "ATRIUM_TEXT_UTILS_SHORT_GARBAGE_WITNESS_ENABLE"

# The flag is read at IMPORT time, so exercising both states means importing
# text_util twice. Doing that in-process (sys.modules.pop + re-import) leaves
# other already-imported modules holding references to the discarded module
# object, and the damage lands on whatever test runs next -- it silently broke
# four TestLangRemap cases when this module first went in. A subprocess is the
# only way to get a genuinely clean import without making the rest of the suite
# order-dependent.
_CHILD = r"""
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
import recategorize_from_csv as rc

expected, known = rc._load_lang_config({config!r})
out = {{}}
for text, ppl, lang_score, original_lang in json.loads(sys.argv[1]):
    row = {{
        "text": text,
        "original_text": text,
        "original_lang": original_lang,
        "orig_lang_score": str(lang_score),
        "perplex": str(ppl),
        "categ": "Noisy",
        "word_count": str(len(text.split())),
    }}
    out[text] = rc._rescore_row(row, expected, known)["categ"]
print(json.dumps(out))
"""


def _categ_under_flag(rows, *, enabled: bool):
    """Re-score `rows` through the production path with the witness on or off."""
    env = dict(os.environ)
    env[_ENV_FLAG] = "true" if enabled else "false"
    script = _CHILD.format(
        root=str(_ROOT),
        tools=str(_ROOT / "tools"),
        config=str(_ROOT / "setup" / "config.txt"),
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, json.dumps(rows)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_ROOT),
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(f"re-score subprocess failed (flag={enabled}):\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# The disputed population, with the production signal vectors recorded in
# tests/calibration_fixtures.py. Each row carries its OWN language: hardcoding
# ces_Latn grants trust tier 1.0 where production applies TRUST_TIER_UNKNOWN
# (0.50), which is the harness bug this repository has fixed three times.
_GARBAGE = [
    ("oueussd", 850.00, 0.9163, "ces_Latn"),
    ("sektlll", 850.00, 0.60, "ces_Latn"),
]
_KEEP = [
    ("malakofauna", 1210.00, 0.5600, "isl_Latn"),
    ("Equus caballus", 640.00, 0.7700, "ast_Latn"),
    ("diapozitiv", 900.00, 0.9700, "ron_Latn"),
    ("II/C", 6.0e7, 0.4000, "ces_Latn"),
    ("1 ks", 5.0e6, 0.5000, "ces_Latn"),
]
_RESIDUE = [("edelite", 850.00, 0.60, "ces_Latn")]


@pytest.fixture(scope="module")
def flag_off():
    return _categ_under_flag(_GARBAGE + _KEEP + _RESIDUE, enabled=False)


@pytest.fixture(scope="module")
def flag_on():
    return _categ_under_flag(_GARBAGE + _KEEP + _RESIDUE, enabled=True)


def test_wiring_is_inert_while_the_flag_is_off(flag_off):
    """The default build must be byte-identical to the pre-D15 behaviour.

    Wiring and enabling are separate commits on purpose: the flag must not be
    flipped until the witness is measured against a gold set, and until then a
    contributor upgrading must see no category move at all.
    """
    for text, _ppl, _ls, _lang in _GARBAGE + _KEEP + _RESIDUE:
        assert flag_off[text] == "Clear", f"{text!r} moved with the flag off: {flag_off[text]}"


def test_the_witness_reaches_a_site_that_returns_trash(flag_on):
    """Gate 6 is the only site in this path that can return `Trash`.

    An earlier revision of the plan proposed section 7 instead. Section 7's
    `damage` branch returns `Noisy`, so the debt's strict xfail would never have
    flipped while the visible improvement (`Clear` -> `Noisy`) was mistaken for
    the debt being paid. Asserting `Trash` here is what makes that mistake
    impossible to repeat silently.
    """
    for text, _ppl, _ls, _lang in _GARBAGE:
        assert flag_on[text] == "Trash", f"{text!r} did not reach Trash with the witness armed"


def test_vocabulary_and_notation_survive_the_witness(flag_on):
    """The trade only pays if the correct lifts are kept.

    `malakofauna` and `Equus caballus` are the lines issue #30 exists to protect;
    `II/C` and `1 ks` are notation recovered by `is_domain_notation()`. If any of
    them regress, the narrowing has become the very demotion the issue was opened
    about.
    """
    for text, _ppl, _ls, _lang in _KEEP:
        assert flag_on[text] == "Clear", f"{text!r} regressed to {flag_on[text]} under the witness"


def test_the_lexicon_residue_is_still_out_of_reach(flag_on):
    """`edelite` is phonotactically legal, so shape cannot separate it.

    Asserted as part of the contract rather than quietly omitted: a lexicon-free
    approach has to own this residue instead of pretending it away. If this ever
    starts passing, a lexical signal landed and D14's scope changed.
    """
    for text, _ppl, _ls, _lang in _RESIDUE:
        assert flag_on[text] == "Clear", (
            f"{text!r} is now {flag_on[text]}; shape alone should not separate the "
            "phonotactically legal residue -- check what else changed"
        )


def test_the_flag_actually_changes_something(flag_off, flag_on):
    """Guards against the predicate silently becoming decorative again.

    `_has_shape_garbage_evidence()` shipped with NO call site at all, and two
    other features in this thread shipped unable to change any outcome. A wiring
    test that cannot tell the two flag states apart would be the third.
    """
    assert flag_off != flag_on, "flipping SHORT_GARBAGE_WITNESS_ENABLE changed no category"
