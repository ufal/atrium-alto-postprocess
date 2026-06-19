"""
calibration_fixtures.py
=======================
Real-text fixtures harvested from CTX192100040 (typed archaeological report)
and CTX192601143 (handwritten scribble page) for the #3 categorisation suite.

Each fixture is (text, frozen_ppl, frozen_orig_lang_score, expected_categ, note).
`ppl` and `lang_score` are the values already produced by the GPU/FastText
stage for that exact line, so a test can feed them straight into the
pre_filter -> compute_quality_score -> categorize_line path WITHOUT any model,
exactly like tests/test_smoke.py::_process_mocked_line.

Ground-truth labels are deliberately conservative: only lines whose correct
category is not in reasonable dispute are included. Borderline 0.88-0.90 lines
(genuine "could be Clear or Noisy") are intentionally excluded so the suite
never encodes a coin-flip as truth.
"""

# ── CLEAR: clean, confident Czech prose. Regression guard against demotion. ──
CLEAR = [
    ("v klášteře Strahovském.",                                    119.50, 1.0000, "Clear",
     "short clean Czech with a trailing period"),
    ("republiky československé",                                    15.88, 0.9136, "Clear",
     "two-word clean Czech, low ppl"),
    ("Laskavostí tohoto pána bylo mi dovoleno již roku 1919 na",   60.00, 1.0000, "Clear",
     "long clean prose with an embedded year (no LDL false-trigger)"),
    ("Opomenulé nebo opozděné ohlášení trestú se pokutou peněžitou",153.00, 1.0000, "Clear",
     "long clean prose, rot_ratio 0.64 — must NOT be rot-penalised"),
    ("svým jménem, nýbrž i lidovým podáním,které tvrdí,že v místech těchto stávala",
     26.62, 1.0000, "Clear", "clean prose dense in short function words (valid_ratio guard)"),
]

# ── NOISY: readable but degraded; must stay usable, never Clear, never Trash. ─
NOISY = [
    ("Maždý taxou vojenskou povinný jest povinen až do",          268.00, 0.6481, "Noisy",
     "leading typo Maždy<-Každy lowers valid_ratio just under Clear"),
    ("Pončeni o povinnosti ku taxo vojenské.",                    304.00, 0.8557, "Noisy",
     "two OCR substitutions, otherwise readable"),
    ("taxo vojenské.",                                            223.00, 0.9191, "Noisy",
     "short readable Czech fragment — regression guard vs short-garbage route"),
    ("statků v Praze.",                                           644.00, 1.0000, "Noisy",
     "3-word clean fragment, must not be Trashed by short/rot penalties"),
]

# ── TRASH (handwriting garbage): word-like random letters + high ppl. ────────
# These are the current false-Noisy survivors in CTX192601143. The combined
# "FastText-uncertain AND high perplexity" signal should route them to Trash.
TRASH_GARBAGE = [
    ("C LaN-n 0(/r\u201c (A 30 Gx A 25 so pgAuc4pi) dato md3\u00f3ny",
     1064.00, 0.3385, "Trash", "uncertain lang + high ppl, word-like blobs"),
    ("kioum Lly ad luo a/l6 707 (woln.",                          1000.00, 0.4715, "Trash",
     "structurally word-like garbage that currently scores Noisy 0.61"),
    ("go04 344* Au\u00fd- Nudky oi oiti 0dCla. AKog,ndg\u00e9 Pe* 63 /\u0161bo0d",
     1032.00, 0.2033, "Trash", "uncertain lang + high ppl"),
    ("' ' \" k4\u017ee /olonbka,\"3 Ege 94%",                      1648.00, 0.2013, "Trash",
     "uncertain lang + high ppl, currently Noisy 0.50"),
]

# ── TRASH (inverted / 180-rotated scan): upside-down Czech. ──────────────────
TRASH_INVERTED = [
    ("noywqued noqnsoa es yasoq yuasvyo quqpzodo oqou onuauodo",   856.00, 0.7500, "Trash",
     "inverted prose, remap-capped lang 0.75 — sweep must catch via orig/diacritics"),
    ("nupoy yoysqu A n7o. ouPpze\" yuAoxw gsouutod nxyaya",       1928.00, 0.2245, "Trash",
     "inverted prose, uncertain lang"),
    ("oueussd",                                                    850.00, 0.9163, "Trash",
     "single inverted token, rot_ratio 1.0"),
]

# ── NON-TEXT: numeric / code / stamp content. ───────────────────────────────
NON_TEXT = [
    ("\u010d: 6694 /1920.",  None, None, "Non-text", "file-number stamp"),
    ("434.",                 None, None, "Non-text", "bare numeral + period"),
    ("2742/2%",              None, None, "Non-text", "stamp-like ratio pattern"),
    ("P. T. Pan",            None, None, "Non-text", "spaced-initials salutation, no real prose"),
]

# ── REGRESSION GUARDS: clean high-rot Czech that the rot penalty must spare. ─
# Currently 10,68 ("eni - trestá so pokutou penéžitou") is wrongly Trashed.
# Whatever the rot fix is, these must remain >= Noisy.
ROT_FALSE_POSITIVE_GUARDS = [
    ("eni - trest\u00e1 so pokutou pen\u00e9\u017eitou",          624.00, 0.6824, "Noisy",
     "rot 0.59, weird 0 — currently Trash 0.45, MUST recover to >= Noisy"),
    ("Sm\u011brem sev.z\u00e1p.od m\u011bstyse Lod\u011bnice,1 hod.cesty - vzdu\u0161nou",
     644.00, 1.0000, "Noisy", "rot 0.71, fully readable — must not be depressed"),
    ("spoustu st\u0159epin z n\u00e1dob, popel d\u0159ev\u011bnn\u00fd a rozli\u010dn\u00e9 k\u016fstky.N\u00e1doby J soux",
     203.00, 1.0000, "Noisy", "rot 0.64, readable prose"),
]

ALL_FIXTURES = (
    CLEAR + NOISY + TRASH_GARBAGE + TRASH_INVERTED
    + NON_TEXT + ROT_FALSE_POSITIVE_GUARDS
)