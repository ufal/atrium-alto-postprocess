"""
tests/test_pipeline_parity.py

Characterization ("golden file") tests that pin the *current* behaviour of the
line-categorisation decision path.

Why this exists
---------------
Three call sites need per-line signals assembled before handing them to
``compute_quality_score()`` / ``categorize_line()``:

* ``classify_TEXT.py``                  — the batch pipeline (reference behaviour)
* ``tools/recategorize_from_csv.py``    — the offline re-scorer
* ``service/text_inference.py``         — the FastAPI ``/process`` endpoint

Each used to hand-maintain its own copy, and they drifted: the re-scorer agreed
only because it kept being patched to match, and the service had silently lost
the language remap, the trust tiers, ``SHORT_PPL_CAP`` and ``orig_lang_score``
altogether. All three now call ``classify_TEXT.score_line()``, so parity is
maintained by construction; ``tests/test_scoring_single_source.py`` holds that
property directly.

This module remains the behavioural pin underneath it. A shared helper only
guarantees the three callers agree with *each other* — it cannot tell you they
still agree with what the pipeline emitted before the refactor. That is what
the golden file below is for.

``tests/test_recategorize_parity.py`` checks that the re-scorer does not *flip
categories* relative to the stored CSVs, but with a tolerance (5% flip rate
allowed, 30% per document). That is a drift alarm, not a specification.

This module is stricter and narrower. It pins:

1. the exact ``(categ, quality_score, reason)`` triple **and the set of rules
   that fired** for a curated case per route through ``determine_category()``;
2. the exact triple for every line of the committed sample documents.

Pinning the *fired rules*, not just the outcome, is the point. Several gates
produce the same label by different routes — ``Trash``/``trash_threshold`` is
reachable from at least six of them — so the label alone cannot tell you the
decision path was preserved. This is what makes the test able to prove that a
later refactor extracting a shared ``score_line()`` helper is behaviour-
preserving rather than merely plausible.

Hermeticity
-----------
The sample pin reads an explicit allowlist of committed documents
(``PINNED_DOCS``), NOT a directory glob. A working copy normally also contains
untracked CTX documents from local runs; globbing made the golden file record
whatever happened to be on the generating machine, so it passed on a fresh CI
clone and failed everywhere else. Worse, the obvious way to "fix" that red test
is to regenerate the golden file — which silently re-baselines the categoriser.
The allowlist removes that whole failure mode.

Regenerating the golden file
----------------------------
Only when a categorisation change is intentional, and review the diff line by
line::

    python -m tests.test_pipeline_parity --regenerate

A change here is a change to what the pipeline emits for the ARUB/ARUP
collections. It must never be regenerated to make a red test pass.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

# Stub the GPU/ML stack before importing the tool (it imports classify_TEXT).
# Same convention as tests/test_recategorize_parity.py, so this module stays on
# the hermetic "not slow" lane that CI runs.
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN_PATH = pathlib.Path(__file__).parent / "fixtures" / "pipeline_characterization.json"
SAMPLES_DIR = REPO_ROOT / "data_samples" / "DOC_LINE_CATEG"
CONFIG_PATH = REPO_ROOT / "setup" / "config.txt"

# Committed sample documents, per `git ls-files data_samples/DOC_LINE_CATEG/`.
# Untracked documents in the same directory are deliberately NOT pinned.
PINNED_DOCS = ("CTX000000001", "CTX000000002", "CTX000000003")

_TOOLS = REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

pd = pytest.importorskip("pandas", reason="pandas is required to replay the sample CSVs")

import recategorize_from_csv as R  # noqa: E402
import rule_coverage_report as RC  # noqa: E402

import text_util as tu  # noqa: E402

# Rules that cannot fire through categorize_line() under the default config.
# Kept explicit so an unreachable rule is a recorded fact with a reason, not a
# silent hole in the coverage assertion below.
UNREACHABLE_RULES: dict[str, str] = {
    "rule_mid_uppercase": (
        "Gate 9d requires word_count <= 2, but gate 7 (rule_short_line) has the same "
        "condition and always returns first, so 9d is shadowed. See "
        "test_mid_uppercase_is_shadowed_by_short_line, which pins that shadowing."
    ),
}

# ── Curated edge cases ──────────────────────────────────────────────────────
# One case per route through determine_category(). The inputs were derived from
# the gate conditions and verified to fire the intended rule, not guessed.
EDGE_CASES: list[tuple[str, dict]] = [
    # ── Pre-filter / empty ──
    ("empty", dict(qs=0.0, txt="", wc=0, vowel_ratio=0.0, perplexity=0.0)),
    ("whitespace_only", dict(qs=0.0, txt="   ", wc=0, vowel_ratio=0.0, perplexity=0.0)),
    # ── Gate 1: hard-Trash perplexity routes ──
    (
        "hard_sweep_low_lang_high_ppl",
        dict(
            qs=0.20,
            txt="qw xz vbn mkl",
            wc=4,
            vowel_ratio=0.05,
            perplexity=5000.0,
            lang_score=0.20,
            orig_lang_score=0.20,
        ),
    ),
    (
        "extreme_ppl_mid_lang_conf",
        dict(
            qs=0.30,
            txt="mesto ulice dum zahrada",
            wc=4,
            vowel_ratio=0.40,
            perplexity=5000.0,
            lang_score=0.70,
            orig_lang_score=0.70,
        ),
    ),
    (
        "absolute_ppl",
        dict(
            qs=0.30,
            txt="zzz qqq www vvv",
            wc=4,
            vowel_ratio=0.0,
            perplexity=99000.0,
            lang_score=0.95,
            orig_lang_score=0.95,
        ),
    ),
    # ── Gate 2: inverted / mirrored scan ──
    (
        "inverted_ghost_dominated",
        dict(
            qs=0.40,
            txt="ou pue dn sdn qwou",
            wc=5,
            vowel_ratio=0.35,
            perplexity=900.0,
            lang_score=0.40,
            orig_lang_score=0.40,
            ghost_dominated=True,
            weird_ratio=0.50,
        ),
    ),
    # ── Gate 3: all-caps vowel-less ──
    ("allcaps_novowel", dict(qs=0.50, txt="BCDFG HJKLM", wc=2, vowel_ratio=0.0, perplexity=300.0)),
    # ── Gate 4: garbage-density hard override ──
    (
        "garbage_density_high",
        dict(
            qs=0.25,
            txt="*#@ /// %%% ~~~",
            wc=4,
            vowel_ratio=0.0,
            perplexity=900.0,
            garbage_density=0.90,
            lang_score=0.50,
            orig_lang_score=0.50,
        ),
    ),
    # ── Gate 5b: zero alphabetic content ──
    (
        "zero_alpha_digits_only",
        dict(qs=0.20, txt="123 456 789", wc=3, vowel_ratio=0.0, perplexity=900.0, lang_score=0.5, orig_lang_score=0.5),
    ),
    # ── Gate 6: short-line garbage ──
    (
        "short_garbage",
        dict(
            qs=0.25,
            txt="ab c9 xz",
            wc=3,
            vowel_ratio=0.20,
            perplexity=800.0,
            valid_word_ratio=0.0,
            lang_score=0.50,
            orig_lang_score=0.50,
            weird_ratio=0.6,
            gibberish_present=True,
        ),
    ),
    # ── Gate 7: short lines (1-2 words) ──
    (
        "short_line_clear",
        dict(
            qs=0.95,
            txt="Praha 1",
            wc=2,
            vowel_ratio=0.40,
            perplexity=200.0,
            valid_word_ratio=1.0,
            lang_score=1.0,
            orig_lang_score=1.0,
        ),
    ),
    (
        "short_line_catalogue",
        dict(
            qs=0.70,
            txt="Tab. 3",
            wc=2,
            vowel_ratio=0.33,
            perplexity=400.0,
            valid_word_ratio=0.5,
            lang_score=0.75,
            orig_lang_score=0.75,
        ),
    ),
    (
        "short_line_measurement_glued_metre",
        dict(
            qs=0.60,
            txt="hl. 0,46m",
            wc=2,
            vowel_ratio=0.20,
            perplexity=900.0,
            valid_word_ratio=0.5,
            lang_score=0.75,
            orig_lang_score=0.75,
        ),
    ),
    (
        "short_line_single_solitary_letter",
        dict(
            qs=0.40,
            txt="k",
            wc=1,
            vowel_ratio=0.0,
            perplexity=900.0,
            valid_word_ratio=0.0,
            lang_score=0.5,
            orig_lang_score=0.5,
        ),
    ),
    # ── Gate 8: high LM confidence ──
    (
        "lowppl_clear",
        dict(
            qs=0.85,
            txt="Lokalita: okr. Horní Mezi",
            wc=4,
            vowel_ratio=0.41,
            perplexity=20.0,
            valid_word_ratio=1.0,
            lang_score=1.0,
            orig_lang_score=1.0,
        ),
    ),
    (
        "lowppl_capped_to_noisy",
        dict(
            qs=0.85,
            txt="nekterá slova zde ctitelná",
            wc=4,
            vowel_ratio=0.40,
            perplexity=20.0,
            valid_word_ratio=0.50,
            lang_score=0.9,
            orig_lang_score=0.9,
        ),
    ),
    # ── Late structural gates (9a-9f) ──
    (
        "wqx_rotation",
        dict(
            qs=0.60,
            txt="wxq wqx xwq qxw",
            wc=4,
            vowel_ratio=0.0,
            perplexity=900.0,
            valid_word_ratio=0.2,
            lang_score=0.5,
            orig_lang_score=0.5,
            weird_ratio=0.6,
        ),
    ),
    (
        "vowelless_allcaps_short",
        dict(
            qs=0.60,
            txt="KRK STRP BRN",
            wc=3,
            vowel_ratio=0.20,
            perplexity=800.0,
            valid_word_ratio=0.2,
            lang_score=0.80,
            orig_lang_score=0.80,
            weird_ratio=0.0,
        ),
    ),
    (
        "ledger_fragmentation",
        dict(
            qs=0.60,
            txt="12 3 45 6 78 9 ab",
            wc=7,
            vowel_ratio=0.10,
            perplexity=900.0,
            valid_word_ratio=0.1,
            lang_score=0.5,
            orig_lang_score=0.5,
            weird_ratio=0.5,
        ),
    ),
    (
        "bigram_run",
        dict(
            qs=0.60,
            txt="ababab mesto ulice zahrada",
            wc=4,
            vowel_ratio=0.35,
            perplexity=800.0,
            valid_word_ratio=0.2,
            lang_score=0.60,
            orig_lang_score=0.60,
            weird_ratio=0.5,
        ),
    ),
    (
        "fragment_tokens",
        dict(
            qs=0.60,
            txt="a b c",
            wc=3,
            vowel_ratio=0.50,
            perplexity=800.0,
            valid_word_ratio=0.1,
            lang_score=0.80,
            orig_lang_score=0.80,
            weird_ratio=0.0,
        ),
    ),
    # ── check_rescues(): the three rescue routes, in firing order ──
    (
        "rescue_trailing_fill",
        dict(
            qs=0.30,
            txt="Přílohy a mapy .....",
            wc=4,
            vowel_ratio=0.40,
            perplexity=800.0,
            garbage_density=0.10,
            valid_word_ratio=0.90,
            lang_score=0.90,
            orig_lang_score=0.90,
        ),
    ),
    (
        "rescue_forgiven_headline",
        dict(
            qs=0.30,
            txt="4. Literatura 5",
            wc=3,
            vowel_ratio=0.40,
            perplexity=900.0,
            valid_word_ratio=0.4,
            lang_score=0.75,
            orig_lang_score=0.75,
        ),
    ),
    (
        "rescue_reference_floor",
        dict(
            qs=0.30,
            txt="inv. 145 kont. 12 nál. 9",
            wc=6,
            vowel_ratio=0.30,
            perplexity=800.0,
            garbage_density=0.10,
            valid_word_ratio=0.40,
            lang_score=0.90,
            orig_lang_score=0.90,
        ),
    ),
    (
        "rescue_none_falls_to_trash",
        dict(
            qs=0.30,
            txt="xkq zvm bpr wtn",
            wc=4,
            vowel_ratio=0.05,
            perplexity=900.0,
            garbage_density=0.20,
            valid_word_ratio=0.0,
            lang_score=0.80,
            orig_lang_score=0.80,
        ),
    ),
    # ── Gate 10: band routing and the damage / readability caps ──
    (
        "damaged_token_caps_to_noisy",
        dict(
            qs=0.85,
            txt="Praha ne3jvice ulice",
            wc=3,
            vowel_ratio=0.40,
            perplexity=300.0,
            valid_word_ratio=0.9,
            lang_score=0.9,
            orig_lang_score=0.9,
        ),
    ),
    (
        "mostly_readable_noisy",
        dict(
            qs=0.88,
            txt="nekterá slova jsou zde ctitelná ale ne vsechna",
            wc=8,
            vowel_ratio=0.40,
            perplexity=600.0,
            valid_word_ratio=0.60,
            lang_score=0.75,
            orig_lang_score=0.75,
        ),
    ),
    (
        "clean_czech_headline_clear",
        dict(
            qs=0.96,
            txt="HRADIŠTĚ U HORNÍ MEZÍ",
            wc=4,
            vowel_ratio=0.44,
            perplexity=237.0,
            valid_word_ratio=1.0,
            lang_score=1.0,
            orig_lang_score=1.0,
        ),
    ),
    (
        "lm_confident_czech_relaxation",
        dict(
            qs=0.82,
            txt="Na lokalitě byla objevena keramika",
            wc=5,
            vowel_ratio=0.42,
            perplexity=100.0,
            valid_word_ratio=0.80,
            lang_score=1.0,
            orig_lang_score=1.0,
            is_upright_czech=True,
        ),
    ),
    (
        "clean_prose_clear",
        dict(
            qs=0.93,
            txt="Na severním okraji zkoumané plochy byla odkryta část příkopu",
            wc=9,
            vowel_ratio=0.43,
            perplexity=180.0,
            valid_word_ratio=0.95,
            lang_score=1.0,
            orig_lang_score=1.0,
        ),
    ),
    # ── Boundary cases straddling the two score thresholds ──
    (
        "boundary_just_below_trash_max",
        dict(
            qs=0.5499,
            txt="mesto ulice dum zahrada park",
            wc=5,
            vowel_ratio=0.40,
            perplexity=500.0,
            valid_word_ratio=0.90,
            lang_score=0.90,
            orig_lang_score=0.90,
        ),
    ),
    (
        "boundary_at_trash_max",
        dict(
            qs=0.5500,
            txt="mesto ulice dum zahrada park",
            wc=5,
            vowel_ratio=0.40,
            perplexity=500.0,
            valid_word_ratio=0.90,
            lang_score=0.90,
            orig_lang_score=0.90,
        ),
    ),
    (
        "boundary_just_below_noisy_max",
        dict(
            qs=0.7999,
            txt="mesto ulice dum zahrada park",
            wc=5,
            vowel_ratio=0.40,
            perplexity=500.0,
            valid_word_ratio=0.90,
            lang_score=0.90,
            orig_lang_score=0.90,
        ),
    ),
    (
        "boundary_at_noisy_max",
        dict(
            qs=0.8000,
            txt="mesto ulice dum zahrada park",
            wc=5,
            vowel_ratio=0.40,
            perplexity=500.0,
            valid_word_ratio=0.90,
            lang_score=0.90,
            orig_lang_score=0.90,
        ),
    ),
]


def _edge_case_rows() -> dict[str, list]:
    """Run each curated case through categorize_line(), capturing fired rules."""
    out: dict[str, list] = {}
    for label, kwargs in EDGE_CASES:
        with tu.rule_fire_capture() as counts:
            categ, qs, reason = tu.categorize_line(**kwargs, return_reason=True)
        out[label] = [categ, round(float(qs), 4), reason, sorted(counts)]
    return out


def _sample_rows() -> dict[str, list]:
    """Replay the committed sample documents through the re-scorer path.

    Keyed by ``<file>:<page>:<line>`` so a diff points at an identifiable line
    rather than a list index.
    """
    expected_langs, known_bases = R._load_lang_config(str(CONFIG_PATH))
    out: dict[str, list] = {}
    for doc in PINNED_DOCS:
        csv_path = SAMPLES_DIR / f"{doc}.csv"
        frame = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            key = f"{row_dict.get('file', doc)}:{row_dict.get('page_num', '')}:{row_dict.get('line_num', '')}"
            if R._is_fast_track(row_dict):
                # Empty / Non-text lines never reach the scorer; pin them as-is
                # so a regression that starts scoring them is still caught.
                out[key] = [row_dict.get("categ", ""), None, "fast_track"]
                continue
            scored = R._rescore_row(row_dict, expected_langs, known_bases)
            reason = next(
                (
                    name
                    for name in (
                        "allcaps_novowel",
                        "lowppl_clear",
                        "trash_threshold",
                        "noisy_threshold",
                        "clear_threshold",
                    )
                    if scored.get(name)
                ),
                "",
            )
            out[key] = [scored["categ"], round(float(scored["quality_score"]), 4), reason]
    return out


def build_snapshot() -> dict:
    return {
        "_comment": (
            "Golden file for tests/test_pipeline_parity.py. Regenerate ONLY for an "
            "intentional categorisation change: python -m tests.test_pipeline_parity --regenerate"
        ),
        "edge_cases": _edge_case_rows(),
        "samples": _sample_rows(),
    }


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"golden file missing: {GOLDEN_PATH}\nGenerate it with: python -m tests.test_pipeline_parity --regenerate"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_pinned_sample_documents_exist():
    """The allowlisted CSVs must be present.

    Deliberately a failure, not a skip: a pinning test that silently stops
    pinning when its inputs disappear is indistinguishable from a passing one.
    """
    missing = [doc for doc in PINNED_DOCS if not (SAMPLES_DIR / f"{doc}.csv").is_file()]
    assert not missing, f"committed sample documents missing from {SAMPLES_DIR}: {missing}"


def test_edge_case_routes_are_pinned(golden):
    """Each curated case keeps its label, score, reason AND decision path."""
    actual = _edge_case_rows()
    expected = golden["edge_cases"]
    assert set(actual) == set(expected), "edge-case set changed; update EDGE_CASES and regenerate"
    drifted = {k: {"expected": expected[k], "actual": actual[k]} for k in expected if expected[k] != actual[k]}
    assert not drifted, "categorisation drift on curated edge cases:\n" + json.dumps(
        drifted, indent=2, ensure_ascii=False
    )


def test_every_rule_is_exercised_or_documented_unreachable(golden):
    """Every rule in the canonical list is either covered or explicitly unreachable.

    Stops the edge-case set from silently rotting as rules are added: a new
    rule with no case here fails until someone either covers it or records why
    it cannot be reached.
    """
    fired = {rule for row in golden["edge_cases"].values() for rule in row[3]}
    uncovered = set(RC.RULES) - fired - set(UNREACHABLE_RULES)
    assert not uncovered, (
        f"rules never exercised by any edge case: {sorted(uncovered)}\n"
        "Add a case to EDGE_CASES, or record it in UNREACHABLE_RULES with the reason."
    )
    # An "unreachable" rule that starts firing means the gate order changed.
    wrongly_unreachable = fired & set(UNREACHABLE_RULES)
    assert not wrongly_unreachable, (
        f"rules marked unreachable but now firing: {sorted(wrongly_unreachable)} — "
        "gate ordering changed; remove them from UNREACHABLE_RULES."
    )


def test_mid_uppercase_is_shadowed_by_short_line():
    """Pin the gate-ordering fact that makes rule_mid_uppercase unreachable.

    Gate 9d fires only for ``word_count <= 2``, but gate 7 (``rule_short_line``)
    tests the same condition and always returns, so 9d can never be reached
    under the default config. That is a real shadowing introduced when the
    short-line gate landed, not an intentional deactivation — this test exists
    so that reordering the gates surfaces it as a decision instead of silently
    switching a dormant rule back on.
    """
    fired_any = False
    for txt in ("pRaha", "pRaha nEjvice", "aBc dEf", "xY zW"):
        for wc in (1, 2):
            with tu.rule_fire_capture() as counts:
                tu.categorize_line(
                    qs=0.50,
                    txt=txt,
                    wc=wc,
                    vowel_ratio=0.30,
                    perplexity=500.0,
                    valid_word_ratio=0.20,
                    lang_score=0.50,
                    orig_lang_score=0.50,
                    weird_ratio=0.50,
                )
            fired_any |= "rule_mid_uppercase" in counts
    assert not fired_any, (
        "rule_mid_uppercase fired — gate 9d is no longer shadowed by gate 7. "
        "Update UNREACHABLE_RULES and add a real edge case for it."
    )


def test_sample_corpus_output_is_pinned(golden):
    """Every line of the committed sample docs keeps its exact (categ, qs, reason)."""
    actual = _sample_rows()
    expected = golden["samples"]
    assert set(actual) == set(expected), (
        f"sample line set changed: only-in-golden={sorted(set(expected) - set(actual))[:5]} "
        f"only-in-actual={sorted(set(actual) - set(expected))[:5]}"
    )
    drifted = {k: {"expected": expected[k], "actual": actual[k]} for k in expected if expected[k] != actual[k]}
    assert not drifted, f"{len(drifted)}/{len(expected)} sample lines changed category or score:\n" + json.dumps(
        dict(list(drifted.items())[:20]), indent=2, ensure_ascii=False
    )


def test_quality_score_is_consistent_with_its_label(golden):
    """A pinned score must sit inside the band its pinned label implies.

    Guards the clamp in ``categorize_line()``: a label/score pair that drifts
    apart is a bug even when both values individually look plausible.
    """
    bad = []
    rows = [(k, v[0], v[1]) for k, v in golden["samples"].items()]
    rows += [(k, v[0], v[1]) for k, v in golden["edge_cases"].items()]
    for key, categ, score in rows:
        if score is None:
            continue
        if categ == "Trash" and not score < tu.CATEG_TRASH_SCORE_MAX:
            bad.append((key, categ, score))
        elif categ == "Noisy" and not (tu.CATEG_TRASH_SCORE_MAX <= score < tu.CATEG_NOISY_SCORE_MAX):
            bad.append((key, categ, score))
        elif categ == "Clear" and not score >= tu.CATEG_NOISY_SCORE_MAX:
            bad.append((key, categ, score))
    assert not bad, "label/score band mismatch:\n" + "\n".join(f"  {k}: {c} @ {s}" for k, c, s in bad)


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(build_snapshot(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {GOLDEN_PATH}")
    else:
        print(__doc__)
