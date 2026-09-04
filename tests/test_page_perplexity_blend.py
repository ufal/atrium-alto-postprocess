"""
tests/test_page_perplexity_blend.py

Covers `classify_TEXT.apply_page_perplexity_blend()` — the page-relative
perplexity blend added for issue #30, **disabled by default**.

Why it exists
-------------
`SHORT_PPL_CAP` (850) flattens perplexity for `wc <= 2`, and the capped value is
what is both scored *and stored*, so the raw LM number never reaches the CSV.
850 sits below `HARD_SWEEP_PPL_MIN` (1000), `PPL_EXTREME_MIN` (3000) and
`PPL_GARBAGE_ABSOLUTE` (30000), so no perplexity rule can fire on a short line
at all. The blend restores a usable number by reading a short line against the
long lines on its own page.

What these tests pin
--------------------
1. With the flag off, nothing moves and `perplex_blend` stays blank — the
   default path must be byte-identical to the pre-change tree.
2. The reference is a *median*, so one enormous outlier cannot poison a page.
3. The blend is geometric; an arithmetic mean would be swallowed by the tail.
4. Blending reads the immutable `perplex_raw`, so repeated passes are a fixed
   point rather than a slow drift.
5. **What it does not do**: it is a page-consistency prior, not a garbage
   detector, and on its own it does not close the gap that gating
   `rule_short_garbage` would open. That limit is asserted, not left implicit.
"""

import sys
import types
from pathlib import Path

import pytest

for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from recategorize_from_csv import _load_lang_config  # noqa: E402

import classify_TEXT as LC  # noqa: E402
import text_util as tu  # noqa: E402

EXPECTED_LANGS, KNOWN_BASES = _load_lang_config(str(_ROOT / "setup" / "config.txt"))

LONG_CLEAN = [
    ("Laskavostí tohoto pána bylo mi dovoleno již roku 1919 na", "ces_Latn", 1.0, 36.0),
    ("Opomenulé nebo opozděné ohlášení trestú se pokutou peněžitou", "ces_Latn", 1.0, 34.0),
    ("svým jménem, nýbrž i lidovým podáním které tvrdí že v místech", "ces_Latn", 1.0, 38.0),
]
LONG_GARBAGE = [
    ("qw xz vbn mkl asd fgh jkl", "vie_Latn", 0.20, 7000.0),
    ("zxcv bnm asdf qwer tyui ghjk", "vie_Latn", 0.20, 6800.0),
    ("mnbv cxz lkjh gfds poiu ytre", "vie_Latn", 0.20, 7200.0),
]


def _page(rows):
    """Build a page exactly as the live pipeline writes it."""
    built = []
    for i, (text, lang, lang_score, ppl) in enumerate(rows, start=1):
        sig = LC.score_line(
            text_content=text,
            original_text=text,
            original_lang=lang,
            original_lang_score=lang_score,
            perplexity=ppl,
            known_lang_bases=KNOWN_BASES,
            expected_langs=EXPECTED_LANGS,
        )
        built.append(
            LC.row_from_signals(
                sig,
                file_id="CTX1",
                page_id=1,
                line_num=i,
                text_content=text,
                original_text=text,
                split_ws="",
                split_we="",
                original_lang=lang,
                original_lang_score=lang_score,
            )
        )
    return pd.DataFrame(built)


def _blend(df):
    with tu.override_constants({"PAGE_PPL_BLEND_ENABLE": True}, modules=(tu, LC)):
        return LC.apply_page_perplexity_blend(df.copy(), known_lang_bases=KNOWN_BASES, expected_langs=EXPECTED_LANGS)


class TestDisabledByDefault:
    def test_flag_off_is_a_no_op(self):
        """The default path must be indistinguishable from the pre-change tree."""
        df = _page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)])
        out = LC.apply_page_perplexity_blend(df.copy(), known_lang_bases=KNOWN_BASES, expected_langs=EXPECTED_LANGS)
        pd.testing.assert_frame_equal(df, out)

    def test_flag_off_leaves_perplex_blend_blank(self):
        df = _page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)])
        assert (df["perplex_blend"] == "").all()

    def test_perplex_raw_records_the_uncapped_value(self):
        """`perplex` is capped at wc <= 2; `perplex_raw` must not be.

        Without this column the blend would have nothing to work from — the
        stored `perplex` is already the constant 850, and blending from a
        constant blends a constant.
        """
        df = _page([("oueussd", "isl_Latn", 0.9163, 4600.0)])
        assert float(df.iloc[0]["perplex"]) == LC.SHORT_PPL_CAP
        assert float(df.iloc[0]["perplex_raw"]) == 4600.0

    def test_long_lines_are_uncapped_in_both_columns(self):
        df = _page([LONG_CLEAN[0]])
        assert float(df.iloc[0]["perplex"]) == float(df.iloc[0]["perplex_raw"]) == 36.0


class TestBlendBehaviour:
    def test_clean_page_pulls_a_short_line_down(self):
        out = _blend(_page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        blended = float(out.iloc[-1]["perplex_blend"])
        assert 300.0 < blended < 500.0, blended

    def test_garbage_page_pushes_the_same_line_up(self):
        out = _blend(_page(LONG_GARBAGE + [("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        blended = float(out.iloc[-1]["perplex_blend"])
        assert 5000.0 < blended < 6500.0, blended

    def test_identical_line_lands_differently_per_page(self):
        """The whole point: the same token is read against its own page."""
        clean = float(_blend(_page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)])).iloc[-1]["perplex_blend"])
        dirty = float(_blend(_page(LONG_GARBAGE + [("oueussd", "isl_Latn", 0.9163, 4600.0)])).iloc[-1]["perplex_blend"])
        assert dirty > clean * 10

    def test_median_reference_survives_one_huge_outlier(self):
        """`II/C` measures ~6e7. Under a mean the page reference would be ~1.5e7
        and every short line on the page would be dragged with it."""
        rows = LONG_CLEAN + [("II/C", "ces_Latn", 0.40, 6.0e7), ("oueussd", "isl_Latn", 0.9163, 4600.0)]
        out = _blend(_page(rows))
        blended = float(out.iloc[-1]["perplex_blend"])
        assert blended < 1000.0, f"outlier poisoned the page reference: {blended}"

    def test_blend_is_geometric_not_arithmetic(self):
        out = _blend(_page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        blended = float(out.iloc[-1]["perplex_blend"])
        reference = float(np.median([36.0, 34.0, 38.0]))
        assert blended == pytest.approx(np.exp(0.5 * np.log(4600.0) + 0.5 * np.log(reference)), rel=1e-3)
        assert blended < (4600.0 + reference) / 2.0

    def test_long_lines_are_never_blended(self):
        out = _blend(_page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        assert (out[pd.to_numeric(out["word_count"]) >= 3]["perplex_blend"] == "").all()

    def test_page_with_too_few_long_lines_is_left_alone(self):
        """Below `PAGE_PPL_MIN_LONG_LINES` there is no trustworthy reference, so
        the page keeps today's SHORT_PPL_CAP behaviour."""
        out = _blend(_page([LONG_CLEAN[0], ("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        assert (out["perplex_blend"] == "").all()

    def test_repeated_application_is_a_fixed_point(self):
        """Blending reads `perplex_raw`, never `perplex`, so a second pass over
        an already-blended frame must not move anything further."""
        once = _blend(_page(LONG_CLEAN + [("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        twice = _blend(once)
        assert list(twice["perplex"]) == list(once["perplex"])
        assert list(twice["perplex_blend"]) == list(once["perplex_blend"])
        assert list(twice["categ"]) == list(once["categ"])


class TestKnownLimits:
    """What the blend does NOT do. Asserted so nobody has to rediscover it."""

    def test_does_not_rescue_phonotactic_garbage_on_its_own(self):
        """A usable perplexity is necessary but not sufficient.

        Both lang-gated Trash routes refuse `oueussd` because FastText is
        *confidently wrong* about it (`orig_lang_score` 0.9163):
        `rule_hard_sweep` wants < 0.45, `rule_extreme_ppl` wants < 0.85. The only
        ungated route, `rule_absolute_ppl`, needs 30000 and the blend lands
        around 5700. So on a garbage page the blend raises the number and still
        cannot convict.

        Consequence for issue #30: this change does **not**, by itself, make it
        safe to gate `rule_short_garbage`. It removes one blocker (short lines
        having no perplexity at all); the language gates remain.
        """
        out = _blend(_page(LONG_GARBAGE + [("oueussd", "isl_Latn", 0.9163, 4600.0)]))
        blended = float(out.iloc[-1]["perplex_blend"])

        assert blended > tu.PPL_EXTREME_MIN, "blend should clear the extreme-ppl bar"
        assert blended < tu.PPL_GARBAGE_ABSOLUTE, "but not the only lang-ungated bar"
        assert 0.9163 > tu.EXTREME_LANG_CONF, "and the lang gate refuses it anyway"

    def test_is_a_page_consistency_prior_not_a_quality_signal(self):
        """On a garbage page a *clean* short line is pushed up too. That is the
        prior working as designed, and the reason the constants need calibrating
        on a real collection before the flag is turned on."""
        out = _blend(_page(LONG_GARBAGE + [("kontext", "ces_Latn", 0.95, 120.0)]))
        blended = float(out.iloc[-1]["perplex_blend"])
        assert blended > 120.0
