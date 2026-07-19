"""
tests/test_quality_model_corrupt.py
===================================
Fast, model-free tests for the issue #23 corruption engine
(``tools/quality_model/corrupt.py``).

Two guarantees are locked here:

1. **Determinism** — the same ``(seed, doc, line, variant)`` key always yields the
   same variant, so a dataset is reproducible from its manifest.
2. **Detector alignment** — each corruption op actually moves the production
   detector it claims to target (the detectors are the oracles).

All pure-Python: no torch / FastText / Qwen.
"""

from __future__ import annotations

import sys
from pathlib import Path

# conftest.py puts the repo root on sys.path; add tools/quality_model too so the
# tool modules import by name (mirrors tests/test_recategorize_parity.py).
_ROOT = Path(__file__).resolve().parent.parent
_QM = _ROOT / "tools" / "quality_model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

import random  # noqa: E402

import corrupt as C  # noqa: E402

import text_util_langID as tu  # noqa: E402


def _rng(seed: int = 1) -> random.Random:
    return random.Random(seed)


# ── Determinism ────────────────────────────────────────────────────────────


def test_derive_rng_is_deterministic():
    a = C.derive_rng(23, "CTX1", 5, 0).random()
    b = C.derive_rng(23, "CTX1", 5, 0).random()
    assert a == b


def test_derive_rng_varies_by_key():
    base = C.derive_rng(23, "CTX1", 5, 0).random()
    assert C.derive_rng(23, "CTX1", 5, 1).random() != base
    assert C.derive_rng(23, "CTX1", 6, 0).random() != base
    assert C.derive_rng(99, "CTX1", 5, 0).random() != base


def test_make_variants_reproducible():
    text = "Mocnost kulturní vrstvy činila 40 cm."
    v1 = C.make_variants(text, "CTX2", 3, global_seed=23, n_variants=3)
    v2 = C.make_variants(text, "CTX2", 3, global_seed=23, n_variants=3)
    assert [v.text for v in v1] == [v.text for v in v2]
    assert [v.ops for v in v1] == [v.ops for v in v2]


def test_make_variants_covers_bands():
    variants = C.make_variants("Popis nálezu v sondě.", "CTX2", 4, global_seed=7, n_variants=3)
    assert [v.band for v in variants] == ["light", "medium", "heavy"]


# ── Detector alignment (each op moves its target signal) ────────────────────


def test_diacritic_strip_removes_czech_diacritics():
    src = "čeština má diakritiku á"
    assert tu.has_cz_diacs(src)
    out = C.op_diacritic_strip(src, eps=1.0, rng=_rng())
    assert not tu.has_cz_diacs(out)


def test_case_flip_creates_mid_uppercase():
    src = "slova domu praha"
    assert tu.detect_mid_uppercase(src) == 0
    out = C.op_case_flip(src, eps=1.0, rng=_rng())
    assert tu.detect_mid_uppercase(out) >= 1


def test_word_fusion_creates_fused_words_and_drops_word_count():
    src = "aaa bbb ccc ddd eee fff"
    out = C.op_word_fusion(src, eps=1.0, rng=_rng())
    assert len(out.split()) < len(src.split())
    assert tu.detect_fused_words(out) >= 1


def test_word_split_increases_word_count():
    src = "kostelvaclava"
    out = C.op_word_split(src, eps=1.0, rng=_rng())
    assert len(out.split()) > 1


def test_symbol_injection_raises_garbage_density():
    src = "cista veta bez chyb"
    assert tu.compute_garbage_density(src) == 0.0
    out = C.op_symbol_injection(src, eps=1.0, rng=_rng())
    assert tu.compute_garbage_density(out) > 0.0


def test_char_double_creates_repeated_run():
    src = "kostel domu"
    assert tu.detect_repeated_chars(src) == 0
    out = C.op_char_double(src, eps=1.0, rng=_rng())
    assert tu.detect_repeated_chars(out) >= 1


def test_vowel_strip_lowers_vowel_ratio():
    src = "kulturni vrstva"
    out = C.op_vowel_strip(src, eps=1.0, rng=_rng())
    assert tu.compute_vowel_ratio(out) < tu.compute_vowel_ratio(src)


def test_char_drop_shortens_line():
    src = "Mocnost kulturní vrstvy"
    out = C.op_char_drop(src, eps=1.0, rng=_rng())
    assert len(out) < len(src)


def test_ledger_fill_raises_garbage_density():
    src = "Polozka"
    out = C.op_ledger_fill(src, eps=0.8, rng=_rng())
    assert tu.compute_garbage_density(out) > tu.compute_garbage_density(src)


def test_truncate_caps_length_at_heavy_severity():
    src = "Toto je pomerne dlouhy radek textu ke zkraceni"
    out = C.op_truncate(src, eps=0.9, rng=_rng())
    assert len(out) <= 12


def test_char_confusion_changes_confusable_text_deterministically():
    src = "kostel Ilona 105"
    out1 = C.op_char_confusion(src, eps=1.0, rng=_rng(4))
    out2 = C.op_char_confusion(src, eps=1.0, rng=_rng(4))
    assert out1 == out2
    assert out1 != src


def test_rotation_ghost_is_deterministic_and_maps_glyphs():
    src = "bdnms podnebi"
    out1 = C.op_rotation_ghost(src, eps=1.0, rng=_rng(2))
    out2 = C.op_rotation_ghost(src, eps=1.0, rng=_rng(2))
    assert out1 == out2
    assert out1 != src


# ── Composition ────────────────────────────────────────────────────────────


def test_corrupt_line_records_ops_and_eps():
    v = C.corrupt_line("Popis nálezu v sondě.", _rng(3), band="medium")
    assert v.band == "medium"
    assert len(v.ops) == len(v.eps)
    assert all(0.0 <= e <= 1.0 for e in v.eps)


def _count_cz_diacs(text: str) -> int:
    return sum(1 for ch in text if ch in tu.CZ_DIACS)


def test_corrupt_line_explicit_ops_are_applied():
    # In explicit-ops mode the per-op eps is still drawn from the band (< 1.0), so
    # diacritic_strip thins the diacritics rather than removing every one. Assert
    # the op measurably fired on a diacritic-dense line.
    src = "řeka žila šišky čáp ďas ťuká ňaká áéíóú"
    v = C.corrupt_line(src, _rng(1), band="heavy", ops=["diacritic_strip"])
    assert v.ops == ["diacritic_strip"]
    assert _count_cz_diacs(v.text) < _count_cz_diacs(src)
