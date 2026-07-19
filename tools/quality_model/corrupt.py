"""OCR-realistic corruption engine (issue #23, Phase 1).

Turns a ``Clear`` OCR line into synthetic ``Noisy``/``Trash`` variants so the
regression model sees a smooth score continuum instead of only the three clamped
bands the production algorithm emits. Each operation is parameterized by an
intensity ``eps in [0, 1]`` and is **deliberately aligned with a production
detector** in ``text_util_langID`` — so the detectors double as the unit-test
oracles that prove each generator does what it claims (see
``tests/test_quality_model_corrupt.py``).

The engine only *generates text*. It never assigns a score: corrupted variants
are (re)labelled by a fresh FastText+Qwen pass in ``score_texts.py`` (strategy
decision D2). This keeps the "one scoring engine" invariant intact.

Determinism: every variant is produced from a ``random.Random`` seeded by a
SHA-256 digest of ``(global_seed, doc_id, line_num, variant_idx)`` — never the
builtin salted ``hash()`` — so a dataset is exactly reproducible from its
manifest.

Run standalone::

    python tools/quality_model/corrupt.py --input clear_lines.csv \\
        --text-col text --variants 3 --seed 23 --out variants.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the repo root importable whether this file is run as a script or imported
# as ``tools.quality_model.corrupt`` — mirrors how the other tools/ scripts reach
# the production modules.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import text_util_langID as tu  # noqa: E402  (path bootstrap must run first)

# ---------------------------------------------------------------------------
# Character tables — reuse the production sets so the generators and detectors
# always agree on which glyphs are diacritics / vowels / rotatable / fill.
# ---------------------------------------------------------------------------

# Fold every Czech/German diacritic to its ASCII base (scanner dropping accents).
_DIACRITIC_FOLD: dict[str, str] = {
    "á": "a",
    "č": "c",
    "ď": "d",
    "é": "e",
    "ě": "e",
    "í": "i",
    "ň": "n",
    "ó": "o",
    "ř": "r",
    "š": "s",
    "ť": "t",
    "ú": "u",
    "ů": "u",
    "ý": "y",
    "ž": "z",
    "Á": "A",
    "Č": "C",
    "Ď": "D",
    "É": "E",
    "Ě": "E",
    "Í": "I",
    "Ň": "N",
    "Ó": "O",
    "Ř": "R",
    "Š": "S",
    "Ť": "T",
    "Ú": "U",
    "Ů": "U",
    "Ý": "Y",
    "Ž": "Z",
    "ä": "a",
    "ö": "o",
    "ü": "u",
    "ß": "ss",
    "Ä": "A",
    "Ö": "O",
    "Ü": "U",
}

# Bigram-level OCR confusions applied before single-char ones (a single pass each).
_BIGRAM_CONFUSIONS: list[tuple[str, str]] = [("rn", "m"), ("cl", "d"), ("vv", "w")]

# Single-char OCR confusions (each maps to plausible mis-reads).
_CHAR_CONFUSIONS: dict[str, list[str]] = {
    "l": ["1", "I", "|"],
    "I": ["l", "1"],
    "1": ["l", "I"],
    "O": ["0"],
    "0": ["O"],
    "o": ["e", "c"],
    "c": ["e"],
    "e": ["c"],
    "S": ["5"],
    "5": ["S"],
    "B": ["8"],
    "8": ["B"],
    "g": ["9"],
    "9": ["g"],
    "m": ["rn"],
    "u": ["n"],
    "n": ["u"],
    "h": ["b"],
    "í": ["1"],
    "ě": ["e"],
}

# Injected specks / marginalia (all non-alnum and outside ALLOWED_INTERNAL, so
# they register as both garbage density and strange symbols).
_INJECT_SYMBOLS: str = "*»~§¤#^`¬°|<>"

# Mirror map for inverted/rotated scans (a superset restricted to the alphabet).
_ROTATION_MAP: dict[str, str] = {
    "b": "q",
    "q": "b",
    "d": "p",
    "p": "d",
    "n": "u",
    "u": "n",
    "m": "w",
    "w": "m",
    "s": "z",
    "z": "s",
}

_VOWELS_LOWER = "aeiouyáéíóúýěů"


# ---------------------------------------------------------------------------
# Individual corruption operations.  Signature: (text, eps, rng) -> text
# ---------------------------------------------------------------------------


def op_diacritic_strip(text: str, eps: float, rng: random.Random) -> str:
    """Drop diacritics → ASCII base (target: ``has_cz_diacs``, vowel ratio, lang)."""
    return "".join(_DIACRITIC_FOLD[ch] if ch in _DIACRITIC_FOLD and rng.random() < eps else ch for ch in text)


def op_char_confusion(text: str, eps: float, rng: random.Random) -> str:
    """Classic OCR glyph confusions (target: word weirdness, perplexity)."""
    s = text
    for a, b in _BIGRAM_CONFUSIONS:
        if a in s and rng.random() < eps:
            s = s.replace(a, b, 1)
    out = []
    for ch in s:
        subs = _CHAR_CONFUSIONS.get(ch)
        out.append(rng.choice(subs) if subs and rng.random() < eps else ch)
    return "".join(out)


def op_case_flip(text: str, eps: float, rng: random.Random) -> str:
    """Interior capital letters, e.g. ``slOva`` (target: ``detect_mid_uppercase``)."""
    out = []
    for word in text.split(" "):
        # Interior lowercase letters whose predecessor is also lowercase — flipping
        # one there is guaranteed to read as a mid-word capital.
        interior = [i for i in range(1, len(word)) if word[i].islower() and word[i].isalpha() and word[i - 1].islower()]
        if interior and rng.random() < eps:
            i = rng.choice(interior)
            word = word[:i] + word[i].upper() + word[i + 1 :]
        out.append(word)
    return " ".join(out)


def op_word_fusion(text: str, eps: float, rng: random.Random) -> str:
    """Drop inter-word spaces → long fused tokens (target: ``detect_fused_words``)."""
    words = text.split(" ")
    if len(words) < 2:
        return text
    out = [words[0]]
    for w in words[1:]:
        if rng.random() < eps:
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return " ".join(out)


def op_word_split(text: str, eps: float, rng: random.Random) -> str:
    """Insert spurious spaces inside words (target: word count, split logic)."""
    out = []
    for word in text.split(" "):
        if len(word) >= 4 and rng.random() < eps:
            i = rng.randint(1, len(word) - 1)
            out.append(word[:i] + " " + word[i:])
        else:
            out.append(word)
    return " ".join(out)


def op_symbol_injection(text: str, eps: float, rng: random.Random) -> str:
    """Sprinkle garbage symbols (target: ``compute_garbage_density``, strange symbols)."""
    if not text:
        return text
    n = max(1, int(round(eps * len(text) * 0.5)))
    chars = list(text)
    for _ in range(n):
        pos = rng.randint(0, len(chars))
        chars.insert(pos, rng.choice(_INJECT_SYMBOLS))
    return "".join(chars)


def op_char_drop(text: str, eps: float, rng: random.Random) -> str:
    """Delete characters — faint ink (target: valid-word ratio, length)."""
    return "".join(ch for ch in text if not (ch != " " and rng.random() < eps))


def op_char_double(text: str, eps: float, rng: random.Random) -> str:
    """Triple a letter → a 3-run (target: ``detect_repeated_chars``)."""
    out = []
    for word in text.split(" "):
        letters = [i for i, c in enumerate(word) if c.isalpha()]
        if letters and rng.random() < eps:
            i = rng.choice(letters)
            word = word[:i] + word[i] * 3 + word[i + 1 :]
        out.append(word)
    return " ".join(out)


def op_rotation_ghost(text: str, eps: float, rng: random.Random) -> str:
    """Mirror glyphs for inverted scans (target: ``compute_rotatable_ratio``)."""
    return "".join(_ROTATION_MAP[ch] if ch in _ROTATION_MAP and rng.random() < eps else ch for ch in text)


def op_ledger_fill(text: str, eps: float, rng: random.Random) -> str:
    """Append trailing fill + digit fragments — table rows (target: ledger rules)."""
    fill_chars = tu.TRAILING_FILL_CHARS.replace(" ", "") or ".-:"
    run = "".join(rng.choice(fill_chars) for _ in range(max(3, int(round(eps * 12)))))
    frag = " " + str(rng.randint(1, 9)) if rng.random() < eps else ""
    return f"{text} {run}{frag}"


def op_vowel_strip(text: str, eps: float, rng: random.Random) -> str:
    """Remove vowels → vowel-starved scramble (target: ``compute_vowel_ratio``)."""
    return "".join("" if ch.lower() in _VOWELS_LOWER and rng.random() < eps else ch for ch in text)


def op_truncate(text: str, eps: float, rng: random.Random) -> str:
    """Cut the line short (target: short-line penalty regime)."""
    if len(text) <= 12:
        return text
    keep = max(4, int(round(len(text) * (1.0 - eps))))
    keep = min(keep, 12) if eps >= 0.5 else keep
    return text[:keep].rstrip()


# Registry: op name -> callable. Weights bias how often each op is sampled when
# composing a variant (structural/heavy corruptions are rarer than light ones).
OPS: dict[str, "CorruptionOp"] = {}


@dataclass(frozen=True)
class CorruptionOp:
    name: str
    fn: object
    weight: float = 1.0


def _register(name: str, fn, weight: float = 1.0) -> None:
    OPS[name] = CorruptionOp(name=name, fn=fn, weight=weight)


_register("diacritic_strip", op_diacritic_strip, 1.5)
_register("char_confusion", op_char_confusion, 1.5)
_register("case_flip", op_case_flip, 1.0)
_register("word_fusion", op_word_fusion, 0.8)
_register("word_split", op_word_split, 0.8)
_register("symbol_injection", op_symbol_injection, 1.0)
_register("char_drop", op_char_drop, 1.0)
_register("char_double", op_char_double, 0.6)
_register("rotation_ghost", op_rotation_ghost, 0.5)
_register("ledger_fill", op_ledger_fill, 0.4)
_register("vowel_strip", op_vowel_strip, 0.5)
_register("truncate", op_truncate, 0.4)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# Stratified severity bands so a batch of variants sweeps the score range smoothly
# instead of clustering. Each band is an (eps_low, eps_high, n_ops_low, n_ops_high).
SEVERITY_BANDS: dict[str, tuple[float, float, int, int]] = {
    "light": (0.05, 0.25, 1, 2),  # -> mostly still Noisy/Clear
    "medium": (0.25, 0.55, 2, 3),  # -> Noisy
    "heavy": (0.55, 0.90, 3, 4),  # -> Trash
}


def derive_rng(global_seed: int, doc_id: str, line_num, variant_idx: int) -> random.Random:
    """Deterministic per-variant RNG from a SHA-256 digest (stable across machines)."""
    key = f"{global_seed}|{doc_id}|{line_num}|{variant_idx}".encode("utf-8")
    seed_int = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return random.Random(seed_int)


@dataclass
class Variant:
    text: str
    band: str
    ops: list[str] = field(default_factory=list)
    eps: list[float] = field(default_factory=list)


def corrupt_line(
    text: str,
    rng: random.Random,
    band: str = "medium",
    ops: list[str] | None = None,
) -> Variant:
    """Produce one corrupted variant of *text* under a severity *band*.

    If *ops* is given, exactly those operations are applied (each with an eps drawn
    from the band); otherwise 1-4 ops are sampled by weight from the registry.
    """
    eps_low, eps_high, n_low, n_high = SEVERITY_BANDS[band]
    if ops is None:
        names = list(OPS)
        weights = [OPS[n].weight for n in names]
        k = rng.randint(n_low, n_high)
        k = min(k, len(names))
        # Weighted sampling without replacement.
        chosen: list[str] = []
        pool, pool_w = names[:], weights[:]
        for _ in range(k):
            idx = _weighted_index(pool_w, rng)
            chosen.append(pool.pop(idx))
            pool_w.pop(idx)
        ops = chosen

    out = text
    used_eps: list[float] = []
    for name in ops:
        eps = rng.uniform(eps_low, eps_high)
        used_eps.append(round(eps, 3))
        out = OPS[name].fn(out, eps, rng)
    return Variant(text=out, band=band, ops=list(ops), eps=used_eps)


def _weighted_index(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    r = rng.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if upto >= r:
            return i
    return len(weights) - 1


def make_variants(
    text: str,
    doc_id: str,
    line_num,
    global_seed: int,
    n_variants: int = 3,
    bands: list[str] | None = None,
) -> list[Variant]:
    """Deterministically make *n_variants* variants of a line, one per severity band
    (cycled if ``n_variants`` exceeds the number of bands)."""
    bands = bands or list(SEVERITY_BANDS)
    variants = []
    for idx in range(n_variants):
        band = bands[idx % len(bands)]
        rng = derive_rng(global_seed, doc_id, line_num, idx)
        variants.append(corrupt_line(text, rng, band=band))
    return variants


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate OCR-corruption variants of clean text lines.")
    p.add_argument("--input", required=True, type=Path, help="CSV with a text column (e.g. a DOC_LINE_CATEG file).")
    p.add_argument("--text-col", default="text", help="Name of the text column (default: text).")
    p.add_argument("--categ-col", default="categ", help="Category column; only Clear rows are corrupted if present.")
    p.add_argument("--doc-col", default="file", help="Document id column, used for deterministic seeding.")
    p.add_argument("--line-col", default="line_num", help="Line id column, used for deterministic seeding.")
    p.add_argument("--variants", type=int, default=3, help="Variants per source line (default: 3).")
    p.add_argument("--seed", type=int, default=23, help="Global seed (default: 23).")
    p.add_argument("--out", required=True, type=Path, help="Output CSV of variants.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    with args.input.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames or []

    has_categ = args.categ_col in fields
    out_rows = []
    for i, row in enumerate(rows):
        text = (row.get(args.text_col) or "").strip()
        if not text:
            continue
        if has_categ and row.get(args.categ_col) != "Clear":
            continue
        doc_id = row.get(args.doc_col, "doc")
        line_num = row.get(args.line_col, i)
        for v in make_variants(text, doc_id, line_num, args.seed, n_variants=args.variants):
            out_rows.append(
                {
                    "source_doc": doc_id,
                    "source_line": line_num,
                    "provenance": "corrupt",
                    "band": v.band,
                    "ops": ";".join(v.ops),
                    "eps": ";".join(str(e) for e in v.eps),
                    "source_text": text,
                    "text": v.text,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["source_doc", "source_line", "provenance", "band", "ops", "eps", "source_text", "text"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} variants from {len(rows)} source rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
