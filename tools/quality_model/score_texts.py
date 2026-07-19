"""Relabel arbitrary text lines with the production quality engine (issue #23, Phase 0).

The offline replay tool (``tools/recategorize_from_csv.py``) can only re-score lines
whose FastText + Qwen signals are already frozen in a CSV. Corrupted/corrected
variants are *new strings*, so they need a fresh FastText + Qwen pass before
``compute_quality_score`` can run. This module is that pass.

``build_line_record`` is a **faithful, dependency-light mirror** of the per-line
orchestration currently inlined in
``langID_classify.process_and_write_batch_cpu`` (``langID_classify.py:315-437``).
It calls the REAL production leaf functions in ``text_util_langID`` — it is not a
second engine. Phase 0's refactor task is to make production *import this function*
and lock the equivalence with ``tests/test_line_record_parity.py``.

Unlike the stored CSV ``quality_score`` (which is band-clamped and later mutated by
document-level post-processing), this returns BOTH:

* ``score_raw``     — the pre-clamp ``compute_quality_score`` output (the model's
  regression target, strategy decision D1), and
* ``score_clamped`` — the band-aligned score ``categorize_line`` stores.

Run standalone (needs the ML stack from setup/requirements.txt)::

    python tools/quality_model/score_texts.py --input variants.csv --text-col text \\
        --model Qwen/Qwen2.5-0.5B --fasttext lid.176.bin --out scored.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import text_util_langID as tu  # noqa: E402  (path bootstrap must run first)

# Trust-tier multipliers — the same values langID_classify reads from config
# ([CLASSIFY] section, langID_classify.py:240-241). Read from the shared config so
# this tool cannot silently diverge from production.
TRUST_TIER_TRUSTED = tu._get_float("CLASSIFY", "TRUST_TIER_TRUSTED", 0.85)
TRUST_TIER_UNKNOWN = tu._get_float("CLASSIFY", "TRUST_TIER_UNKNOWN", 0.50)

# Columns this tool adds on top of the raw text.
SCORE_COLUMNS = [
    "categ",
    "score_raw",
    "score_clamped",
    "reason",
    "route",
    "lang",
    "lang_score",
    "original_lang",
    "orig_lang_score",
    "perplex",
    "word_count",
    "char_count",
    "garbage_density",
    "word_weird",
    "vowel_ratio",
    "rot_ratio",
    "fused_words",
    "gibberish",
    "weird_wx",
    "upper",
    "repeated",
    "ldl_fuses",
    "caps_header",
]


def _known_lang_bases(expected_langs: list[str], trusted_langs: list[str]) -> frozenset:
    return frozenset(tu._lang_base(lng) for lng in (list(trusted_langs) + list(expected_langs)))


def build_line_record(
    raw_line: str,
    original_lang: str,
    original_lang_score: float,
    perplexity: float,
    expected_langs: list[str],
    trusted_langs: list[str],
) -> dict:
    """Score a single line exactly the way production does.

    ``original_lang`` / ``original_lang_score`` are the *pre-remap* FastText
    prediction; ``perplexity`` is the fresh Qwen (or distilgpt2) value. Empty and
    ``Non-text`` lines short-circuit through ``pre_filter_line`` with a 0.0 score,
    mirroring the fast-track rows in production (``langID_classify.py:447``).
    """
    original_text = raw_line
    merged, split_ws, split_we = tu.parse_line_splits(raw_line)
    route, clean_text = tu.pre_filter_line(merged)

    if route in ("Empty", "Non-text"):
        return {
            "categ": route,
            "score_raw": 0.0,
            "score_clamped": 0.0,
            "reason": route.lower(),
            "route": route,
            "text": clean_text,
            "original_text": original_text,
            "split_ws": split_ws,
            "split_we": split_we,
            "lang": "N/A",
            "lang_score": 0.0,
            "original_lang": "N/A",
            "orig_lang_score": 0.0,
            "perplex": 0.0,
            "word_count": 0,
            "char_count": len(clean_text),
            "garbage_density": 0.0,
            "word_weird": 0.0,
            "vowel_ratio": 0.0,
            "rot_ratio": 0.0,
            "fused_words": 0,
            "gibberish": 0,
            "weird_wx": 0,
            "upper": 0,
            "repeated": 0,
            "ldl_fuses": 0,
            "caps_header": False,
        }

    text_content = clean_text
    known = _known_lang_bases(expected_langs, trusted_langs)

    wc = len(text_content.split())
    cc = len(text_content)

    lang, lang_score = tu.remap_lang(original_lang, original_lang_score, known, expected_langs[0])

    ppl_val = perplexity
    if wc <= 2 and ppl_val > tu.SHORT_PPL_CAP:
        ppl_val = tu.SHORT_PPL_CAP

    # Garbage density and vowel ratio are computed on the ORIGINAL (pre-repair)
    # text so cleaning never hides noise — identical to production.
    g_density = tu.compute_garbage_density(original_text)
    vowel_ratio = tu.compute_vowel_ratio(original_text)

    upper_count = tu.detect_mid_uppercase(text_content)
    rep_count = tu.detect_repeated_chars(text_content)
    fuse_count = tu.detect_letter_digit_letter(text_content)
    fused_words = tu.detect_fused_words(text_content)
    gibb_count = tu.detect_gibberish_words(text_content)
    wx_count = tu.detect_wx_words(text_content)
    rot_ratio = tu.compute_rotatable_ratio(text_content)

    is_upright_czech, ghost_dominated = tu.analyze_rotation_signals(text_content)
    caps_header = tu.is_all_caps_line(text_content)
    word_scores = tu.score_words_in_line(text_content)
    weird_ratio = tu.compute_word_weird_ratio(word_scores)
    valid_ratio = tu.compute_valid_ratio(text_content)

    # Two-tier trust on the ORIGINAL FastText score (feeds QS_WEIGHT_LANG).
    base_lang = tu._lang_base(original_lang)
    if base_lang in known:
        trust_lang_score = (
            original_lang_score if base_lang in expected_langs else original_lang_score * TRUST_TIER_TRUSTED
        )
    else:
        trust_lang_score = original_lang_score * TRUST_TIER_UNKNOWN

    score_raw = tu.compute_quality_score(
        valid_word_ratio=valid_ratio,
        perplexity=ppl_val,
        text_length=cc,
        weird_ratio=weird_ratio,
        vowel_ratio=vowel_ratio,
        garbage_density=g_density,
        lang_score=trust_lang_score,
        gibberish_ratio=(gibb_count + wx_count) / max(wc, 1),
        fused_ratio=fused_words / max(wc, 1),
        is_upright_czech=is_upright_czech,
    )

    categ, score_clamped, reason = tu.categorize_line(
        score_raw,
        text_content,
        wc,
        vowel_ratio,
        ppl_val,
        weird_ratio=weird_ratio,
        return_reason=True,
        valid_word_ratio=valid_ratio,
        lang_score=trust_lang_score,
        orig_lang_score=original_lang_score,
        gibberish_present=(gibb_count + wx_count) > 0,
        garbage_density=g_density,
        is_upright_czech=is_upright_czech,
        ghost_dominated=ghost_dominated,
    )

    return {
        "categ": categ,
        "score_raw": round(float(score_raw), 4),
        "score_clamped": round(float(score_clamped), 4),
        "reason": reason,
        "route": route,
        "text": text_content,
        "original_text": original_text,
        "split_ws": split_ws,
        "split_we": split_we,
        "lang": lang,
        "lang_score": round(float(lang_score), 4),
        "original_lang": original_lang,
        "orig_lang_score": round(float(original_lang_score), 4),
        "perplex": round(float(ppl_val), 2),
        "word_count": wc,
        "char_count": cc,
        "garbage_density": round(float(g_density), 4),
        "word_weird": round(float(weird_ratio), 4),
        "vowel_ratio": round(float(vowel_ratio), 4),
        "rot_ratio": round(float(rot_ratio), 4),
        "fused_words": fused_words,
        "gibberish": gibb_count,
        "weird_wx": wx_count,
        "upper": upper_count,
        "repeated": rep_count,
        "ldl_fuses": fuse_count,
        "caps_header": caps_header,
    }


# ---------------------------------------------------------------------------
# Model loading + batch scoring (heavy deps imported lazily so the module — and
# build_line_record — import cleanly without the ML stack, e.g. in fast tests).
# ---------------------------------------------------------------------------


def load_fasttext(model_path: str):
    import fasttext  # noqa: PLC0415  (lazy: only needed for a real scoring run)

    return fasttext.load_model(model_path)


def predict_langs(ft, lines: list[str]) -> list[tuple[str, float]]:
    labels, scores = ft.predict([ln.lower() for ln in lines], k=1)
    return [(lbl[0].replace("__label__", ""), float(sc[0])) for lbl, sc in zip(labels, scores, strict=True)]


def load_perplexity_model(model_name: str):
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)
    model.eval()
    return model, tokenizer, device


def compute_perplexities(texts: list[str], model, tokenizer, device, batch_size: int = 32) -> list[float]:
    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        out.extend(tu.calculate_perplexity_batch(chunk, model, tokenizer, device))
    return out


def score_lines(
    lines: list[str],
    ft,
    ppl_bundle,
    expected_langs: list[str],
    trusted_langs: list[str],
    batch_size: int = 32,
) -> list[dict]:
    """Score a list of raw text lines end-to-end (FastText + perplexity + engine)."""
    preds = predict_langs(ft, lines)
    model, tokenizer, device = ppl_bundle
    ppls = compute_perplexities(lines, model, tokenizer, device, batch_size=batch_size)
    records = []
    for line, (lang, score), ppl in zip(lines, preds, ppls, strict=True):
        records.append(build_line_record(line, lang, score, ppl, expected_langs, trusted_langs))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_lines(path: Path, text_col: str) -> tuple[list[str], list[dict]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return [(r.get(text_col) or "") for r in rows], rows
    with path.open(encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    return lines, [{} for _ in lines]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Relabel text lines with the production quality engine.")
    p.add_argument("--input", required=True, type=Path, help="CSV (with --text-col) or newline-delimited .txt.")
    p.add_argument("--text-col", default="text", help="Text column when --input is CSV (default: text).")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Perplexity LM (default: Qwen/Qwen2.5-0.5B).")
    p.add_argument("--fasttext", default="lid.176.bin", help="FastText language-id model path.")
    p.add_argument("--expected-langs", default="ces,deu,eng", help="Comma-separated expected base langs.")
    p.add_argument("--trusted-langs", default="deu,eng,fra,pol,ita,slk", help="Comma-separated trusted foreign langs.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", required=True, type=Path, help="Output CSV with the added score columns.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    lines, src_rows = _read_lines(args.input, args.text_col)
    expected = [s.strip() for s in args.expected_langs.split(",") if s.strip()]
    trusted = [s.strip() for s in args.trusted_langs.split(",") if s.strip()]

    ft = load_fasttext(args.fasttext)
    ppl_bundle = load_perplexity_model(args.model)
    records = score_lines(lines, ft, ppl_bundle, expected, trusted, batch_size=args.batch_size)

    # Preserve any provenance columns present in the source CSV.
    passthrough = [c for c in (src_rows[0].keys() if src_rows and src_rows[0] else []) if c not in SCORE_COLUMNS]
    fieldnames = passthrough + ["text"] + SCORE_COLUMNS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for src, rec in zip(src_rows, records, strict=True):
            writer.writerow({**src, **rec})

    print(f"Scored {len(records)} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
