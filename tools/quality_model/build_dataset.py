"""Assemble the quality-score training dataset (issue #23, Phase 1).

Composes the corruption engine (``corrupt.py``) and the relabeller
(``score_texts.build_line_record``) into the pipeline from strategy §1.5:

    select sources → generate variants → relabel → dedup → split-by-document
    → balance → write.

Two relabelling backends implement the same ``scorer(items) -> None`` contract:

* ``offline_scorer`` — model-free. Reuses each *original* line's frozen
  perplexity/lang from the source CSV and applies a fixed default to synthetic
  variants. Deterministic, no GPU — used for dry runs and the fast tests. The
  non-perplexity detectors still react to corruption, so the score continuum
  (and the monotonicity report) is meaningful even without Qwen.
* ``ModelScorer`` — the real fresh FastText + Qwen pass (``score_texts.py``),
  used for production dataset builds.

Splitting is **by document** so no line and no near-duplicate leaks between
train/val/test; the expert gold documents are forced into the held-out ``test``
split and never seen in training (strategy Phase 4 gate).

Run standalone (offline dry run, no models)::

    python tools/quality_model/build_dataset.py \\
        --input data_samples/DOC_LINE_CATEG/CTX000000002.csv \\
        --scorer offline --seed 23 --out /tmp/dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import corrupt as C  # noqa: E402
import score_texts as S  # noqa: E402

# Categories excluded from the model's domain (strategy D3): the cheap pre-filter
# keeps handling these in production, so they never enter the training set.
EXCLUDED_CATEGS = frozenset({"Empty", "Non-text"})

DEFAULT_EXPECTED = ["ces", "deu", "eng"]
DEFAULT_TRUSTED = ["deu", "eng", "fra", "pol", "ita", "slk"]

# Feature columns copied from the relabelled record into each dataset row.
FEATURE_COLUMNS = [
    "lang",
    "lang_score",
    "perplex",
    "word_count",
    "char_count",
    "garbage_density",
    "word_weird",
    "vowel_ratio",
    "rot_ratio",
    "fused_words",
    "gibberish",
]

OUTPUT_COLUMNS = (
    ["text", "score_raw", "score_clamped", "categ", "reason", "provenance", "ops", "band"]
    + ["source_doc", "source_line", "source_categ"]
    + FEATURE_COLUMNS
    + ["split"]
)


def normalize_text(text: str) -> str:
    """Whitespace-normalised key for deduplication."""
    return " ".join((text or "").split())


# ---------------------------------------------------------------------------
# 1. Select source lines
# ---------------------------------------------------------------------------


def select_sources(
    rows: list[dict],
    per_doc_cap: int | None,
    *,
    text_col: str = "text",
    categ_col: str = "categ",
    doc_col: str = "file",
    line_col: str = "line_num",
    wc_col: str = "word_count",
) -> list[dict]:
    """Drop Empty/Non-text/blank lines and optionally cap lines per document."""
    kept: list[dict] = []
    per_doc = Counter()
    for row in rows:
        categ = (row.get(categ_col) or "").strip()
        text = (row.get(text_col) or "").strip()
        if not text or categ in EXCLUDED_CATEGS:
            continue
        try:
            wc = int(float(row.get(wc_col) or 0))
        except (TypeError, ValueError):
            wc = len(text.split())
        if wc == 0:
            continue
        doc = row.get(doc_col, "doc")
        if per_doc_cap is not None and per_doc[doc] >= per_doc_cap:
            continue
        per_doc[doc] += 1
        kept.append(
            {
                "text": text,
                "source_categ": categ,
                "source_doc": doc,
                "source_line": row.get(line_col, ""),
                "frozen_perplex": _to_float(row.get("perplex"), None),
                "frozen_lang": row.get("original_lang") or row.get("lang") or "",
                "frozen_lang_score": _to_float(row.get("orig_lang_score") or row.get("lang_score"), 0.0),
            }
        )
    return kept


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 2. Generate original + corrupted variant items
# ---------------------------------------------------------------------------


def generate_items(sources: list[dict], seed: int, variants_per_clear: int) -> list[dict]:
    """Emit the original for every source line plus corrupted variants of Clear lines.

    Noisy correction variants are Phase 2 (``correct.py``) and are not produced
    here; Trash lines are used as-is (strategy §1.5).
    """
    items: list[dict] = []
    for src in sources:
        items.append(_item_from(src, provenance="original", text=src["text"], ops=[], band="none"))
        if src["source_categ"] == "Clear" and variants_per_clear > 0:
            variants = C.make_variants(
                src["text"], str(src["source_doc"]), src["source_line"], seed, n_variants=variants_per_clear
            )
            for v in variants:
                items.append(_item_from(src, provenance="corrupt", text=v.text, ops=v.ops, band=v.band))
    return items


def _item_from(src: dict, *, provenance: str, text: str, ops: list[str], band: str) -> dict:
    return {
        "text": text,
        "provenance": provenance,
        "ops": ";".join(ops),
        "band": band,
        "source_categ": src["source_categ"],
        "source_doc": src["source_doc"],
        "source_line": src["source_line"],
        "frozen_perplex": src["frozen_perplex"],
        "frozen_lang": src["frozen_lang"],
        "frozen_lang_score": src["frozen_lang_score"],
    }


# ---------------------------------------------------------------------------
# 3. Relabel (two interchangeable scorers)
# ---------------------------------------------------------------------------


def make_offline_scorer(
    default_perplexity: float = 150.0,
    default_lang: str = "ces_Latn",
    default_lang_score: float = 0.90,
    expected_langs: list[str] | None = None,
    trusted_langs: list[str] | None = None,
):
    """Model-free scorer: originals reuse their frozen perplexity/lang; synthetic
    variants (no frozen signal) get a fixed default. Recomputes the score with the
    live engine, so it is config-consistent, not the stored clamped value."""
    expected = expected_langs or DEFAULT_EXPECTED
    trusted = trusted_langs or DEFAULT_TRUSTED

    def _score(items: list[dict]) -> None:
        for it in items:
            ppl = it["frozen_perplex"] if it["frozen_perplex"] is not None else default_perplexity
            lang = it["frozen_lang"] or default_lang
            lang_score = it["frozen_lang_score"] if it["frozen_lang_score"] else default_lang_score
            rec = S.build_line_record(it["text"], lang, lang_score, ppl, expected, trusted)
            _merge_record(it, rec)

    return _score


class ModelScorer:
    """Real fresh FastText + Qwen relabelling pass (strategy D2)."""

    def __init__(
        self,
        model_name: str,
        fasttext_path: str,
        expected_langs: list[str] | None = None,
        trusted_langs: list[str] | None = None,
        batch_size: int = 32,
    ):
        self.expected = expected_langs or DEFAULT_EXPECTED
        self.trusted = trusted_langs or DEFAULT_TRUSTED
        self.batch_size = batch_size
        self.ft = S.load_fasttext(fasttext_path)
        self.ppl_bundle = S.load_perplexity_model(model_name)

    def __call__(self, items: list[dict]) -> None:
        texts = [it["text"] for it in items]
        recs = S.score_lines(texts, self.ft, self.ppl_bundle, self.expected, self.trusted, batch_size=self.batch_size)
        for it, rec in zip(items, recs, strict=True):
            _merge_record(it, rec)


def _merge_record(item: dict, rec: dict) -> None:
    item["score_raw"] = rec["score_raw"]
    item["score_clamped"] = rec["score_clamped"]
    item["categ"] = rec["categ"]
    item["reason"] = rec["reason"]
    for col in FEATURE_COLUMNS:
        item[col] = rec.get(col)


def score_items(items: list[dict], scorer) -> tuple[list[dict], int]:
    """Relabel every item, then drop those that fell out of the model domain
    (corruption pushed them to Empty/Non-text). Returns (kept, dropped_count)."""
    scorer(items)
    kept = [it for it in items if it.get("categ") not in EXCLUDED_CATEGS]
    return kept, len(items) - len(kept)


# ---------------------------------------------------------------------------
# 4. Dedup (before splitting — the main leakage vector)
# ---------------------------------------------------------------------------


def dedup_items(items: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept: list[dict] = []
    for it in items:
        key = normalize_text(it["text"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(it)
    return kept, len(items) - len(kept)


# ---------------------------------------------------------------------------
# 5. Split by document
# ---------------------------------------------------------------------------


def assign_splits(
    items: list[dict],
    gold_docs: set[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Deterministically assign each document (hence all its lines) to a split.
    Gold documents are forced into ``test``. Returns the doc→split map."""
    train_r, val_r, _ = ratios
    train_cut = train_r * 100
    val_cut = (train_r + val_r) * 100
    doc_split: dict[str, str] = {}
    for it in items:
        doc = str(it["source_doc"])
        if doc not in doc_split:
            if doc in gold_docs:
                doc_split[doc] = "test"
            else:
                bucket = int(hashlib.sha256(f"{seed}|{doc}".encode()).hexdigest(), 16) % 100
                doc_split[doc] = "train" if bucket < train_cut else ("val" if bucket < val_cut else "test")
        it["split"] = doc_split[doc]
    return doc_split


# ---------------------------------------------------------------------------
# 6. Balance the train split by score bin (soft cap; never duplicate)
# ---------------------------------------------------------------------------


def balance_train(items: list[dict], n_bins: int, bin_cap: int | None) -> tuple[list[dict], int]:
    """Soft-cap the number of TRAIN rows per score bin. val/test are untouched."""
    if bin_cap is None:
        return items, 0
    per_bin: dict[int, int] = defaultdict(int)
    kept: list[dict] = []
    removed = 0
    for it in items:
        if it["split"] != "train":
            kept.append(it)
            continue
        b = min(n_bins - 1, max(0, int(float(it["score_raw"]) * n_bins)))
        if per_bin[b] >= bin_cap:
            removed += 1
            continue
        per_bin[b] += 1
        kept.append(it)
    return kept, removed


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_dataset(
    rows: list[dict],
    scorer,
    *,
    seed: int = 23,
    variants_per_clear: int = 3,
    per_doc_cap: int | None = None,
    gold_docs: set[str] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    n_bins: int = 20,
    bin_cap: int | None = None,
) -> tuple[list[dict], dict]:
    gold_docs = gold_docs or set()
    sources = select_sources(rows, per_doc_cap)
    items = generate_items(sources, seed, variants_per_clear)
    items, dropped_prefilter = score_items(items, scorer)
    items, dedup_removed = dedup_items(items)
    doc_split = assign_splits(items, gold_docs, ratios, seed)
    items, balance_removed = balance_train(items, n_bins, bin_cap)

    manifest = {
        "config": {
            "seed": seed,
            "variants_per_clear": variants_per_clear,
            "per_doc_cap": per_doc_cap,
            "ratios": list(ratios),
            "n_bins": n_bins,
            "bin_cap": bin_cap,
            "gold_docs": sorted(gold_docs),
        },
        "counts": {
            "total": len(items),
            "by_split": dict(Counter(it["split"] for it in items)),
            "by_provenance": dict(Counter(it["provenance"] for it in items)),
            "by_category": dict(Counter(it["categ"] for it in items)),
            "sources_selected": len(sources),
        },
        "dropped_prefilter": dropped_prefilter,
        "dedup_removed": dedup_removed,
        "balance_removed": balance_removed,
        "doc_splits": doc_split,
    }
    return items, manifest


def write_dataset(items: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for it in items:
            writer.writerow(it)


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble the quality-score training dataset.")
    p.add_argument("--input", nargs="*", type=Path, default=[], help="DOC_LINE_CATEG CSV file(s).")
    p.add_argument("--input-glob", default=None, help="Glob for DOC_LINE_CATEG CSVs (e.g. 'DOC_LINE_CATEG/*.csv').")
    p.add_argument("--scorer", choices=["offline", "model"], default="offline", help="Relabelling backend.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Perplexity LM for --scorer model.")
    p.add_argument("--fasttext", default="lid.176.bin", help="FastText model path for --scorer model.")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--variants-per-clear", type=int, default=3)
    p.add_argument("--per-doc-cap", type=int, default=None)
    p.add_argument("--gold-docs", default="", help="Comma-separated document ids forced into the test split.")
    p.add_argument("--bins", type=int, default=20, help="Score bins used for train balancing.")
    p.add_argument("--bin-cap", type=int, default=None, help="Max train rows per score bin (default: no cap).")
    p.add_argument("--out", required=True, type=Path, help="Output dataset CSV.")
    p.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON (default: <out>.manifest.json).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    paths = list(args.input)
    if args.input_glob:
        paths += [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise SystemExit("No input CSVs given (use --input or --input-glob).")

    rows = read_rows(paths)
    if args.scorer == "model":
        scorer = ModelScorer(args.model, args.fasttext)
    else:
        scorer = make_offline_scorer()

    gold = {d.strip() for d in args.gold_docs.split(",") if d.strip()}
    items, manifest = build_dataset(
        rows,
        scorer,
        seed=args.seed,
        variants_per_clear=args.variants_per_clear,
        per_doc_cap=args.per_doc_cap,
        gold_docs=gold,
        n_bins=args.bins,
        bin_cap=args.bin_cap,
    )

    write_dataset(items, args.out)
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(items)} rows -> {args.out}")
    print(f"  splits: {manifest['counts']['by_split']}")
    print(f"  provenance: {manifest['counts']['by_provenance']}")
    print(f"  dropped_prefilter={manifest['dropped_prefilter']} dedup_removed={manifest['dedup_removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
