"""Score-delta report for auto-corrected Noisy lines (issue #23, Phase 2).

The issue asks explicitly: *"The change of score computed by the algo after
auto-correction should be checked."* This tool answers it. For every
(source, corrected) pair it relabels **both** texts with the production engine and
reports:

* mean / median Δ  (Δ = ``score_raw(corrected) − score_raw(source)``),
* share improved, share degraded by more than 0.05 (a corrector turning garbage
  into different garbage),
* the Noisy→{Clear,Noisy,Trash} band-transition matrix,
* per-backend breakdown,
* the top / bottom example lines for manual review.

**Go/no-go gate:** a backend whose median Δ ≤ 0 is not helping and its variants
should be excluded from the training set (``report["gate"]["passed"]``).

Scoring uses the same ``scorer(items) -> None`` contract as ``build_dataset.py``;
the model-free ``offline`` scorer makes this runnable (and testable) without a GPU
— the non-perplexity detectors already move when correction removes garbage.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_dataset as B  # noqa: E402

DEGRADE_THRESHOLD = 0.05


def _item(text: str, lang: str) -> dict:
    return {"text": text, "frozen_perplex": None, "frozen_lang": lang, "frozen_lang_score": 0.0}


def score_pairs(pairs: list[dict], scorer) -> list[dict]:
    """Relabel source and corrected text for each pair; return delta records.

    ``pairs`` items need ``source_text`` and ``corrected_text`` (plus optional
    ``lang`` and ``backend``).
    """
    before = [_item(p["source_text"], p.get("lang", "")) for p in pairs]
    after = [_item(p["corrected_text"], p.get("lang", "")) for p in pairs]
    scorer(before)
    scorer(after)

    records = []
    for p, b, a in zip(pairs, before, after, strict=True):
        records.append(
            {
                "backend": p.get("backend", "unknown"),
                "source_text": p["source_text"],
                "corrected_text": p["corrected_text"],
                "changed": p["corrected_text"].strip() != p["source_text"].strip(),
                "score_before": b["score_raw"],
                "score_after": a["score_raw"],
                "delta": round(a["score_raw"] - b["score_raw"], 4),
                "categ_before": b["categ"],
                "categ_after": a["categ"],
            }
        )
    return records


def build_delta_report(records: list[dict], top_n: int = 20) -> dict:
    deltas = [r["delta"] for r in records]
    changed = [r for r in records if r["changed"]]
    n = len(records)
    median_delta = statistics.median(deltas) if deltas else 0.0

    transitions = Counter((r["categ_before"], r["categ_after"]) for r in records)
    per_backend: dict[str, dict] = {}
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_backend[r["backend"]].append(r)
    for backend, recs in by_backend.items():
        d = [r["delta"] for r in recs]
        per_backend[backend] = {
            "n": len(recs),
            "median_delta": round(statistics.median(d), 4) if d else 0.0,
            "mean_delta": round(statistics.fmean(d), 4) if d else 0.0,
            "share_improved": round(sum(1 for x in d if x > 0) / len(d), 4) if d else 0.0,
        }

    ranked = sorted(records, key=lambda r: r["delta"])
    return {
        "n": n,
        "n_changed": len(changed),
        "mean_delta": round(statistics.fmean(deltas), 4) if deltas else 0.0,
        "median_delta": round(median_delta, 4),
        "share_improved": round(sum(1 for d in deltas if d > 0) / n, 4) if n else 0.0,
        "share_degraded_gt_thresh": round(sum(1 for d in deltas if d < -DEGRADE_THRESHOLD) / n, 4) if n else 0.0,
        "band_transitions": {f"{a}->{b}": c for (a, b), c in sorted(transitions.items())},
        "per_backend": per_backend,
        "gate": {"median_delta": round(median_delta, 4), "passed": median_delta > 0},
        "worst_examples": ranked[:top_n],
        "best_examples": list(reversed(ranked[-top_n:])),
    }


def format_report(report: dict) -> str:
    lines = ["Correction score-delta report", "=" * 32, ""]
    lines.append(f"pairs={report['n']}  changed={report['n_changed']}")
    lines.append(f"mean Δ={report['mean_delta']}  median Δ={report['median_delta']}")
    lines.append(
        f"improved={report['share_improved']}  degraded(>{DEGRADE_THRESHOLD})={report['share_degraded_gt_thresh']}"
    )
    gate = report["gate"]
    lines.append(f"GATE: median Δ={gate['median_delta']}  -> {'PASS' if gate['passed'] else 'FAIL (exclude backend)'}")
    lines.append("")
    lines.append("Noisy→? band transitions:")
    for k, v in report["band_transitions"].items():
        lines.append(f"    {k:<20} {v}")
    lines.append("")
    lines.append("Per-backend:")
    for backend, s in report["per_backend"].items():
        lines.append(f"    {backend:<28} n={s['n']} median Δ={s['median_delta']} improved={s['share_improved']}")
    return "\n".join(lines)


def read_pairs(path: Path, source_col: str, corrected_col: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pairs = []
    for r in rows:
        src = (r.get(source_col) or "").strip()
        corr = (r.get(corrected_col) or "").strip()
        if not src or not corr:
            continue
        pairs.append(
            {
                "source_text": src,
                "corrected_text": corr,
                "lang": r.get("lang", ""),
                "backend": r.get("backend", "unknown"),
            }
        )
    return pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Report the algo score delta after auto-correction.")
    p.add_argument("--input", required=True, type=Path, help="CSV from correct.py (source_text + corrected_text).")
    p.add_argument("--source-col", default="source_text")
    p.add_argument("--corrected-col", default="corrected_text")
    p.add_argument("--scorer", choices=["offline", "model"], default="offline")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--fasttext", default="lid.176.bin")
    p.add_argument("--json", type=Path, default=None, help="Optional path to write the full report as JSON.")
    args = p.parse_args(argv)

    pairs = read_pairs(args.input, args.source_col, args.corrected_col)
    scorer = B.ModelScorer(args.model, args.fasttext) if args.scorer == "model" else B.make_offline_scorer()
    records = score_pairs(pairs, scorer)
    report = build_delta_report(records)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
