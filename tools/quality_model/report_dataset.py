"""Sanity report for a built quality-score dataset (issue #23, Phase 1).

Two checks from strategy §1.2 plus a distribution summary:

1. **Monotonicity** — for each source line, heavier corruption must not *raise* the
   relabelled score. Reported as pairwise concordance accuracy (a low number means
   a corruption op is inverted or the scorer is misbehaving).
2. **Realism** — compares synthetic Noisy/Trash variants against *real* Noisy/Trash
   lines on the noise-bearing features. Large gaps flag synthetic-vs-real
   distribution shift (the KS test is a Phase-later upgrade; this uses mean/median
   deltas so it stays dependency-free).
3. **Distribution** — row counts by split / provenance / category and a score
   histogram.

Run standalone::

    python tools/quality_model/report_dataset.py --input /tmp/dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Severity ordering for the monotonicity check (originals are severity 0).
_SEVERITY = {"none": 0, "light": 1, "medium": 2, "heavy": 3}

# Noise-bearing features compared between synthetic and real noisy lines.
REALISM_FEATURES = ["garbage_density", "word_weird", "vowel_ratio", "rot_ratio", "perplex"]

NOISY_TRASH = frozenset({"Noisy", "Trash"})


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Monotonicity
# ---------------------------------------------------------------------------


def monotonicity(rows: list[dict]) -> dict:
    """Pairwise concordance: within a source line, a heavier variant should score
    <= a lighter one. Returns counts + accuracy."""
    groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        sev = _SEVERITY.get(r.get("band", "none"), 0)
        groups[(r.get("source_doc"), r.get("source_line"))].append((sev, _f(r, "score_raw")))

    total = 0
    concordant = 0
    for members in groups.values():
        for i in range(len(members)):
            for j in range(len(members)):
                sev_a, score_a = members[i]
                sev_b, score_b = members[j]
                if sev_a < sev_b:  # a is lighter than b
                    total += 1
                    if score_a >= score_b - 1e-9:
                        concordant += 1
    accuracy = concordant / total if total else 1.0
    return {"pairs": total, "concordant": concordant, "accuracy": round(accuracy, 4)}


# ---------------------------------------------------------------------------
# 2. Realism
# ---------------------------------------------------------------------------


def realism(rows: list[dict], features: list[str] | None = None) -> dict:
    features = features or REALISM_FEATURES
    synthetic = [r for r in rows if r.get("provenance") == "corrupt" and r.get("categ") in NOISY_TRASH]
    real = [r for r in rows if r.get("provenance") == "original" and r.get("categ") in NOISY_TRASH]

    out: dict = {"n_synthetic": len(synthetic), "n_real": len(real), "features": {}}
    for feat in features:
        syn_vals = [_f(r, feat) for r in synthetic]
        real_vals = [_f(r, feat) for r in real]
        out["features"][feat] = {
            "synthetic_mean": round(statistics.fmean(syn_vals), 4) if syn_vals else None,
            "real_mean": round(statistics.fmean(real_vals), 4) if real_vals else None,
            "abs_mean_delta": (
                round(abs(statistics.fmean(syn_vals) - statistics.fmean(real_vals)), 4)
                if syn_vals and real_vals
                else None
            ),
        }
    return out


# ---------------------------------------------------------------------------
# 3. Distribution
# ---------------------------------------------------------------------------


def distribution(rows: list[dict], n_bins: int = 10) -> dict:
    hist = Counter()
    for r in rows:
        b = min(n_bins - 1, max(0, int(_f(r, "score_raw") * n_bins)))
        hist[b] += 1
    return {
        "total": len(rows),
        "by_split": dict(Counter(r.get("split") for r in rows)),
        "by_provenance": dict(Counter(r.get("provenance") for r in rows)),
        "by_category": dict(Counter(r.get("categ") for r in rows)),
        "score_histogram": {f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}": hist.get(b, 0) for b in range(n_bins)},
    }


def build_report(rows: list[dict]) -> dict:
    return {
        "monotonicity": monotonicity(rows),
        "realism": realism(rows),
        "distribution": distribution(rows),
    }


def format_report(report: dict) -> str:
    lines = ["Quality-score dataset report", "=" * 32, ""]
    mono = report["monotonicity"]
    lines.append(f"Monotonicity: accuracy={mono['accuracy']} ({mono['concordant']}/{mono['pairs']} pairs)")
    lines.append("")
    dist = report["distribution"]
    lines.append(f"Rows: {dist['total']}  splits={dist['by_split']}")
    lines.append(f"  provenance={dist['by_provenance']}")
    lines.append(f"  category={dist['by_category']}")
    lines.append("  score histogram:")
    for band, count in dist["score_histogram"].items():
        bar = "#" * min(40, count)
        lines.append(f"    {band}  {count:>5}  {bar}")
    lines.append("")
    rea = report["realism"]
    lines.append(f"Realism (synthetic n={rea['n_synthetic']} vs real n={rea['n_real']} Noisy/Trash):")
    for feat, stats in rea["features"].items():
        lines.append(
            f"    {feat:<16} synthetic={stats['synthetic_mean']} "
            f"real={stats['real_mean']} |Δ|={stats['abs_mean_delta']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Report monotonicity / realism / distribution of a built dataset.")
    p.add_argument("--input", required=True, type=Path, help="Dataset CSV from build_dataset.py.")
    p.add_argument("--json", type=Path, default=None, help="Optional path to also write the report as JSON.")
    args = p.parse_args(argv)

    rows = load(args.input)
    report = build_report(rows)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
