#!/usr/bin/env python3
"""
tools/importance_consensus.py
=============================
Cross-backend consensus tool for parameter importance.
Loads importance JSONs from different backend runs and identifies
robust parameters that consistently rank in the top-K.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def calculate_consensus(run_dirs: list[Path], top_k: int = 10) -> dict[str, Any]:
    importances = {}
    for d in run_dirs:
        p = d / "param_importance.json"
        if not p.exists():
            print(f"Skipping {d}: param_importance.json not found")
            continue
        importances[d.name] = json.loads(p.read_text(encoding="utf-8"))

    if not importances:
        raise ValueError("No valid importance data found.")

    # Aggregate parameters and their rankings
    param_ranks = {}
    backends = list(importances.keys())

    for backend, imp_dict in importances.items():
        sorted_params = sorted(imp_dict.keys(), key=lambda k: imp_dict[k], reverse=True)
        for rank, param in enumerate(sorted_params, start=1):
            if param not in param_ranks:
                param_ranks[param] = {}
            param_ranks[param][backend] = rank

    consensus = []
    for param, ranks in param_ranks.items():
        appearances = sum(1 for r in ranks.values() if r <= top_k)
        is_robust = appearances >= max(2, len(backends) // 2)
        avg_rank = sum(ranks.values()) / len(ranks) if len(ranks) == len(backends) else 999.0

        consensus.append(
            {
                "param": param,
                "top_k_appearances": appearances,
                "is_robust": is_robust,
                "average_rank": float(f"{avg_rank:.2f}"),
                "ranks": ranks,
            }
        )

    consensus.sort(key=lambda x: (-x["top_k_appearances"], x["average_rank"]))
    return {"consensus": consensus, "backends_evaluated": backends}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-backend consensus generator.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Sweep output directories")
    parser.add_argument("--top-k", type=int, default=10, help="Rank threshold for robustness")
    parser.add_argument("--out", type=Path, default=Path("importance_consensus.json"))
    args = parser.parse_args(argv)

    results = calculate_consensus(args.run_dirs, top_k=args.top_k)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nConsensus saved to {args.out}")
    print(f"{'Parameter':<35} | {'Robust'} | {'Avg Rank':<8} | {'Details'}")
    print("-" * 80)
    for c in results["consensus"]:
        if c["top_k_appearances"] > 0:
            robust = "YES" if c["is_robust"] else "NO "
            ranks_str = ", ".join([f"{b}: {r}" for b, r in c["ranks"].items()])
            print(f"{c['param']:<35} | {robust:<6} | {c['average_rank']:<8} | {ranks_str}")


if __name__ == "__main__":
    main()
