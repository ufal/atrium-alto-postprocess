#!/usr/bin/env python3
"""
run_pipeline.py — end-to-end ALTO XML postprocessing orchestrator.

Runs the repository's processing scripts sequentially on a directory of
document-level ALTO XMLs and, at the end, merges every per-stage paradata log
into ONE summary JSON.

Pipeline stages
---------------
  1. page_split.py            ALTO/            -> PAGE_ALTO/        (split into pages)
  2. alto_stats_create.py     PAGE_ALTO/       -> <stats>.csv       (page statistics)   [paradata]
  3. extract text             <stats>.csv      -> PAGE_TXT*/        (text extraction)   [paradata]
       --method alto-tools  -> extract_ALTO_2_TXT.py        (PAGE_TXT/,     Apache-2.0)
       --method layoutreader-> extract_LytRdr_ALTO_2_TXT.py (PAGE_TXT_LR/,  CC BY-NC-SA 4.0)  [default]
       --method glm         -> extract_LLM_ALTO_2_TXT.py    (PAGE_TXT_LLM/, glm-4)
  4. langID_classify.py       PAGE_TXT*/       -> DOC_LINE_CATEG/   (line classify)     [paradata]
  5. langID_aggregate_STAT.py DOC_LINE_CATEG/  -> DOC_LINE_STATS/   (page aggregate)    [paradata]

(#4) The classify stage reads its text input from the LANGID_TEXT_DIR env var,
which this orchestrator sets to the SELECTED method's output directory. Without
this, langID_classify always read the LayoutReader dir and silently ignored
alto-tools / glm output.

Configuration
-------------
Every setting is read from config_langID.txt. Precedence: CLI flag > config > default.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from atrium_paradata import merge_run_paradata

CONFIG_PATH = os.getenv("LANGID_CONFIG", "config_langID.txt")

# method -> (script, [EXTRACT] output-dir key, default output dir)
EXTRACT_METHODS = {
    "alto-tools":   ("extract_ALTO_2_TXT.py",        "OUTPUT_TXT",     "./data_samples/PAGE_TXT"),
    "layoutreader": ("extract_LytRdr_ALTO_2_TXT.py", "OUTPUT_TXT_LR",  "./data_samples/PAGE_TXT_LR"),
    "glm":          ("extract_LLM_ALTO_2_TXT.py",    "OUTPUT_TXT_LLM", "./data_samples/PAGE_TXT_LLM"),
}

_DEFAULTS = {
    "method":        "layoutreader",
    "input_dir":     "data_samples/ALTO",
    "page_alto_dir": "data_samples/PAGE_ALTO",
    "skip_split":    False,
    "paradata_dir":  "paradata",
    "input_csv":     "test_alto_stats.csv",
}


def _load_config(config_path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(config_path, encoding="utf-8")
    return cfg


def _cfg_get(cfg, section, key, default):
    if cfg.has_section(section):
        return cfg.get(section, key, fallback=default)
    return default


def _cfg_getbool(cfg, section, key, default):
    if cfg.has_section(section) and cfg.has_option(section, key):
        return cfg.getboolean(section, key)
    return default


def _resolve_extract_outdir(method: str, cfg: configparser.ConfigParser) -> str:
    """The text-output directory the chosen extraction method writes to.

    Used to point the classify stage at the right text source (#4).
    """
    _script, key, default = EXTRACT_METHODS[method]
    return (_cfg_get(cfg, "EXTRACT", key, default) or default).strip()


def resolve_settings(args, cfg: configparser.ConfigParser) -> Dict:
    method = (args.method or _cfg_get(cfg, "PIPELINE", "METHOD", _DEFAULTS["method"]))
    method = method.strip()
    if method not in EXTRACT_METHODS:
        raise SystemExit(
            f"Unknown extraction method '{method}'. "
            f"Choose one of: {', '.join(EXTRACT_METHODS)}."
        )

    input_dir = (args.input_dir
                 or _cfg_get(cfg, "PIPELINE", "INPUT_DIR", _DEFAULTS["input_dir"]))
    page_alto = (args.page_alto_dir
                 or _cfg_get(cfg, "PIPELINE", "PAGE_ALTO_DIR", _DEFAULTS["page_alto_dir"]))
    paradata_dir = (args.paradata_dir
                    or _cfg_get(cfg, "PIPELINE", "PARADATA_DIR", _DEFAULTS["paradata_dir"]))
    input_csv = (args.input_csv
                 or _cfg_get(cfg, "EXTRACT", "INPUT_CSV", _DEFAULTS["input_csv"]))
    skip_split = args.skip_split or _cfg_getbool(cfg, "PIPELINE", "SKIP_SPLIT",
                                                 _DEFAULTS["skip_split"])

    return {
        "method": method,
        "input_dir": input_dir.strip(),
        "page_alto_dir": page_alto.strip(),
        "paradata_dir": paradata_dir.strip(),
        "input_csv": input_csv.strip(),
        "skip_split": skip_split,
        "text_dir": _resolve_extract_outdir(method, cfg),
    }


def _snapshot(paradata_dir: Path) -> set:
    if not paradata_dir.exists():
        return set()
    return {p.name for p in paradata_dir.glob("*.json")}


def _run_stage(name: str, cmd: List[str], paradata_dir: Path) -> List[str]:
    """Run one stage as a subprocess; return NEW paradata JSON paths it produced."""
    print(f"\n{'='*78}\n> STAGE: {name}\n  $ {' '.join(cmd)}\n{'='*78}", flush=True)

    before = _snapshot(paradata_dir)
    time.sleep(1.1)  # run_id has 1-second resolution; avoid collisions
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {result.returncode}")

    after = _snapshot(paradata_dir)
    new = sorted(after - before)
    new_paths = [str(paradata_dir / n) for n in new]
    print(f"  -> paradata: {', '.join(new)}" if new_paths
          else "  -> (no paradata emitted by this stage)")
    return new_paths


def build_plan(settings: Dict, config_path: str) -> List[Dict]:
    py = sys.executable or "python3"
    extract_script = EXTRACT_METHODS[settings["method"]][0]

    plan: List[Dict] = []
    if not settings["skip_split"]:
        plan.append({
            "name": "1. page_split (ALTO -> PAGE_ALTO)",
            "cmd": [py, "page_split.py", settings["input_dir"], settings["page_alto_dir"]],
            "logged": False,
        })
    plan.append({
        "name": "2. alto_stats_create (PAGE_ALTO -> stats.csv)",
        "cmd": [py, "alto_stats_create.py", settings["page_alto_dir"],
                "-o", settings["input_csv"]],
        "logged": True,
    })
    plan.append({
        "name": f"3. extract text [{settings['method']}] (stats.csv -> {settings['text_dir']})",
        "cmd": [py, extract_script],
        "logged": True,
    })
    plan.append({
        "name": "4. langID_classify (PAGE_TXT* -> DOC_LINE_CATEG)",
        "cmd": [py, "langID_classify.py"],
        "logged": True,
    })
    plan.append({
        "name": "5. langID_aggregate_STAT (DOC_LINE_CATEG -> DOC_LINE_STATS)",
        "cmd": [py, "langID_aggregate_STAT.py", "--config", config_path],
        "logged": True,
    })
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=CONFIG_PATH,
                    help=f"Config file to read settings from (default: {CONFIG_PATH}).")
    ap.add_argument("--method", choices=list(EXTRACT_METHODS), default=None,
                    help="Override the extraction backend ([PIPELINE].METHOD; default layoutreader).")
    ap.add_argument("--input-dir", default=None,
                    help="Override [PIPELINE].INPUT_DIR (document-level ALTO XMLs).")
    ap.add_argument("--page-alto-dir", default=None,
                    help="Override [PIPELINE].PAGE_ALTO_DIR (per-page ALTO dir).")
    ap.add_argument("--input-csv", default=None,
                    help="Override [EXTRACT].INPUT_CSV (page-stats CSV).")
    ap.add_argument("--skip-split", action="store_true",
                    help="Force-skip page_split (also settable via [PIPELINE].SKIP_SPLIT).")
    ap.add_argument("--paradata-dir", default=None,
                    help="Override [PIPELINE].PARADATA_DIR.")
    ap.add_argument("--summary-out", default=None,
                    help="Path for the merged run summary "
                         "(default: <paradata-dir>/<run_id>_pipeline-run.json).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the resolved plan without running anything.")
    args = ap.parse_args()

    config_path = args.config

    cfg = _load_config(config_path)
    settings = resolve_settings(args, cfg)
    paradata_dir = Path(settings["paradata_dir"])
    plan = build_plan(settings, config_path)

    # (#4) Propagate config + the SELECTED method's text dir to every child stage.
    # extract_* and langID_classify read LANGID_CONFIG; langID_classify reads
    # LANGID_TEXT_DIR for its input text directory. Subprocesses inherit os.environ.
    os.environ["LANGID_CONFIG"] = config_path
    os.environ["LANGID_TEXT_DIR"] = settings["text_dir"]

    cfg_note = config_path if Path(config_path).exists() else f"{config_path} (missing - using defaults)"
    print(f"Config: {cfg_note}")
    print(f"Pipeline plan ({len(plan)} stages, extraction method='{settings['method']}'):")
    for stage in plan:
        tag = "[paradata]" if stage["logged"] else "[no log]  "
        print(f"  {tag} {stage['name']}")
    print(f"Resolved settings: input_dir={settings['input_dir']} "
          f"page_alto_dir={settings['page_alto_dir']} input_csv={settings['input_csv']} "
          f"text_dir={settings['text_dir']} "
          f"skip_split={settings['skip_split']} paradata_dir={settings['paradata_dir']}")

    if args.dry_run:
        print("\nDry run - nothing executed.")
        return 0

    collected: List[str] = []
    run_started = time.strftime("%y%m%d-%H%M%S")
    try:
        for stage in plan:
            collected.extend(_run_stage(stage["name"], stage["cmd"], paradata_dir))
    except RuntimeError as exc:
        print(f"\nx Pipeline aborted: {exc}", file=sys.stderr)
        if collected:
            print("  Merging paradata from completed stages before exiting...")
        else:
            return 1

    if not collected:
        print("\nNo paradata logs were produced; nothing to merge.")
        return 0

    summary_out = args.summary_out or str(paradata_dir / f"{run_started}_pipeline-run.json")
    merged = merge_run_paradata(
        json_paths=collected,
        out_path=summary_out,
        pipeline="alto-postprocess",
        method=settings["method"],
    )

    data = json.loads(Path(merged).read_text(encoding="utf-8"))
    print(f"\n{'='*78}\n> PIPELINE COMPLETE - merged {data['stage_count']} logged stage(s)")
    print(f"  Effective output license : {data['license']}  ({data['license_url']})")
    fmts = ", ".join(f"{k}x{v}" for k, v in data["intermediate_formats"].items()) or "-"
    print(f"  Intermediate formats     : {fmts}")
    print(f"  Total duration           : {data['total_duration_seconds']} s")
    print(f"  Run summary              : {merged}\n{'='*78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())