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

Stage skipping (#6)
-------------------
Any stage can be skipped with --skip-<stage> where <stage> is one of
split|stats|extract|classify|aggregate (each also settable as [PIPELINE].SKIP_<STAGE>).
--start-from <stage> is a convenience that skips every EARLIER stage. A skipped
stage's outputs must already exist on disk; run_pipeline prints a non-fatal
warning if they are missing. Because a skipped stage emits no paradata, the merged
run summary lists `skipped_stages` and its license / intermediate_formats reflect
only the stages that actually ran.

Configuration
-------------
Every setting is read from config_langID.txt. Precedence: CLI flag > config > default.

Usage
-----
  python3 run_pipeline.py                        # all settings from config ([PIPELINE].METHOD)
  python3 run_pipeline.py --method glm           # override just the extraction backend
  python3 run_pipeline.py --skip-split           # PAGE_ALTO already populated
  python3 run_pipeline.py --skip-extract         # PAGE_TXT* already populated (avoids model load)
  python3 run_pipeline.py --start-from classify  # run classify + aggregate only
  python3 run_pipeline.py --dry-run              # print the resolved plan, run nothing
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
from typing import Dict, List

from atrium_paradata import merge_run_paradata

CONFIG_PATH = os.getenv("LANGID_CONFIG", "setup/config_langID.txt")

# Canonical stage order; the keys also drive --skip-<key> / [PIPELINE].SKIP_<KEY>.
STAGE_ORDER = ["split", "stats", "extract", "classify", "aggregate"]

# method -> (script, [EXTRACT] output-dir key, default output dir)
EXTRACT_METHODS = {
    "alto-tools": ("extract_ALTO_2_TXT.py", "OUTPUT_TXT", "./data_samples/PAGE_TXT"),
    "layoutreader": ("extract_LytRdr_ALTO_2_TXT.py", "OUTPUT_TXT_LR", "./data_samples/PAGE_TXT_LR"),
    "glm": ("extract_LLM_ALTO_2_TXT.py", "OUTPUT_TXT_LLM", "./data_samples/PAGE_TXT_LLM"),
}

_DEFAULTS = {
    "method": "layoutreader",
    "input_dir": "data_samples/ALTO",
    "page_alto_dir": "data_samples/PAGE_ALTO",
    "skip_split": False,
    "paradata_dir": "paradata",
    "input_csv": "test_alto_stats.csv",
    "categ_dir": "data_samples/DOC_LINE_CATEG",
    "stats_dir": "data_samples/DOC_LINE_STATS",
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

    Used to point the classify stage at the right text source (#4) and to check
    the extract stage's output when it is skipped (#6).
    """
    _script, key, default = EXTRACT_METHODS[method]
    return (_cfg_get(cfg, "EXTRACT", key, default) or default).strip()


def _resolve_skips(args, cfg: configparser.ConfigParser) -> Dict[str, bool]:
    """Per-stage skip map: CLI --skip-<stage> OR [PIPELINE].SKIP_<STAGE>.

    --start-from <stage> additionally forces every EARLIER stage to be skipped.
    getattr() keeps this robust to partial argparse Namespaces used in unit tests.
    """
    skip = {
        s: bool(getattr(args, f"skip_{s}", False)) or _cfg_getbool(cfg, "PIPELINE", f"SKIP_{s.upper()}", False)
        for s in STAGE_ORDER
    }
    start_from = getattr(args, "start_from", None)
    if start_from:
        for s in STAGE_ORDER[: STAGE_ORDER.index(start_from)]:
            skip[s] = True
    return skip


def resolve_settings(args, cfg: configparser.ConfigParser) -> Dict:
    method = args.method or _cfg_get(cfg, "PIPELINE", "METHOD", _DEFAULTS["method"])
    method = method.strip()
    if method not in EXTRACT_METHODS:
        raise SystemExit(f"Unknown extraction method '{method}'. Choose one of: {', '.join(EXTRACT_METHODS)}.")

    input_dir = (args.input_dir or _cfg_get(cfg, "PIPELINE", "INPUT_DIR", _DEFAULTS["input_dir"])).strip()
    page_alto = (args.page_alto_dir or _cfg_get(cfg, "PIPELINE", "PAGE_ALTO_DIR", _DEFAULTS["page_alto_dir"])).strip()
    paradata_dir = (args.paradata_dir or _cfg_get(cfg, "PIPELINE", "PARADATA_DIR", _DEFAULTS["paradata_dir"])).strip()
    input_csv = (args.input_csv or _cfg_get(cfg, "EXTRACT", "INPUT_CSV", _DEFAULTS["input_csv"])).strip()
    text_dir = _resolve_extract_outdir(method, cfg)
    categ_dir = (
        _cfg_get(cfg, "CLASSIFY", "OUTPUT_LINES_LOG", _DEFAULTS["categ_dir"]) or _DEFAULTS["categ_dir"]
    ).strip()
    stats_dir = (_cfg_get(cfg, "AGGREGATE", "OUTPUT_DOC_DIR", _DEFAULTS["stats_dir"]) or _DEFAULTS["stats_dir"]).strip()

    skip = _resolve_skips(args, cfg)

    return {
        "method": method,
        "input_dir": input_dir,
        "page_alto_dir": page_alto,
        "paradata_dir": paradata_dir,
        "input_csv": input_csv,
        "text_dir": text_dir,
        "skip": skip,
        # Back-compat: callers/tests that read settings["skip_split"] still work.
        "skip_split": skip["split"],
        "start_from": getattr(args, "start_from", None),
        # Resolved output location per stage (used for the pre-flight existence check).
        "outputs": {
            "split": page_alto,
            "stats": input_csv,
            "extract": text_dir,
            "classify": categ_dir,
            "aggregate": stats_dir,
        },
    }


def _snapshot(paradata_dir: Path) -> set:
    if not paradata_dir.exists():
        return set()
    return {p.name for p in paradata_dir.glob("*.json")}


def _output_present(path: str) -> bool:
    """True if a stage output already exists: a non-empty file or non-empty dir."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        return any(p.iterdir())
    return p.stat().st_size > 0


def _run_stage(name: str, cmd: List[str], paradata_dir: Path) -> List[str]:
    """Run one stage as a subprocess; return NEW paradata JSON paths it produced."""
    print(f"\n{'=' * 78}\n> STAGE: {name}\n  $ {' '.join(cmd)}\n{'=' * 78}", flush=True)

    before = _snapshot(paradata_dir)
    time.sleep(1.1)  # run_id has 1-second resolution; avoid collisions
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {result.returncode}")

    after = _snapshot(paradata_dir)
    new = sorted(after - before)
    new_paths = [str(paradata_dir / n) for n in new]
    print(f"  -> paradata: {', '.join(new)}" if new_paths else "  -> (no paradata emitted by this stage)")
    return new_paths


def build_plan(settings: Dict, config_path: str) -> List[Dict]:
    """All five stages in order, each tagged with its skip flag (no filtering)."""
    py = sys.executable or "python3"
    extract_script = EXTRACT_METHODS[settings["method"]][0]

    stages: List[Dict] = [
        {
            "key": "split",
            "name": "1. page_split (ALTO -> PAGE_ALTO)",
            "cmd": [py, "page_split.py", settings["input_dir"], settings["page_alto_dir"]],
            "logged": False,
        },
        {
            "key": "stats",
            "name": "2. alto_stats_create (PAGE_ALTO -> stats.csv)",
            "cmd": [py, "alto_stats_create.py", settings["page_alto_dir"], "-o", settings["input_csv"]],
            "logged": True,
        },
        {
            "key": "extract",
            "name": f"3. extract text [{settings['method']}] (stats.csv -> {settings['text_dir']})",
            "cmd": [py, extract_script],
            "logged": True,
        },
        {
            "key": "classify",
            "name": "4. langID_classify (PAGE_TXT* -> DOC_LINE_CATEG)",
            "cmd": [py, "langID_classify.py"],
            "logged": True,
        },
        {
            "key": "aggregate",
            "name": "5. langID_aggregate_STAT (DOC_LINE_CATEG -> DOC_LINE_STATS)",
            "cmd": [py, "langID_aggregate_STAT.py", "--config", config_path],
            "logged": True,
        },
    ]
    for st in stages:
        st["skip"] = settings["skip"][st["key"]]
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config", default=CONFIG_PATH, help=f"Config file to read settings from (default: {CONFIG_PATH})."
    )
    ap.add_argument(
        "--method",
        choices=list(EXTRACT_METHODS),
        default=None,
        help="Override the extraction backend ([PIPELINE].METHOD; default layoutreader).",
    )
    ap.add_argument("--input-dir", default=None, help="Override [PIPELINE].INPUT_DIR (document-level ALTO XMLs).")
    ap.add_argument("--page-alto-dir", default=None, help="Override [PIPELINE].PAGE_ALTO_DIR (per-page ALTO dir).")
    ap.add_argument("--input-csv", default=None, help="Override [EXTRACT].INPUT_CSV (page-stats CSV).")
    ap.add_argument("--paradata-dir", default=None, help="Override [PIPELINE].PARADATA_DIR.")

    # --- Stage skipping / starting points (#6) ---
    ap.add_argument(
        "--start-from",
        choices=STAGE_ORDER,
        default=None,
        help="Run from this stage onward; skip every earlier stage (e.g. 'classify').",
    )
    ap.add_argument(
        "--skip-split", action="store_true", help="Skip page_split (also [PIPELINE].SKIP_SPLIT). PAGE_ALTO ready."
    )
    ap.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip alto_stats_create (also [PIPELINE].SKIP_STATS). stats CSV ready.",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip text extraction (also [PIPELINE].SKIP_EXTRACT). Main use: PAGE_TXT* ready; avoids model load.",
    )
    ap.add_argument(
        "--skip-classify", action="store_true", help="Skip langID_classify (also [PIPELINE].SKIP_CLASSIFY)."
    )
    ap.add_argument(
        "--skip-aggregate", action="store_true", help="Skip langID_aggregate_STAT (also [PIPELINE].SKIP_AGGREGATE)."
    )

    ap.add_argument(
        "--summary-out",
        default=None,
        help="Path for the merged run summary (default: <paradata-dir>/<run_id>_pipeline-run.json).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the resolved plan without running anything.")
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

    run_count = sum(1 for st in plan if not st["skip"])
    skip_count = len(plan) - run_count

    cfg_note = config_path if Path(config_path).exists() else f"{config_path} (missing - using defaults)"
    print(f"Config: {cfg_note}")
    start_note = f", start-from='{settings['start_from']}'" if settings["start_from"] else ""
    print(
        f"Pipeline plan ({run_count} to run, {skip_count} skipped, "
        f"extraction method='{settings['method']}'{start_note}):"
    )
    for st in plan:
        run_tag = "[skip]" if st["skip"] else "[run] "
        log_tag = "[paradata]" if st["logged"] else "[no log]  "
        print(f"  {run_tag} {log_tag} {st['name']}")
    print(
        f"Resolved settings: input_dir={settings['input_dir']} "
        f"page_alto_dir={settings['page_alto_dir']} input_csv={settings['input_csv']} "
        f"text_dir={settings['text_dir']} paradata_dir={settings['paradata_dir']}"
    )

    # Pre-flight: a skipped stage's output must already exist for downstream stages.
    for st in plan:
        if st["skip"] and not _output_present(settings["outputs"][st["key"]]):
            print(
                f"  ! WARNING: stage '{st['key']}' is skipped but its output "
                f"'{settings['outputs'][st['key']]}' is missing/empty; later stages may fail.",
                file=sys.stderr,
            )

    if args.dry_run:
        print("\nDry run - nothing executed.")
        return 0

    collected: List[str] = []
    skipped_names = [st["name"] for st in plan if st["skip"]]
    run_started = time.strftime("%y%m%d-%H%M%S")
    try:
        for st in plan:
            if st["skip"]:
                print(f"\n-- SKIPPED: {st['name']}")
                continue
            collected.extend(_run_stage(st["name"], st["cmd"], paradata_dir))
    except RuntimeError as exc:
        print(f"\nx Pipeline aborted: {exc}", file=sys.stderr)
        if collected:
            print("  Merging paradata from completed stages before exiting...")
        else:
            return 1

    if not collected:
        print("\nNo paradata logs were produced; nothing to merge.")
        if skipped_names:
            print(f"  Skipped stages: {', '.join(skipped_names)}")
        return 0

    summary_out = args.summary_out or str(paradata_dir / f"{run_started}_pipeline-run.json")
    merged = merge_run_paradata(
        json_paths=collected,
        out_path=summary_out,
        pipeline="alto-postprocess",
        method=settings["method"],
        skipped_stages=skipped_names,
    )

    data = json.loads(Path(merged).read_text(encoding="utf-8"))
    print(f"\n{'=' * 78}\n> PIPELINE COMPLETE - merged {data['stage_count']} logged stage(s)")
    print(f"  Effective output license : {data['license']}  ({data['license_url']})")
    fmts = ", ".join(f"{k}x{v}" for k, v in data["intermediate_formats"].items()) or "-"
    print(f"  Intermediate formats     : {fmts}")
    print(f"  Total duration           : {data['total_duration_seconds']} s")
    if skipped_names:
        print(f"  Skipped stages           : {', '.join(skipped_names)}")
        print("  NOTE: license/formats above reflect EXECUTED stages only.")
    print(f"  Run summary              : {merged}\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
