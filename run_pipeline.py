#!/usr/bin/env python3
"""
run_pipeline.py — end-to-end OCR-output postprocessing orchestrator.

Runs the repository's processing scripts sequentially on a directory of
input documents and, at the end, merges every per-stage paradata log into
ONE summary JSON. Supports three input formats, selected implicitly via
--method (each method is tagged with the format it consumes):

  ALTO XML (--method alto-tools|layoutreader|glm):
    1. page_split.py            ALTO/            -> PAGE_ALTO/        (split into pages)
    2. alto_stats_create.py     PAGE_ALTO/       -> <stats>.csv       (page statistics)   [paradata]
    3. extract text             <stats>.csv      -> PAGE_TXT*/        (text extraction)   [paradata]
         --method alto-tools  -> extract_ALTO_2_TXT.py        (PAGE_TXT/,     Apache-2.0)
         --method layoutreader-> extract_LytRdr_ALTO_2_TXT.py (PAGE_TXT_LR/,  CC BY-NC-SA 4.0)  [default]
         --method glm         -> extract_LLM_ALTO_2_TXT.py    (PAGE_TXT_LLM/, glm-4)

  Generic JSON (--method json-keys):
    1. page_split.py            JSON/            -> PAGE_JSON/        (split into pages)  (#31)
    2. json_stats_create.py     PAGE_JSON/       -> <stats>.csv       (page statistics)   [paradata]
    3. extract_JSON_2_TXT.py    <stats>.csv      -> PAGE_TXT_JSON/    (text extraction)   [paradata]

  Any other text-bearing file (--method text-lines, #31): PDF, DOCX, ODT, XLSX/ODS,
  PPTX/ODP, EPUB, RTF, HTML/hOCR, PAGE XML, TEI, ABBYY/DjVu XML, Tesseract TSV, XML,
  JSON/JSONL, CSV/TSV, Markdown, SRT/VTT, EML/MBOX, plain text, gzip/bz2/xz-wrapped
  files and ZIP bundles of page files — recognised by content (text_formats.py;
  docs/text_inputs.md). Input dir: --input-dir, else [PIPELINE].INPUT_DIR_TEXT:
    1. text_split.py            TEXT/            -> PAGE_TEXT/        (pages + reports)   [paradata]
    2. text_stats_create.py     PAGE_TEXT/       -> <stats>.csv       (page statistics)   [paradata]
    3. extract_TEXT_2_TXT.py    <stats>.csv      -> PAGE_TXT_TEXT/    (+ DOC_LINES_TEXT/ line tables) [paradata]

  4. classify_TEXT.py       PAGE_TXT*/       -> DOC_LINE_CATEG/   (line classify)     [paradata]
  5. aggregate_STAT.py DOC_LINE_CATEG/  -> DOC_LINE_STATS/   (page aggregate)    [paradata]

(#4) The classify stage reads its text input from the LANGID_TEXT_DIR env var,
which this orchestrator sets to the SELECTED method's output directory. Without
this, classify_TEXT always read the LayoutReader dir and silently ignored
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
Every setting is read from config.txt. Precedence: CLI flag > config > default.
For text-lines, --input-csv also reaches extract_TEXT_2_TXT.py (classify still reads
[CLASSIFY].INPUT_CSV), and --strict/--no-strict reach text_split.py and
extract_TEXT_2_TXT.py. --source-origin reaches the split stage of any method.

Usage
-----
  python3 run_pipeline.py                        # all settings from config ([PIPELINE].METHOD)
  python3 run_pipeline.py --method glm           # override just the extraction backend
  python3 run_pipeline.py --method json-keys --input-dir data_samples/JSON  # generic JSON input
  python3 run_pipeline.py --method text-lines    # PDF/DOCX/TXT/... from [PIPELINE].INPUT_DIR_TEXT (#31)
  python3 run_pipeline.py --skip-split           # PAGE_ALTO already populated
  python3 run_pipeline.py --skip-extract         # PAGE_TXT* already populated (avoids model load)
  python3 run_pipeline.py --start-from classify  # run classify + aggregate only
  python3 run_pipeline.py --dry-run              # print the resolved plan, run nothing
  python3 run_pipeline.py --document-json-dir ./docs  # enable paradata pair accretion
"""

from __future__ import annotations

import argparse
import configparser
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from atrium_document import canonical_doc_id
from atrium_paradata import merge_run_paradata

CONFIG_PATH = os.getenv("LANGID_CONFIG", "setup/config.txt")

# Canonical stage order; the keys also drive --skip-<key> / [PIPELINE].SKIP_<KEY>.
STAGE_ORDER = ["split", "stats", "extract", "classify", "aggregate"]

# method -> (script, [EXTRACT] output-dir key, default output dir, input format)
EXTRACT_METHODS = {
    "alto-tools": ("extract_ALTO_2_TXT.py", "OUTPUT_TXT", "./data_samples/PAGE_TXT", "alto"),
    "layoutreader": ("extract_LytRdr_ALTO_2_TXT.py", "OUTPUT_TXT_LR", "./data_samples/PAGE_TXT_LR", "alto"),
    "glm": ("extract_LLM_ALTO_2_TXT.py", "OUTPUT_TXT_LLM", "./data_samples/PAGE_TXT_LLM", "alto"),
    "json-keys": ("extract_JSON_2_TXT.py", "OUTPUT_TXT_JSON", "./data_samples/PAGE_TXT_JSON", "json"),
    "text-lines": ("extract_TEXT_2_TXT.py", "OUTPUT_TXT_TEXT", "./data_samples/PAGE_TXT_TEXT", "text"),
}

# input format -> its stage-1 split script, stage-2 stats-CSV builder, and the
# settings key of the per-page directory the split writes and the stats stage scans.
# (#31) Every format splits into pages: alto via page_split.split_alto_xml(), json
# via page_split.split_json_document(), text (any other text-bearing file) via
# text_split.py. build_plan() still accepts split_script=None (a permanent no-op
# stage), but no format uses it.
INPUT_FORMATS = {
    "alto": {"split_script": "page_split.py", "stats_script": "alto_stats_create.py", "page_dir": "page_alto_dir"},
    "json": {"split_script": "page_split.py", "stats_script": "json_stats_create.py", "page_dir": "page_json_dir"},
    "text": {"split_script": "text_split.py", "stats_script": "text_stats_create.py", "page_dir": "page_text_dir"},
}

_DEFAULTS = {
    "method": "layoutreader",
    "input_dir": "data_samples/ALTO",
    "input_dir_text": "data_samples/TEXT",
    "page_alto_dir": "data_samples/PAGE_ALTO",
    "page_json_dir": "data_samples/PAGE_JSON",
    "page_text_dir": "data_samples/PAGE_TEXT",
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
    _script, key, default, _fmt = EXTRACT_METHODS[method]
    return (_cfg_get(cfg, "EXTRACT", key, default) or default).strip()


def _page_dir_for(settings: Dict, fmt: str) -> str:
    """The per-page directory for input format `fmt` (INPUT_FORMATS[fmt]["page_dir"])."""
    return settings[INPUT_FORMATS[fmt]["page_dir"]]


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

    input_format = EXTRACT_METHODS[method][3]

    # Handle the JSON paradata pair document directory
    page_json = (
        getattr(args, "page_json_dir", None) or _cfg_get(cfg, "PIPELINE", "PAGE_JSON_DIR", _DEFAULTS["page_json_dir"])
    ).strip()

    # (#31) getattr(): unit tests build partial Namespaces without this flag.
    page_text = (
        getattr(args, "page_text_dir", None) or _cfg_get(cfg, "PIPELINE", "PAGE_TEXT_DIR", _DEFAULTS["page_text_dir"])
    ).strip()

    if input_format == "text":
        # (#31 Phase 4) text-lines has its own default input dir, so a bare
        # `--method text-lines` no longer ingests the ALTO samples; an older config
        # that only sets INPUT_DIR keeps working.
        input_dir = (
            args.input_dir
            or (_cfg_get(cfg, "PIPELINE", "INPUT_DIR_TEXT", "") or "").strip()
            or (_cfg_get(cfg, "PIPELINE", "INPUT_DIR", "") or "").strip()
            or _DEFAULTS["input_dir_text"]
        ).strip()
    else:
        input_dir = (args.input_dir or _cfg_get(cfg, "PIPELINE", "INPUT_DIR", _DEFAULTS["input_dir"])).strip()
    page_alto = (args.page_alto_dir or _cfg_get(cfg, "PIPELINE", "PAGE_ALTO_DIR", _DEFAULTS["page_alto_dir"])).strip()
    # page_json = (
    #     getattr(args, "page_json_dir", None) or _cfg_get(cfg, "PIPELINE", "PAGE_JSON_DIR", _DEFAULTS["page_json_dir"])
    # ).strip()
    paradata_dir = (args.paradata_dir or _cfg_get(cfg, "PIPELINE", "PARADATA_DIR", _DEFAULTS["paradata_dir"])).strip()

    # Handle the JSON paradata pair document directory
    document_json_dir = (
        getattr(args, "document_json_dir", None) or _cfg_get(cfg, "PIPELINE", "DOCUMENT_JSON_DIR", "")
    ).strip()

    input_csv = (args.input_csv or _cfg_get(cfg, "EXTRACT", "INPUT_CSV", _DEFAULTS["input_csv"])).strip()
    text_dir = _resolve_extract_outdir(method, cfg)
    categ_dir = (
        _cfg_get(cfg, "CLASSIFY", "OUTPUT_LINES_LOG", _DEFAULTS["categ_dir"]) or _DEFAULTS["categ_dir"]
    ).strip()
    stats_dir = (_cfg_get(cfg, "AGGREGATE", "OUTPUT_DOC_DIR", _DEFAULTS["stats_dir"]) or _DEFAULTS["stats_dir"]).strip()

    # (#31/D8) The per-page-output dir for the SELECTED format — page_alto_dir
    # for "alto", page_json_dir for "json", page_text_dir for "text". Every format
    # splits into pages, so the stats-CSV builder scans this dir uniformly.
    page_dir = _page_dir_for(
        {"page_alto_dir": page_alto, "page_json_dir": page_json, "page_text_dir": page_text}, input_format
    )
    stats_scan_dir = page_dir

    skip = _resolve_skips(args, cfg)

    return {
        "method": method,
        "input_format": input_format,
        "input_dir": input_dir,
        "page_alto_dir": page_alto,
        "page_json_dir": page_json,
        "page_text_dir": page_text,
        "stats_scan_dir": stats_scan_dir,
        "paradata_dir": paradata_dir,
        "document_json_dir": document_json_dir,
        "input_csv": input_csv,
        "text_dir": text_dir,
        "skip": skip,
        # Back-compat: callers/tests that read settings["skip_split"] still work.
        "skip_split": skip["split"],
        "start_from": getattr(args, "start_from", None),
        # (#31 Phase 4) pass-throughs: None = not given (the stage uses its config).
        "strict": getattr(args, "strict", None),
        "source_origin": (getattr(args, "source_origin", None) or "").strip() or None,
        # Resolved output location per stage (used for the pre-flight existence check).
        # "split" is the per-format page dir for every format (#31); None
        # remains supported here for a format with no split-stage concept.
        "outputs": {
            "split": page_dir,
            "stats": input_csv,
            "extract": text_dir,
            "classify": categ_dir,
            "aggregate": stats_dir,
        },
    }


def input_csv_mismatches(
    cli_input_csv: Optional[str], cfg: configparser.ConfigParser, sections=("EXTRACT", "CLASSIFY")
) -> List[str]:
    """Warnings for each config INPUT_CSV that a --input-csv override does NOT reach
    (text-lines passes it to its extract stage, so only [CLASSIFY] is checked there)."""
    if not cli_input_csv:
        return []
    out = []
    for section in sections:
        configured = (_cfg_get(cfg, section, "INPUT_CSV", "") or "").strip()
        if configured and os.path.abspath(configured) != os.path.abspath(cli_input_csv.strip()):
            out.append(
                f"--input-csv {cli_input_csv} is written by the stats stage, but [{section}].INPUT_CSV "
                f"is {configured} and that is what the {section.lower()} stage reads — set it in the config too."
            )
    return out


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


def _single_input_doc_id(input_dir: str, input_format: str) -> Optional[str]:
    """Doc id of the lone file in input_dir, or None if it's not exactly one file.

    (atrium-project#10 D3) Delegates to the hub's `canonical_doc_id()`, which is what
    page_split.py's `_doc_id_from_filename` now calls too. This used to hand-roll
    `splitext()` + `.replace(".alto", "")` on the stated grounds of "kept local to
    avoid coupling run_pipeline.py to that module's internals" — a reasonable-sounding
    argument that produced the coupling it was avoiding, by convention instead of by
    import: the orchestrator seeds the bridge directory under the doc_id it derives
    here, and page_split then writes its record under the doc_id it derives there. Two
    copies of one convention silently fork on any multi-dot name, and the whole record
    for that document is orphaned. Sharing the hub's single derivation is the coupling
    that is actually safe, because it is the contract both stages are held to.
    """
    if input_format == "text":
        # (#31) Any file text_split.py would read — the same candidate filter
        # (regular, non-hidden, non-lock files), so both derive the same doc_id.
        from text_split import iter_candidate_files

        try:
            matches = [path for _name, path in iter_candidate_files(input_dir)]
        except OSError:
            return None
        return canonical_doc_id(matches[0]) if len(matches) == 1 else None
    pattern = "*.xml" if input_format == "alto" else "*.json"
    matches = sorted(glob.glob(str(Path(input_dir) / pattern)))
    if len(matches) != 1:
        return None
    return canonical_doc_id(matches[0])


def _prepare_document_json_bridge(document_json: Optional[str], doc_id: Optional[str]) -> Optional[Path]:
    """Seed a scratch directory for the existing --document-json-dir/DOCUMENT_JSON_DIR
    mechanism that every stage script already honors, so single-file --document-json
    can reuse it instead of duplicating the accretion logic."""
    if not doc_id:
        print(
            "[document] --document-json/-out require --input-dir to contain exactly one "
            "input file — skipping the document record for this run",
            file=sys.stderr,
        )
        return None
    scratch_dir = Path(tempfile.mkdtemp(prefix="atrium_document_json_"))
    if document_json:
        baseline = Path(document_json)
        if not baseline.exists():
            print(f"[document] baseline {baseline} not found — emitting this repo's own part only", file=sys.stderr)
        else:
            shutil.copyfile(baseline, scratch_dir / f"{doc_id}.document.json")
    return scratch_dir


def _collect_document_json_output(scratch_dir: Path, doc_id: str, document_json_out: str) -> None:
    record = scratch_dir / f"{doc_id}.document.json"
    if not record.exists():
        print(
            f"[document] no document record was produced in {scratch_dir} — {document_json_out} was NOT written",
            file=sys.stderr,
        )
        return
    out_path = Path(document_json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record, out_path)
    print(f"[document] Record written -> {out_path}", flush=True)


def build_plan(settings: Dict, config_path: str) -> List[Dict]:
    """All five stages in order, each tagged with its skip flag (no filtering)."""
    py = sys.executable or "python3"
    extract_script = EXTRACT_METHODS[settings["method"]][0]
    fmt = settings["input_format"]
    split_script = INPUT_FORMATS[fmt]["split_script"]
    stats_script = INPUT_FORMATS[fmt]["stats_script"]

    strict = settings.get("strict")
    strict_flag = [] if strict is None or fmt != "text" else ["--strict" if strict else "--no-strict"]
    origin_flag = ["--source-origin", settings["source_origin"]] if settings.get("source_origin") else []

    if split_script:
        page_out_dir = _page_dir_for(settings, fmt)
        split_stage = {
            "key": "split",
            "name": f"1. {split_script} ({fmt} -> {page_out_dir})",
            "cmd": [py, split_script, settings["input_dir"], page_out_dir] + origin_flag + strict_flag,
            # text_split.py always emits paradata; page_split's tag is left as it was.
            "logged": fmt == "text",
        }
    else:
        split_stage = {
            "key": "split",
            "name": f"1. page_split (not applicable for format '{fmt}')",
            "cmd": None,
            "logged": False,
        }

    stages: List[Dict] = [
        split_stage,
        {
            "key": "stats",
            "name": f"2. {stats_script} ({settings['stats_scan_dir']} -> stats.csv)",
            "cmd": [py, stats_script, settings["stats_scan_dir"], "-o", settings["input_csv"]],
            "logged": True,
        },
        {
            "key": "extract",
            "name": f"3. extract text [{settings['method']}] (stats.csv -> {settings['text_dir']})",
            # (#31 Phase 4) extract_TEXT_2_TXT.py takes the stats CSV as an argument, so
            # --input-csv reaches it; the ALTO/JSON extractors read the config only.
            "cmd": [py, extract_script]
            + (["--input-csv", settings["input_csv"]] + strict_flag if fmt == "text" else []),
            "logged": True,
        },
        {
            "key": "classify",
            "name": "4. classify_TEXT (PAGE_TXT* -> DOC_LINE_CATEG)",
            "cmd": [py, "classify_TEXT.py"],
            "logged": True,
        },
        {
            "key": "aggregate",
            "name": "5. aggregate_STAT (DOC_LINE_CATEG -> DOC_LINE_STATS)",
            "cmd": [py, "aggregate_STAT.py", "--config", config_path],
            "logged": True,
        },
    ]
    for st in stages:
        # A stage with cmd=None has nothing to run regardless of --skip-* flags.
        st["skip"] = st["cmd"] is None or settings["skip"][st["key"]]
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
    ap.add_argument(
        "--input-dir",
        default=None,
        help="Override [PIPELINE].INPUT_DIR (document-level ALTO XMLs, or JSON files for --method json-keys) or, "
        "for --method text-lines, [PIPELINE].INPUT_DIR_TEXT (any text-bearing files — PDF, DOCX, TXT, ...).",
    )
    ap.add_argument("--page-alto-dir", default=None, help="Override [PIPELINE].PAGE_ALTO_DIR (per-page ALTO dir).")
    ap.add_argument("--page-json-dir", default=None, help="Override [PIPELINE].PAGE_JSON_DIR (per-page JSON dir, #31).")
    ap.add_argument(
        "--page-text-dir",
        default=None,
        help="Override [PIPELINE].PAGE_TEXT_DIR (per-page text dir written by text_split.py, --method text-lines, #31).",
    )
    ap.add_argument("--input-csv", default=None, help="Override [EXTRACT].INPUT_CSV (page-stats CSV).")
    ap.add_argument("--paradata-dir", default=None, help="Override [PIPELINE].PARADATA_DIR.")
    ap.add_argument(
        "--document-json-dir",
        default=None,
        help="Directory containing baseline AtriumDocument JSON files for accretion (enables paradata pair).",
    )

    # --- Stage skipping / starting points (#6) ---
    ap.add_argument(
        "--start-from",
        choices=STAGE_ORDER,
        default=None,
        help="Run from this stage onward; skip every earlier stage (e.g. 'classify').",
    )
    ap.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip stage 1, the page split (page_split.py; text_split.py for text-lines; also [PIPELINE].SKIP_SPLIT)."
        " PAGE_ALTO / PAGE_JSON / PAGE_TEXT must be ready.",
    )
    ap.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip stage 2 (alto_/json_/text_stats_create.py; also [PIPELINE].SKIP_STATS). The stats CSV must be ready.",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip text extraction (also [PIPELINE].SKIP_EXTRACT). Main use: PAGE_TXT* ready; avoids model load.",
    )
    ap.add_argument("--skip-classify", action="store_true", help="Skip classify_TEXT (also [PIPELINE].SKIP_CLASSIFY).")
    ap.add_argument(
        "--skip-aggregate", action="store_true", help="Skip aggregate_STAT (also [PIPELINE].SKIP_AGGREGATE)."
    )

    ap.add_argument(
        "--summary-out",
        default=None,
        help="Path for the merged run summary (default: <paradata-dir>/<run_id>_pipeline-run.json).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the resolved plan without running anything.")
    ap.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="text-lines only: passed to text_split.py and extract_TEXT_2_TXT.py, which then exit non-zero (and "
        "stop the pipeline) when a file failed or was read partially (default: [TEXT_INGEST].STRICT).",
    )
    ap.add_argument(
        "--source-origin",
        default=None,
        help="Passed to the split stage (page_split.py / text_split.py): the source.origin recorded for every "
        "input of this run, e.g. ocr:pero.",
    )
    ap.add_argument(
        "--document-json",
        type=str,
        default=None,
        help="Single-file convenience form of --document-json-dir (issue #13): baseline "
        "ATRIUM Document JSON for a ONE-document run (--input-dir must contain exactly "
        "one input file). Seeds a scratch --document-json-dir internally.",
    )
    ap.add_argument(
        "--document-json-out",
        type=str,
        default=None,
        help="Exact path to write the updated ATRIUM Document JSON. Pairs with --document-json "
        "or with --input-dir pointed at a single file.",
    )
    args = ap.parse_args()

    config_path = args.config

    cfg = _load_config(config_path)
    settings = resolve_settings(args, cfg)
    paradata_dir = Path(settings["paradata_dir"])
    plan = build_plan(settings, config_path)

    # Single-file --document-json/-out: seed a scratch dir and route it through the
    # same document_json_dir mechanism every stage script already honors, rather than
    # duplicating the accretion logic here.
    doc_json_scratch_dir: Optional[Path] = None
    doc_json_doc_id: Optional[str] = None
    if args.document_json or args.document_json_out:
        doc_json_doc_id = _single_input_doc_id(settings["input_dir"], settings["input_format"])
        doc_json_scratch_dir = _prepare_document_json_bridge(args.document_json, doc_json_doc_id)
        if doc_json_scratch_dir is not None:
            settings["document_json_dir"] = str(doc_json_scratch_dir)

    # (#4) Propagate config + the SELECTED method's text dir to every child stage.
    # extract_* and classify_TEXT read LANGID_CONFIG; classify_TEXT reads
    # LANGID_TEXT_DIR for its input text directory. Subprocesses inherit os.environ.
    os.environ["LANGID_CONFIG"] = config_path
    os.environ["LANGID_TEXT_DIR"] = settings["text_dir"]

    # Document JSON directory propagation for accretion hook support
    if settings.get("document_json_dir"):
        doc_dir_path = Path(settings["document_json_dir"]).resolve()
        doc_dir_path.mkdir(parents=True, exist_ok=True)
        os.environ["DOCUMENT_JSON_DIR"] = str(doc_dir_path)

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
        f"page_alto_dir={settings['page_alto_dir']} page_json_dir={settings['page_json_dir']} "
        f"input_csv={settings['input_csv']} "
        f"text_dir={settings['text_dir']} paradata_dir={settings['paradata_dir']} "
        f"document_json_dir={settings['document_json_dir']}"
        + (f" page_text_dir={settings['page_text_dir']}" if settings["input_format"] == "text" else "")
    )

    # (#31) --input-csv only redirects the stats stage's `-o`: the extract scripts and
    # classify_TEXT read [EXTRACT].INPUT_CSV / [CLASSIFY].INPUT_CSV from the config file
    # themselves (run_pipeline passes them no arguments). Say so when the two disagree —
    # otherwise the later stages silently process whatever CSV the config names.
    sections = ("CLASSIFY",) if settings["input_format"] == "text" else ("EXTRACT", "CLASSIFY")
    for warning in input_csv_mismatches(args.input_csv, cfg, sections):
        print(f"  ! WARNING: {warning}", file=sys.stderr)
    if args.strict is not None and settings["input_format"] != "text":
        print(f"  ! WARNING: --strict/--no-strict only applies to --method text-lines; ignored for {settings['method']}.",
              file=sys.stderr)  # fmt: skip

    # Pre-flight: a skipped stage's output must already exist for downstream stages.
    # A None output (e.g. "split" for formats with no split stage) has nothing
    # to check — it's a permanent no-op, not a stage the user chose to skip.
    for st in plan:
        stage_output = settings["outputs"][st["key"]]
        if st["skip"] and stage_output is not None and not _output_present(stage_output):
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

    if doc_json_scratch_dir is not None and args.document_json_out:
        _collect_document_json_output(doc_json_scratch_dir, doc_json_doc_id, args.document_json_out)

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
