"""
Tests for the run_pipeline.py orchestrator configuration precedence.
"""

import configparser
import json
from argparse import Namespace

from atrium_paradata import merge_run_paradata
from run_pipeline import STAGE_ORDER, _resolve_extract_outdir, build_plan, resolve_settings


def _args(**over):
    """A fully-populated argparse.Namespace with every flag defaulted to None/False."""
    base = dict(
        method=None,
        input_dir=None,
        page_alto_dir=None,
        input_csv=None,
        paradata_dir=None,
        skip_split=False,
        skip_stats=False,
        skip_extract=False,
        skip_classify=False,
        skip_aggregate=False,
        start_from=None,
    )
    base.update(over)
    return Namespace(**base)


def test_skip_flag_sets_single_stage():
    settings = resolve_settings(_args(skip_extract=True), configparser.ConfigParser())
    assert settings["skip"]["extract"] is True
    assert settings["skip"]["split"] is False
    assert settings["skip"]["classify"] is False


def test_skip_config_fallback():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"PIPELINE": {"SKIP_EXTRACT": "true", "SKIP_AGGREGATE": "true"}})
    settings = resolve_settings(_args(), cfg)
    assert settings["skip"]["extract"] is True
    assert settings["skip"]["aggregate"] is True
    assert settings["skip"]["stats"] is False


def test_skip_cli_overrides_config():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"PIPELINE": {"SKIP_CLASSIFY": "false"}})
    settings = resolve_settings(_args(skip_classify=True), cfg)
    assert settings["skip"]["classify"] is True


def test_start_from_skips_earlier_stages():
    settings = resolve_settings(_args(start_from="extract"), configparser.ConfigParser())
    assert settings["skip"]["split"] is True
    assert settings["skip"]["stats"] is True
    assert settings["skip"]["extract"] is False
    assert settings["skip"]["classify"] is False
    assert settings["skip"]["aggregate"] is False


def test_outputs_resolved_from_config_and_defaults():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"CLASSIFY": {"OUTPUT_LINES_LOG": "cfg/categ"}})
    settings = resolve_settings(_args(), cfg)
    assert settings["outputs"]["classify"] == "cfg/categ"
    assert settings["outputs"]["aggregate"] == "data_samples/DOC_LINE_STATS"


def test_build_plan_returns_all_stages_with_skip_flags():
    settings = resolve_settings(_args(start_from="classify"), configparser.ConfigParser())
    plan = build_plan(settings, "config_langID.txt")
    assert [s["key"] for s in plan] == STAGE_ORDER
    assert [s["key"] for s in plan if not s["skip"]] == ["classify", "aggregate"]


def test_merge_paradata_records_skipped_stages(tmp_path):
    stage_json = tmp_path / "stage.json"
    stage_json.write_text(
        json.dumps({"program": "alto-postprocess", "statistics": {"output_counts_by_type": {"csv": 1}}}),
        encoding="utf-8",
    )
    out = tmp_path / "merged.json"
    merge_run_paradata(
        json_paths=[str(stage_json)],
        out_path=str(out),
        pipeline="alto-postprocess",
        method="layoutreader",
        skipped_stages=["3. extract text", "1. page_split"],
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["skipped_stages"] == ["3. extract text", "1. page_split"]
    assert "EXECUTED stages only" in data["license_note"]


def test_resolve_extract_outdir():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"EXTRACT": {"OUTPUT_TXT_LR": "custom/path"}})

    # Resolves to the config override for LayoutReader
    assert _resolve_extract_outdir("layoutreader", cfg) == "custom/path"
    # Falls back to default for alto-tools
    assert _resolve_extract_outdir("alto-tools", cfg) == "./data_samples/PAGE_TXT"


def test_resolve_settings_cli_precedence():
    """CLI arguments should strictly override config file values."""
    cfg = configparser.ConfigParser()
    cfg.read_dict({"PIPELINE": {"METHOD": "glm", "SKIP_SPLIT": "False"}})

    args = Namespace(
        method="layoutreader",  # CLI overrides config's "glm"
        input_dir="cli/input",
        page_alto_dir="cli/page_alto",
        input_csv="cli/input.csv",
        skip_split=True,  # CLI overrides config's "False"
        paradata_dir="cli/paradata",
    )

    settings = resolve_settings(args, cfg)
    assert settings["method"] == "layoutreader"
    assert settings["input_dir"] == "cli/input"
    assert settings["skip_split"] is True


def test_resolve_settings_config_fallback():
    """Missing CLI args should safely fall back to the config, then defaults."""
    cfg = configparser.ConfigParser()
    cfg.read_dict(
        {
            "PIPELINE": {
                "METHOD": "glm",
                "INPUT_DIR": "cfg/input",
                "PAGE_ALTO_DIR": "cfg/page",
                "PARADATA_DIR": "cfg/para",
                "SKIP_SPLIT": "True",
            },
            "EXTRACT": {"INPUT_CSV": "cfg/stats.csv", "OUTPUT_TXT_LLM": "cfg/out_llm"},
        }
    )

    args = Namespace(
        method=None, input_dir=None, page_alto_dir=None, input_csv=None, skip_split=False, paradata_dir=None
    )

    settings = resolve_settings(args, cfg)
    assert settings["method"] == "glm"
    assert settings["input_dir"] == "cfg/input"
    assert settings["skip_split"] is True
    assert settings["text_dir"] == "cfg/out_llm"
