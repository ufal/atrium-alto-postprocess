"""
Tests for the run_pipeline.py orchestrator configuration precedence.
"""

import configparser
import json
import sys
from argparse import Namespace

from atrium_paradata import merge_run_paradata
from run_pipeline import STAGE_ORDER, _resolve_extract_outdir, build_plan, resolve_settings


def _args(**over):
    base = dict(
        method=None,
        input_dir=None,
        page_alto_dir=None,
        page_json_dir=None,
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


def test_resolve_settings_json_keys_format():
    """(#31) --method json-keys resolves input_format='json' and splits into pages."""
    settings = resolve_settings(_args(method="json-keys", input_dir="data/JSON"), configparser.ConfigParser())
    assert settings["input_format"] == "json"
    assert settings["page_json_dir"] == "data_samples/PAGE_JSON"
    assert settings["stats_scan_dir"] == "data_samples/PAGE_JSON"
    assert settings["text_dir"] == "./data_samples/PAGE_TXT_JSON"
    assert settings["outputs"]["split"] == "data_samples/PAGE_JSON"


def test_build_plan_json_keys_routes_split_and_stats():
    """(#31) The split stage runs for json-keys, writing to PAGE_JSON."""
    settings = resolve_settings(_args(method="json-keys", input_dir="data/JSON"), configparser.ConfigParser())
    plan = build_plan(settings, "config.txt")

    split_stage = next(s for s in plan if s["key"] == "split")
    assert split_stage["skip"] is False
    assert split_stage["cmd"] == [sys.executable or "python3", "page_split.py", "data/JSON", "data_samples/PAGE_JSON"]

    stats_stage = next(s for s in plan if s["key"] == "stats")
    assert stats_stage["cmd"][:2] == [sys.executable or "python3", "json_stats_create.py"]
    assert "data_samples/PAGE_JSON" in stats_stage["cmd"]


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
    plan = build_plan(settings, "config.txt")
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


# def test_resolve_settings_json_keys_format():
#     """(#31) --method json-keys resolves input_format='json' and now splits
#     into pages just like alto — stats scan the split (PAGE_JSON) output dir,
#     not the raw input_dir directly."""
#     settings = resolve_settings(_args(method="json-keys", input_dir="data/JSON"), configparser.ConfigParser())
#     assert settings["input_format"] == "json"
#     assert settings["page_json_dir"] == "data_samples/PAGE_JSON"
#     assert settings["stats_scan_dir"] == "data_samples/PAGE_JSON"
#     assert settings["text_dir"] == "./data_samples/PAGE_TXT_JSON"
#     assert settings["outputs"]["split"] == "data_samples/PAGE_JSON"


def test_resolve_settings_page_json_dir_cli_override():
    """--page-json-dir (or [PIPELINE].PAGE_JSON_DIR) overrides the default,
    parallel to --page-alto-dir."""
    settings = resolve_settings(
        _args(method="json-keys", input_dir="data/JSON", page_json_dir="custom/page_json"),
        configparser.ConfigParser(),
    )
    assert settings["page_json_dir"] == "custom/page_json"
    assert settings["stats_scan_dir"] == "custom/page_json"


# def test_build_plan_json_keys_routes_split_and_stats():
#     """(#31) The split stage now actually runs for json-keys, writing to
#     PAGE_JSON, and the stats stage scans that same directory."""
#     settings = resolve_settings(_args(method="json-keys", input_dir="data/JSON"), configparser.ConfigParser())
#     plan = build_plan(settings, "config.txt")
#
#     split_stage = next(s for s in plan if s["key"] == "split")
#     assert split_stage["skip"] is False
#     assert split_stage["cmd"] == [sys.executable or "python3", "page_split.py", "data/JSON", "data_samples/PAGE_JSON"]
#
#     stats_stage = next(s for s in plan if s["key"] == "stats")
#     assert stats_stage["cmd"][:2] == [sys.executable or "python3", "json_stats_create.py"]
#     assert "data_samples/PAGE_JSON" in stats_stage["cmd"]
#
#     extract_stage = next(s for s in plan if s["key"] == "extract")
#     assert extract_stage["cmd"][-1] == "extract_JSON_2_TXT.py"


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


# ── (#31) --method text-lines: any other text-bearing input ──────────────────


def test_resolve_settings_text_lines_format():
    settings = resolve_settings(_args(method="text-lines", input_dir="data/TEXT"), configparser.ConfigParser())
    assert settings["input_format"] == "text"
    assert settings["page_text_dir"] == "data_samples/PAGE_TEXT"
    assert settings["stats_scan_dir"] == "data_samples/PAGE_TEXT"
    assert settings["text_dir"] == "./data_samples/PAGE_TXT_TEXT"
    assert settings["outputs"]["split"] == "data_samples/PAGE_TEXT"


def test_build_plan_text_lines_routes_all_three_new_stages():
    settings = resolve_settings(_args(method="text-lines", input_dir="data/TEXT"), configparser.ConfigParser())
    plan = {s["key"]: s for s in build_plan(settings, "config.txt")}
    py = sys.executable or "python3"
    assert plan["split"]["cmd"] == [py, "text_split.py", "data/TEXT", "data_samples/PAGE_TEXT"]
    assert plan["split"]["logged"] is True
    assert plan["stats"]["cmd"][:3] == [py, "text_stats_create.py", "data_samples/PAGE_TEXT"]
    # (#31 Phase 4) the stats CSV is passed to the text-lines extractor, so --input-csv reaches it
    assert plan["extract"]["cmd"] == [py, "extract_TEXT_2_TXT.py", "--input-csv", "test_alto_stats.csv"]
    # (#31 Phase 5) ... and to classify, which used to read [CLASSIFY].INPUT_CSV only
    assert plan["classify"]["cmd"] == [py, "classify_TEXT.py", "--input-csv", "test_alto_stats.csv"]


def test_resolve_settings_page_text_dir_cli_and_config():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"PIPELINE": {"PAGE_TEXT_DIR": "cfg/PAGE_TEXT"}})
    assert resolve_settings(_args(method="text-lines"), cfg)["page_text_dir"] == "cfg/PAGE_TEXT"
    settings = resolve_settings(_args(method="text-lines", page_text_dir="cli/PT"), cfg)
    assert settings["page_text_dir"] == settings["stats_scan_dir"] == "cli/PT"


def test_alto_and_json_page_dirs_unchanged_by_the_text_format():
    """The lookup table that replaced the binary alto/json choice resolves exactly as before."""
    alto = resolve_settings(_args(method="alto-tools"), configparser.ConfigParser())
    assert alto["stats_scan_dir"] == alto["outputs"]["split"] == "data_samples/PAGE_ALTO"
    plan = {s["key"]: s for s in build_plan(alto, "config.txt")}
    assert plan["split"]["cmd"][1:] == ["page_split.py", "data_samples/ALTO", "data_samples/PAGE_ALTO"]
    assert plan["split"]["logged"] is True  # page_split writes paradata (the tag said "[no log]")


def test_input_csv_reaches_extract_and_classify_for_every_method():
    """(#31 Phase 5) --input-csv used to redirect only the stats stage's output; the
    extract and classify stages kept reading the config's CSV (a warning said so)."""
    py = sys.executable or "python3"
    scripts = {
        "layoutreader": "extract_LytRdr_ALTO_2_TXT.py",
        "alto-tools": "extract_ALTO_2_TXT.py",
        "glm": "extract_LLM_ALTO_2_TXT.py",
        "json-keys": "extract_JSON_2_TXT.py",
        "text-lines": "extract_TEXT_2_TXT.py",
    }
    for method, script in scripts.items():
        settings = resolve_settings(_args(method=method, input_csv="cli/stats.csv"), configparser.ConfigParser())
        plan = {s["key"]: s for s in build_plan(settings, "config.txt")}
        assert plan["stats"]["cmd"][-2:] == ["-o", "cli/stats.csv"]
        assert plan["extract"]["cmd"] == [py, script, "--input-csv", "cli/stats.csv"]
        assert plan["classify"]["cmd"] == [py, "classify_TEXT.py", "--input-csv", "cli/stats.csv"]


def test_text_lines_has_its_own_stats_csv():
    """The stock config wrote text-lines' stats CSV over the ALTO methods' one."""
    cfg = configparser.ConfigParser()
    cfg.read_dict({"EXTRACT": {"INPUT_CSV": "alto.csv", "INPUT_CSV_TEXT": "text.csv"}})
    assert resolve_settings(_args(method="text-lines"), cfg)["input_csv"] == "text.csv"
    assert resolve_settings(_args(method="alto-tools"), cfg)["input_csv"] == "alto.csv"
    assert resolve_settings(_args(method="json-keys"), cfg)["input_csv"] == "alto.csv"
    assert resolve_settings(_args(method="text-lines", input_csv="cli.csv"), cfg)["input_csv"] == "cli.csv"
    cfg.remove_option("EXTRACT", "INPUT_CSV_TEXT")  # an older config keeps sharing INPUT_CSV
    assert resolve_settings(_args(method="text-lines"), cfg)["input_csv"] == "alto.csv"


def test_the_stock_config_separates_the_text_lines_stats_csv():
    from pathlib import Path

    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(Path(__file__).resolve().parent.parent / "setup" / "config.txt", encoding="utf-8")
    text = resolve_settings(_args(method="text-lines"), cfg)["input_csv"]
    alto = resolve_settings(_args(method="alto-tools"), cfg)["input_csv"]
    assert text != alto


# ── (#31 Phase 4) text-lines input dir, pass-throughs, ALTO/JSON plans unchanged ─


def test_text_lines_input_dir_precedence():
    cfg = configparser.ConfigParser()
    assert resolve_settings(_args(method="text-lines"), cfg)["input_dir"] == "data_samples/TEXT"
    cfg.read_dict({"PIPELINE": {"INPUT_DIR": "cfg/ANY"}})
    assert resolve_settings(_args(method="text-lines"), cfg)["input_dir"] == "cfg/ANY"  # an older config
    cfg.read_dict({"PIPELINE": {"INPUT_DIR": "cfg/ALTO", "INPUT_DIR_TEXT": "cfg/TEXT"}})
    assert resolve_settings(_args(method="text-lines"), cfg)["input_dir"] == "cfg/TEXT"
    assert resolve_settings(_args(method="text-lines", input_dir="cli/IN"), cfg)["input_dir"] == "cli/IN"


def test_alto_and_json_input_dirs_ignore_input_dir_text():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"PIPELINE": {"INPUT_DIR": "cfg/ALTO", "INPUT_DIR_TEXT": "cfg/TEXT"}})
    for method in ("layoutreader", "alto-tools", "glm", "json-keys"):
        assert resolve_settings(_args(method=method), cfg)["input_dir"] == "cfg/ALTO"


def test_strict_and_source_origin_pass_through():
    py = sys.executable or "python3"
    text = resolve_settings(
        _args(method="text-lines", input_dir="in", strict=False, source_origin="ocr:pero"), configparser.ConfigParser()
    )
    plan = {s["key"]: s for s in build_plan(text, "config.txt")}
    assert plan["split"]["cmd"] == [
        py, "text_split.py", "in", "data_samples/PAGE_TEXT", "--source-origin", "ocr:pero", "--no-strict"
    ]  # fmt: skip
    assert plan["extract"]["cmd"][-1] == "--no-strict"
    alto = resolve_settings(
        _args(method="alto-tools", strict=True, source_origin="ocr:pero"), configparser.ConfigParser()
    )
    plan = {s["key"]: s for s in build_plan(alto, "config.txt")}
    assert plan["split"]["cmd"][1:] == ["page_split.py", "data_samples/ALTO", "data_samples/PAGE_ALTO",
                                        "--source-origin", "ocr:pero"]  # fmt: skip
    assert plan["extract"]["cmd"] == [py, "extract_ALTO_2_TXT.py"]


def test_alto_and_json_plans_are_unchanged_without_the_new_flags():
    py = sys.executable or "python3"
    expected_extract = {
        "layoutreader": "extract_LytRdr_ALTO_2_TXT.py",
        "alto-tools": "extract_ALTO_2_TXT.py",
        "glm": "extract_LLM_ALTO_2_TXT.py",
        "json-keys": "extract_JSON_2_TXT.py",
    }
    for method, script in expected_extract.items():
        settings = resolve_settings(_args(method=method), configparser.ConfigParser())
        plan = {s["key"]: s for s in build_plan(settings, "config.txt")}
        page_dir = "data_samples/PAGE_JSON" if method == "json-keys" else "data_samples/PAGE_ALTO"
        assert plan["split"]["cmd"] == [py, "page_split.py", "data_samples/ALTO", page_dir]
        assert plan["extract"]["cmd"] == [py, script]
        assert plan["classify"]["cmd"] == [py, "classify_TEXT.py"]
