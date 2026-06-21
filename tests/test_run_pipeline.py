"""
Tests for the run_pipeline.py orchestrator configuration precedence.
"""

import configparser
from argparse import Namespace

from run_pipeline import _resolve_extract_outdir, resolve_settings


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
