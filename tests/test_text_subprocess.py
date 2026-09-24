"""
tests/test_text_subprocess.py — E2E subprocess chain for the text-lines method (#31).

Runs stages 1-3 exactly as run_pipeline.py does — `text_split.py <in> <PAGE_TEXT>`,
`text_stats_create.py <PAGE_TEXT> -o <csv>`, `extract_TEXT_2_TXT.py` with no arguments
(all settings through LANGID_CONFIG) — as real processes, over a mixed directory of
inputs, and checks exit codes, outputs and the line-table invariant. Categorization
(classify_TEXT.py) is NOT run: it needs the models.

Every process runs with `cwd` = a fresh tmp_path (paradata/ lands there) and a scratch
config, so nothing here reads or writes the repo's config or data_samples/.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.text_format_fixtures import docx_bytes, odf_bytes, pdf_bytes, w_p, w_t, xlsx_bytes

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(script: str, args: list, cwd: Path, config: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, LANGID_CONFIG=str(config))
    env.pop("DOCUMENT_JSON_DIR", None)
    env.pop("DOCUMENT_SOURCE_ORIGIN", None)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture
def chain(tmp_path):
    pytest.importorskip("pypdfium2")
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "zprava.txt").write_bytes("Zpráva o výzkumu\n\nSonda číslo 1.\fDruhá strana\n".encode("cp1250"))
    (inp / "denik.docx").write_bytes(
        docx_bytes(w_p(w_t("Den první"), '<w:r><w:br w:type="page"/></w:r>') + w_p(w_t("Den druhý")))
    )
    (inp / "sken.pdf").write_bytes(pdf_bytes([["OCR layer line one.", "OCR l1ne tw0."], []], invisible=True))
    (inp / "nalezy.xlsx").write_bytes(xlsx_bytes([("Nálezy", [["Popis"], ["Zlomek nádoby"]])]))
    (inp / "poznamky.odt").write_bytes(odf_bytes("text", "<text:p>Poznámka</text:p>"))
    (inp / "tabulka.csv").write_text("page,text\n1,Řádek jedna\n2,Řádek dva\n", encoding="utf-8")
    (inp / "poskozeny.pdf").write_bytes(b"%PDF-1.4 truncated")
    (inp / "obrazek.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)

    stats = tmp_path / "stats.csv"
    config = tmp_path / "config.txt"
    config.write_text(
        "\n".join(
            [
                "[EXTRACT]",
                f"INPUT_CSV = {stats}",
                f"OUTPUT_TXT_TEXT = {tmp_path / 'PAGE_TXT_TEXT'}",
                f"OUTPUT_LINES_TEXT = {tmp_path / 'DOC_LINES_TEXT'}",
                "[CLASSIFY]",
                f"OUTPUT_LINES_LOG = {tmp_path / 'DOC_LINE_CATEG'}",
                "[DOCUMENT]",
                "JSON_DIR =",
                "[TEXT_INGEST]",
                "READER_TIMEOUT_S = 120",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path, config, stats


def test_split_stats_extract_chain_over_mixed_inputs(chain):
    tmp_path, config, stats = chain
    page_text = tmp_path / "PAGE_TEXT"

    split = _run("text_split.py", [str(tmp_path / "in"), str(page_text)], tmp_path, config)
    assert split.returncode == 0, split.stderr
    assert "6 document(s) split, 2 failed" in split.stdout

    with open(page_text / "ingest_report.csv", encoding="utf-8", newline="") as fh:
        report = {r["filename"]: r for r in csv.DictReader(fh)}
    assert report["poskozeny.pdf"]["reason"] == "corrupt"
    assert report["obrazek.jpg"]["reason"] == "image_needs_ocr"
    assert report["zprava.txt"]["encoding"] in ("cp1250", "iso8859_2")
    assert report["sken.pdf"]["pages_ocr_layer"] == "1" and report["sken.pdf"]["pages_no_text"] == "1"

    stats_run = _run("text_stats_create.py", [str(page_text), "-o", str(stats)], tmp_path, config)
    assert stats_run.returncode == 0, stats_run.stderr

    extract = _run("extract_TEXT_2_TXT.py", [], tmp_path, config)
    assert extract.returncode == 0, extract.stderr
    assert "Success rate: 100.00%" in extract.stdout

    out = tmp_path / "PAGE_TXT_TEXT"
    assert (out / "zprava" / "zprava-1.txt").read_text(encoding="utf-8") == "Zpráva o výzkumu\nSonda číslo 1."
    assert (out / "denik" / "denik-2.txt").read_text(encoding="utf-8") == "Den druhý"
    assert (out / "sken" / "sken-2.txt").read_text(encoding="utf-8") == ""  # the page without a text layer
    assert (out / "nalezy" / "nalezy-1.txt").read_text(encoding="utf-8") == "Popis\nZlomek nádoby"

    tables = sorted(p.name for p in (tmp_path / "DOC_LINES_TEXT").iterdir())
    assert tables == ["denik.csv", "nalezy.csv", "poznamky.csv", "sken.csv", "tabulka.csv", "zprava.csv"]
    for table in (tmp_path / "DOC_LINES_TEXT").iterdir():
        with open(table, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                page_file = out / row["file"] / f"{row['file']}-{row['page_num']}.txt"
                assert (
                    page_file.open(encoding="utf-8").readlines()[int(row["line_num"]) - 1].rstrip("\n") == row["text"]
                )
    assert not (tmp_path / "DOC_LINE_CATEG").exists()  # categorization was not run


def test_strict_mode_fails_the_process(chain):
    tmp_path, config, _stats = chain
    result = _run("text_split.py", [str(tmp_path / "in"), str(tmp_path / "PT"), "--strict"], tmp_path, config)
    assert result.returncode == 1
    assert "corrupt" in result.stderr


def test_cli_misuse_exit_codes(chain):
    tmp_path, config, _stats = chain
    assert _run("text_split.py", [], tmp_path, config).returncode == 2
    assert _run("text_split.py", [str(tmp_path / "missing"), str(tmp_path / "o")], tmp_path, config).returncode == 1
    assert _run("text_stats_create.py", [str(tmp_path / "missing")], tmp_path, config).returncode == 1
    assert _run("extract_TEXT_2_TXT.py", ["--bogus"], tmp_path, config).returncode == 2
