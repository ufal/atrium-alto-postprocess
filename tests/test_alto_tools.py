"""
tests/test_alto_tools.py
========================
Parity tests for the vendored ALTO extractor (issue #50).

`alto_tools.py` replaced two `alto-tools` CLI subprocesses with two function
calls. The whole point of the change is that *nothing downstream moves*: the
`.txt` files this pipeline writes and the counts in its statistics CSV must be
what the pinned upstream CLI produced. A vendoring that subtly reshapes text is a
data-quality regression no other test in this ecosystem would catch, so this file
pins the output three ways:

1. **Golden literals** — the exact stdout of
   `alto-tools -t` / `-s` at upstream commit
   ``1f4f01e5f6ac3562740e39948442973b9cb94be4``, captured on 2026-09-10 and
   pasted in below. These pin the paths no sample file reaches: an explicit
   `<ReadingOrder>`, an `@IDNEXT` chain, `<HYP>`, and the glyph / illustration /
   graphic counters.
2. **Independent re-derivation** — every ALTO file in `data_samples/` is
   re-extracted here by a second, deliberately naive implementation written from
   the ALTO schema rather than from the vendored code, and the two must agree.
   A tautology would not catch a bad copy; this does.
3. **Call-site contracts** — `extract_ALTO_2_TXT.extract_single_page()` and
   `alto_stats_create.run_alto_tools_stats()` still produce exactly what their
   consumers (the `.txt` files, the stats CSV columns) expect.

No models, no GPU, no network, no `alto-tools` on PATH.
"""

from __future__ import annotations

import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import alto_tools

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "alto"

SAMPLE_ALTO = sorted(glob.glob(str(_ROOT / "data_samples" / "**" / "*.xml"), recursive=True))

# ── 1. Golden literals from the pinned upstream CLI ──────────────────────────
#
#   $ python -m alto_tools.alto_tools -t tests/fixtures/alto/<file> | od -c
#
# at cneud/alto-tools@1f4f01e5f6ac3562740e39948442973b9cb94be4. Block separator
# "\n", line separator "\n", no trailing newline.

CLI_TEXT_READINGORDER = "\n\nsecond blo-\nck\n\nfirst block"
CLI_TEXT_IDNEXT = "\n\nalpha\n\nbeta\n\ngamma"
CLI_STATS_IDNEXT = {
    "textlines": 3,
    "strings": 3,
    "glyphs": 1,
    "illustrations": 1,
    "graphics": 1,
}


def test_reading_order_matches_cli_byte_for_byte():
    """<ReadingOrder> overrides document order, and <HYP> yields an ASCII '-'."""
    got = alto_tools.text_from_file(_FIXTURES / "readingorder.alto.xml")
    assert got == CLI_TEXT_READINGORDER


def test_idnext_chain_matches_cli_byte_for_byte():
    """With no <ReadingOrder>, the @IDNEXT chain orders the blocks: B1 -> B3 -> B2."""
    got = alto_tools.text_from_file(_FIXTURES / "idnext.alto.xml")
    assert got == CLI_TEXT_IDNEXT


def test_statistics_match_cli():
    """All five counters, including the three no data_samples/ file exercises."""
    assert alto_tools.statistics_from_file(_FIXTURES / "idnext.alto.xml") == CLI_STATS_IDNEXT


# ── 2. Independent re-derivation over the real samples ───────────────────────


def _namespace(root: ET.Element) -> str:
    return root.tag.split("}")[0].strip("{") if "}" in root.tag else ""


def _reference_text(xml_path) -> str:
    """Re-implement `alto-tools -t` from the ALTO schema, not from alto_tools.py.

    Only valid for files with neither <ReadingOrder> nor @IDNEXT — i.e. plain
    document order, which is every file in data_samples/. The fixtures above
    cover the other two orderings.
    """
    root = ET.parse(xml_path).getroot()
    ns = _namespace(root)
    q = (lambda tag: f"{{{ns}}}{tag}") if ns else (lambda tag: tag)

    assert root.find(f".//{q('ReadingOrder')}") is None, f"{xml_path} has a ReadingOrder"
    out: list[str] = []
    for block in root.iter(q("TextBlock")):
        assert not block.get("IDNEXT"), f"{xml_path} has an IDNEXT chain"
        out.append("\n")
        for line in block.iter(q("TextLine")):
            out.append("\n")
            words = [w.attrib.get("CONTENT", "") for w in line.findall(q("String"))]
            text = " ".join(words)
            if line.find(q("HYP")) is not None:
                text += "-"
            out.append(text)
    return "".join(out)


def _reference_counts(xml_path) -> dict[str, int]:
    root = ET.parse(xml_path).getroot()
    ns = _namespace(root)
    q = (lambda tag: f"{{{ns}}}{tag}") if ns else (lambda tag: tag)
    return {
        "textlines": sum(1 for _ in root.iter(q("TextLine"))),
        "strings": sum(1 for _ in root.iter(q("String"))),
        "glyphs": sum(1 for _ in root.iter(q("Glyph"))),
        "illustrations": sum(1 for _ in root.iter(q("Illustration"))),
        "graphics": sum(1 for _ in root.iter(q("GraphicalElement"))),
    }


def test_sample_alto_files_are_present():
    """Guard against the parametrised tests below silently collecting nothing."""
    assert SAMPLE_ALTO, "no ALTO samples found under data_samples/"


@pytest.mark.parametrize("xml_path", SAMPLE_ALTO, ids=lambda p: Path(p).name)
def test_text_matches_independent_reimplementation(xml_path):
    assert alto_tools.text_from_file(xml_path) == _reference_text(xml_path)


@pytest.mark.parametrize("xml_path", SAMPLE_ALTO, ids=lambda p: Path(p).name)
def test_statistics_match_independent_reimplementation(xml_path):
    assert alto_tools.statistics_from_file(xml_path) == _reference_counts(xml_path)


# ── 3. Failure modes: what used to be a non-zero exit status ─────────────────


def test_unparseable_xml_raises_alto_tools_error(tmp_path):
    bad = tmp_path / "broken.alto.xml"
    bad.write_text("<alto><unclosed>", encoding="utf-8")
    with pytest.raises(alto_tools.AltoToolsError):
        alto_tools.text_from_file(bad)


def test_unregistered_namespace_raises_alto_tools_error(tmp_path):
    alien = tmp_path / "alien.alto.xml"
    alien.write_text('<alto xmlns="http://example.org/not-alto"><Layout/></alto>', encoding="utf-8")
    with pytest.raises(alto_tools.AltoToolsError):
        alto_tools.statistics_from_file(alien)


def test_upstream_provenance_is_recorded():
    """The paradata component `alto_tools` claims Apache-2.0; the tree must back it."""
    assert alto_tools.UPSTREAM_COMMIT == "1f4f01e5f6ac3562740e39948442973b9cb94be4"
    assert alto_tools.UPSTREAM_LICENSE == "Apache-2.0"
    licence = _ROOT / "LICENSES" / "alto-tools-Apache-2.0.txt"
    assert licence.is_file(), "vendored Apache-2.0 licence text is missing"
    assert "Apache License" in licence.read_text(encoding="utf-8")


def test_alto_tools_is_no_longer_a_declared_dependency():
    """Acceptance pin for #50: nothing may re-resolve alto-tools at build or run time."""
    for req in (_ROOT / "setup").glob("requirements*.txt"):
        for line in req.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                assert "alto-tools" not in stripped, f"{req.name}: {stripped}"


# ── 4. Call-site contracts ───────────────────────────────────────────────────


def test_extract_single_page_writes_dehyphenated_cli_text(tmp_path):
    from extract_ALTO_2_TXT import _dehyphenate, extract_single_page

    xml_path = _FIXTURES / "readingorder.alto.xml"
    assert extract_single_page(("DOC1", "1", str(xml_path), str(tmp_path))) is True

    written = (tmp_path / "DOC1" / "DOC1-1.txt").read_text(encoding="utf-8")
    assert written == _dehyphenate(CLI_TEXT_READINGORDER)
    # The hyphenated word is rejoined across the line break, as before #50.
    assert "block" in written and "blo-" not in written


def test_extract_single_page_returns_false_on_bad_xml(tmp_path):
    from extract_ALTO_2_TXT import extract_single_page

    bad = tmp_path / "broken.alto.xml"
    bad.write_text("<alto><unclosed>", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert extract_single_page(("DOC1", "1", str(bad), str(out_dir))) is False
    assert not (out_dir / "DOC1" / "DOC1-1.txt").exists()


def test_run_alto_tools_stats_returns_the_csv_columns():
    from alto_stats_create import run_alto_tools_stats

    stats = run_alto_tools_stats(str(_FIXTURES / "idnext.alto.xml"))
    # The four columns _process_single_xml() writes to the CSV, plus glyphs.
    assert stats == CLI_STATS_IDNEXT
    for key in ("textlines", "illustrations", "graphics", "strings"):
        assert isinstance(stats[key], int)


def test_run_alto_tools_stats_returns_none_on_bad_xml(tmp_path, capsys):
    from alto_stats_create import run_alto_tools_stats

    bad = tmp_path / "broken.alto.xml"
    bad.write_text("<alto><unclosed>", encoding="utf-8")
    assert run_alto_tools_stats(str(bad)) is None
    assert "Error reading ALTO statistics" in capsys.readouterr().out
