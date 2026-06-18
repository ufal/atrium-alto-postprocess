"""
tests/test_alto_text_preservation.py
=====================================
Intermediate input -> output content-preservation tests for the
ALTO text-extraction step (atrium issue #14).

These guard the invariant that ALTO -> TXT extraction *reshapes* text
(reading-order reconstruction, whitespace from <SP>, hyphenation join)
without ever dropping, merging, or silently altering meaningful content.

Design notes
------------
* No ML models, no GPU, no network — runs in the default ("not slow") lane.
* The module under test (extract_LytRdr_ALTO_2_TXT.py) imports torch,
  transformers and `from v3.helpers import ...`. The `v3` package is not
  vendored in this repo, so a bare import would hit `sys.exit(1)`. We register
  lightweight stub modules *before* importing so only the two pure functions
  (parse_alto_xml, post_process_text) load. No production code is changed.
* The strong char-conservation tests use only the real sample files, which
  contain no hyphenation — hyphenation legitimately injects '-', '{', '}' and a
  duplicated prefix, so character multisets are conserved only on the
  non-hyphenated inputs. Hyphenation is covered separately by substring tests.
"""
import sys
import types
import glob
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

# ── Import shim: stub heavy / missing deps before importing the module ───────
for _n in ("torch", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
# parse_alto_xml / post_process_text never touch this attribute, but the module
# references it at import time.
sys.modules["transformers"].LayoutLMv3ForTokenClassification = object  # type: ignore[attr-defined]
_v3 = types.ModuleType("v3")
_v3h = types.ModuleType("v3.helpers")
_v3h.prepare_inputs = _v3h.boxes2inputs = _v3h.parse_logits = lambda *a, **k: None  # type: ignore[attr-defined]
sys.modules.setdefault("v3", _v3)
sys.modules.setdefault("v3.helpers", _v3h)

from extract_LytRdr_ALTO_2_TXT import parse_alto_xml, post_process_text  # noqa: E402


# ── Sample discovery & helpers ───────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ALTO = sorted(glob.glob(str(_ROOT / "data_samples" / "PAGE_ALTO" / "CTX*" / "*.alto.xml")))

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v3#"


def _string_contents(xml_path):
    """Return the list of non-empty <String> CONTENT values, in document order."""
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    tag = f"{{{ns}}}String" if ns else "String"
    out = []
    for s in root.iter(tag):
        c = s.attrib.get("CONTENT")
        if c:
            out.append(c)
    return out


def _expected_line_count(xml_path):
    """
    Mirror the module's line-acceptance rule: a TextLine is emitted only if it
    accumulates at least one String with non-empty CONTENT *and* valid integer
    coordinates (so min_x != inf).
    """
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    tl_tag = f"{{{ns}}}TextLine" if ns else "TextLine"
    count = 0
    for tl in root.iter(tl_tag):
        kept = False
        for child in list(tl):
            if child.tag.split("}")[-1] != "String":
                continue
            content = child.attrib.get("CONTENT")
            if not content:
                continue
            try:
                int(float(child.attrib["HPOS"]))
                int(float(child.attrib["VPOS"]))
                int(float(child.attrib["WIDTH"]))
                int(float(child.attrib["HEIGHT"]))
            except (ValueError, TypeError, KeyError):
                continue
            kept = True
        if kept:
            count += 1
    return count


def _nonspace(s):
    return "".join(s.split())


NS = ALTO_NS


def _alto_doc(body, width=1000, height=1000):
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<alto xmlns="{NS}"><Layout>'
        f'<Page ID="Page1" PHYSICAL_IMG_NR="1" HEIGHT="{height}" WIDTH="{width}">'
        f"{body}</Page></Layout></alto>"
    )


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# Guard: if samples are missing the suite should fail loudly, not silently pass.
def test_sample_alto_files_present():
    assert SAMPLE_ALTO, "Expected real ALTO samples under data_samples/PAGE_ALTO/CTX*/"
    assert len(SAMPLE_ALTO) >= 7


# ════════════════════════════════════════════════════════════════════════════
# parse_alto_xml — content preservation on real samples
# ════════════════════════════════════════════════════════════════════════════
class TestParseAltoContentPresence:

    @pytest.mark.parametrize("xml_path", SAMPLE_ALTO, ids=lambda p: Path(p).name)
    def test_every_string_content_present(self, xml_path):
        """Every <String CONTENT> value must survive as a substring of the output."""
        lines, _boxes, _dims = parse_alto_xml(xml_path)
        joined = "\n".join(lines)
        for content in _string_contents(xml_path):
            assert content in joined, (
                f"CONTENT {content!r} dropped from extracted text of {Path(xml_path).name}"
            )

    @pytest.mark.parametrize("xml_path", SAMPLE_ALTO, ids=lambda p: Path(p).name)
    def test_no_word_characters_lost(self, xml_path):
        """
        Strong conservation: the multiset of non-whitespace characters in the
        extracted output must be a superset of the raw ALTO CONTENT strings.
        Hyphenation reconstruction legally injects duplicate letters, '{', '}',
        and dashes, but NO original characters may be dropped.
        """
        lines, _boxes, _dims = parse_alto_xml(xml_path)

        out_chars = Counter(_nonspace("".join(lines)))
        in_chars = Counter(_nonspace("".join(_string_contents(xml_path))))

        # Subtract the output characters from the input characters.
        # If any remain, it means they were dropped during extraction.
        missing = in_chars - out_chars

        assert not missing, (
            f"Characters DROPPED during extraction of {Path(xml_path).name}! "
            f"Missing characters: {missing}"
        )

    @pytest.mark.parametrize("xml_path", SAMPLE_ALTO, ids=lambda p: Path(p).name)
    def test_line_count_matches_nonempty_textlines(self, xml_path):
        """Number of emitted lines equals the number of content-bearing TextLines."""
        lines, _boxes, _dims = parse_alto_xml(xml_path)
        assert len(lines) == _expected_line_count(xml_path)

    @pytest.mark.parametrize("xml_path", SAMPLE_ALTO, ids=lambda p: Path(p).name)
    def test_lines_and_boxes_aligned(self, xml_path):
        """Each emitted line has exactly one bounding box (no desync)."""
        lines, boxes, _dims = parse_alto_xml(xml_path)
        assert len(lines) == len(boxes)


# ════════════════════════════════════════════════════════════════════════════
# parse_alto_xml — crafted edge cases
# ════════════════════════════════════════════════════════════════════════════
class TestParseAltoEdgeCases:

    def test_sp_elements_keep_words_separated(self, tmp_path):
        """An <SP> between two Strings must keep the words separated (no merge)."""
        body = (
            "<TextLine>"
            '<String CONTENT="hradiste" HEIGHT="30" WIDTH="100" VPOS="10" HPOS="10"/>'
            '<SP WIDTH="10" VPOS="10" HPOS="110"/>'
            '<String CONTENT="okres" HEIGHT="30" WIDTH="80" VPOS="10" HPOS="120"/>'
            "</TextLine>"
        )
        path = _write(tmp_path, "sp.alto.xml", _alto_doc(body))
        lines, _b, _d = parse_alto_xml(path)
        assert lines == ["hradiste okres"]
        # both words present and separated — they did not fuse into "hradisteokres"
        assert "hradiste okres" in lines[0]
        assert "hradisteokres" not in lines[0]

    def test_missing_sp_does_not_invent_separator(self, tmp_path):
        """Without an <SP>, adjacent Strings concatenate (engine-faithful)."""
        body = (
            "<TextLine>"
            '<String CONTENT="abc" HEIGHT="30" WIDTH="50" VPOS="10" HPOS="10"/>'
            '<String CONTENT="def" HEIGHT="30" WIDTH="50" VPOS="10" HPOS="60"/>'
            "</TextLine>"
        )
        path = _write(tmp_path, "nosp.alto.xml", _alto_doc(body))
        lines, _b, _d = parse_alto_xml(path)
        assert lines == ["abcdef"]

    def test_hyphenation_subs_content_preserved(self, tmp_path):
        """
        A HypPart1 split with SUBS_CONTENT must keep BOTH the visible fragment
        and the reconstructed full word — nothing is lost in the join.
        """
        body = (
            "<TextLine>"
            '<String CONTENT="za" SUBS_TYPE="HypPart1" SUBS_CONTENT="zacatek" '
            'HEIGHT="30" WIDTH="50" VPOS="10" HPOS="10"/>'
            "</TextLine>"
        )
        path = _write(tmp_path, "hyp.alto.xml", _alto_doc(body))
        lines, _b, _d = parse_alto_xml(path)
        assert len(lines) == 1
        assert "za" in lines[0]          # original fragment retained
        assert "zacatek" in lines[0]     # reconstructed full form retained
        assert "{zacatek}" in lines[0]   # full form recorded in the {..} marker

    def test_hyp_child_tag_appends_hyphen(self, tmp_path):
        """An explicit <HYP> child tag appends its CONTENT to the preceding String."""
        body = (
            "<TextLine>"
            '<String CONTENT="za" HEIGHT="30" WIDTH="50" VPOS="10" HPOS="10"/>'
            '<HYP CONTENT="-"/>'
            "</TextLine>"
        )
        path = _write(tmp_path, "hypchild.alto.xml", _alto_doc(body))
        lines, _b, _d = parse_alto_xml(path)
        assert lines == ["za-"]

    def test_empty_content_string_line_dropped(self, tmp_path):
        """A TextLine whose Strings all have empty CONTENT yields no line."""
        body = (
            "<TextLine>"
            '<String CONTENT="" HEIGHT="30" WIDTH="50" VPOS="10" HPOS="10"/>'
            '<SP WIDTH="10" VPOS="10" HPOS="60"/>'
            '<String CONTENT="" HEIGHT="30" WIDTH="50" VPOS="10" HPOS="80"/>'
            "</TextLine>"
        )
        path = _write(tmp_path, "empty.alto.xml", _alto_doc(body))
        lines, boxes, _d = parse_alto_xml(path)
        assert lines == []
        assert boxes == []

    def test_unicode_content_preserved_verbatim(self, tmp_path):
        """Czech diacritics and dashes survive byte-for-byte."""
        body = (
            "<TextLine>"
            '<String CONTENT="Příkop" HEIGHT="30" WIDTH="80" VPOS="10" HPOS="10"/>'
            '<SP WIDTH="10" VPOS="10" HPOS="90"/>'
            '<String CONTENT="—" HEIGHT="30" WIDTH="20" VPOS="10" HPOS="100"/>'
            '<SP WIDTH="10" VPOS="10" HPOS="120"/>'
            '<String CONTENT="naleziště" HEIGHT="30" WIDTH="120" VPOS="10" HPOS="130"/>'
            "</TextLine>"
        )
        path = _write(tmp_path, "uni.alto.xml", _alto_doc(body))
        lines, _b, _d = parse_alto_xml(path)
        assert lines == ["Příkop — naleziště"]

    def test_malformed_xml_returns_empty(self, tmp_path):
        """A broken XML file is handled gracefully (no exception, empty result)."""
        path = _write(tmp_path, "broken.alto.xml", "<alto><not-closed>")
        lines, boxes, dims = parse_alto_xml(path)
        assert lines == [] and boxes == [] and dims == (0, 0)


# ════════════════════════════════════════════════════════════════════════════
# post_process_text — only inserts whitespace, never drops/merges characters
# ════════════════════════════════════════════════════════════════════════════
class TestPostProcessText:

    def test_post_process_empty_returns_empty(self):
        assert post_process_text([], []) == ""

    def test_single_line_returned_stripped(self):
        out = post_process_text(["hello world"], [[0, 0, 100, 30]])
        assert out == "hello world"

    @pytest.mark.parametrize(
        "lines,boxes",
        [
            (["abc", "def"], [[0, 0, 10, 10], [0, 20, 10, 30]]),                # normal flow
            (["one", "two", "three"],
             [[0, 0, 10, 10], [0, 200, 10, 230], [0, 240, 10, 270]]),          # big gap -> blank line
            (["right", "left"], [[0, 100, 10, 130], [0, 0, 10, 30]]),          # upward jump -> column reset
            (["Příkop", "naleziště", "kostra"],
             [[0, 0, 10, 30], [0, 40, 10, 70], [0, 80, 10, 110]]),            # unicode
        ],
    )
    def test_post_process_only_inserts_whitespace(self, lines, boxes):
        """
        Reconstruction may only add separators: the non-whitespace character
        stream of the output must equal that of the concatenated input lines.
        """
        out = post_process_text(lines, boxes)
        assert _nonspace(out) == _nonspace("".join(lines))

    def test_post_process_preserves_line_order(self):
        """Lines appear in the same order they were supplied."""
        lines = ["alpha", "beta", "gamma"]
        boxes = [[0, 0, 10, 10], [0, 20, 10, 30], [0, 40, 10, 50]]
        out = post_process_text(lines, boxes)
        positions = [out.find(w) for w in lines]
        assert positions == sorted(positions)
        assert all(p != -1 for p in positions)

    def test_post_process_no_box_data_still_conserves_chars(self):
        """Even with degenerate boxes, no characters are dropped."""
        lines = ["foo", "bar"]
        boxes = [[0, 0, 0, 0], [0, 0, 0, 0]]
        out = post_process_text(lines, boxes)
        assert _nonspace(out) == _nonspace("".join(lines))