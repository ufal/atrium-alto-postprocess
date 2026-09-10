#!/usr/bin/env python3
"""
alto_tools.py — vendored ALTO XML text extraction and element statistics.

Upstream origin
---------------
Derived from **ALTO Tools** by Clemens Neudecker and contributors:

* project : https://github.com/cneud/alto-tools
* commit  : ``1f4f01e5f6ac3562740e39948442973b9cb94be4`` (``__version__ = "0.1.0"``)
* file    : ``src/alto_tools/alto_tools.py``
* licence : **Apache License 2.0** — full text in ``LICENSES/alto-tools-Apache-2.0.txt``

Modifications are noted inline with ``VENDORED:`` comments, as the Apache-2.0
licence requires for a derived work.

Why this file exists (issue #50)
--------------------------------
Two scripts used to shell out to the ``alto-tools`` console script:

* ``extract_ALTO_2_TXT.py`` — ``alto-tools -t <xml>`` (page text on stdout)
* ``alto_stats_create.py``  — ``alto-tools -s <xml>`` (element counts on stdout)

That made the published Docker image depend on a ``git+https://…@<commit>``
requirement, which the E2E lane then re-resolved *inside the released image* from
a moving ``refs/heads/master``. Vendoring the two used code paths removes the
``git+`` requirement, the ``shutil.which()`` guard, the ``PATH`` export and the
run-time ``pip install`` in one move, and turns two subprocess boundaries into
two function calls.

What was vendored
-----------------
Only the code reachable from ``alto-tools -t`` and ``alto-tools -s``:

* ``alto_parse``      — namespace detection + ``ElementTree`` parse
* ``alto_text``       — reading-order text reconstruction (the ``-t`` path)
* ``alto_statistics`` — ``<TextLine>``/``<String>``/``<Glyph>``/``<Illustration>``/
  ``<GraphicalElement>`` counts (the ``-s`` path)

Deliberately **not** vendored, because nothing in this repository calls it:
``alto_confidence`` (``-c``), ``alto_illustrations`` (``-i``), ``alto_graphics``
(``-g``), the ``--dehyphenate`` / ``--detect-hyphens`` branches of ``alto_text``
(this repo does its own de-hyphenation in ``extract_ALTO_2_TXT._dehyphenate``),
the ``argparse`` CLI, stdin input, directory walking and the ``-x/--xml-encoding``
sniffing path.

Output compatibility
--------------------
``alto_text()`` returns exactly the byte sequence the ``alto-tools -t`` CLI wrote
to stdout for the same file (block separator ``"\\n"``, line separator ``"\\n"``,
no trailing newline), so the ``.txt`` files this pipeline produces are unchanged.
``alto_statistics()`` returns the counts the ``-s`` CLI printed, already keyed the
way ``alto_stats_create.py`` needs them. ``tests/test_alto_tools.py`` pins both.

Error handling
--------------
Upstream reported failures by printing and then crashing, which the callers
observed as a non-zero exit code from the subprocess. Here every failure raises
``AltoToolsError`` (or a plain parse/attribute error for the pathological inputs
that crashed upstream too), which both call sites translate into the same "skip
this page" outcome they had before.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Provenance of the vendored code, recorded next to it so a reader (and the
# paradata `alto_tools` component in setup/para_config.txt) can trace it.
UPSTREAM_URL = "https://github.com/cneud/alto-tools"
UPSTREAM_COMMIT = "1f4f01e5f6ac3562740e39948442973b9cb94be4"
UPSTREAM_VERSION = "0.1.0"
UPSTREAM_LICENSE = "Apache-2.0"

# ALTO namespaces registered upstream — copied verbatim. A file whose root
# namespace is not in this list is rejected, exactly as the CLI rejected it.
ALTO_NAMESPACES = {
    # ALTO @ CCS Content Conversion Specialists GmbH
    # https://content-conversion.com/mets-alto/
    "alto-1": "http://schema.ccs-gmbh.com/ALTO",
    "alto-1-xsd": "http://schema.ccs-gmbh.com/ALTO/alto-1-4.xsd",
    # ALTO @ Bibliothèque nationale de France
    # https://bibnum.bnf.fr/alto_prod/documentation/alto_prod.html
    "alto-bnf": "http://bibnum.bnf.fr/ns/alto_prod",
    "alto-bnf-xsd": "http://bibnum.bnf.fr/ns/alto_prod.xsd",
    # ALTO @ Library of Congress
    # https://www.loc.gov/standards/alto/
    # https://altoxml.github.io/
    "alto-2": "http://www.loc.gov/standards/alto/ns-v2#",
    "alto-2-xsd": "https://www.loc.gov/standards/alto/alto.xsd",
    "alto-3": "http://www.loc.gov/standards/alto/ns-v3#",
    "alto-3-xsd": "http://www.loc.gov/standards/alto/v3/alto.xsd",
    "alto-4": "http://www.loc.gov/standards/alto/ns-v4#",
    "alto-4-xsd": "http://www.loc.gov/standards/alto/v4/alto.xsd",
}


class AltoToolsError(Exception):
    """An ALTO file could not be parsed, or is not in a registered ALTO namespace.

    VENDORED: upstream printed to stdout/stderr and then died with an unhandled
    exception, so the caller saw a non-zero exit status. In-process the callers
    need something catchable, and the resulting behaviour is identical: the
    offending page is skipped and logged.
    """


def _source_name(source) -> str:
    """Best-effort label for an error message (a path, or an open file's name)."""
    return getattr(source, "name", None) or str(source)


def alto_parse(source, **kwargs) -> tuple[ET.ElementTree, str]:
    """Parse an ALTO file and return ``(element_tree, namespace_uri)``.

    ``source`` is anything ``ElementTree.parse`` accepts — a path or an open
    text-mode file object.

    VENDORED: upstream returned ``(alto, xml, xmlns)`` and signalled both failure
    modes (parse error, unregistered namespace) by printing and then falling off
    the end of the function. The ``alto`` passthrough is dropped (only the
    removed ``-c/-i/-g`` paths used it, for their ``File: …`` prefixes) and the
    failures raise.
    """
    try:
        xml = ET.parse(source, **kwargs)
    except ET.ParseError as exc:
        raise AltoToolsError(f"Parser error in file '{_source_name(source)}': {exc}") from exc

    # Extract namespace from document root.
    root_tag = xml.getroot().tag
    if "http://" in str(root_tag.split("}")[0].strip("{")):
        xmlns = root_tag.split("}")[0].strip("{")
    else:
        try:
            ns = xml.getroot().attrib
            xmlns = str(ns).split(" ")[1].strip("}").strip("'")
        except IndexError as exc:
            raise AltoToolsError(f"File '{_source_name(source)}': no namespace declaration found.") from exc

    if xmlns not in ALTO_NAMESPACES.values():
        raise AltoToolsError(f"File '{_source_name(source)}': namespace {xmlns} is not registered.")
    return xml, xmlns


def _reading_order(xml: ET.ElementTree, xmlns: str, blocks: dict) -> list[str]:
    """Return the ``<TextBlock>`` @IDs in reading order.

    Three sources, in upstream's order of preference: an explicit
    ``<ReadingOrder>``, an ``@IDNEXT`` chain, or document order.
    """
    readingorder = xml.find(".//{%s}ReadingOrder" % xmlns)
    if readingorder is not None:
        # VENDORED: upstream wrote `ordered or unordered`. An Element with no
        # children is falsy, so an empty <OrderedGroup> silently falls through to
        # <UnorderedGroup>; that fall-through is preserved explicitly here
        # because implicit Element truthiness is deprecated since Python 3.12.
        ordered = readingorder.find("{%s}OrderedGroup" % xmlns)
        unordered = readingorder.find("{%s}UnorderedGroup" % xmlns)
        group = ordered if (ordered is not None and len(ordered)) else unordered
        if group is None:
            raise AltoToolsError("ReadingOrder contains no OrderedGroup or UnorderedGroup")
        order = [groupref.get("REF") for groupref in group.iter("*") if "REF" in groupref.attrib]
        # FIXME (upstream): ALTO ReadingOrder @REF can also target TextLine and
        # String. Adding TextLine and String elements to the block list is not
        # sufficient though: we would need to iterate over the lowest-level
        # sequence below.
        if not all(block_id in blocks for block_id in order):
            raise AltoToolsError("ReadingOrder references below block level currently not supported")
        return order

    if any(block.get("IDNEXT") for block in blocks.values()):
        pairs = {block_id: block.get("IDNEXT") for block_id, block in blocks.items()}
        block = next(block for block in pairs if block not in pairs.values())
        order = [block]
        while block := pairs.get(block, None):
            order.append(block)
        return order

    return list(blocks.keys())


def alto_text(xml: ET.ElementTree, xmlns: str, pb: str = "\n", lb: str = "\n") -> str:
    """Extract text content from a parsed ALTO tree, in reading order.

    Returns the string the ``alto-tools -t`` CLI wrote to stdout: ``pb`` before
    every ``<TextBlock>``, ``lb`` before every ``<TextLine>``, ``<String>``
    @CONTENT values joined with single spaces, and a plain ASCII hyphen-minus
    appended when the line ends in a ``<HYP>`` element. There is no trailing
    newline — ``extract_ALTO_2_TXT._dehyphenate`` adds one.

    VENDORED: returns the text instead of writing it to ``sys.stdout`` (and so
    drops upstream's ``codecs.getwriter("utf-8")`` stdout re-wrapping, which only
    existed to make that write UTF-8-safe). The ``dehyphenate`` /
    ``detect_hyphens`` branches are dropped: this pipeline never passed ``-H``,
    and repairs hyphenation itself afterwards.
    """
    out: list[str] = []
    # Find all <TextBlock> elements.
    blocks = {block.get("ID"): block for block in xml.iterfind(".//{%s}TextBlock" % xmlns)}
    for block_id in _reading_order(xml, xmlns, blocks):
        block = blocks[block_id]
        out.append(pb)
        # Find all <TextLine> elements.
        for line in block.iterfind(".//{%s}TextLine" % xmlns):
            # New line after every <TextLine> element.
            out.append(lb)
            text = ""
            # Find all <String> elements. Do not rely on interspersed <SP>
            # elements — https://github.com/altoxml/schema/issues/54
            words = list(line.findall("{%s}String" % xmlns))
            for word in words:
                if word is not words[0]:
                    text += " "
                # Get value of attribute @CONTENT.
                text += word.attrib.get("CONTENT")
            hyp = line.find("{%s}HYP" % xmlns)
            if hyp is not None:
                # Use plain ASCII hyphen-minus instead of the annotated hyphen
                # (which could be a soft hyphen, a historical form, etc.).
                text += "-"
            out.append(text)
    return "".join(out)


def alto_statistics(xml: ET.ElementTree, xmlns: str) -> dict[str, int]:
    """Count the ALTO elements the page-statistics stage records.

    Returns the same five keys the ``alto-tools -s`` CLI printed, already under
    the names ``alto_stats_create.py`` writes to its CSV:
    ``textlines``, ``strings``, ``glyphs``, ``illustrations``, ``graphics``.

    VENDORED: returns the dict only. Upstream also printed a ``File: …,
    Statistics:`` header plus one ``# of <Element> elements: N`` line per counter,
    which ``alto_stats_create.parse_alto_tools_stats_line()`` then had to
    regex back into this very dict; both sides of that round-trip are gone.
    """
    return {
        "textlines": sum(1 for _ in xml.iterfind(".//{%s}TextLine" % xmlns)),
        "strings": sum(1 for _ in xml.iterfind(".//{%s}String" % xmlns)),
        "glyphs": sum(1 for _ in xml.iterfind(".//{%s}Glyph" % xmlns)),
        "illustrations": sum(1 for _ in xml.iterfind(".//{%s}Illustration" % xmlns)),
        "graphics": sum(1 for _ in xml.iterfind(".//{%s}GraphicalElement" % xmlns)),
    }


# ── File-level entry points used by this repository ──────────────────────────
#
# VENDORED: these two replace the CLI. They reproduce what `alto-tools -t` and
# `alto-tools -s` did per file, including opening the file in text mode with the
# CLI's default `--file-encoding UTF-8` (so, as upstream, an `encoding=` in the
# XML declaration is not honoured).


def text_from_file(xml_path, file_encoding: str = "UTF-8", pb: str = "\n", lb: str = "\n") -> str:
    """Equivalent of ``alto-tools -t <xml_path>``; returns its stdout as a string."""
    with open(xml_path, encoding=file_encoding) as alto:
        xml, xmlns = alto_parse(alto)
    return alto_text(xml, xmlns, pb=pb, lb=lb)


def statistics_from_file(xml_path, file_encoding: str = "UTF-8") -> dict[str, int]:
    """Equivalent of ``alto-tools -s <xml_path>``; returns its counts as a dict."""
    with open(xml_path, encoding=file_encoding) as alto:
        xml, xmlns = alto_parse(alto)
    return alto_statistics(xml, xmlns)
