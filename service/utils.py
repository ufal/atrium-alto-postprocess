"""
service/utils.py
Helper functions for ALTO parsing, box normalization, and text reconstruction.
"""

import logging
from typing import List, Tuple

# Use lxml for highly efficient XML parsing
import lxml.etree as ET
import numpy as np

logger = logging.getLogger(__name__)

# (#5) Hardened parser for UNTRUSTED ALTO uploaded via the FastAPI /process
# endpoint. The default lxml parser resolves entities and may hit the network,
# which exposes XXE and entity-expansion ("billion laughs") attacks. We disable
# all of that: no entity resolution, no external DTD loading, no network, and
# huge_tree stays off so pathological documents are rejected rather than
# expanded. resolve_entities=False keeps any internal entity references inert.
_SAFE_PARSER = ET.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    dtd_validation=False,
    huge_tree=False,
)


def parse_alto_xml(xml_path: str) -> Tuple[List[str], List[List[int]], Tuple[int, int]]:
    """
    Parses ALTO XML from a file path using fast lxml bindings.
    Returns: words (list), boxes (list), (width, height)

    Parsing is performed with a hardened parser (no entity resolution, no
    external DTDs, no network) so hostile uploads cannot trigger XXE or
    entity-expansion attacks.
    """
    try:
        tree = ET.parse(xml_path, parser=_SAFE_PARSER)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"XML Parse Error in {xml_path}: {e}")
        return [], [], (0, 0)

    # Use lxml's native namespace handling
    ns = {"alto": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    # Optimized pre-compiled tags
    page_tag = ".//alto:Page" if ns else ".//Page"
    text_line_tag = ".//alto:TextLine" if ns else ".//TextLine"

    page = root.find(page_tag, ns)
    if page is None:
        return [], [], (0, 0)

    try:
        page_w = int(float(page.attrib.get("WIDTH", 0)))
        page_h = int(float(page.attrib.get("HEIGHT", 0)))
    except (ValueError, TypeError):
        return [], [], (0, 0)

    words: List[str] = []
    boxes: List[List[int]] = []

    text_lines = root.findall(text_line_tag, ns)

    for line in text_lines:
        children = list(line)
        for i, child in enumerate(children):
            tag_name = child.tag.split("}")[-1]

            if tag_name == "String":
                content = child.attrib.get("CONTENT")
                if not content:
                    continue

                # ALTO Hyphenation Logic
                subs_type = child.attrib.get("SUBS_TYPE")
                subs_content = child.attrib.get("SUBS_CONTENT")

                try:
                    x = int(float(child.attrib.get("HPOS", 0)))
                    y = int(float(child.attrib.get("VPOS", 0)))
                    w = int(float(child.attrib.get("WIDTH", 0)))
                    h = int(float(child.attrib.get("HEIGHT", 0)))
                except (ValueError, TypeError):
                    continue

                # Check for explicit visual hyphen tag
                has_hyp_tag = False
                if i + 1 < len(children):
                    next_tag = children[i + 1].tag.split("}")[-1]
                    if next_tag == "HYP":
                        content += children[i + 1].attrib.get("CONTENT", "-")
                        has_hyp_tag = True

                if subs_type == "HypPart1" and subs_content:
                    if not has_hyp_tag and not content.endswith("-"):
                        content += "-"
                    content = f"{content} {{{subs_content}}}"

                words.append(content)
                boxes.append([x, y, x + w, y + h])

    return words, boxes, (page_w, page_h)


def parse_alto_page_labels(xml_path: str) -> List[str]:
    """Every ``<Page>``'s LABEL in an ALTO upload, in document order.

    ``PHYSICAL_IMG_NR`` when present, else the 1-based index — the exact convention
    ``page_split.split_alto_xml`` uses to name its per-page outputs, so a label
    produced here is the same string the batch pipeline would key that page on.

    Added for atrium-project#10 (J1): the /process accretion has to say WHICH page its
    ``pages[]``/``lines[]`` contribution describes, and the label is in the ALTO — it is
    simply not in ``text_inference``'s return value, which reports lines only. Returns
    [] on an unparseable or page-less document; the same hardened parser as the rest of
    this module, since this also sees untrusted uploads.
    """
    try:
        root = ET.parse(xml_path, parser=_SAFE_PARSER).getroot()
    except Exception as e:
        logger.error(f"XML Parse Error in {xml_path}: {e}")
        return []

    ns = {"alto": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    page_tag = ".//alto:Page" if ns else ".//Page"
    return [page.attrib.get("PHYSICAL_IMG_NR") or str(i) for i, page in enumerate(root.findall(page_tag, ns), 1)]


def parse_alto_xml_lines(xml_path: str) -> Tuple[List[str], List[List[int]], Tuple[int, int]]:
    """
    Parses ALTO XML from a file path, grouping text by <TextLine> (one
    bounding box per line spanning all of its words) instead of parse_alto_xml's
    word-level output.

    Line granularity is what the LayoutReader reading-order model and the
    per-line classification pipeline expect (process_alto in text_inference.py);
    the grouping logic mirrors extract_LytRdr_ALTO_2_TXT.parse_alto_xml so the
    service and the batch pipeline reconstruct lines identically, including the
    word-{ground_truth} hyphenation annotation text_util.parse_line_splits
    consumes. Uses the same hardened, XXE-safe parser as parse_alto_xml since
    this also handles untrusted uploads.
    """
    try:
        tree = ET.parse(xml_path, parser=_SAFE_PARSER)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"XML Parse Error in {xml_path}: {e}")
        return [], [], (0, 0)

    ns = {"alto": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    page_tag = ".//alto:Page" if ns else ".//Page"
    text_line_tag = ".//alto:TextLine" if ns else ".//TextLine"

    page = root.find(page_tag, ns)
    if page is None:
        return [], [], (0, 0)

    try:
        page_w = int(float(page.attrib.get("WIDTH", 0)))
        page_h = int(float(page.attrib.get("HEIGHT", 0)))
    except (ValueError, TypeError):
        return [], [], (0, 0)

    lines: List[str] = []
    boxes: List[List[int]] = []

    for line_elem in root.findall(text_line_tag, ns):
        line_text = ""
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        children = list(line_elem)
        for i, child in enumerate(children):
            tag_name = child.tag.split("}")[-1]

            if tag_name == "String":
                content = child.attrib.get("CONTENT")
                if not content:
                    continue

                subs_type = child.attrib.get("SUBS_TYPE")
                subs_content = child.attrib.get("SUBS_CONTENT")

                try:
                    x = int(float(child.attrib.get("HPOS", 0)))
                    y = int(float(child.attrib.get("VPOS", 0)))
                    w = int(float(child.attrib.get("WIDTH", 0)))
                    h = int(float(child.attrib.get("HEIGHT", 0)))
                except (ValueError, TypeError):
                    continue

                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

                has_hyp_tag = False
                if i + 1 < len(children):
                    next_tag = children[i + 1].tag.split("}")[-1]
                    if next_tag == "HYP":
                        content += children[i + 1].attrib.get("CONTENT", "-")
                        has_hyp_tag = True

                if subs_type == "HypPart1" and subs_content:
                    if not has_hyp_tag and not content.endswith("-"):
                        content += "-"
                    content = f"{content} {{{subs_content}}}"

                line_text += content

            elif tag_name == "SP":
                # Preserve native spaces provided by the OCR engine.
                line_text += " "

        line_text = line_text.strip()
        if line_text and min_x != float("inf"):
            lines.append(line_text)
            boxes.append([int(min_x), int(min_y), int(max_x), int(max_y)])

    return lines, boxes, (page_w, page_h)


def post_process_text(ordered_lines: List[str], ordered_boxes: List[List[int]]) -> str:
    """
    Reconstructs text from LayoutReader-reordered line elements, inserting a
    paragraph break ("\\n\\n") on a large vertical gap or a reading-order jump
    upward, and a plain line break otherwise.

    Mirrors extract_LytRdr_ALTO_2_TXT.post_process_text so the service and the
    batch pipeline join lines identically.
    """
    if not ordered_lines:
        return ""

    if ordered_boxes:
        heights = [(b[3] - b[1]) for b in ordered_boxes]
        valid_heights = [h for h in heights if h > 5]
        if not valid_heights:
            valid_heights = heights
        median_height = np.median(valid_heights) if valid_heights else 10
    else:
        median_height = 10

    block_gap_threshold = median_height * 1.5

    result_tokens: List[str] = []
    prev_box = None

    for line_text, box in zip(ordered_lines, ordered_boxes, strict=True):
        if prev_box is None:
            separator = ""
        else:
            curr_top = box[1]
            prev_bottom = prev_box[3]
            vertical_gap = curr_top - prev_bottom

            if vertical_gap < -median_height:
                separator = "\n\n"
            elif vertical_gap > block_gap_threshold:
                separator = "\n\n"
            else:
                separator = "\n"

        result_tokens.append(separator)
        result_tokens.append(line_text)
        prev_box = box

    return "".join(result_tokens).strip()


def normalize_boxes(boxes: List[List[int]], width: int, height: int) -> List[List[int]]:
    """Normalise pixel boxes to the 0-1000 scale LayoutLMv3 expects.

    Mirrors extract_LytRdr_ALTO_2_TXT.normalize_boxes so the service and the
    batch pipeline feed the layout model identical inputs (#8).
    """
    if not boxes or width == 0 or height == 0:
        return [[0, 0, 0, 0] for _ in boxes]
    x_scale = 1000.0 / width
    y_scale = 1000.0 / height
    out: List[List[int]] = []
    for x1, y1, x2, y2 in boxes:
        out.append(
            [
                max(0, min(1000, int(round(x1 * x_scale)))),
                max(0, min(1000, int(round(y1 * y_scale)))),
                max(0, min(1000, int(round(x2 * x_scale)))),
                max(0, min(1000, int(round(y2 * y_scale)))),
            ]
        )
    return out
