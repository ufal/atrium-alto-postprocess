"""
service/utils.py
Helper functions for ALTO parsing, box normalization, and text reconstruction.
"""
import re
import sys
import logging
import configparser
from pathlib import Path
from typing import List, Tuple

# Use lxml for highly efficient XML parsing
import lxml.etree as ET
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
_config = configparser.RawConfigParser()
# Locate config_langID.txt in the project root
_config_path = Path(__file__).resolve().parent.parent / "config_langID.txt"
if _config_path.exists():
    _config.read(_config_path)

def _get_float(section, key, default):
    return _config.getfloat(section, key, fallback=default) if _config.has_section(section) else default

if _config.has_section("CLASSIFY") and _config.has_option("CLASSIFY", "EXPECTED_LANGS"):
    COMMON_LANGS = [lang.strip() for lang in _config.get("CLASSIFY", "EXPECTED_LANGS").split(",") if lang.strip()]
else:
    COMMON_LANGS = ["ces", "deu", "eng"]

# Sourced directly from config_langID.txt rather than hard-coding stale values
PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)
PERPLEXITY_THRESHOLD_MIN = _get_float("TEXT_UTILS", "SHORT_PPL_CAP", 850.0)
LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)
LANG_SCORE_CLEAR = _get_float("TEXT_UTILS", "LANG_SCORE_CLEAR", 0.75)

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
    ns = {'alto': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    # Optimized pre-compiled tags
    page_tag = './/alto:Page' if ns else './/Page'
    text_line_tag = './/alto:TextLine' if ns else './/TextLine'

    page = root.find(page_tag, ns)
    if page is None:
        return [], [], (0, 0)

    try:
        page_w = int(float(page.attrib.get('WIDTH', 0)))
        page_h = int(float(page.attrib.get('HEIGHT', 0)))
    except (ValueError, TypeError):
        return [], [], (0, 0)

    words: List[str] = []
    boxes: List[List[int]] = []

    text_lines = root.findall(text_line_tag, ns)

    for line in text_lines:
        children = list(line)
        for i, child in enumerate(children):
            tag_name = child.tag.split('}')[-1]

            if tag_name == 'String':
                content = child.attrib.get('CONTENT')
                if not content: continue

                # ALTO Hyphenation Logic
                subs_type = child.attrib.get('SUBS_TYPE')
                subs_content = child.attrib.get('SUBS_CONTENT')

                try:
                    x = int(float(child.attrib.get('HPOS', 0)))
                    y = int(float(child.attrib.get('VPOS', 0)))
                    w = int(float(child.attrib.get('WIDTH', 0)))
                    h = int(float(child.attrib.get('HEIGHT', 0)))
                except (ValueError, TypeError):
                    continue

                # Check for explicit visual hyphen tag
                has_hyp_tag = False
                if i + 1 < len(children):
                    next_tag = children[i + 1].tag.split('}')[-1]
                    if next_tag == 'HYP':
                        content += children[i + 1].attrib.get('CONTENT', '-')
                        has_hyp_tag = True

                if subs_type == 'HypPart1' and subs_content:
                    if not has_hyp_tag and not content.endswith('-'):
                        content += "-"
                    content = f"{content} {{{subs_content}}}"

                words.append(content)
                boxes.append([x, y, x + w, y + h])

    return words, boxes, (page_w, page_h)


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
        out.append([
            max(0, min(1000, int(round(x1 * x_scale)))),
            max(0, min(1000, int(round(y1 * y_scale)))),
            max(0, min(1000, int(round(x2 * x_scale)))),
            max(0, min(1000, int(round(y2 * y_scale)))),
        ])
    return out


def categorize_line(lang_code: str, score: float, ppl: float, text: str, weird_ratio: float = 0.0) -> str:
    """Legacy fallback classifier (Clear / Noisy / Trash).

    Only used by service/text_inference.py when text_util_langID cannot be
    imported. The primary path uses text_util_langID.categorize_line.
    """
    is_common = any(lang_code.startswith(cl) for cl in COMMON_LANGS)

    # Hard override matching the new logic
    if ppl > PERPLEXITY_THRESHOLD_MAX and weird_ratio > 0.4:
        return "Trash"

    # Heuristic for short lines
    words_count = len(text.split())
    short_line_coef = 2.0 if len(text) < 20 or words_count < 4 else 1.0

    if score > LANG_SCORE_CLEAR and is_common:
        return "Clear"

    if score > LANG_SCORE_ROUGH and ppl < (PERPLEXITY_THRESHOLD_MIN * short_line_coef):
        return "Noisy"

    return "Trash"