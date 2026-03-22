"""
text_util.py
Helper functions for ALTO parsing, box normalization, and text reconstruction.
"""
import re
import sys
import logging
from typing import List, Tuple

# Use lxml for highly efficient XML parsing
import lxml.etree as ET
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# --- Constants ---
COMMON_LANGS = ["ces", "deu", "eng"]
PERPLEXITY_THRESHOLD_MAX = 5000
PERPLEXITY_THRESHOLD_MIN = 1500
LANG_SCORE_ROUGH = 0.45
LANG_SCORE_CLEAR = 0.75

def parse_alto_xml(xml_path: str) -> Tuple[List[str], List[List[int]], Tuple[int, int]]:
    """
    Parses ALTO XML from a file path using fast lxml bindings.
    Returns: words (list), boxes (list), (width, height)
    """
    try:
        tree = ET.parse(xml_path)
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

def categorize_line(lang_code: str, score: float, ppl: float, text: str) -> str:
    """Classifies line as Clear, Noisy, or Trash."""
    is_common = any(lang_code.startswith(cl) for cl in COMMON_LANGS)

    # Heuristic for short lines
    words_count = len(text.split())
    short_line_coef = 2.0 if len(text) < 20 or words_count < 4 else 1.0

    if score > LANG_SCORE_CLEAR and is_common:
        return "Clear"

    if score > LANG_SCORE_ROUGH and ppl < (PERPLEXITY_THRESHOLD_MIN * short_line_coef):
        return "Noisy"

    return "Trash"

# ... (Rest of utility functions similarly typed)