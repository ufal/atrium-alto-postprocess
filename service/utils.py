"""
text_util.py
Helper functions for ALTO parsing, box normalization, and text reconstruction.
"""
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch import nn

# --- Constants ---
COMMON_LANGS = ["ces", "deu", "eng"]
PERPLEXITY_THRESHOLD_MAX = 5000
PERPLEXITY_THRESHOLD_MIN = 1500
LANG_SCORE_ROUGH = 0.45
LANG_SCORE_CLEAR = 0.75


def parse_alto_xml(xml_path):
    """
    Parses ALTO XML from a file path.
    Returns: words (list), boxes (list), (width, height)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"XML Parse Error: {e}")
        return [], [], (0, 0)

    ns = {'alto': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    def find_all(node, tag):
        return node.findall(f'.//alto:{tag}', ns) if ns else node.findall(f'.//{tag}')

    page = root.find('.//alto:Page', ns) if ns else root.find('.//Page')
    if page is None:
        return [], [], (0, 0)

    try:
        page_w = int(float(page.attrib.get('WIDTH')))
        page_h = int(float(page.attrib.get('HEIGHT')))
    except (ValueError, TypeError):
        return [], [], (0, 0)

    words = []
    boxes = []

    text_lines = find_all(root, 'TextLine')
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
                    x = int(float(child.attrib.get('HPOS')))
                    y = int(float(child.attrib.get('VPOS')))
                    w = int(float(child.attrib.get('WIDTH')))
                    h = int(float(child.attrib.get('HEIGHT')))
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


def normalize_boxes(boxes, width, height):
    """Normalizes coordinates to 0-1000 scale."""
    if width == 0 or height == 0:
        return [[0, 0, 0, 0] for _ in boxes]

    x_scale = 1000.0 / width
    y_scale = 1000.0 / height

    norm_boxes = []
    for (x1, y1, x2, y2) in boxes:
        nx1 = max(0, min(1000, int(round(x1 * x_scale))))
        ny1 = max(0, min(1000, int(round(y1 * y_scale))))
        nx2 = max(0, min(1000, int(round(x2 * x_scale))))
        ny2 = max(0, min(1000, int(round(y2 * y_scale))))
        norm_boxes.append([nx1, ny1, nx2, ny2])
    return norm_boxes


def post_process_layout(ordered_words, ordered_boxes):
    """
    Reconstructs text with visual logic (newlines vs spaces).
    """
    if not ordered_words: return ""

    # Calculate Median Height for relative spacing
    heights = [(b[3] - b[1]) for b in ordered_boxes]
    valid_heights = [h for h in heights if h > 5]
    median_height = np.median(valid_heights) if valid_heights else 10

    OVERLAP_THRESHOLD = 0.5
    WIDE_GAP_THRESHOLD = median_height * 2.0
    BLOCK_GAP_THRESHOLD = median_height * 1.5

    result_tokens = []
    prev_box = None

    for i, (word, box) in enumerate(zip(ordered_words, ordered_boxes)):
        separator = ""
        if prev_box:
            curr_top = box[1]
            prev_bottom = prev_box[3]

            # Vertical Overlap
            y1_max = max(prev_box[1], box[1])
            y2_min = min(prev_box[3], box[3])
            intersection = max(0, y2_min - y1_max)
            min_h = min(prev_box[3] - prev_box[1], box[3] - box[1])
            overlap_ratio = intersection / min_h if min_h > 0 else 0

            if overlap_ratio > OVERLAP_THRESHOLD:
                # Same line
                h_gap = box[0] - prev_box[2]
                if h_gap > WIDE_GAP_THRESHOLD:
                    separator = "  "
                elif h_gap < 0:
                    separator = ""
                else:
                    separator = " "
            else:
                # New line
                vertical_gap = curr_top - prev_bottom
                if vertical_gap < -median_height:
                    separator = "\n"  # Column jump
                elif vertical_gap > BLOCK_GAP_THRESHOLD:
                    separator = "\n"  # Paragraph break
                else:
                    separator = " "  # Line wrap

        if separator == "\n":
            while result_tokens and result_tokens[-1] in [" ", "  "]: result_tokens.pop()
            if result_tokens and result_tokens[-1] != "\n": result_tokens.append("\n")
        elif separator == "  ":
            if result_tokens and result_tokens[-1] != "\n": result_tokens.append("  ")
        elif separator == " ":
            if result_tokens and result_tokens[-1] not in ["\n", " ", "  "]: result_tokens.append(" ")

        result_tokens.append(word)
        prev_box = box

    return "".join(result_tokens).strip()


def parse_line_splits(line_text: str):
    """
    Detects and merges split words like "divi- {divided}".
    Returns: merged_text, prefix, suffix
    """
    clean_line = line_text.strip()
    # Regex for "prefix- {complete}"
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"

    matches = list(re.finditer(pattern, clean_line))
    if not matches: return clean_line, "", ""

    last_prefix = ""
    last_suffix = ""

    def replace_match(match):
        nonlocal last_prefix, last_suffix
        prefix = match.group(1)
        content = match.group(2)
        if content.startswith(prefix):
            last_suffix = content[len(prefix):]
        last_prefix = prefix
        return content

    merged_text = re.sub(pattern, replace_match, clean_line)
    return merged_text, last_prefix, last_suffix


def calculate_perplexity_batch(texts, model, tokenizer, device):
    """Batched Perplexity calculation using DistilGPT2."""
    if not texts: return []
    try:
        max_len = model.config.max_position_embeddings
        tokenizer.pad_token = tokenizer.eos_token

        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            # Shift logits for token-by-token prediction
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(target_ids.size(0), -1)

            # Mask out padding
            non_masked = (shift_labels != -100)
            seq_loss = (loss * non_masked).sum(dim=1)
            num_tokens = non_masked.sum(dim=1).clamp(min=1)

            ppl = torch.exp(seq_loss / num_tokens)
            return ppl.tolist()

    except Exception:
        return [0.0] * len(texts)


def categorize_line(lang_code, score, ppl, text):
    """Classifies line as Clear, Noisy, or Trash."""
    is_common = any(lang_code.startswith(cl) for cl in COMMON_LANGS)

    # Heuristic for short lines
    words_count = len(text.split())
    short_line_coef = 2.0 if len(text) < 20 or words_count < 4 else 1.0

    if score > LANG_SCORE_CLEAR and is_common:
        return "Clear"

    if (ppl >= PERPLEXITY_THRESHOLD_MAX * short_line_coef or score <= LANG_SCORE_ROUGH) and not is_common:
        return "Trash"

    if ppl >= PERPLEXITY_THRESHOLD_MIN * short_line_coef or score <= LANG_SCORE_CLEAR or not is_common:
        return "Noisy"

    return "Clear"