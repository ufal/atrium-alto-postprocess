#!/usr/bin/env python3
"""
extract_LytRdr_ALTO_2_TXT.py

Step 1: Extract and reorder text from ALTO XML files using LayoutReader in parallel.
Fixed: Switched from word-level to line-level extraction. Trusts ABBYY's <TextLine>
       grouping to preserve tables and justified text structures.
"""

import pandas as pd
import concurrent.futures
import os
import sys
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import torch
from transformers import LayoutLMv3ForTokenClassification
import numpy as np

# --- Path Setup to find 'v3' ---
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
if str(script_dir.parent) not in sys.path:
    sys.path.append(str(script_dir.parent))

try:
    # Assumes you have the 'v3' folder from the LayoutReader repo available
    from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
except ImportError:
    try:
        from layoutreader.v3.helpers import prepare_inputs, boxes2inputs, parse_logits
    except ImportError:
        print("\nCRITICAL ERROR: Could not import 'v3.helpers'.")
        print("Ensure the 'v3' folder from the LayoutReader repository is in your python path.")
        sys.exit(1)

# --- Configuration ---
INPUT_CSV = "alto_statistics.csv"
OUTPUT_TEXT_DIR = "../PAGE_TXT_LR"
MAX_WORKERS = 1  # Set to 1 for GPU, higher for CPU

# Global variables
model = None
device = None


def init_worker():
    """Initializer for worker processes."""
    global model, device

    # LIMIT THREADS: Crucial for CPU parallel inference stability
    torch.set_num_threads(4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["transformers_verbosity"] = "error"
    try:
        model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model in worker: {e}")
        sys.exit(1)


def parse_alto_xml(xml_path):
    """Parses ALTO XML to extract unified TextLines and their collective bounding boxes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
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

    lines = []
    boxes = []

    text_lines_elements = find_all(root, 'TextLine')
    for line_elem in text_lines_elements:
        line_text = ""
        # Initialize extremes for the unified line bounding box
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        children = list(line_elem)
        for i, child in enumerate(children):
            tag_name = child.tag.split('}')[-1]

            if tag_name == 'String':
                content = child.attrib.get('CONTENT')
                if not content:
                    continue

                # Get Hyphenation Ground Truth
                subs_type = child.attrib.get('SUBS_TYPE')
                subs_content = child.attrib.get('SUBS_CONTENT')

                try:
                    x = int(float(child.attrib.get('HPOS')))
                    y = int(float(child.attrib.get('VPOS')))
                    w = int(float(child.attrib.get('WIDTH')))
                    h = int(float(child.attrib.get('HEIGHT')))
                except (ValueError, TypeError):
                    continue

                # Expand the line's bounding box to encompass this string
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

                # Check for explicit hyphenation tag in ALTO
                has_hyp_tag = False
                if i + 1 < len(children):
                    next_child = children[i + 1]
                    next_tag = next_child.tag.split('}')[-1]
                    if next_tag == 'HYP':
                        content += next_child.attrib.get('CONTENT', '-')
                        has_hyp_tag = True

                # --- Inject Ground Truth ---
                if subs_type == 'HypPart1' and subs_content:
                    if not has_hyp_tag and not content.endswith('-'):
                        content += "-"
                    content = f"{content} {{{subs_content}}}"

                line_text += content

            elif tag_name == 'SP':
                # Preserve native spaces provided by OCR engine
                line_text += " "

        line_text = line_text.strip()
        # Only append if we actually found valid text and coordinates
        if line_text and min_x != float('inf'):
            lines.append(line_text)
            boxes.append([min_x, min_y, max_x, max_y])

    return lines, boxes, (page_w, page_h)


def normalize_boxes(boxes, width, height):
    """Normalize boxes to 0-1000 scale."""
    normalized = []
    if width == 0 or height == 0:
        return [[0, 0, 0, 0] for _ in boxes]
    x_scale = 1000.0 / width
    y_scale = 1000.0 / height
    for box in boxes:
        x1, y1, x2, y2 = box
        nx1 = max(0, min(1000, int(round(x1 * x_scale))))
        ny1 = max(0, min(1000, int(round(y1 * y_scale))))
        nx2 = max(0, min(1000, int(round(x2 * x_scale))))
        ny2 = max(0, min(1000, int(round(y2 * y_scale))))
        normalized.append([nx1, ny1, nx2, ny2])
    return normalized


def post_process_text(ordered_lines, ordered_boxes):
    """
    Reconstructs text from reordered line elements.
    LayoutReader has sequenced the lines; we just need to join them logically.
    """
    if not ordered_lines:
        return ""

    # Calculate Median Height (Proxy for standard line height)
    if ordered_boxes:
        heights = [(b[3] - b[1]) for b in ordered_boxes]
        valid_heights = [h for h in heights if h > 5]
        if not valid_heights: valid_heights = heights
        median_height = np.median(valid_heights) if valid_heights else 10
    else:
        median_height = 10

    # Vertical distance to denote a new text block/paragraph.
    BLOCK_GAP_THRESHOLD = median_height * 1.5

    result_tokens = []
    prev_box = None

    for i, (line_text, box) in enumerate(zip(ordered_lines, ordered_boxes)):
        separator = ""

        if prev_box is None:
            separator = ""
        else:
            curr_top = box[1]
            prev_bottom = prev_box[3]
            vertical_gap = curr_top - prev_bottom

            # Logic: Should we start a new block or just a new line?

            # Case A: Reading order jumps UP (New Column or Page Reset)
            if vertical_gap < -median_height:
                separator = "\n\n"

            # Case B: Significant Drop (New Paragraph / Separated Text Area)
            elif vertical_gap > BLOCK_GAP_THRESHOLD:
                separator = "\n\n"

            # Case C: Standard Line Continuation
            else:
                separator = "\n"

        result_tokens.append(separator)
        result_tokens.append(line_text)
        prev_box = box

    final_text = "".join(result_tokens)
    return final_text.strip()


def extract_single_page(args):
    """Worker function to process one page."""
    file_id, page_id, xml_path_str, output_dir = args
    global model, device

    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    if txt_path.exists():
        return True

    xml_path = Path(xml_path_str)
    # Basic fallback logic
    if not xml_path.exists():
        backup_xml_path = xml_path.parents[1] / "onepagers" / xml_path.name
        if backup_xml_path.exists():
            xml_path = backup_xml_path
        else:
            return False

    try:
        # 1. Parse content (Now yielding lines, not words)
        lines, boxes, (page_w, page_h) = parse_alto_xml(xml_path)

        if not lines:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("")
            return True

        # 2. Normalize boxes
        norm_boxes = normalize_boxes(boxes, page_w, page_h)

        full_ordered_lines = []
        full_ordered_boxes = []

        # 3. Process in Chunks
        # CHUNK_SIZE of 350 will easily process entire pages in one go now
        # since we are passing lines, not words.
        CHUNK_SIZE = 350

        for i in range(0, len(lines), CHUNK_SIZE):
            chunk_lines = lines[i: i + CHUNK_SIZE]
            chunk_boxes = norm_boxes[i: i + CHUNK_SIZE]

            if not chunk_lines:
                continue

            try:
                inputs = boxes2inputs(chunk_boxes)
                inputs = prepare_inputs(inputs, model)

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)

                with torch.no_grad():
                    logits = model(**inputs).logits.cpu().squeeze(0)

                # Reordering
                order_indices = parse_logits(logits, len(chunk_boxes))

                chunk_ordered_lines = [chunk_lines[idx] for idx in order_indices]
                chunk_ordered_boxes = [chunk_boxes[idx] for idx in order_indices]

                full_ordered_lines.extend(chunk_ordered_lines)
                full_ordered_boxes.extend(chunk_ordered_boxes)

            except RuntimeError as e:
                if "memory" in str(e).lower():
                    print(f"Skipping {file_id}-{page_id} due to Memory Error.")
                    return False
                raise e

        # 4. Generate text
        final_text = post_process_text(full_ordered_lines, full_ordered_boxes)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        return True

    except Exception as e:
        print(f"Error processing {file_id}-{page_id}: {e}")
        return False


def main():
    if not Path(INPUT_CSV).exists():
        print(f"Error: {INPUT_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} pages to extract.")

    tasks = []
    for _, row in df.iterrows():
        # Ensure your CSV has these columns
        tasks.append((row['file'], row['page'], row['path'], OUTPUT_TEXT_DIR))

    use_cuda = torch.cuda.is_available()
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    if use_cuda:
        # CUDA: Sequential execution
        print("CUDA detected: Running sequentially.")
        init_worker()
        results = []
        for task in tqdm(tasks, desc="Processing (GPU)"):
            results.append(extract_single_page(task))
    else:
        # CPU: Parallel execution
        print(f"CPU detected: Extracting with {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                initializer=init_worker
        ) as executor:
            results = list(
                tqdm(executor.map(extract_single_page, tasks, chunksize=1), total=len(tasks), desc="Processing (CPU)"))

    success_count = sum(results)
    print(f"Extraction complete. Success rate: {success_count / len(results):.2%}")


if __name__ == "__main__":
    main()