#!/usr/bin/env python3
"""
extract_LLM_ALTO_2_TXT_fixed.py

Extract text from Page Images using GLM-4v (Multimodal LLM).
Refactored to fix ChatGLMConfig errors and input formatting.
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageFile, ImageOps
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# --- Configuration ---
INPUT_CSV = "alto_statistics_pages.csv"
OUTPUT_TEXT_DIR = "../PAGE_TXT_LLM"
MODEL_PATH = "THUDM/glm-4v-9b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image settings
MAX_RESOLUTION = 3840
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def trim_whitespace(image, padding=20):
    """Crops the image to content bounding box."""
    try:
        gray = ImageOps.grayscale(image)
        inverted = ImageOps.invert(gray)
        bbox = inverted.getbbox()
        if bbox:
            left, upper, right, lower = bbox
            width, height = image.size
            left = max(0, left - padding)
            upper = max(0, upper - padding)
            right = min(width, right + padding)
            lower = min(height, lower + padding)
            return image.crop((left, upper, right, lower))
    except Exception:
        pass # Fallback to original if trim fails
    return image

def resize_if_huge(image, max_dim=MAX_RESOLUTION):
    """Downscales image if too large."""
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > max_dim:
        scale = max_dim / longest_side
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_model():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"Loading configuration from {MODEL_PATH}...")
    # Load the configuration explicitly first
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # --- START OF FIX ---
    # Check if the attribute is missing and patch it
    if not hasattr(config, "num_hidden_layers"):
        # If 'num_layers' exists (ChatGLM standard), use it
        if hasattr(config, "num_layers"):
            print(f"Patching Config: Mapping num_layers ({config.num_layers}) to num_hidden_layers")
            config.num_hidden_layers = config.num_layers
        else:
            # Fallback: Hardcode to 40 (standard for GLM-4-9B) if even num_layers is missing
            # This handles cases where the config object is severely malformed
            print("Patching Config: Hardcoding num_hidden_layers to 40")
            config.num_hidden_layers = 40
            config.num_layers = 40  # Ensure consistency
    # --- END OF FIX ---

    print(f"Loading model from {MODEL_PATH} with patched config...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,  # Pass the patched config object here
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    # Double-check the model's internal config reference just to be safe
    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = model.config.num_layers

    model.eval()
    return tokenizer, model


def extract_single_page_glm(tokenizer, model, image_path):
    """
    Runs inference using tokenizer.apply_chat_template
    """
    try:
        # 1. Load & Preprocess
        image = Image.open(image_path).convert("RGB")
        image = trim_whitespace(image)
        image = resize_if_huge(image)

        # 2. Construct Chat Input
        # GLM-4v expects the image object inside the message list
        messages = [
            {
                "role": "user",
                "image": image,
                "content": "OCR: Transcribe all text on this page exactly as it appears."
            }
        ]

        # 3. Format Inputs using the Chat Template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(DEVICE)

        # 4. Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False, # Deterministic
                temperature=0.1  # Low temperature for accuracy if do_sample were True
            )

        # 5. Decode
        # Slice outputs to remove the input tokens (echo)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    except Exception as e:
        # Check for CUDA OOM specifically to give better feedback
        if "CUDA out of memory" in str(e):
            print(f"⚠️ OOM Error on {image_path}. Skipping.")
            torch.cuda.empty_cache()
        else:
            print(f"⚠️ Error processing {image_path}: {e}")
        return None

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    has_image_col = 'image_path' in df.columns

    tokenizer, model = load_model()

    Path(OUTPUT_TEXT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Starting extraction for {len(df)} pages on {DEVICE}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row['file']
        page_id = row['page']

        # --- Path Logic ---
        if has_image_col and pd.notna(row['image_path']):
            image_path = Path(row['image_path'].replace(".alto", ""))
        else:
            xml_path = Path(row['path'])
            image_dir = xml_path.parent
            page_str = str(page_id).zfill(2)
            filename = f"{file_id}-{page_str}.png"
            image_path = image_dir / filename

        # --- Validation ---
        if not image_path.exists():
            backup_image_path = image_path.parents[1] / "onepagers" / image_path.name
            if backup_image_path.exists():
                image_path = backup_image_path
            else:
                # Silent skip to keep progress bar clean, enable logging if needed
                continue

        # --- Output Check ---
        save_dir = Path(OUTPUT_TEXT_DIR) / str(file_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        txt_path = save_dir / f"{file_id}-{page_id}.txt"

        if txt_path.exists():
            continue

        # --- Inference ---
        text = extract_single_page_glm(tokenizer, model, image_path)

        if text:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

    print("Done.")

if __name__ == "__main__":
    main()