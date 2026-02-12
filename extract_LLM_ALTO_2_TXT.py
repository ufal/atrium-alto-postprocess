#!/usr/bin/env python3
"""
extract_LLM_ALTO_2_TXT_fixed.py

Extract text from Page Images using GLM-4v (Multimodal LLM).
Refactored to fix ChatGLMConfig errors and input formatting.
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from PIL import Image, ImageFile, ImageOps
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "alto_statistics_pages.csv"
OUTPUT_TEXT_DIR = "../PAGE_TXT_LLM"
MODEL_PATH = "THUDM/glm-4v-9b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image settings
MAX_RESOLUTION = 1344
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
        pass
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
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # --- FIX 1: Patch Tokenizer ---
    if not hasattr(tokenizer, "batch_encode_plus"):
        def patched_batch_encode_plus(batch_text_or_text_pairs, **kwargs):
            return tokenizer(batch_text_or_text_pairs, **kwargs)

        tokenizer.batch_encode_plus = patched_batch_encode_plus

    # --- FIX 2: Patch Config for Architecture ---
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = getattr(config, "num_layers", 40)

    # --- FIX 3: Patch Config for Init ---
    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 8192)

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    # --- FIX 4: Cleanup Config for Generation ---
    # 1. Remove max_length to prevent the "modified config" error
    if hasattr(model.config, "max_length"):
        del model.config.max_length

    # 2. Silence warnings about 'temperature' and 'top_p' being used with do_sample=False
    # We explicitly unset them in the model's internal generation config
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    model.eval()
    return tokenizer, model


def extract_single_page_glm(tokenizer, model, image_path):
    try:
        # 1. Load & Preprocess
        image = Image.open(image_path).convert("RGB")
        image = trim_whitespace(image)
        image = resize_if_huge(image)

        # 2. Construct Chat Input
        messages = [
            {
                "role": "user",
                "image": image,
                "content": "OCR: Transcribe all text on this page exactly as it appears."
            }
        ]

        # 3. Format Inputs
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
                max_new_tokens=4096,
                do_sample=False,  # Deterministic (Greedy Search)
                # temperature=0.1,      <-- REMOVED to stop warning
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 5. Decode
        input_length = inputs['input_ids'].shape[1]
        output_tokens = outputs[:, input_length:]

        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return generated_text

    except Exception as e:
        if "CUDA out of memory" in str(e):
            print(f"⚠️ OOM Error on {image_path}. Skipping.")
            torch.cuda.empty_cache()
        else:
            print(f"⚠️ Error processing {image_path}: {e}")
        return None

    except Exception as e:
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