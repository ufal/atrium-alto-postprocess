"""
text_inference.py
Manages the LayoutReader, FastText, and GPT2 models.
"""
import os
import sys
import logging
from pathlib import Path
import torch
import fasttext
from transformers import LayoutLMv3ForTokenClassification, AutoModelForCausalLM, AutoTokenizer

# Import utility functions
try:
    from .utils import *
    # Assumes v3/helpers exist in path for LayoutReader
    from .v3.helpers import prepare_inputs, boxes2inputs, parse_logits
except ImportError:
    from utils import *

    # Fallback/Mock for testing if v3 is missing
    try:
        from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
    except:
        print("Warning: LayoutReader v3.helpers not found.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
# Adjust these to where your models actually live
MODEL_DIR = Path(__file__).parent / "models"
FASTTEXT_MODEL_PATH = MODEL_DIR / "lid.176.bin"


class TextModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layout_model = None
        self.ft_model = None
        self.ppl_model = None
        self.ppl_tokenizer = None
        self._models_loaded = False

    def load_models(self):
        """Lazy loader for models to save startup time."""
        if self._models_loaded: return

        logger.info(f"Loading Text Processing Models on {self.device}...")

        # 1. LayoutReader (LayoutLMv3)
        try:
            self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
            self.layout_model.to(self.device)
            self.layout_model.eval()
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3: {e}")

        # 2. FastText (Language ID)
        if FASTTEXT_MODEL_PATH.exists():
            self.ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
        else:
            logger.warning(f"FastText model not found at {FASTTEXT_MODEL_PATH}")

        # 3. DistilGPT2 (Perplexity)
        try:
            self.ppl_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.ppl_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)
            self.ppl_model.eval()
        except Exception as e:
            logger.error(f"Failed to load DistilGPT2: {e}")

        self._models_loaded = True
        logger.info("Models loaded successfully.")

    def process_alto(self, file_path):
        """Full pipeline: ALTO -> LayoutReader -> Raw Text -> Cleaned Text"""
        self.load_models()

        # 1. Extract Words & Boxes
        words, boxes, (w, h) = parse_alto_xml(file_path)
        if not words:
            return {"raw_text": "", "cleaned_lines": []}

        # 2. Normalize
        norm_boxes = normalize_boxes(boxes, w, h)

        # 3. Layout Inference (Chunked for memory safety)
        full_ordered_words = []
        full_ordered_boxes = []
        CHUNK_SIZE = 350  # Safe limit for LayoutLM

        for i in range(0, len(words), CHUNK_SIZE):
            b_words = words[i:i + CHUNK_SIZE]
            b_boxes = norm_boxes[i:i + CHUNK_SIZE]

            if not b_words: continue

            # LayoutReader Inference logic
            try:
                inputs = boxes2inputs(b_boxes)
                inputs = prepare_inputs(inputs, self.layout_model)

                # Move tensors to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)

                with torch.no_grad():
                    logits = self.layout_model(**inputs).logits.cpu().squeeze(0)

                order_indices = parse_logits(logits, len(b_boxes))

                full_ordered_words.extend([b_words[idx] for idx in order_indices])
                full_ordered_boxes.extend([b_boxes[idx] for idx in order_indices])
            except Exception as e:
                logger.error(f"LayoutReader Error on chunk {i}: {e}")
                # Fallback: append in original order
                full_ordered_words.extend(b_words)
                full_ordered_boxes.extend(b_boxes)

        # 4. Reconstruct Raw Text
        raw_text = post_process_layout(full_ordered_words, full_ordered_boxes)

        # 5. Clean / Filter Text
        cleaned_data = self._clean_text_lines(raw_text.split('\n'))

        return {
            "type": "alto_xml",
            "raw_text": raw_text,
            "cleaned_lines": cleaned_data
        }

    def process_text_file(self, file_path):
        """Pipeline for raw text files."""
        self.load_models()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # Fallback for older encodings
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        cleaned_data = self._clean_text_lines(lines)

        return {
            "type": "plain_text",
            "line_count": len(lines),
            "cleaned_lines": cleaned_data
        }

    def _clean_text_lines(self, lines, batch_size=64):
        """Internal batch processing for LangID and Perplexity."""
        results = []
        batch_text = []
        batch_indices = []

        # State for split word reconstruction
        expected_suffix = ""

        for i, line in enumerate(lines):
            # 1. Merge Splits
            merged, prefix, suffix = parse_line_splits(line)

            current_ws = prefix
            current_we = ""

            if expected_suffix:
                stripped = merged.lstrip()
                if stripped.startswith(expected_suffix):
                    merged = merged.replace(expected_suffix, "", 1).strip()
                    current_we = expected_suffix

            expected_suffix = suffix

            # 2. Filter Empty/Garbage
            if len(merged) < 3:
                continue

            batch_text.append(merged)
            batch_indices.append({
                "line_num": i + 1,
                "text": merged,
                "split_start": current_ws,
                "split_end": current_we
            })

            # Process Batch
            if len(batch_text) >= batch_size:
                self._run_batch_metrics(batch_text, batch_indices, results)
                batch_text = []
                batch_indices = []

        # Final Batch
        if batch_text:
            self._run_batch_metrics(batch_text, batch_indices, results)

        return results

    def _run_batch_metrics(self, texts, metadata, output_list):
        """Runs GPU models on a batch."""
        if not self.ft_model or not self.ppl_model:
            # If models failed to load, return raw
            for m in metadata:
                m.update({"category": "Unknown", "lang": "N/A", "ppl": 0})
                output_list.append(m)
            return

        # 1. Perplexity
        ppls = calculate_perplexity_batch(texts, self.ppl_model, self.ppl_tokenizer, self.device)

        # 2. FastText
        # FastText expects lowercase usually, check your specific model requirements
        preds = self.ft_model.predict([t.lower().replace("\n", " ") for t in texts], k=1)
        labels, scores = preds

        for i, meta in enumerate(metadata):
            lang = labels[i][0].replace("__label__", "")
            score = scores[i][0]
            ppl = ppls[i]

            category = categorize_line(lang, score, ppl, meta['text'])

            meta.update({
                "lang": lang,
                "lang_conf": round(float(score), 4),
                "perplexity": round(ppl, 2),
                "category": category
            })
            output_list.append(meta)


# Singleton Instance
text_manager = TextModelManager()