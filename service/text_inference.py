"""
service/text_inference.py
Manages the LayoutReader, FastText, and GPT2 models.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import fasttext
from transformers import LayoutLMv3ForTokenClassification, AutoModelForCausalLM, AutoTokenizer

# --- PATH SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Explicitly import only necessary utilities
try:
    from .utils import parse_alto_xml, categorize_line
    from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
except ImportError:
    from utils import parse_alto_xml, categorize_line
    try:
        from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
    except ImportError:
        print("CRITICAL: 'v3' folder not found in project root.")

try:
    from text_util_langID import (
        categorize_line as _categorize_line_struct,
        detect_strange_symbols,
        detect_mid_uppercase,
    )
except ImportError:
    import warnings
    warnings.warn(
        "text_util_langID not found; falling back to service/utils.py categorize_line.",
        ImportWarning,
        stacklevel=2,
    )
    _categorize_line_struct = None
    detect_strange_symbols = None
    detect_mid_uppercase = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Parameterized paths allowing override via Docker/deployment variables
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(project_root / "models")))
FASTTEXT_MODEL_PATH = MODEL_DIR / "lid.176.bin"

class TextModelManager:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layout_model: Optional[LayoutLMv3ForTokenClassification] = None
        self.ft_model: Optional[fasttext.FastText._FastText] = None
        self.ppl_model: Optional[AutoModelForCausalLM] = None
        self.ppl_tokenizer: Optional[AutoTokenizer] = None
        self._models_loaded = False

    def load_models(self) -> None:
        """Loads models synchronously. Raises RuntimeError if loading fails."""
        if self._models_loaded:
            return

        logger.info(f"Loading Text Processing Models on {self.device}...")

        try:
            # 1. LayoutReader (LayoutLMv3)
            # Add specific paths to layout model logic
            self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained(...) # Specify Path
            self.layout_model.to(self.device)
            self.layout_model.eval()

            # 2. FastText
            self.ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

            # 3. GPT-2 Perplexity Model
            gpt2_path = os.getenv("GPT2_MODEL_NAME", "distilgpt2")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            self.ppl_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
            self.ppl_model.to(self.device)
            self.ppl_model.eval()

            self._models_loaded = True
            logger.info("All models loaded successfully.")

        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            self._models_loaded = False
            # Bubble up the exception to halt the startup sequence
            raise RuntimeError(f"Failed to load core text-processing models: {e}")

    # ... (Rest of the class implementation follows, maintaining type hints)