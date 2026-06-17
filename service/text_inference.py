
"""
service/text_inference.py
Manages the LayoutReader, FastText, and DistilGPT2 models.

Classification is fully aligned with the main pipeline (langID_classify.py):
  - Unified penalty path : categorize_line() from text_util_langID
  - New API fields       : word_weird, garbage_density, ldl_fuses, etc.
"""
"""
service/text_inference.py
Manages the LayoutReader, FastText, and DistilGPT2/Qwen models.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from .utils import parse_alto_xml, normalize_boxes
except ImportError:
    from utils import parse_alto_xml, normalize_boxes

try:
    from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
except ImportError:
    print("CRITICAL: 'v3' folder not found in project root — layout reordering unavailable.")
    prepare_inputs = boxes2inputs = parse_logits = None  # type: ignore[assignment]

# Import the full quality-analysis toolkit
try:
    from text_util_langID import (
        compute_garbage_density, detect_strange_symbols, detect_mid_uppercase,
        detect_repeated_chars, detect_letter_digit_letter, detect_gibberish_words,
        score_words_in_line, compute_word_weird_ratio, compute_valid_ratio,
        compute_symbol_ratio, compute_quality_score, categorize_line as _categorize_line_struct,
        pre_filter_line, calculate_perplexity_batch, COMMON_LANGS
    )

    _UTIL_AVAILABLE = True
except ImportError as _err:
    logging.getLogger(__name__).warning(
        "text_util_langID not found (%s); falling back to legacy utils.", _err
    )
    _UTIL_AVAILABLE = False
    from utils import categorize_line as _legacy_categorize  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(project_root / "models")))
FASTTEXT_MODEL_PATH = MODEL_DIR / "lid.176.bin"


class TextModelManager:
    def __init__(self) -> None:
        self.device = "cpu"  # Initialized here, updated properly in load_models
        self.layout_model: Optional[Any] = None
        self.ft_model: Optional[Any] = None
        self.ppl_model: Optional[Any] = None
        self.ppl_tokenizer: Optional[Any] = None
        self._models_loaded = False

    def load_models(self) -> None:
        """Load all models synchronously; raise RuntimeError on failure."""
        if self._models_loaded:
            return

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Text Processing Models on %s …", self.device)

        try:
            # LAZY LOAD heavy ML libraries strictly inside this method
            import fasttext
            from transformers import LayoutLMv3ForTokenClassification, AutoModelForCausalLM, AutoTokenizer

            # 1. LayoutReader (LayoutLMv3)
            layout_model_path = os.getenv("LAYOUT_MODEL_PATH", "hantian/layoutreader")
            self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained(layout_model_path)
            self.layout_model.to(self.device)
            self.layout_model.eval()

            # 2. FastText language identification
            self.ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

            # 3. DistilGPT2 perplexity model
            gpt2_path = os.getenv("GPT2_MODEL_NAME", "distilgpt2")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            self.ppl_tokenizer.pad_token = self.ppl_tokenizer.eos_token
            self.ppl_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
            self.ppl_model.to(self.device)
            self.ppl_model.eval()

            self._models_loaded = True
            logger.info("All models loaded successfully.")

        except Exception as exc:
            logger.error("Critical error loading models: %s", exc)
            self._models_loaded = False
            raise RuntimeError(f"Failed to load core text-processing models: {exc}") from exc


# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from .utils import parse_alto_xml
except ImportError:
    from utils import parse_alto_xml

try:
    from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
except ImportError:
    print("CRITICAL: 'v3' folder not found in project root — layout reordering unavailable.")
    prepare_inputs = boxes2inputs = parse_logits = None  # type: ignore[assignment]

# Import the full quality-analysis toolkit from the main pipeline module.
try:
    from text_util_langID import (
        compute_garbage_density,
        detect_strange_symbols,
        detect_mid_uppercase,
        detect_repeated_chars,
        detect_letter_digit_letter,
        detect_gibberish_words,
        score_words_in_line,
        compute_word_weird_ratio,
        compute_valid_ratio,
        compute_symbol_ratio,
        compute_quality_score,
        categorize_line as _categorize_line_struct,
        pre_filter_line,
        calculate_perplexity_batch,
        COMMON_LANGS
    )

    _UTIL_AVAILABLE = True
except ImportError as _err:
    logging.getLogger(__name__).warning(
        "text_util_langID not found (%s); falling back to legacy utils.categorize_line.", _err
    )
    _UTIL_AVAILABLE = False
    from utils import categorize_line as _legacy_categorize  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(project_root / "models")))
FASTTEXT_MODEL_PATH = MODEL_DIR / "lid.176.bin"


# ---------------------------------------------------------------------------
# Helper: classify one line (mirrors process_and_write_batch in langID_classify)
# ---------------------------------------------------------------------------

def _classify_line(
        text: str,
        ppl: float,
        *,
        ft_model,
        ppl_model,
        tokenizer,
        device: str,
) -> Dict[str, Any]:
    """
    Run the full unified classification pipeline on a single text line and
    return all quality metrics.
    """
    # 1. Language Identification
    labels, scores = ft_model.predict([text.lower()], k=1)
    lang = labels[0][0].replace("__label__", "")
    lang_score = float(scores[0][0])

    # 2. Extract Structural Metrics
    sym_count = detect_strange_symbols(text)
    upper_count = detect_mid_uppercase(text)
    rep_count = detect_repeated_chars(text)
    fuse_count = detect_letter_digit_letter(text)
    gibb_count = detect_gibberish_words(text)
    g_density = compute_garbage_density(text)

    # 3. Weirdness and Quality Scores
    word_scores = score_words_in_line(text)
    weird_ratio = compute_word_weird_ratio(word_scores)
    q_score = compute_quality_score(
        valid_word_ratio=compute_valid_ratio(text),
        symbol_ratio=compute_symbol_ratio(text),
        perplexity=ppl,
        text_length=len(text),
    )

    # 4. Unified Categorization Logic (passes weird_ratio to prevent flip-flopping)
    categ = _categorize_line_struct(
        ppl=ppl,
        text_source=text,
        lang=lang,
        lang_score=lang_score,
        weird_ratio=weird_ratio,
        expected_langs=COMMON_LANGS
    )

    return {
        "text": text,
        "lang": lang,
        "lang_score": round(lang_score, 4),
        "perplexity": round(ppl, 2),
        "garbage_density": round(g_density, 4),
        "sym_count": sym_count,
        "upper_count": upper_count,
        "repeated_count": rep_count,
        "ldl_fuses": fuse_count,
        "gibberish": gibb_count,
        "word_weird": round(weird_ratio, 4),
        "quality_score": round(q_score, 4),
        "category": categ,
    }


def _classify_line_legacy(text: str, ppl: float, ft_model) -> Dict[str, Any]:
    """
    Fallback when text_util_langID is unavailable.
    """
    labels, scores = ft_model.predict([text.lower()], k=1)
    lang = labels[0][0].replace("__label__", "")
    lang_score = float(scores[0][0])
    categ = _legacy_categorize(lang, lang_score, ppl, text, weird_ratio=0.0)  # type: ignore[call-arg]

    return {
        "text": text,
        "lang": lang,
        "lang_score": round(lang_score, 4),
        "perplexity": round(ppl, 2),
        "garbage_density": None,
        "sym_count": None,
        "upper_count": None,
        "repeated_count": None,
        "ldl_fuses": None,
        "gibberish": None,
        "word_weird": None,
        "quality_score": None,
        "category": categ,
    }

# Module-level singleton used by text_api.py
text_manager = TextModelManager()