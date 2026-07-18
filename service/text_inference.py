"""
service/text_inference.py
Manages the LayoutReader, FastText, and Qwen2.5-0.5B (default) perplexity models.

Classification is fully aligned with the main pipeline (langID_classify.py):
  - Unified penalty path : categorize_line() from text_util_langID
  - New API fields       : word_weird, garbage_density, ldl_fuses, etc.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from v3.helpers import boxes2inputs, parse_logits, prepare_inputs
except ImportError:
    print("CRITICAL: 'v3' folder not found in project root — layout reordering unavailable.")
    prepare_inputs = boxes2inputs = parse_logits = None  # type: ignore[assignment]

# Import the full quality-analysis toolkit from the main pipeline module.
try:
    from text_util_langID import (
        analyze_rotation_signals,
        compute_garbage_density,
        compute_quality_score,
        compute_valid_ratio,
        compute_vowel_ratio,
        compute_word_weird_ratio,
        detect_fused_words,
        detect_gibberish_words,
        detect_letter_digit_letter,
        detect_mid_uppercase,
        detect_repeated_chars,
        detect_strange_symbols,
        detect_wx_words,
        score_words_in_line,
    )
    from text_util_langID import categorize_line as _categorize_line_struct

    _UTIL_AVAILABLE = True
except ImportError as _err:
    logging.getLogger(__name__).warning(
        "text_util_langID not found (%s); falling back to legacy utils.categorize_line.",
        _err,
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
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                LayoutLMv3ForTokenClassification,
            )

            # 1. LayoutReader (LayoutLMv3)
            layout_model_path = os.getenv("LAYOUT_MODEL_PATH", "hantian/layoutreader")
            self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained(layout_model_path)
            self.layout_model.to(self.device)
            self.layout_model.eval()

            # 2. FastText language identification
            self.ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

            # 3. Perplexity model (Qwen2.5-0.5B by default; override with GPT2_MODEL_NAME,
            #    e.g. distilgpt2 for English-only collections).
            #    Loaded in full precision and moved explicitly to a single device (no 4-bit
            #    bitsandbytes / device_map="auto", which placed layers non-deterministically).
            gpt2_path = os.getenv("GPT2_MODEL_NAME", "Qwen/Qwen2.5-0.5B")
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            self.ppl_tokenizer.pad_token = self.ppl_tokenizer.eos_token

            ppl_dtype = "auto" if self.device == "cuda" else torch.float32
            self.ppl_model = AutoModelForCausalLM.from_pretrained(gpt2_path, dtype=ppl_dtype)
            self.ppl_model.to(self.device)

            self.ppl_model.eval()

            self._models_loaded = True
            logger.info("All models loaded successfully.")

        except Exception as exc:
            logger.error("Critical error loading models: %s", exc)
            self._models_loaded = False
            raise RuntimeError(f"Failed to load core text-processing models: {exc}") from exc

    # -- request-level entry points used by text_api.py ---------------------

    def _line_perplexities(self, texts: list) -> list:
        """Batch perplexity for a list of lines; 0.0 per line when the unified
        toolkit is unavailable (legacy fallback path)."""
        if not texts:
            return []
        if _UTIL_AVAILABLE and self.ppl_model is not None:
            from text_util_langID import calculate_perplexity_batch

            return calculate_perplexity_batch(texts, self.ppl_model, self.ppl_tokenizer, self.device)
        return [0.0] * len(texts)

    def _classify_lines(self, texts: list) -> list:
        """Run the full per-line quality pipeline; returns 1-based row dicts."""
        rows = []
        for i, (text, ppl) in enumerate(zip(texts, self._line_perplexities(texts)), start=1):
            if _UTIL_AVAILABLE:
                row = _classify_line(
                    text,
                    float(ppl),
                    ft_model=self.ft_model,
                    ppl_model=self.ppl_model,
                    tokenizer=self.ppl_tokenizer,
                    device=self.device,
                )
            else:
                row = _classify_line_legacy(text, float(ppl), self.ft_model)
            rows.append({"line_num": i, **row})
        return rows

    def process_text_file(self, path: str) -> Dict[str, Any]:
        """Classify every non-empty line of a plain-text file."""
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
        lines = self._classify_lines(texts)
        return {"task_type": "text", "num_lines": len(lines), "reading_order": "document", "lines": lines}

    def process_alto(self, path: str) -> Dict[str, Any]:
        """Parse an ALTO XML page, reorder lines with LayoutReader when
        available (document order otherwise), and classify each line."""
        from utils import normalize_boxes, parse_alto_xml

        texts, boxes, (page_w, page_h) = parse_alto_xml(path)
        reading_order = "document"
        if texts and self.layout_model is not None and boxes2inputs is not None:
            try:
                import torch

                norm_boxes = normalize_boxes(boxes, page_w, page_h)
                inputs = prepare_inputs(boxes2inputs(norm_boxes), self.layout_model)
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(self.device)
                with torch.no_grad():
                    logits = self.layout_model(**inputs).logits.cpu().squeeze(0)
                order = parse_logits(logits, len(norm_boxes))
                texts = [texts[i] for i in order]
                reading_order = "layout-reader"
            except Exception as exc:
                logger.warning("Layout reordering failed (%s); keeping document order.", exc)
        lines = self._classify_lines(texts)
        return {"task_type": "alto", "num_lines": len(lines), "reading_order": reading_order, "lines": lines}


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

    categorize_line signature (from text_util_langID):
        categorize_line(qs, txt, wc, vowel_ratio, perplexity, *, weird_ratio=0.0,
                        return_reason=False, valid_word_ratio=1.0, lang_score=1.0,
                        orig_lang_score=1.0, gibberish_present=False,
                        garbage_density=0.0, is_upright_czech=False,
                        ghost_dominated=False)
    """
    # 1. Language Identification
    labels, scores = ft_model.predict([text.lower()], k=1)
    lang = labels[0][0].replace("__label__", "")
    lang_score = float(scores[0][0])

    # 2. Structural Metrics
    sym_count = detect_strange_symbols(text)
    upper_count = detect_mid_uppercase(text)
    rep_count = detect_repeated_chars(text)
    fuse_count = detect_letter_digit_letter(text)
    gibb_count = detect_gibberish_words(text)
    wx_count = detect_wx_words(text)
    fused_words = detect_fused_words(text)
    g_density = compute_garbage_density(text)
    vowel_ratio = compute_vowel_ratio(text)

    wc = len(text.split())
    cc = len(text)

    # 3. Weirdness, validity, rotation
    word_scores = score_words_in_line(text)
    weird_ratio = compute_word_weird_ratio(word_scores)
    valid_ratio = compute_valid_ratio(text)
    is_upright_czech, ghost_dominated = analyze_rotation_signals(text)

    # 4. Quality score
    q_score = compute_quality_score(
        valid_word_ratio=valid_ratio,
        perplexity=ppl,
        text_length=cc,
        weird_ratio=weird_ratio,
        vowel_ratio=vowel_ratio,
        garbage_density=g_density,
        lang_score=lang_score,
        gibberish_ratio=(gibb_count + wx_count) / max(wc, 1),
        fused_ratio=fused_words / max(wc, 1),
        is_upright_czech=is_upright_czech,
    )

    # 5. Categorisation — positional args match the real signature exactly
    categ, q_score = _categorize_line_struct(
        q_score,  # qs
        text,  # txt
        wc,  # wc
        vowel_ratio,  # vowel_ratio
        ppl,  # perplexity
        weird_ratio=weird_ratio,
        valid_word_ratio=valid_ratio,
        lang_score=lang_score,
        gibberish_present=(gibb_count + wx_count) > 0,
        garbage_density=g_density,
        is_upright_czech=is_upright_czech,
        ghost_dominated=ghost_dominated,
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
