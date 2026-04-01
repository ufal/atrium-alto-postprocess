
"""
service/text_inference.py
Manages the LayoutReader, FastText, and DistilGPT2 models.

Classification is fully aligned with the main pipeline (langID_classify.py):
  - Unified penalty path : categorize_line() from text_util_langID
  - New API fields       : word_weird, garbage_density, ldl_fuses, etc.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import fasttext
from transformers import LayoutLMv3ForTokenClassification, AutoModelForCausalLM, AutoTokenizer

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


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

class TextModelManager:
    def __init__(self) -> None:
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.layout_model: Optional[LayoutLMv3ForTokenClassification] = None
        self.ft_model:     Optional[fasttext.FastText._FastText]      = None
        self.ppl_model:    Optional[AutoModelForCausalLM]             = None
        self.ppl_tokenizer: Optional[AutoTokenizer]                   = None
        self._models_loaded = False

    # ------------------------------------------------------------------
    def load_models(self) -> None:
        """Load all models synchronously; raise RuntimeError on failure."""
        if self._models_loaded:
            return

        logger.info("Loading Text Processing Models on %s …", self.device)

        try:
            # 1. LayoutReader (LayoutLMv3)
            layout_model_path = os.getenv(
                "LAYOUT_MODEL_PATH", "hantian/layoutreader"
            )
            self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained(
                layout_model_path
            )
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

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

    # ------------------------------------------------------------------
    def _get_perplexity(self, texts: List[str]) -> List[float]:
        """Batch perplexity via the main pipeline helper when available."""
        if _UTIL_AVAILABLE:
            return calculate_perplexity_batch(
                texts, self.ppl_model, self.ppl_tokenizer, self.device
            )
        # Inline fallback (identical algorithm, kept for safety)
        from torch import nn
        import torch as _torch
        max_len = self.ppl_model.config.max_position_embeddings
        enc = self.ppl_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        )
        input_ids     = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)
        target_ids    = input_ids.clone()
        target_ids[target_ids == self.ppl_tokenizer.pad_token_id] = -100
        with _torch.no_grad():
            logits       = self.ppl_model(input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct     = nn.CrossEntropyLoss(reduction="none")
            loss         = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(target_ids.size(0), -1)
            non_masked = shift_labels != -100
            ppl = _torch.exp(
                (loss * non_masked).sum(1) / non_masked.sum(1).clamp(min=1)
            )
        return ppl.tolist()

    # ------------------------------------------------------------------
    def _classify_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a list of pre-cleaned text lines.

        Applies pre_filter_line first (Empty / Non-text, no GPU) then
        batches remaining lines through perplexity scoring and the full
        hybrid classifier, matching langID_classify.py exactly.
        """
        results     : List[Dict[str, Any]] = [{}] * len(lines)
        gpu_indices : List[int]            = []
        gpu_texts   : List[str]            = []

        # CPU pre-filter pass
        for idx, raw in enumerate(lines):
            if _UTIL_AVAILABLE:
                categ, clean = pre_filter_line(raw)
            else:
                clean = raw.strip()
                categ = "Process" if clean else "Empty"

            if categ != "Process":
                results[idx] = {
                    "text":          clean,
                    "lang":          "N/A",
                    "lang_score":    0.0,
                    "perplexity":    0.0,
                    "sym_count":     0,
                    "upper_count":   0,
                    "word_weird":    0.0,
                    "quality_score": 0.0,
                    "category":      categ,
                }
            else:
                gpu_indices.append(idx)
                gpu_texts.append(clean)

        if not gpu_texts:
            return results

        # Batch perplexity (GPU)
        ppls = self._get_perplexity(gpu_texts)

        # Per-line classification
        for i, (orig_idx, text, ppl) in enumerate(zip(gpu_indices, gpu_texts, ppls)):
            if _UTIL_AVAILABLE:
                results[orig_idx] = _classify_line(
                    text, ppl,
                    ft_model=self.ft_model,
                    ppl_model=self.ppl_model,
                    tokenizer=self.ppl_tokenizer,
                    device=self.device,
                )
            else:
                results[orig_idx] = _classify_line_legacy(text, ppl, self.ft_model)

        return results

    # ------------------------------------------------------------------
    def process_alto(self, xml_path: str) -> Dict[str, Any]:
        """
        Parse an ALTO XML file, reorder tokens with LayoutLMv3, reconstruct
        lines, then classify each line with the full hybrid pipeline.
        """
        self._ensure_loaded()

        words, boxes, (page_w, page_h) = parse_alto_xml(xml_path)
        if not words:
            return {"type": "alto_xml", "cleaned_lines": [], "raw_text": ""}

        # Layout reordering
        ordered_words = words  # fallback: keep ALTO order
        if prepare_inputs is not None and page_w > 0 and page_h > 0:
            try:
                inputs = boxes2inputs(words, boxes, page_w, page_h)
                model_inputs = prepare_inputs(inputs, self.layout_model)
                logits = self.layout_model(**{
                    k: v.to(self.device) for k, v in model_inputs.items()
                    if k != "bbox"
                }).logits
                order = parse_logits(logits, len(words))
                ordered_words = [words[i] for i in order]
            except Exception as exc:
                logger.warning("LayoutLMv3 reordering failed (%s); using ALTO order.", exc)

        # Group words back into lines by proximity (simple vertical-gap split)
        raw_text   = " ".join(ordered_words)
        text_lines = [raw_text]   # single-line fallback; refine if box data available

        classified = self._classify_lines(text_lines)
        cleaned_lines = [
            {"line_num": i + 1, **row}
            for i, row in enumerate(classified)
        ]

        return {
            "type":          "alto_xml",
            "cleaned_lines": cleaned_lines,
            "raw_text":      raw_text,
        }

    # ------------------------------------------------------------------
    def process_text_file(self, txt_path: str) -> Dict[str, Any]:
        """
        Read a plain-text file (one logical line per newline) and classify
        every line with the full hybrid pipeline.
        """
        self._ensure_loaded()

        with open(txt_path, "r", encoding="utf-8") as fh:
            raw_lines = fh.readlines()

        stripped = [ln.rstrip("\n") for ln in raw_lines]
        classified = self._classify_lines(stripped)

        cleaned_lines = [
            {"line_num": i + 1, **row}
            for i, row in enumerate(classified)
        ]

        return {
            "type":          "plain_text",
            "cleaned_lines": cleaned_lines,
            "raw_text":      "\n".join(stripped),
        }


# Module-level singleton used by text_api.py
text_manager = TextModelManager()