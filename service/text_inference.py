"""
service/text_inference.py
Manages the LayoutReader, FastText, and Qwen2.5-0.5B (default) perplexity models.

Classification is fully aligned with the main pipeline (classify_TEXT.py):
  - Unified penalty path : categorize_line() from text_util
  - New API fields       : word_weird, garbage_density, ldl_fuses, etc.
"""

import configparser
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# (12-factor XI) No basicConfig() here. This module is imported as a library by
# text_api.py and by tests; configuring the ROOT logger at import time is a
# side effect that silently overrides whatever the host application chose.
# Emit to a named logger and let the entry point decide handlers and level.
# Declared here, ahead of the imports below, so the one call that can fire at
# import time (the CRITICAL in the except branch immediately below) has a
# logger to use instead of falling back to print(). (issue #61)
logger = logging.getLogger(__name__)

try:
    from v3.helpers import boxes2inputs, parse_logits, prepare_inputs
except ImportError:
    logger.critical("'v3' folder not found in project root — layout reordering unavailable.")
    prepare_inputs = boxes2inputs = parse_logits = None  # type: ignore[assignment]

# Import the full quality-analysis toolkit from the main pipeline module.
# Unconditional on purpose: the service must never silently fall back to a
# stale secondary categoriser — a broken import has to fail loud at startup.
# extract_JSON_2_TXT's key-whitelist walk has no heavy dependencies (stdlib
# json only), so it's imported directly rather than re-implemented — unlike
# the ALTO parsing below, which is deliberately mirrored (#8) to avoid pulling
# extract_LytRdr_ALTO_2_TXT's eager torch/transformers/pandas imports into a
# module that must stay importable without ML libraries installed.
from classify_TEXT import score_line  # noqa: E402
from document_hook import parse_origin_by_kind, resolve_input_origin  # noqa: E402
from extract_JSON_2_TXT import TARGET_KEYS, _yield_json_text_by_keys  # noqa: E402
from service.utils import normalize_boxes, parse_alto_xml_lines, post_process_text  # noqa: E402

# (#31) The text-lines readers (PDF, DOCX, ODT, XLSX, PPTX, EPUB, RTF, HTML/hOCR, PAGE
# XML, TEI, CSV, Markdown, ...). Stdlib + lxml at import; pypdfium2/charset-normalizer
# load lazily and only for the uploads that need them.
from text_formats import (  # noqa: E402
    READERS,
    IngestError,
    decode_bytes,
    default_source_origin,
    load_settings,
    no_text_message,
    read_document_isolated,
    shape_lines,
)
from text_util import (  # noqa: E402
    _lang_base,
    calculate_perplexity_batch,
    detect_strange_symbols,
    parse_line_splits,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(project_root / "models")))
FASTTEXT_MODEL_PATH = MODEL_DIR / "lid.176.bin"

# LayoutReader chunk sizes (#8): same defaults as extract_LytRdr_ALTO_2_TXT's
# [EXTRACT] LR_CHUNK_SIZE/LR_MIN_CHUNK_SIZE, but env-var configured here since
# the service otherwise has no dependency on setup/config.txt.
LR_CHUNK_SIZE = int(os.getenv("LR_CHUNK_SIZE", 350))
LR_MIN_CHUNK_SIZE = int(os.getenv("LR_MIN_CHUNK_SIZE", 50))

# (#31) Lines per perplexity forward pass for document uploads: calculate_perplexity_batch
# pads one batch to its longest line, so a 500-page PDF must not be one batch. Same as
# the batch pipeline's [CLASSIFY].BATCH_SIZE default.
DOCUMENT_BATCH_LINES = 128


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

    def _classify_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Score and categorise a list of already-split text lines.

        Shared by process_text_file / process_json / process_alto so all three
        formats classify identically once they've each produced an ordered
        list of lines. Perplexity is computed once for the whole batch (#8),
        mirroring how classify_TEXT.py favours batched GPU perplexity over a
        per-line model call.
        """
        if not lines:
            return []

        ppls = calculate_perplexity_batch(lines, self.ppl_model, self.ppl_tokenizer, self.device)
        cleaned_lines: List[Dict[str, Any]] = []
        for line_num, (text, ppl) in enumerate(zip(lines, ppls, strict=True), start=1):
            entry = _classify_line(
                text,
                ppl,
                ft_model=self.ft_model,
                ppl_model=self.ppl_model,
                tokenizer=self.ppl_tokenizer,
                device=self.device,
            )
            entry["line_num"] = line_num
            cleaned_lines.append(entry)
        return cleaned_lines

    def process_text_file(self, path: str) -> Dict[str, Any]:
        """Classify a plain-text upload, one line per non-empty line.

        (#31 Phase 4) Decoded and shaped like the batch text-lines path: any common
        encoding (cp1250 first), lines normalized and wrapped at MAX_LINE_CHARS. It
        used to be read as UTF-8 only, so a legacy-encoded upload failed with a 500.
        Raises text_formats.IngestError for bytes that are not text.
        """
        _limits, options, _configured, _by_kind = ingest_settings()
        with open(path, "rb") as f:
            data = f.read()
        lines: List[str] = []
        if data:
            text, _enc, _flags = decode_bytes(data, options.fallback_encodings)
            lines = shape_lines([text], options.max_line_chars, keep_blank=False)
        return {"type": "plain_text", "cleaned_lines": self._classify_lines(lines)}

    def process_document(self, path: str, kind: Optional[str] = None) -> Dict[str, Any]:
        """Classify any other text-bearing upload (#31): PDF, DOCX, ODT, XLSX, PPTX, ...

        Reads the file with the same text_formats readers and the same line shaping
        (shape_lines: blank lines dropped, long lines wrapped) as the text-lines batch
        method, so one file yields the same lines through either path. Pages are kept:
        every entry carries `page` (1-based index, the batch path's page id) and
        `page_label` (the source's own label), and `line_num` restarts per page like
        DOC_LINE_CATEG's. Raises text_formats.IngestError for unreadable input, and
        `no_text` for a document without a single text line (a scan with no text layer).
        The [TEXT_INGEST] settings and the origin keys come from the config
        (ingest_settings); blank lines are always dropped here, since this path has no
        `Empty` fast track and each one would go through the perplexity model.
        """
        limits, options, configured, by_kind = ingest_settings()
        doc = read_document_isolated(path, limits, options, kind=kind)
        if doc.line_count() == 0:
            raise IngestError("no_text", no_text_message(doc))
        cleaned: List[Dict[str, Any]] = []
        pages: List[Dict[str, Any]] = []
        for n, page in enumerate(doc.pages, 1):
            lines = shape_lines(page.lines, options.max_line_chars, keep_blank=False)
            entries: List[Dict[str, Any]] = []
            for start in range(0, len(lines), DOCUMENT_BATCH_LINES):
                entries.extend(self._classify_lines(lines[start : start + DOCUMENT_BATCH_LINES]))
            for line_num, entry in enumerate(entries, start=1):
                entry["line_num"] = line_num
                entry["page"] = str(n)
                entry["page_label"] = page.label
            cleaned.extend(entries)
            pages.append(
                {
                    "page": str(n),
                    "page_label": page.label,
                    "lines": len(entries),
                    "text_layer": page.text_layer,
                    "needs_ocr_reason": page.needs_ocr_reason,
                }
            )
        return {
            "type": "document",
            "format": doc.kind,
            "media_type": doc.media_type,
            "origin": resolve_input_origin(
                doc.kind, default_source_origin(doc), configured=configured, by_kind=by_kind
            ),
            "pages": pages,
            "cleaned_lines": cleaned,
        }

    def process_json(self, path: str) -> Dict[str, Any]:
        """Classify a generic JSON OCR upload.

        Extracts ordered text leaves with the same TARGET_KEYS whitelist walk
        extract_JSON_2_TXT.py uses for the batch pipeline, so a given file
        yields the same lines through either path. Each leaf is treated as
        one line, matching the pipeline's "one JSON file = one page" model.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines = list(_yield_json_text_by_keys(data, TARGET_KEYS))
        return {"type": "json", "cleaned_lines": self._classify_lines(lines)}

    def process_alto(self, path: str) -> Dict[str, Any]:
        """Classify an ALTO XML upload: parse -> reorder -> dehyphenate -> classify.

        Uses the line-level parser (parse_alto_xml_lines), not the word-level
        parse_alto_xml also exported by service/utils.py — line granularity is
        what LayoutReader reordering and per-line classification both expect
        here (#8).
        """
        lines, boxes, (page_w, page_h) = parse_alto_xml_lines(path)
        if not lines:
            return {"type": "alto_xml", "cleaned_lines": []}

        norm_boxes = normalize_boxes(boxes, page_w, page_h)

        if boxes2inputs is not None and self.layout_model is not None:
            ordered_lines, ordered_boxes = _run_layout_reader(lines, norm_boxes, self.layout_model, self.device)
        else:
            # v3.helpers unavailable: fall back to document order rather than
            # failing the whole request (#8; mirrors the startup warning above).
            ordered_lines, ordered_boxes = lines, norm_boxes

        full_text = post_process_text(ordered_lines, ordered_boxes)

        # Reconstruct hyphen-split words and classify line by line, carrying
        # the split-suffix state across lines exactly like classify_TEXT.py's
        # per-page loop (so a word split across a line break isn't duplicated
        # or left broken).
        raw_lines = full_text.splitlines()
        resolved_lines: List[str] = []
        expected_incoming_suffix = ""
        for raw_line in raw_lines:
            if not raw_line.strip():
                continue
            merged_text, _outgoing_prefix, outgoing_suffix = parse_line_splits(raw_line)
            if expected_incoming_suffix:
                stripped = merged_text.lstrip()
                if stripped.startswith(expected_incoming_suffix):
                    merged_text = merged_text.replace(expected_incoming_suffix, "", 1).strip()
            expected_incoming_suffix = outgoing_suffix
            if merged_text.strip():
                resolved_lines.append(merged_text.strip())

        return {"type": "alto_xml", "cleaned_lines": self._classify_lines(resolved_lines)}


def _run_layout_reader(lines: List[str], norm_boxes: List[List[int]], layout_model, device):
    """Predict LayoutReader reading order for one page's lines, chunked with
    CUDA-OOM halving/retry. Mirrors extract_LytRdr_ALTO_2_TXT.extract_single_page's
    inference loop (#8), factored out as a reusable function since the service
    processes one page per request rather than a CSV of many.
    """
    import torch

    full_ordered_lines: List[str] = []
    full_ordered_boxes: List[List[int]] = []

    chunk_size = LR_CHUNK_SIZE
    i = 0
    while i < len(lines):
        chunk_lines = lines[i : i + chunk_size]
        chunk_boxes = norm_boxes[i : i + chunk_size]
        if not chunk_lines:
            i += chunk_size
            continue

        try:
            inputs = boxes2inputs(chunk_boxes)
            inputs = prepare_inputs(inputs, layout_model)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                logits = layout_model(**inputs).logits.cpu().squeeze(0)

            order_indices = parse_logits(logits, len(chunk_boxes))
            full_ordered_lines.extend([chunk_lines[idx] for idx in order_indices])
            full_ordered_boxes.extend([chunk_boxes[idx] for idx in order_indices])
            i += chunk_size

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                isinstance(e, RuntimeError) and "memory" in str(e).lower()
            )
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            chunk_size = chunk_size // 2
            if chunk_size < LR_MIN_CHUNK_SIZE:
                logger.error("LayoutReader OOM even at minimum chunk size; falling back to document order.")
                return lines, norm_boxes
            logger.warning("LayoutReader OOM: retrying at i=%d with chunk_size=%d.", i, chunk_size)

    return full_ordered_lines, full_ordered_boxes


# ---------------------------------------------------------------------------
# Helper: classify one line (mirrors process_and_write_batch in classify_TEXT)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def ingest_settings():
    """(limits, options, SOURCE_ORIGIN, SOURCE_ORIGIN_BY_KIND) for /process uploads (#31 Phase 4).

    The same [TEXT_INGEST] and [DOCUMENT] keys the batch text-lines stages read, from
    the same config (LANGID_CONFIG), read once — so an upload and a batch run of one
    file yield the same lines. The DOCUMENT_SOURCE_ORIGIN env var is a batch-run knob
    and is not read here. Raises ValueError, naming the key, for a malformed value.
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(os.getenv("LANGID_CONFIG", str(project_root / "setup" / "config.txt")), encoding="utf-8")
    limits, options = load_settings(cfg)
    configured = cfg.get("DOCUMENT", "SOURCE_ORIGIN", fallback="").strip()
    by_kind = parse_origin_by_kind(cfg.get("DOCUMENT", "SOURCE_ORIGIN_BY_KIND", fallback=""), READERS)
    return limits, options, configured, by_kind


@lru_cache(maxsize=1)
def _lang_config() -> "tuple[list[str], frozenset]":
    """Resolve EXPECTED_LANGS / TRUSTED_FOREIGN_LANGS like the batch pipeline.

    The service previously had no language configuration at all, which is why
    it skipped the remap and the trust tiers entirely. It reads the same
    ``setup/config.txt`` the pipeline does so the two agree on which languages
    are expected, merely trusted, or unknown.
    """
    config = configparser.ConfigParser()
    config.read(os.getenv("LANGID_CONFIG", str(project_root / "setup" / "config.txt")))
    expected = [
        s.strip() for s in config.get("CLASSIFY", "EXPECTED_LANGS", fallback="ces,deu,eng").split(",") if s.strip()
    ]
    trusted = [
        s.strip()
        for s in config.get("CLASSIFY", "TRUSTED_FOREIGN_LANGS", fallback="deu,eng,fra,pol,ita").split(",")
        if s.strip()
    ]
    known_bases = frozenset(_lang_base(code) for code in (trusted + expected))
    return expected, known_bases


def _classify_line(
    text: str,
    ppl: float,
    *,
    ft_model,
    ppl_model,
    tokenizer,
    device: str,
) -> Dict[str, Any]:
    """Classify a single line through the pipeline's own scoring step.

    This used to hand-roll its own signal assembly, and it drifted: it never
    applied ``remap_lang`` or the two-tier trust scaling, never applied
    ``SHORT_PPL_CAP`` to one- and two-token lines, and never passed
    ``orig_lang_score`` to ``categorize_line`` at all -- so that argument sat at
    its 1.0 default and silently disabled three Trash routes the batch pipeline
    relies on (``rule_hard_sweep``, ``rule_extreme_ppl``, ``rule_wqx_rot``). The
    API could therefore return a different category than the pipeline for the
    same line.

    It now calls ``classify_TEXT.score_line``, the single scoring step shared
    with the batch pipeline and the offline re-scorer.

    The service sees one text per line with no separate pre-repair variant, so
    the same string is passed as both ``text_content`` and ``original_text``;
    that is what the previous code did implicitly by computing every signal on
    ``text``.
    """
    # 1. Language identification (unchanged: raw FastText prediction)
    labels, scores = ft_model.predict([text.lower()], k=1)
    original_lang = labels[0][0].replace("__label__", "")
    original_lang_score = float(scores[0][0])

    expected_langs, known_bases = _lang_config()

    # 2-5. The ONE scoring step, shared with classify_TEXT / recategorize_from_csv.
    sig = score_line(
        text_content=text,
        original_text=text,
        original_lang=original_lang,
        original_lang_score=original_lang_score,
        perplexity=ppl,
        known_lang_bases=known_bases,
        expected_langs=expected_langs,
    )

    # detect_strange_symbols is service-only (reported, never scored).
    sym_count = detect_strange_symbols(text)

    return {
        "text": text,
        "lang": sig["lang"],
        "lang_score": round(sig["lang_score"], 4),
        "original_lang": original_lang,
        "orig_lang_score": round(original_lang_score, 4),
        "perplexity": round(sig["perplex"], 2),
        "garbage_density": round(sig["garbage_density"], 4),
        "sym_count": sym_count,
        "upper_count": sig["upper"],
        "repeated_count": sig["repeated"],
        "ldl_fuses": sig["ldl_fuses"],
        "gibberish": sig["gibberish"],
        "word_weird": round(sig["word_weird"], 4),
        "quality_score": round(sig["quality_score"], 4),
        "category": sig["categ"],
    }


# Module-level singleton used by text_api.py
text_manager = TextModelManager()
