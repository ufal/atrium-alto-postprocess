"""Auto-correction backends for Noisy lines (issue #23, Phase 2).

The issue proposes turning ``Noisy`` OCR lines into semi-``Clear`` variants by
auto-correcting them — korektor for Czech (the ÚFAL statistical spell-checker),
optionally an LLM (e.g. GLM) — and then **checking how the algorithm score moves
after correction** (see ``report_correction_delta.py``).

All backends share one contract (``CorrectionBackend.correct_batch``) and are
wrapped by a resumable JSONL disk cache so a re-run never re-queries a line.
Correction is offline data-generation only: it never touches the production
pipeline, and (per strategy D2) the corrected text is relabelled from scratch by
``score_texts.py`` — the corrector is not trusted to have improved anything, the
scorer decides.

Backends
--------
* ``KorektorRestBackend`` — LINDAT REST (`.../services/korektor/api/correct`,
  model ``czech-spellchecker-130202`` by default). Czech only.
* ``KorektorLocalBackend`` — a subprocess wrapper around a locally installed
  korektor binary, for cluster-scale runs. Text in on stdin, corrected out on
  stdout.
* ``LlmBackend`` — an OpenAI-compatible chat endpoint (self-hosted GLM / vLLM).
  Handles the deu/eng minority korektor does not cover.
* ``NoopBackend`` — identity, for tests and dry runs.

Language routing: ``route_backend`` sends ``ces`` lines to korektor and everything
else to the LLM (or skips, if no LLM backend is configured).

Network dependencies (``httpx``) are imported lazily, so this module — and the
fast tests, which monkeypatch the two HTTP helpers — import with no extra deps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

KOREKTOR_API = "https://lindat.mff.cuni.cz/services/korektor/api"
KOREKTOR_DEFAULT_MODEL = "czech-spellchecker-130202"

LLM_CORRECTION_PROMPT = (
    "You are correcting OCR errors in a single line of scanned text. "
    "Fix obvious character misreads, split/merged words and dropped diacritics. "
    "Preserve the original language and meaning. If the line is already correct or "
    "is unrecoverable garbage, return it unchanged. Return ONLY the corrected line, "
    "with no quotes, labels or commentary."
)


# ---------------------------------------------------------------------------
# Low-level HTTP helpers (monkeypatched in tests; httpx imported lazily)
# ---------------------------------------------------------------------------


def _post_form(url: str, data: dict, timeout: float) -> dict:
    import httpx  # noqa: PLC0415  (lazy: only needed for a real REST call)

    resp = httpx.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    import httpx  # noqa: PLC0415

    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class CorrectionBackend(ABC):
    """A text-correction backend. ``id`` must be stable — it keys the disk cache."""

    id: str = "base"

    @abstractmethod
    def correct_batch(self, texts: list[str]) -> list[str]:
        """Return one corrected string per input string (same length, same order)."""

    def correct(self, text: str) -> str:
        return self.correct_batch([text])[0]


class NoopBackend(CorrectionBackend):
    """Identity corrector — returns input unchanged. For tests / dry runs."""

    id = "noop"

    def correct_batch(self, texts: list[str]) -> list[str]:
        return list(texts)


class KorektorRestBackend(CorrectionBackend):
    """Czech spell-correction via the LINDAT korektor REST service."""

    def __init__(
        self,
        model: str = KOREKTOR_DEFAULT_MODEL,
        base_url: str = KOREKTOR_API,
        timeout: float = 30.0,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.id = f"korektor-rest:{model}"

    def _correct_one(self, text: str) -> str:
        url = f"{self.base_url}/correct"
        params = {"data": text, "model": self.model}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                payload = _post_form(url, params, self.timeout)
                # LINDAT services return the output under the "result" key.
                return str(payload.get("result", text)).rstrip("\n")
            except Exception as exc:  # noqa: BLE001 (retry any transport/HTTP error)
                last_exc = exc
                if attempt < self.max_retries - 1 and self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError(f"korektor REST failed after {self.max_retries} attempts: {last_exc}")

    def correct_batch(self, texts: list[str]) -> list[str]:
        return [self._correct_one(t) for t in texts]


class KorektorLocalBackend(CorrectionBackend):
    """Local korektor binary wrapper (text on stdin, corrected text on stdout).

    The exact invocation depends on the local build; ``command`` is a template
    list. Any ``{model}`` element is substituted with ``model_path``. Defaults
    assume ``korektor <model_path>`` reading untokenized text from stdin.
    """

    def __init__(
        self,
        model_path: str,
        command: list[str] | None = None,
        timeout: float = 60.0,
    ):
        self.model_path = model_path
        self.command = command or ["korektor", "{model}"]
        self.timeout = timeout
        self.id = f"korektor-local:{Path(model_path).name}"

    def _cmd(self) -> list[str]:
        return [self.model_path if part == "{model}" else part for part in self.command]

    def correct_batch(self, texts: list[str]) -> list[str]:
        # korektor processes newline-delimited lines; keep the 1:1 mapping.
        joined = "\n".join(texts) + "\n"
        proc = subprocess.run(
            self._cmd(),
            input=joined,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"korektor exited {proc.returncode}: {proc.stderr.strip()}")
        out_lines = proc.stdout.split("\n")
        # Trim the trailing empty element from the final newline; pad defensively.
        out_lines = out_lines[: len(texts)] + [""] * (len(texts) - len(out_lines))
        return [ln if ln else texts[i] for i, ln in enumerate(out_lines)]


class LlmBackend(CorrectionBackend):
    """OpenAI-compatible chat endpoint (e.g. self-hosted GLM via vLLM)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        prompt: str = LLM_CORRECTION_PROMPT,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.temperature = temperature
        self.timeout = timeout
        self.id = f"llm:{model}"

    def _correct_one(self, text: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ],
        }
        data = _post_json(self.endpoint, payload, headers, self.timeout)
        return str(data["choices"][0]["message"]["content"]).strip()

    def correct_batch(self, texts: list[str]) -> list[str]:
        return [self._correct_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Resumable JSONL disk cache
# ---------------------------------------------------------------------------


class DiskCache:
    """Append-only JSONL cache keyed by ``sha1(backend_id + '|' + text)``."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._store: dict[str, str] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._store[rec["key"]] = rec["value"]

    @staticmethod
    def key(backend_id: str, text: str) -> str:
        return hashlib.sha1(f"{backend_id}|{text}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        self._store[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class CachedCorrector:
    """Wrap a backend with a disk cache: only cache-missing texts hit the backend."""

    def __init__(self, backend: CorrectionBackend, cache: DiskCache | None = None):
        self.backend = backend
        self.cache = cache or DiskCache(None)

    def correct_batch(self, texts: list[str]) -> list[str]:
        keys = [DiskCache.key(self.backend.id, t) for t in texts]
        results: list[str | None] = [self.cache.get(k) for k in keys]
        miss_idx = [i for i, r in enumerate(results) if r is None]
        if miss_idx:
            fresh = self.backend.correct_batch([texts[i] for i in miss_idx])
            for i, value in zip(miss_idx, fresh, strict=True):
                self.cache.put(keys[i], value)
                results[i] = value
        return [r if r is not None else texts[i] for i, r in enumerate(results)]


# ---------------------------------------------------------------------------
# Language routing
# ---------------------------------------------------------------------------


def route_backend(
    lang: str,
    korektor: CorrectionBackend | None,
    llm: CorrectionBackend | None,
    czech_bases: frozenset[str] = frozenset({"ces", "slk"}),
) -> CorrectionBackend | None:
    """Pick a backend for a line's language: Czech/Slovak → korektor, else → LLM."""
    base = (lang or "").split("_")[0].lower()
    if base in czech_bases and korektor is not None:
        return korektor
    return llm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_backend(args) -> CorrectionBackend:
    if args.backend == "noop":
        return NoopBackend()
    if args.backend == "korektor-rest":
        return KorektorRestBackend(model=args.korektor_model)
    if args.backend == "korektor-local":
        if not args.korektor_model_path:
            raise SystemExit("--korektor-model-path is required for korektor-local")
        return KorektorLocalBackend(args.korektor_model_path)
    if args.backend == "llm":
        import os

        return LlmBackend(args.llm_endpoint, args.llm_model, api_key=os.environ.get(args.llm_api_key_env, ""))
    raise SystemExit(f"unknown backend {args.backend}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-correct Noisy OCR lines (korektor / LLM).")
    p.add_argument("--input", required=True, type=Path, help="CSV with a text column (Noisy lines).")
    p.add_argument("--text-col", default="text")
    p.add_argument("--categ-col", default="categ", help="If present, only rows with this category are corrected.")
    p.add_argument("--only-categ", default="Noisy", help="Category to correct (default: Noisy). Use '' for all rows.")
    p.add_argument("--backend", choices=["noop", "korektor-rest", "korektor-local", "llm"], default="korektor-rest")
    p.add_argument("--korektor-model", default=KOREKTOR_DEFAULT_MODEL)
    p.add_argument("--korektor-model-path", default=None, help="Local korektor model file (korektor-local).")
    p.add_argument("--llm-endpoint", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--llm-model", default="glm-4")
    p.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    p.add_argument("--cache", type=Path, default=None, help="JSONL cache path (resumable).")
    p.add_argument("--out", required=True, type=Path, help="Output CSV: adds source_text + corrected_text.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    with args.input.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    corrector = CachedCorrector(_make_backend(args), DiskCache(args.cache))

    to_correct = []
    for i, row in enumerate(rows):
        text = (row.get(args.text_col) or "").strip()
        if not text:
            continue
        if args.only_categ and row.get(args.categ_col) not in (None, args.only_categ):
            continue
        to_correct.append((i, text))

    corrected = corrector.correct_batch([t for _, t in to_correct])

    out_rows = []
    for (i, src), corr in zip(to_correct, corrected, strict=True):
        out_rows.append(
            {
                **rows[i],
                "source_text": src,
                "corrected_text": corr,
                "backend": corrector.backend.id,
                "changed": corr.strip() != src.strip(),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["source_text", "corrected_text", "backend", "changed"] if rows else []
    # de-dup fieldnames while preserving order
    seen: set[str] = set()
    fieldnames = [c for c in fieldnames if not (c in seen or seen.add(c))]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    changed = sum(1 for r in out_rows if r["changed"])
    print(f"Corrected {len(out_rows)} lines ({changed} changed) with {corrector.backend.id} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
