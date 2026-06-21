"""
atrium_paradata.py  –  Unified provenance/paradata logger for ATRIUM pipelines.

DROP THIS FILE AS-IS into every ATRIUM repository root, alongside
para_licenses.py and a repository-specific para_config.txt.

Resolves ATRIUM issue #9:
  * license is no longer hardcoded – it is computed per-run from the components
    actually used (see para_licenses.py + para_config.txt);
  * a tool VERSION tag is recorded;
  * the repository/runner reference is resolved DYNAMICALLY (env override) so it
    can point at the published container actually executing, not a static fork;
  * a docker image placeholder field is emitted;
  * paradata is intended to live in the OUTPUT directory, not the GH repo
    (default paradata_dir is now resolved relative to the output location);
  * single-file workflows can merge the per-tool logs into ONE json per input
    file via merge_paradata_files().

Backward compatibility
-----------------------
The constructor and log_success/log_skip/finalize/context-manager API are
unchanged for existing callers.  finalize() gains an OPTIONAL processed_total
keyword (default None preserves the previous max(output_counts) behaviour).
"""

from __future__ import annotations

import configparser
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from para_licenses import merge_effective_licenses, resolve_effective_license
except ImportError:  # keep logging functional even if the helper is missing
    resolve_effective_license = None  # type: ignore
    merge_effective_licenses = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Static fallbacks (used only when para_config.txt is absent)
# ──────────────────────────────────────────────────────────────────────────────

_REPO_URLS: Dict[str, str] = {
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "alto-postprocess": "https://github.com/ufal/atrium-alto-postprocess",
    "nlp-enrich": "https://github.com/ufal/atrium-nlp-enrich",
    "translator": "https://github.com/ufal/atrium-translator",
}

_ENV_RUNNER_IMAGE = "ATRIUM_RUNNER_IMAGE"
_ENV_RUNNER_REPO = "ATRIUM_RUNNER_REPO"
_ENV_RUNNER_REF = "ATRIUM_RUNNER_REF"


def _load_para_config(start_dir: str = ".") -> Dict[str, Any]:
    """Load repository-specific para_config.txt if present."""
    path = os.path.join(start_dir, "para_config.txt")
    out: Dict[str, Any] = {"components": []}
    if not os.path.exists(path):
        return out

    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    if cfg.has_section("tool"):
        out["program"] = cfg.get("tool", "program", fallback=None)
        out["version"] = cfg.get("tool", "version", fallback=None)
        out["repository_fallback"] = cfg.get("tool", "repository_fallback", fallback=None)

    if cfg.has_section("components"):
        for name, spec in cfg.items("components"):
            fields = [s.strip() for s in spec.split(";")]
            lic = fields[0] if len(fields) > 0 else ""
            loaded = fields[1] if len(fields) > 1 else "always"
            role = fields[2] if len(fields) > 2 else ""
            out["components"].append(
                {
                    "name": name.strip(),
                    "license": lic,
                    "loaded": loaded,
                    "role": role,
                }
            )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ParadataLogger
# ──────────────────────────────────────────────────────────────────────────────


class ParadataLogger:
    """Context-manager-friendly paradata recorder."""

    def __init__(
        self,
        program: str,
        config: Dict[str, Any],
        paradata_dir: str = "paradata",
        output_types: Optional[List[str]] = None,
        version: Optional[str] = None,
        docker_image: Optional[str] = None,
        config_dir: str = ".",
    ) -> None:
        self.program = program
        self.paradata_dir = paradata_dir
        self._start_dt = datetime.now(tz=timezone.utc)
        self._run_id = self._start_dt.strftime("%y%m%d-%H%M%S")

        self._para_cfg = _load_para_config(config_dir)

        self.version = version or self._para_cfg.get("version") or "unknown"

        self.docker_image = docker_image or os.environ.get(_ENV_RUNNER_IMAGE) or ""

        self.config = _sanitise(config)

        self._output_counts: Dict[str, int] = {}
        if output_types:
            for t in output_types:
                self._output_counts[t] = 0

        self._skipped: List[Dict[str, str]] = []
        self._input_total: int = 0
        self._finalised: bool = False

        self._components_used: Dict[str, str] = {}
        for comp in self._para_cfg.get("components", []):
            if comp.get("loaded") == "always":
                self._components_used[comp["name"]] = comp["license"]

        os.makedirs(paradata_dir, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def log_skip(self, filepath: str, reason: str) -> None:
        self._skipped.append(
            {
                "file": str(filepath),
                "reason": str(reason),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    def log_success(self, output_type: str, count: int = 1) -> None:
        self._output_counts[output_type] = self._output_counts.get(output_type, 0) + count

    def log_component(self, name: str, license: Optional[str] = None) -> None:
        """Record that a licensed component was ACTUALLY exercised this run."""
        if license is None:
            for comp in self._para_cfg.get("components", []):
                if comp["name"] == name:
                    license = comp["license"]
                    break
        self._components_used[name] = license or "UNKNOWN"

    # ── reference / license resolution ────────────────────────────────────────

    def _resolve_repository(self) -> str:
        """Dynamic runner reference: env > para_config fallback > static map."""
        return (
            os.environ.get(_ENV_RUNNER_REPO)
            or self._para_cfg.get("repository_fallback")
            or _REPO_URLS.get(self.program, "https://github.com/ufal")
        )

    def _license_block(self) -> Dict[str, Any]:
        comps = list(self._components_used.items())
        if resolve_effective_license is not None and comps:
            return resolve_effective_license(comps)
        return {
            "effective_license": "CC BY-NC 4.0",
            "effective_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "is_non_commercial": True,
            "is_share_alike": False,
            "determined_by": [],
            "components": [{"name": n, "license": lic} for n, lic in comps],
            "unknown_licenses": [],
            "notes": "License helper unavailable or no components recorded; defaulted conservatively to CC BY-NC 4.0.",
        }

    def finalize(
        self,
        input_total: Optional[int] = None,
        processed_total: Optional[int] = None,
    ) -> str:
        """Write the paradata JSON.

        Parameters
        ----------
        input_total : number of input units (e.g. source documents). If None it
            is inferred as processed + skipped.
        processed_total : (#10) explicit count of successfully processed input
            units. When given it is used verbatim for `successfully_processed`,
            decoupling it from output-file counts so the figure can never exceed
            input_total for stages that fan one input out to many outputs
            (e.g. page_split: 1 document -> N pages). When None, the previous
            behaviour (max of output counts) is kept for backward compatibility.
        """
        if self._finalised:
            raise RuntimeError("finalize() has already been called.")

        end_dt = datetime.now(tz=timezone.utc)
        duration_sec = (end_dt - self._start_dt).total_seconds()
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

        skipped_count = len(self._skipped)
        if processed_total is not None:
            processed_docs = processed_total
        else:
            processed_docs = max(self._output_counts.values()) if self._output_counts else 0
        if input_total is None:
            input_total = processed_docs + skipped_count

        perf_per_min: Dict[str, float] = {}
        for otype, cnt in self._output_counts.items():
            perf_per_min[otype] = round(cnt / duration_min, 4) if duration_min > 0 else 0.0

        lic = self._license_block()

        payload = {
            "schema_version": "2.0",
            "program": self.program,
            "tool_version": self.version,
            "repository": self._resolve_repository(),
            "runner_ref": os.environ.get(_ENV_RUNNER_REF, ""),
            "docker_image": self.docker_image,
            "python_version": sys.version,
            "run_id": self._run_id,
            "license": lic["effective_license"],
            "license_url": lic["effective_license_url"],
            "license_detail": lic,
            "start_time": self._start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "duration_seconds": round(duration_sec, 3),
            "config": self.config,
            "statistics": {
                "input_files_total": input_total,
                "successfully_processed": processed_docs,
                "skipped_files": skipped_count,
                "output_counts_by_type": dict(self._output_counts),
                "performance_per_minute": perf_per_min,
            },
            "skipped_files_detail": self._skipped,
        }

        out_path = os.path.join(self.paradata_dir, f"{self._run_id}_{self.program}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        self._finalised = True
        print(f"[paradata] Log written → {out_path}", flush=True)
        return out_path

    # ── context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "ParadataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._finalised:
            try:
                self.finalize()
            except Exception as e:
                print(f"[paradata] WARNING – could not write log: {e}", file=sys.stderr)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Single-file workflow: merge per-tool logs into ONE json per input file
# ──────────────────────────────────────────────────────────────────────────────


def merge_paradata_files(
    json_paths: List[str],
    input_file: str,
    out_path: str,
) -> str:
    """Merge several per-tool paradata JSONs into a single per-file record."""
    steps: List[Dict[str, Any]] = []
    license_blocks: List[Dict[str, Any]] = []
    total_duration = 0.0

    for p in json_paths:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        steps.append(
            {
                "program": data.get("program"),
                "tool_version": data.get("tool_version"),
                "repository": data.get("repository"),
                "docker_image": data.get("docker_image"),
                "run_id": data.get("run_id"),
                "duration_seconds": data.get("duration_seconds"),
                "license": data.get("license"),
                "config": data.get("config"),
            }
        )
        if data.get("license_detail"):
            license_blocks.append(data["license_detail"])
        total_duration += float(data.get("duration_seconds") or 0.0)

    if merge_effective_licenses is not None and license_blocks:
        merged_lic = merge_effective_licenses(license_blocks)
    else:
        merged_lic = {
            "effective_license": "CC BY-NC 4.0",
            "effective_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "notes": "License helper unavailable; defaulted to CC BY-NC 4.0.",
        }

    payload = {
        "schema_version": "2.0",
        "record_type": "single-file-merged",
        "input_file": input_file,
        "pipeline_steps": steps,
        "step_count": len(steps),
        "total_duration_seconds": round(total_duration, 3),
        "license": merged_lic["effective_license"],
        "license_url": merged_lic["effective_license_url"],
        "license_detail": merged_lic,
        "merged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[paradata] Merged single-file log → {out_path}", flush=True)
    return out_path


def merge_run_paradata(
    json_paths: List[str],
    out_path: str,
    pipeline: Optional[str] = None,
    method: Optional[str] = None,
) -> str:
    """Merge per-stage paradata JSONs of ONE end-to-end run into one summary.

    The effective license is re-derived from the UNION of all components used
    across stages. merge_effective_licenses() now deduplicates the union before
    resolving (#12), so the component catalogue and the "N component(s)" note
    reflect the unique set rather than repeating always-on components per stage.
    """
    stages: List[Dict[str, Any]] = []
    license_blocks: List[Dict[str, Any]] = []
    formats: "Dict[str, int]" = {}
    total_duration = 0.0
    total_inputs = 0
    total_processed = 0
    total_skipped = 0
    all_skips: List[Dict[str, Any]] = []
    repo = ""
    tool_version = ""
    earliest: Optional[str] = None
    latest: Optional[str] = None

    for order, p in enumerate(json_paths, 1):
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        repo = repo or data.get("repository", "")
        tool_version = tool_version or data.get("tool_version", "")

        cfg = data.get("config", {}) or {}
        stats = data.get("statistics", {}) or {}
        out_counts = stats.get("output_counts_by_type", {}) or {}

        for ftype, cnt in out_counts.items():
            formats[ftype] = formats.get(ftype, 0) + int(cnt or 0)

        total_duration += float(data.get("duration_seconds") or 0.0)
        total_inputs += int(stats.get("input_files_total") or 0)
        total_processed += int(stats.get("successfully_processed") or 0)
        total_skipped += int(stats.get("skipped_files") or 0)
        all_skips.extend(data.get("skipped_files_detail", []) or [])

        st = data.get("start_time")
        en = data.get("end_time")
        if st and (earliest is None or st < earliest):
            earliest = st
        if en and (latest is None or en > latest):
            latest = en

        stages.append(
            {
                "order": order,
                "program": data.get("program"),
                "script": cfg.get("script"),
                "method": cfg.get("method"),
                "run_id": data.get("run_id"),
                "input_dir": cfg.get("input_dir"),
                "input_csv": cfg.get("input_csv"),
                "output_dir": cfg.get("output_dir") or cfg.get("output_csv"),
                "output_formats": out_counts,
                "duration_seconds": data.get("duration_seconds"),
                "license": data.get("license"),
                "input_files_total": stats.get("input_files_total"),
                "successfully_processed": stats.get("successfully_processed"),
                "skipped_files": stats.get("skipped_files"),
            }
        )

        if data.get("license_detail"):
            license_blocks.append(data["license_detail"])

    if merge_effective_licenses is not None and license_blocks:
        # merge_effective_licenses dedups the union internally (#12), so no
        # post-hoc component collapse is needed here.
        merged_lic = merge_effective_licenses(license_blocks)
    else:
        merged_lic = {
            "effective_license": "CC BY-NC 4.0",
            "effective_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "notes": "License helper unavailable; defaulted to CC BY-NC 4.0.",
        }

    payload = {
        "schema_version": "2.0",
        "record_type": "pipeline-run-merged",
        "pipeline": pipeline or "",
        "method": method or "",
        "repository": repo,
        "tool_version": tool_version,
        "run_id": datetime.now(tz=timezone.utc).strftime("%y%m%d-%H%M%S"),
        "stage_count": len(stages),
        "pipeline_stages": stages,
        "intermediate_formats": formats,
        "license": merged_lic["effective_license"],
        "license_url": merged_lic["effective_license_url"],
        "license_detail": merged_lic,
        "start_time": earliest or "",
        "end_time": latest or "",
        "total_duration_seconds": round(total_duration, 3),
        "statistics": {
            "stages_total": len(stages),
            "input_files_total": total_inputs,
            "successfully_processed": total_processed,
            "skipped_files": total_skipped,
        },
        "skipped_files_detail": all_skips,
        "merged_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[paradata] Merged pipeline-run log \u2192 {out_path}", flush=True)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sanitise(obj: Any, _depth: int = 0) -> Any:
    if _depth > 10:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v, _depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
