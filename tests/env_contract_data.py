"""Repo-local declarations for tests/test_env_contract.py (atrium-project#60).

Never vendored, never in para-drift, never in docs/templates/ruff.toml's [format]
exclude — unlike test_env_contract.py itself, this file's SHAPE is per-repo by
design. See the canonical test's module docstring for the full rationale.
"""

from __future__ import annotations

# Read by shipped code but deliberately absent from .env.example, each with a
# reason. All four are batch/tools-layer scripts the api entrypoint
# (service/text_api.py -> text_inference, utils, document_hook) never imports.
NOT_PUBLISHED: dict[str, str] = {
    "DOCUMENT_SOURCE_ORIGIN": "batch-pipeline knob read by page_split.py and text_split.py; the service reads only "
    "[DOCUMENT].SOURCE_ORIGIN(_BY_KIND) from setup/config.txt, never this variable",
    "KOREKTOR_URL": "read only by tools/quality_model/correct.py, a batch/tools-layer script; not reachable from the service entrypoint",
    "LANGID_TEXT_DIR": "batch-pipeline knob read by classify_TEXT.py and run_pipeline.py; not reachable from the service entrypoint",
    "MAX_WORKERS": "batch-pipeline knob read by extract_ALTO_2_TXT.py and extract_JSON_2_TXT.py; not reachable from the service entrypoint",
    # Unlike the four above, these two families ARE reachable from the service
    # entrypoint (text_util.py is imported by service/text_inference.py:53, at
    # import time) -- the exemption reason is different in kind, not degree.
    # text_util.py's ~95 ATRIUM_TEXT_UTILS_*/ATRIUM_CLASSIFY_* names are
    # algorithmic scoring constants (perplexity thresholds, vowel-ratio weights,
    # rotation-detection cutoffs, ...) read via _get_float/_get_int/_get_str/
    # _get_csv_set (text_util.py:193,210,203,220), each overriding one scalar in
    # setup/config.txt (text_util.py:164) -- the file this repo's plans have
    # repeatedly said is the real deployment surface for tuning these, not the
    # environment (atrium-project#53 plan D: "Explicitly not in scope: converting
    # every algorithmic parameter to an env var. Twelve-factor III is about
    # deployment-varying config."). Declared here, not silently unmatched by the
    # scanner, so the .env.example header's "COMPLETE ledger" claim stays true:
    # atrium-project#60 found these invisible to the plain-literal regex (an
    # f-string composes the name, not a literal at the getenv call site) and
    # taught test_env_contract.py's _prefix_reads to resolve them generically
    # before this entry was added -- see that file's module docstring.
    "ATRIUM_TEXT_UTILS_*": "algorithmic scoring constant tuned via setup/config.txt, not a deployment-varying knob (atrium-project#60/#53)",
    "ATRIUM_CLASSIFY_*": "algorithmic scoring constant tuned via setup/config.txt, not a deployment-varying knob (atrium-project#60/#53)",
}

# In .env.example but read by no Python in this repo — each with a reason.
CONSUMED_ELSEWHERE: dict[str, str] = {
    "ATRIUM_VERSION": "read only by docker-compose.yml to pick the image tag; no Python here reads it",
    "HF_HOME": "read by huggingface_hub itself, set by the Dockerfile and docker-compose.yml",
}

# service/README.md or .env.example cells whose value is prose rather than a literal
# the code-default resolver can compare against.
PROSE_DEFAULTS: dict[str, str] = {
    "GPT2_MODEL_NAME": "README gives the default in prose ('see below', explained just under the table) rather than repeating the literal",
    "MODEL_DIR": "README gives the default in prose ('see below') rather than repeating the computed path",
}
