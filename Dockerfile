# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# --- Provenance (flows into atrium_paradata.py via ENV) ---
ARG ATRIUM_RUNNER_IMAGE=""
ARG ATRIUM_RUNNER_REPO="https://github.com/ufal/atrium-alto-postprocess"
ARG ATRIUM_RUNNER_REF=""
# CPU torch by default (alto-tools method needs no GPU; LayoutReader runs on CPU too,
# just slower). Override to a CUDA wheel index (e.g. .../whl/cu121) to build a GPU
# image for fast LayoutReader runs.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

ENV ATRIUM_RUNNER_IMAGE=${ATRIUM_RUNNER_IMAGE} \
    ATRIUM_RUNNER_REPO=${ATRIUM_RUNNER_REPO} \
    ATRIUM_RUNNER_REF=${ATRIUM_RUNNER_REF} \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    MODEL_DIR=/app/models \
    LANGID_CONFIG=/app/setup/config.txt

# (#3) The per-document line CSVs now carry extra diagnostic boolean and string columns
# (including original_text, original_lang, and rule flags). This aggregation
# ignores them — page stats are unchanged — but a future revision could
# emit per-page rule-frequency sums from them.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps first for layer caching. CPU torch pinned, then the unpinned `torch`
#    in the requirements files is already satisfied (stays CPU).
COPY setup/requirements.txt setup/requirements-test.txt setup/requirements-sweep.txt setup/
COPY service/requirements.txt service/requirements.txt
RUN pip install --index-url ${TORCH_INDEX_URL} torch \
    && pip install -r setup/requirements.txt -r service/requirements.txt -r setup/requirements-test.txt -r setup/requirements-sweep.txt

# 2) LayoutReader v3/ (translated from setup_api_server.sh) -> /app/v3 (on sys.path)
#    Required by the GPU extraction method (extract_LytRdr_ALTO_2_TXT.py).
#
#    (#50 rider) Pinned to a commit. `--depth 1` on a branch resolves to whatever
#    the tip is on the day of the build, so two builds a week apart could ship
#    different LayoutReader helpers under the same image tag. A shallow clone
#    cannot take a SHA via `--branch`, hence init + fetch --depth 1 <sha>.
ARG LAYOUTREADER_COMMIT=3b46ee82cd02dcea94ee4064152e5f4de91e205e
RUN git init /tmp/layoutreader \
    && git -C /tmp/layoutreader remote add origin https://github.com/FreeOCR-AI/layoutreader.git \
    && git -C /tmp/layoutreader sparse-checkout init --cone \
    && git -C /tmp/layoutreader sparse-checkout set v3 \
    && git -C /tmp/layoutreader fetch --filter=blob:none --depth 1 origin ${LAYOUTREADER_COMMIT} \
    && git -C /tmp/layoutreader checkout ${LAYOUTREADER_COMMIT} \
    && mv /tmp/layoutreader/v3 /app/v3 \
    && rm -rf /tmp/layoutreader

# 3) FastText LID weights -> $MODEL_DIR/lid.176.bin, symlinked to the bare CWD path
#    the batch pipeline (classify_TEXT.py:99) loads.
#
#    Hardened fetch:
#      * follows the canonical ?download=true redirect to the LFS CDN,
#      * sends a non-empty User-Agent (Cloudflare rejects some empty-UA bots),
#      * retries transient HTTP errors and connection drops with backoff,
#      * resumes (--continue) the ~2 GB download instead of restarting it,
#      * verifies the result is a non-empty file so a truncated body or an HTML
#        error page fails the build here, loudly, rather than producing a broken
#        model that only blows up at runtime.
#    `-nv` keeps the log readable while still surfacing errors (unlike `-q`).
RUN mkdir -p "$MODEL_DIR" \
    && wget -nv --tries=5 --continue --timeout=60 \
            --retry-connrefused --waitretry=10 \
            --retry-on-http-error=403,408,429,500,502,503,504 \
            --header="User-Agent: atrium-alto-postprocess-docker-build/1.0" \
            "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin?download=true" \
            -O "$MODEL_DIR/lid.176.bin" \
    && test -s "$MODEL_DIR/lid.176.bin" \
    && ln -s "$MODEL_DIR/lid.176.bin" /app/lid.176.bin

# 4) source
COPY . .

# 5) non-root runtime user owning app + caches + data mount
RUN useradd --create-home --uid 10001 atrium \
    && mkdir -p /cache/huggingface /data \
    && chown -R atrium:atrium /app /cache /data
USER atrium

ENTRYPOINT ["python", "run_pipeline.py"]
CMD []


# ---------------------------------------------------------------------------
# API surface — published as :<version>-api (issue #55)
#
# Before this stage existed, alto-postprocess's FastAPI service was reachable only
# through a docker-compose `entrypoint:` override on the BATCH image, so no runnable
# API image was ever published and there was nothing for ARÚP/ARÚB to deploy on
# Kubernetes. service/requirements.txt is already installed in `base` (see the pip
# install above), so this stage only has to declare how to serve.
#
# Entrypoint is `python service/text_api.py` rather than a uvicorn CLI invocation:
# this repo's app deliberately configures logging and reads HOST/PORT/RELOAD in its
# own __main__ block (12-factor VII/XI), and that block is the documented production
# start path. It calls uvicorn.run(..., timeout_graceful_shutdown=...) itself.
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000
# STOPSIGNAL is the default (SIGTERM) — declared explicitly so a future edit cannot
# change it silently; service/text_api.py's lifespan chains to uvicorn's own handler
# for it via serve_lifecycle (service/atrium_service.py).
STOPSIGNAL SIGTERM
ENV PORT=8000 GRACEFUL_SHUTDOWN_S=20
ENTRYPOINT ["python", "service/text_api.py"]
CMD []
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "/app/service/healthcheck.py"]
