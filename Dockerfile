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
    LANGID_CONFIG=/app/config_langID.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps first for layer caching. CPU torch pinned, then the unpinned `torch`
#    in the requirements files is already satisfied (stays CPU).
COPY requirements.txt requirements-test.txt ./
COPY service/requirements.txt service/requirements.txt
RUN pip install --index-url ${TORCH_INDEX_URL} torch \
    && pip install -r requirements.txt -r service/requirements.txt -r requirements-test.txt

# 2) LayoutReader v3/ (translated from setup_api_server.sh) -> /app/v3 (on sys.path)
#    Required by the GPU extraction method (extract_LytRdr_ALTO_2_TXT.py).
RUN git clone --filter=blob:none --no-checkout --depth 1 \
        https://github.com/FreeOCR-AI/layoutreader.git /tmp/layoutreader \
    && git -C /tmp/layoutreader sparse-checkout init --cone \
    && git -C /tmp/layoutreader sparse-checkout set v3 \
    && git -C /tmp/layoutreader checkout \
    && mv /tmp/layoutreader/v3 /app/v3 \
    && rm -rf /tmp/layoutreader

# 3) FastText LID weights -> $MODEL_DIR/lid.176.bin, symlinked to the bare CWD path
#    the batch pipeline (langID_classify.py:99) loads.
RUN mkdir -p "$MODEL_DIR" \
    && wget -q "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin" \
            -O "$MODEL_DIR/lid.176.bin" \
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