## 🐳 Run with Docker

The whole pipeline ships as a self-contained image — no manual `setup_api_server.sh`,
`v3/` checkout, or FastText download needed. The image bakes the LayoutReader `v3/`
helpers and the FastText LID weights at build time; LayoutLMv3 and Qwen2.5-0.5B
auto-download to a persistent Hugging Face cache volume on first run.

Two extraction methods are available:

| Method         | Resource | Notes                                                      |
|----------------|----------|------------------------------------------------------------|
| `alto-tools`   | **CPU**  | Fast, low-memory raw extraction. No GPU needed.            |
| `layoutreader` | **GPU**  | Layout-aware reading-order reconstruction (default).       |

Copy the example env file first (optional — sensible defaults apply without it):

```bash
cp .env.example .env
```

**Build the image:**

```bash
docker compose build
```

**Run the full pipeline** over the bundled synthetic samples (LayoutReader method, default):

```bash
docker compose run --rm alto
```

**Run the CPU-only method** (no GPU required):

```bash
docker compose run --rm alto --method alto-tools
```

**Inspect the resolved plan without running anything** (network-free):

```bash
docker compose run --rm alto --dry-run
```

**Run an auxiliary script** (override the entrypoint):

```bash
docker compose run --rm --entrypoint python alto page_split.py --help
```

**Start the API service** (FastAPI on http://localhost:8000):

```bash
docker compose --profile api up
```

### Processing your own data

The pipeline's input/output locations are read from `config_langID.txt`
(`[PIPELINE] INPUT_DIR/PAGE_ALTO_DIR/PARADATA_DIR`, `[EXTRACT] INPUT_CSV/OUTPUT_TXT*`,
`[CLASSIFY] OUTPUT_LINES_LOG`, `[AGGREGATE] OUTPUT_STATS/OUTPUT_DOC_DIR`), which default
to `./data_samples/…`. The host `./data` directory is mounted at `/data` and
`config_langID.txt` is bind-mounted read-only, so to process your own archive edit those
keys to point under `/data/…` (e.g. `INPUT_DIR=/data/input`,
`PARADATA_DIR=/data/output/paradata`) and drop your ALTO XMLs into `./data`.
`--input-dir` / `--paradata-dir` can also be passed on the CLI, but the per-stage
text/categ/stats output dirs are config-only.

### GPU runs (LayoutReader method)

The default image pins **CPU** torch (small, no CUDA wheels), which is all the
`alto-tools` method needs and is enough to run LayoutReader slowly. For fast
LayoutReader extraction on a GPU host, build a CUDA image and run with the GPU overlay
(requires the NVIDIA Container Toolkit):

```bash
docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm alto
```

The `docker-compose.gpu.yml` device exposure is only meaningful with such a CUDA image.

### Provenance

Every output JSON records the running image: the Compose/CI build-args flow through the
Dockerfile `ARG`→`ENV` into `atrium_paradata.py`, so a published `:v0.15.5` image
self-reports `docker_image`, `repository`, and `runner_ref` in its paradata. The canonical
containerization strategy doc lives in the `atrium-project` repo.