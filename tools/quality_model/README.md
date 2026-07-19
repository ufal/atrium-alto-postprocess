# `tools/quality_model/` — quality-score fine-tuning tooling (issue #23)

A data factory + training harness that distils the hand-crafted OCR line-quality
engine (FastText + Qwen2.5-0.5B perplexity + the 9-signal weighted score) into a
single small regression model. The full strategy — design decisions, model
selection, evaluation protocol, phased plan, risks — lives in
[`agent_dev_logs/plans/23.plan.md`](../../agent_dev_logs/plans/23.plan.md).

> **Core invariant (from `tools/SWEEP_NOTES.md`): there is ONE scoring engine.**
> Every module here reuses the production `text_util_langID` /
> `langID_classify` functions. Nothing re-implements the score.

## Why

The production categoriser only ever emits three *clamped* score bands (Trash
`<0.55`, Noisy `[0.55, 0.80)`, Clear `≥0.80`) and depends on a single GPU worker
running Qwen for perplexity — the pipeline bottleneck. Issue #23 asks for a
smaller, single model that predicts the score directly. To train it we need a
*smooth* score continuum, which we manufacture by corrupting Clear lines and
correcting Noisy lines, then **relabelling every variant with the real engine**.

## Install

```bash
pip install -r setup/requirements.txt -r setup/requirements-finetune.txt
```

`torch` / `transformers` / `accelerate` come from `setup/requirements.txt` and are
not repeated in the finetune file.

## Modules

| File                         | Phase | Status    | What it does                                                                                                                                                                                                                                                          |
|------------------------------|-------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `score_texts.py`             | 0     | ✅ drafted | `build_line_record()` — the faithful mirror of the production per-line scorer (`langID_classify.py:315-437`) that Phase 0 will extract; a CLI that loads FastText + Qwen once and relabels arbitrary lines, emitting `score_raw` (pre-clamp) **and** `score_clamped`. |
| `corrupt.py`                 | 1     | ✅ drafted | OCR-realistic corruption engine; each op is aligned with a production detector and deterministically seeded via SHA-256.                                                                                                                                              |
| `build_dataset.py`           | 1     | ✅ drafted | select sources → generate variants → relabel → dedup → split-by-document → balance → CSV + JSON manifest. Model-free `offline` scorer for dry runs/tests; `ModelScorer` for real FastText + Qwen relabelling.                                                          |
| `report_dataset.py`          | 1     | ✅ drafted | severity→score monotonicity + synthetic-vs-real feature deltas + split/provenance/score distribution.                                                                                                                                                                 |
| `correct.py`                 | 2     | ⏳ TODO    | korektor (REST/local) + pluggable LLM correction backends with a JSONL disk cache.                                                                                                                                                                                    |
| `report_correction_delta.py` | 2     | ⏳ TODO    | the issue's explicit check: algo-score `Δ` after correction, band-transition matrix, per-backend go/no-go.                                                                                                                                                            |
| `train_baseline_gbm.py`      | 3     | ⏳ TODO    | `HistGradientBoostingRegressor` baseline (± perplexity feature).                                                                                                                                                                                                      |
| `train.py`                   | 3     | ⏳ TODO    | HF `Trainer` fine-tune of `distilbert-base-multilingual-cased` (regression + category heads, Huber + CE).                                                                                                                                                             |
| `evaluate.py`                | 4     | ⏳ TODO    | metrics vs algorithm (held-out docs) **and** vs expert gold subsets (the only objective gate).                                                                                                                                                                        |

## Usage (drafted modules)

Generate corruption variants of the Clear lines in a `DOC_LINE_CATEG` CSV:

```bash
python tools/quality_model/corrupt.py \
    --input data_samples/DOC_LINE_CATEG/CTX000000002.csv \
    --text-col text --categ-col categ \
    --variants 3 --seed 23 \
    --out /tmp/variants.csv
```

Relabel any list of lines with the production engine (needs the ML stack):

```bash
python tools/quality_model/score_texts.py \
    --input /tmp/variants.csv --text-col text \
    --model Qwen/Qwen2.5-0.5B --fasttext lid.176.bin \
    --out /tmp/scored.csv
```

`build_line_record()` is also importable and model-free (you supply the FastText
label/score and the perplexity), which is how the fast tests exercise the full
engine without a GPU.

Assemble a training dataset (offline dry run — no models, approximate perplexity)
or the real thing:

```bash
# offline: select -> corrupt Clear lines -> relabel -> dedup -> split-by-doc -> balance
python tools/quality_model/build_dataset.py \
    --input-glob 'data_samples/DOC_LINE_CATEG/*.csv' \
    --scorer offline --seed 23 --variants-per-clear 3 \
    --gold-docs CTX192100040 \
    --out /tmp/dataset.csv          # + /tmp/dataset.csv.manifest.json

# real relabelling pass (needs the ML stack + GPU)
python tools/quality_model/build_dataset.py --input-glob 'DOC_LINE_CATEG/*.csv' \
    --scorer model --model Qwen/Qwen2.5-0.5B --fasttext lid.176.bin --out dataset.csv

# monotonicity + realism + distribution report
python tools/quality_model/report_dataset.py --input /tmp/dataset.csv
```

The `offline` scorer holds perplexity fixed, so its *absolute* scores are only
approximate — but the non-perplexity detectors still react to corruption, so it is
faithful enough for pipeline tests and the monotonicity check. Use `--scorer model`
for a dataset you will actually train on (strategy D2).

## Tests

```bash
pytest -m "not slow" tests/test_quality_model_corrupt.py \
    tests/test_quality_model_score_texts.py tests/test_quality_model_dataset.py
```

Fast tests are model-free and never read `data_samples/` directly (house rule).
Model / GPU / network paths are marked `@pytest.mark.slow` in later phases.
