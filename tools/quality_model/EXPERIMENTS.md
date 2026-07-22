# Quality-model experiments log (issue #23)

Committed run-log for the training experiments — same spirit as `tools/SWEEP_NOTES.md`.
Each `train.py` / `train_baseline_gbm.py` run writes `metrics.json` + `run_config.json`
under a gitignored `runs/<id>/`; record the headline numbers here so the comparison
survives without the (large, uncommitted) artifacts.

## How to reproduce a row

```bash
# 1. build the dataset (real relabelling pass — needs FastText + Qwen on a GPU)
python tools/quality_model/build_dataset.py --input-glob 'DOC_LINE_CATEG/*.csv' \
    --scorer model --gold-docs <expert-gold-doc-ids> --out data/qm_dataset.csv

# 2. GBM baselines (± perplexity)
python tools/quality_model/train_baseline_gbm.py --dataset data/qm_dataset.csv --out runs/gbm

# 3. encoder fine-tune
python tools/quality_model/train.py --dataset data/qm_dataset.csv \
    --config setup/config_quality_model.txt --out runs/distilbert
```

## Metric key
- `MAE` / `Spearman` — regression on `score_raw` (held-out split).
- `banded F1` — macro-F1 after banding the predicted score at 0.55 / 0.80.
- `cathead F1` — macro-F1 from the model's category head (encoder only).
- `gold F1` — macro-F1 vs the expert gold subset (Phase 4; the only objective gate).

## Runs

| run id              | date | data (manifest hash) | model                              | loss              | MAE ↓ | Spearman ↑ | banded F1 ↑ | gold F1 ↑ | notes                                          |
|---------------------|------|----------------------|------------------------------------|-------------------|-------|------------|-------------|-----------|------------------------------------------------|
| _(baseline target)_ | —    | —                    | GBM (−perplexity)                  | —                 | —     | —          | —           | —         | text-features-only floor the encoder must beat |
| _(baseline)_        | —    | —                    | GBM (+perplexity)                  | —                 | —     | —          | —           | —         | strongest feature baseline; cannot drop Qwen   |
| _(primary)_         | —    | —                    | distilbert-base-multilingual-cased | Huber(0.1)+0.3·CE | —     | —          | —           | —         | strategy D6 primary                            |
| _(fallback)_        | —    | —                    | google/canine-s                    | Huber(0.1)+0.3·CE | —     | —          | —           | —         | run if subword fragmentation on garbage hurts  |

## Success criteria (from `agent_dev_logs/plans/23.plan.md` §3)

- Held-out MAE ≤ 0.06, Spearman ≥ 0.90, banded macro-F1 ≥ 0.85 (≥ 92% agreement).
- **Beats the GBM(−perplexity) baseline** — otherwise the encoder is not earning its cost.
- Gold-set macro-F1 ≥ the algorithm's own gold-set macro-F1 − 0.01 (parity floor; Phase 4).
- ≤ 150M params and cheaper than the Qwen perplexity path.

_No runs recorded yet — the real relabelling pass and fine-tune need the ML stack + GPU._
