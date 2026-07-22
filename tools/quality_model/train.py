"""Fine-tune a small multilingual encoder to regress the quality score (issue #23, Phase 3).

Primary model: ``distilbert-base-multilingual-cased`` (~134M, < Qwen2.5-0.5B,
covers ces/deu/eng — strategy D6). A sigmoid regression head predicts ``score_raw``
and an optional 3-way head predicts the category band; the loss is
``Huber(delta) + CE_WEIGHT * CrossEntropy`` (D8), because category is not a pure
function of the score (issue #3).

torch / transformers are imported lazily inside the training helpers, so this module
(and the config/metric glue the fast tests exercise) imports without the ML stack.
The real fine-tune needs the ML stack + a GPU and is covered by a single
``@pytest.mark.slow`` smoke.

Run::

    python tools/quality_model/train.py --dataset dataset.csv --out runs/distilbert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common as CM  # noqa: E402

CATEGORY_TO_ID = {c: i for i, c in enumerate(CM.CATEGORIES)}
ID_TO_CATEGORY = {i: c for c, i in CATEGORY_TO_ID.items()}


# ---------------------------------------------------------------------------
# Pure config / metric glue (unit-tested without torch)
# ---------------------------------------------------------------------------


def build_train_params(cfg, args) -> dict:
    """Merge config + CLI into the training hyperparameter dict (CLI wins)."""
    g = CM.cfg_get
    return {
        "model_name": args.model or g(cfg, "MODEL", "NAME", "distilbert-base-multilingual-cased"),
        "max_length": g(cfg, "MODEL", "MAX_LENGTH", 192, int),
        "category_head": g(cfg, "MODEL", "CATEGORY_HEAD", True, bool),
        "num_categories": g(cfg, "MODEL", "NUM_CATEGORIES", 3, int),
        "lr_encoder": g(cfg, "TRAIN", "LR_ENCODER", 2e-5, float),
        "lr_head": g(cfg, "TRAIN", "LR_HEAD", 1e-3, float),
        "batch_size": args.batch_size or g(cfg, "TRAIN", "BATCH_SIZE", 64, int),
        "epochs": args.epochs or g(cfg, "TRAIN", "EPOCHS", 4, int),
        "warmup_ratio": g(cfg, "TRAIN", "WARMUP_RATIO", 0.06, float),
        "weight_decay": g(cfg, "TRAIN", "WEIGHT_DECAY", 0.01, float),
        "huber_delta": g(cfg, "TRAIN", "HUBER_DELTA", 0.1, float),
        "ce_weight": g(cfg, "TRAIN", "CE_WEIGHT", 0.3, float),
        "bf16": g(cfg, "TRAIN", "BF16", True, bool),
        "seed": args.seed or g(cfg, "TRAIN", "SEED", 42, int),
        "target_col": g(cfg, "DATA", "TARGET_COL", CM.DEFAULT_TARGET),
        "text_col": g(cfg, "DATA", "TEXT_COL", CM.DEFAULT_TEXT),
        "categ_col": g(cfg, "DATA", "CATEG_COL", CM.DEFAULT_CATEG),
        "es_metric": g(cfg, "TRAIN", "EARLY_STOPPING_METRIC", "spearman"),
        "es_patience": g(cfg, "TRAIN", "EARLY_STOPPING_PATIENCE", 2, int),
    }


def make_compute_metrics(category_head: bool):
    """Return a compute_metrics(eval_pred) usable by HF Trainer.

    Accepts either an ``EvalPrediction``-like object or a plain
    ``(predictions, labels)`` tuple, so it is testable with pure lists.
    """

    def compute(eval_pred) -> dict:
        preds, labels = _unpack(eval_pred)
        score_pred, cat_logits = _split_preds(preds, category_head)
        score_true, cat_true = _split_labels(labels, category_head)

        score_pred = [max(0.0, min(1.0, float(s))) for s in _flatten(score_pred)]
        score_true = [float(s) for s in _flatten(score_true)]
        metrics = {f"reg_{k}": v for k, v in CM.regression_metrics(score_true, score_pred).items()}
        banded = CM.banded_category_metrics(score_true, score_pred)
        metrics["banded_macro_f1"] = banded["macro_f1"]
        metrics["banded_accuracy"] = banded["accuracy"]
        metrics["spearman"] = metrics["reg_spearman"]  # early-stopping alias

        if category_head and cat_logits is not None and cat_true is not None:
            pred_cats = [ID_TO_CATEGORY.get(_argmax(row), "Noisy") for row in cat_logits]
            true_cats = [ID_TO_CATEGORY.get(int(c), "Noisy") for c in _flatten(cat_true)]
            head = CM.category_metrics(true_cats, pred_cats)
            metrics["cathead_macro_f1"] = head["macro_f1"]
            metrics["cathead_accuracy"] = head["accuracy"]
        return metrics

    return compute


def _unpack(eval_pred):
    if hasattr(eval_pred, "predictions"):
        return eval_pred.predictions, eval_pred.label_ids
    return eval_pred[0], eval_pred[1]


def _split_preds(preds, category_head: bool):
    if category_head and isinstance(preds, (tuple, list)) and len(preds) == 2:
        return preds[0], preds[1]
    return preds, None


def _split_labels(labels, category_head: bool):
    if category_head and isinstance(labels, (tuple, list)) and len(labels) == 2:
        return labels[0], labels[1]
    return labels, None


def _flatten(seq):
    out = []
    for v in seq:
        if isinstance(v, (list, tuple)):
            out.extend(v)
        else:
            out.append(v)
    return out


def _argmax(row) -> int:
    row = list(row)
    best, bi = row[0], 0
    for i, v in enumerate(row):
        if v > best:
            best, bi = v, i
    return bi


def category_ids(rows: list[dict], categ_col: str = CM.DEFAULT_CATEG) -> list[int]:
    """Map each row's category to its class id (unknown/other → Noisy)."""
    return [CATEGORY_TO_ID.get(r.get(categ_col, ""), CATEGORY_TO_ID["Noisy"]) for r in rows]


# ---------------------------------------------------------------------------
# Model + training (torch / transformers imported lazily)
# ---------------------------------------------------------------------------


def build_model(params: dict):
    """Construct the multi-head regression model. Imports torch/transformers."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    class QualityModel(nn.Module):
        def __init__(
            self, model_name: str, category_head: bool, num_categories: int, huber_delta: float, ce_weight: float
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.regressor = nn.Linear(hidden, 1)
            self.category_head = category_head
            self.classifier = nn.Linear(hidden, num_categories) if category_head else None
            self.huber = nn.HuberLoss(delta=huber_delta)
            self.ce = nn.CrossEntropyLoss()
            self.ce_weight = ce_weight

        def forward(self, input_ids=None, attention_mask=None, score_labels=None, category_labels=None, **kwargs):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]  # CLS
            pooled = self.dropout(pooled)
            score = torch.sigmoid(self.regressor(pooled)).squeeze(-1)
            cat_logits = self.classifier(pooled) if self.category_head else None

            loss = None
            if score_labels is not None:
                loss = self.huber(score, score_labels.float())
                if self.category_head and category_labels is not None:
                    loss = loss + self.ce_weight * self.ce(cat_logits, category_labels.long())
            preds = (score, cat_logits) if self.category_head else score
            return {"loss": loss, "predictions": preds}

    return QualityModel(
        params["model_name"],
        params["category_head"],
        params["num_categories"],
        params["huber_delta"],
        params["ce_weight"],
    )


def encode_rows(rows: list[dict], tokenizer, params: dict):
    """Tokenise + attach score/category labels. Imports nothing heavy itself."""
    texts = [r.get(params["text_col"], "") for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=params["max_length"], padding=False)
    scores = [CM._to_float(r.get(params["target_col"])) for r in rows]
    cats = category_ids(rows, params["categ_col"])
    examples = []
    for i in range(len(rows)):
        ex = {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "score_labels": scores[i]}
        if params["category_head"]:
            ex["category_labels"] = cats[i]
        examples.append(ex)
    return examples


def run_training(dataset_path: Path, cfg, args) -> dict:  # pragma: no cover - needs GPU/ML stack
    """Full fine-tune. Exercised by the @slow smoke test, not the fast suite."""
    import numpy as np  # noqa: PLC0415, F401  (transformers needs it)
    from transformers import (  # noqa: PLC0415
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    params = build_train_params(cfg, args)
    set_seed(params["seed"])
    rows = CM.read_dataset(dataset_path)
    splits = CM.split_rows(rows)

    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    train_ds = encode_rows(splits.get("train", []), tokenizer, params)
    val_ds = encode_rows(splits.get("val", []) or splits.get("test", []), tokenizer, params)
    model = build_model(params)

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["epochs"],
        learning_rate=params["lr_encoder"],
        warmup_ratio=params["warmup_ratio"],
        weight_decay=params["weight_decay"],
        bf16=params["bf16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=params["es_metric"],
        greater_is_better=True,
        seed=params["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(params["category_head"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params["es_patience"])],
    )
    trainer.train()
    metrics = trainer.evaluate()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "run_config.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    p = argparse.ArgumentParser(description="Fine-tune the quality-score encoder.")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--config", type=Path, default=_HERE.parents[1] / "setup" / "config_quality_model.txt")
    p.add_argument("--model", default=None, help="Override MODEL.NAME (e.g. google/canine-s).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/distilbert"))
    args = p.parse_args(argv)

    cfg = CM.load_config(args.config) if Path(args.config).exists() else None
    dataset = args.dataset or Path(CM.cfg_get(cfg, "DATA", "DATASET", "dataset.csv"))
    metrics = run_training(dataset, cfg, args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
