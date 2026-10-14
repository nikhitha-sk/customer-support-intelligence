"""Fine-tune the multi-task DistilBERT classifier on CPU or GPU.

Run:  python -m csi.models.train --epochs 3
      python -m csi.models.train --epochs 1 --max-train 2000   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from csi.config import settings
from csi.models.metrics import classification_metrics, coverage_precision_curve
from csi.models.multitask import TASK_LABELS, MultiTaskClassifier

log = logging.getLogger("csi.train")

# Type is the easiest/least business-critical task, so it gets a smaller share of the loss.
TASK_LOSS_WEIGHTS = {"queue": 1.0, "priority": 1.0, "type": 0.5}


def class_weights(labels: pd.Series, classes: list[str]) -> torch.Tensor:
    """Inverse-sqrt-frequency weights: counters the imbalance without letting
    the rarest queues dominate the loss the way fully 'balanced' weights would."""
    counts = labels.value_counts().reindex(classes).fillna(1).to_numpy(dtype=float)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float)


def encode_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {t: df[t].map({l: i for i, l in enumerate(lbls)}).to_numpy() for t, lbls in TASK_LABELS.items()}


def batches(texts: list[str], labels: dict[str, np.ndarray], tokenizer, batch_size: int, shuffle: bool, seed: int):
    """Length-bucketed batching: similar-length tickets share a batch, so padding (and CPU time) is minimal."""
    idx = np.arange(len(texts))
    lengths = np.array([len(t) for t in texts])
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)
        # sort within large windows, then shuffle the batches themselves
        window = batch_size * 50
        idx = np.concatenate([w[np.argsort(lengths[w])] for w in np.array_split(idx, max(1, len(idx) // window))])
        groups = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
        rng.shuffle(groups)
    else:
        idx = idx[np.argsort(lengths)]
        groups = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    for g in groups:
        enc = tokenizer([texts[i] for i in g], truncation=True, max_length=settings.max_length,
                        padding=True, return_tensors="pt")
        yield g, enc, {t: torch.tensor(labels[t][g]) for t in labels}


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 64) -> dict[str, np.ndarray]:
    model.eval()
    dummy = {t: np.zeros(len(texts), dtype=int) for t in TASK_LABELS}
    out = {t: np.zeros((len(texts), len(l)), dtype=np.float32) for t, l in TASK_LABELS.items()}
    for g, enc, _ in batches(texts, dummy, tokenizer, batch_size, shuffle=False, seed=0):
        logits = model(enc["input_ids"], enc["attention_mask"])
        for t in TASK_LABELS:
            out[t][g] = torch.softmax(logits[t], -1).numpy()
    return out


def evaluate(model, tokenizer, df: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray]]:
    probs = predict_proba(model, tokenizer, df["text"].tolist())
    results = {}
    for t, lbls in TASK_LABELS.items():
        pred = np.array(lbls)[probs[t].argmax(1)]
        results[t] = classification_metrics(df[t].to_numpy(), pred, lbls)
    return results, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training rows (0 = all)")
    args = ap.parse_args()

    torch.manual_seed(settings.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    p = settings.paths
    p.ensure()

    df = pd.read_parquet(p.processed / "tickets.parquet")
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if args.max_train:
        train = train.sample(n=min(args.max_train, len(train)), random_state=settings.seed).reset_index(drop=True)
    log.info("train=%d val=%d test=%d", len(train), len(val), len(test))

    tokenizer = AutoTokenizer.from_pretrained(settings.encoder_name)
    model = MultiTaskClassifier(settings.encoder_name, TASK_LABELS)
    loss_fns = {t: nn.CrossEntropyLoss(weight=class_weights(train[t], l)) for t, l in TASK_LABELS.items()}

    texts, labels = train["text"].tolist(), encode_labels(train)
    steps_per_epoch = math.ceil(len(texts) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps), total_steps)

    best_score, history = -1.0, []
    best_dir = p.models / "multitask_distilbert"
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, (_, enc, y) in enumerate(
            batches(texts, labels, tokenizer, args.batch_size, shuffle=True, seed=settings.seed + epoch), 1
        ):
            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = sum(TASK_LOSS_WEIGHTS[t] * loss_fns[t](logits[t], y[t]) for t in TASK_LABELS)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                elapsed = time.time() - start
                log.info("epoch %d step %d/%d loss=%.4f (%.1fs/step, eta %.0f min)", epoch, step,
                         steps_per_epoch, running / step, elapsed / step,
                         (steps_per_epoch - step) * elapsed / step / 60)

        val_metrics, _ = evaluate(model, tokenizer, val)
        score = val_metrics["queue"]["macro_f1"] + val_metrics["priority"]["macro_f1"]
        row = {"epoch": epoch, "train_loss": round(running / steps_per_epoch, 4),
               **{f"val_{t}_macro_f1": val_metrics[t]["macro_f1"] for t in TASK_LABELS},
               "epoch_minutes": round((time.time() - start) / 60, 1)}
        history.append(row)
        log.info("epoch summary %s", row)
        if score > best_score:
            best_score = score
            model.save(best_dir, tokenizer)
            log.info("saved new best model to %s", best_dir)

    model, tokenizer = MultiTaskClassifier.load(best_dir)
    test_metrics, probs = evaluate(model, tokenizer, test)
    qpred = np.array(TASK_LABELS["queue"])[probs["queue"].argmax(1)]
    test_metrics["queue"]["coverage_precision"] = coverage_precision_curve(
        test["queue"], qpred, probs["queue"].max(1))
    test_metrics["history"] = history
    test_metrics["config"] = {**vars(args), "encoder": settings.encoder_name, "max_length": settings.max_length,
                              "train_rows": len(train), "task_loss_weights": TASK_LOSS_WEIGHTS}
    (p.reports / "metrics_distilbert.json").write_text(json.dumps(test_metrics, indent=2))

    pred_df = test[["ticket_id"]].copy()
    for t, lbls in TASK_LABELS.items():
        pred_df[f"pred_{t}"] = np.array(lbls)[probs[t].argmax(1)]
        pred_df[f"conf_{t}"] = probs[t].max(1)
    pred_df.to_parquet(p.processed / "test_predictions.parquet", index=False)
    for t in TASK_LABELS:
        log.info("TEST %-8s acc=%.3f macroF1=%.3f weightedF1=%.3f", t, test_metrics[t]["accuracy"],
                 test_metrics[t]["macro_f1"], test_metrics[t]["weighted_f1"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
