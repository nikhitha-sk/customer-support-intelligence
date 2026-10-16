"""Temperature scaling for the multi-task DistilBERT.

Fine-tuned transformers are usually over-confident: "90% sure" is right less than 90% of the time.
Auto-routing trusts the confidence score, so we learn one temperature T per task on the validation
set (softmax(logits / T)) and check on the test set that confidence now matches accuracy (ECE).
T does not change the predicted label, only how sure the model claims to be.

Run (after training):  python -m csi.models.calibrate
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from csi.config import settings
from csi.models.metrics import coverage_precision_curve
from csi.models.multitask import TASK_LABELS
from csi.models.predictor import TicketClassifier
from csi.models.train import predict_proba

log = logging.getLogger("csi.calibrate")
TEMPERATURE_FILE = "temperature.json"


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """softmax(log p / T) == softmax(logits / T): the per-row log-sum-exp constant cancels out."""
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: average gap between confidence and accuracy, weighted by bin size."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    grid = np.round(np.arange(0.5, 4.01, 0.01), 2)
    return float(grid[np.argmin([nll(apply_temperature(probs, t), y) for t in grid])])


def main() -> dict:
    p = settings.paths
    model_dir = p.models / "multitask_distilbert"
    if not (model_dir / "heads.pt").exists():
        raise SystemExit("Train the transformer first: python -m csi.models.train")
    clf = TicketClassifier(calibrated=False, backend="distilbert")

    df = pd.read_parquet(p.processed / "tickets.parquet")
    val, test = df[df.split == "val"], df[df.split == "test"]
    pv = predict_proba(clf.model, clf.tokenizer, val["text"].tolist())
    pt = predict_proba(clf.model, clf.tokenizer, test["text"].tolist())

    temps, report = {}, {}
    for task, labels in TASK_LABELS.items():
        idx = {l: i for i, l in enumerate(labels)}
        yv, yt = val[task].map(idx).to_numpy(), test[task].map(idx).to_numpy()
        t = fit_temperature(pv[task], yv)
        temps[task] = t
        cal = apply_temperature(pt[task], t)
        report[task] = {
            "temperature": t,
            "test_ece_before": round(ece(pt[task], yt), 4), "test_ece_after": round(ece(cal, yt), 4),
            "test_nll_before": round(nll(pt[task], yt), 4), "test_nll_after": round(nll(cal, yt), 4),
            "mean_confidence_before": round(float(pt[task].max(1).mean()), 4),
            "mean_confidence_after": round(float(cal.max(1).mean()), 4),
            "accuracy": round(float((cal.argmax(1) == yt).mean()), 4),
        }
        if task == "queue":
            pred = np.array(labels)[cal.argmax(1)]
            report[task]["coverage_precision"] = coverage_precision_curve(test[task], pred, cal.max(1))
        log.info("%-8s T=%.2f ECE %.3f -> %.3f", task, t, report[task]["test_ece_before"],
                 report[task]["test_ece_after"])

    (model_dir / TEMPERATURE_FILE).write_text(json.dumps(temps, indent=2))
    (p.reports / "metrics_calibration.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
