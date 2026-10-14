"""Evaluation helpers shared by the baseline and the transformer."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def coverage_precision_curve(y_true, y_pred, confidence, thresholds=None) -> list[dict]:
    """For auto-routing: what share of tickets can be routed automatically, and how accurately,
    if we only trust predictions above a confidence threshold?"""
    y_true, y_pred, confidence = map(np.asarray, (y_true, y_pred, confidence))
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.3, 1.0, 0.05), 2)
    rows = []
    for t in thresholds:
        mask = confidence >= t
        coverage = float(mask.mean())
        precision = float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else float("nan")
        rows.append({"threshold": float(t), "coverage": round(coverage, 4), "precision": round(precision, 4)})
    return rows
