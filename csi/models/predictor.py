"""Inference wrapper: loads a ticket classifier once and serves predictions.

Two backends: the fine-tuned multi-task DistilBERT and the TF-IDF baselines.
With CSI_CLASSIFIER=auto (default) the one with the better test macro F1 on queue routing is served,
so a freshly trained but weaker transformer never silently replaces a stronger baseline.
If the transformer has not been trained or evaluated yet, the baselines are used, so the API and
dashboard always work.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

import joblib
import numpy as np
import torch

from csi.config import settings
from csi.models.multitask import TASK_LABELS, MultiTaskClassifier

log = logging.getLogger("csi.predictor")
DISTILBERT, TFIDF = "distilbert-multitask", "tfidf-logreg"


def _queue_macro_f1(report: str) -> float | None:
    path = settings.paths.reports / report
    return json.loads(path.read_text())["queue"]["macro_f1"] if path.exists() else None


def choose_backend(choice: str | None = None) -> str:
    choice = choice or settings.classifier
    if choice in ("distilbert", DISTILBERT):
        return DISTILBERT
    if choice in ("tfidf", TFIDF):
        return TFIDF
    if not (settings.paths.models / "multitask_distilbert" / "heads.pt").exists():
        return TFIDF
    bert, base = _queue_macro_f1("metrics_distilbert.json"), _queue_macro_f1("metrics_baseline.json")
    if bert is None:  # still training, or never evaluated on the test set
        return TFIDF
    return DISTILBERT if base is None or bert >= base else TFIDF


class TicketClassifier:
    def __init__(self, quantize: bool = True, calibrated: bool = True, backend: str | None = None):
        model_dir = settings.paths.models / "multitask_distilbert"
        self.backend = choose_backend(backend)
        # Per-task softmax temperatures fitted by csi.models.calibrate (1.0 = uncalibrated).
        self.temperature = {t: 1.0 for t in TASK_LABELS}
        temp_file = model_dir / "temperature.json"
        if calibrated and temp_file.exists():
            self.temperature.update(json.loads(temp_file.read_text()))
        if self.backend == DISTILBERT:
            torch.set_num_threads(os.cpu_count() or 4)
            model, self.tokenizer = MultiTaskClassifier.load(model_dir)
            # INT8 dynamic quantisation of the Linear layers: ~2x faster on CPU, negligible accuracy loss.
            self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)                 if quantize else model
            self.baselines = None
        else:
            log.info("Serving the TF-IDF baselines")
            self.baselines = {t: joblib.load(settings.paths.models / f"baseline_{t}.joblib") for t in TASK_LABELS}

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict]:
        if self.baselines is not None:
            probs = {}
            for t, lbls in TASK_LABELS.items():
                m = self.baselines[t]
                p = m.predict_proba(texts)
                order = [list(m.classes_).index(l) for l in lbls]
                probs[t] = p[:, order]
        else:
            enc = self.tokenizer(texts, truncation=True, max_length=settings.max_length,
                                 padding=True, return_tensors="pt")
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            probs = {t: torch.softmax(logits[t] / self.temperature[t], -1).numpy() for t in TASK_LABELS}

        results = []
        for i in range(len(texts)):
            row = {}
            for t, lbls in TASK_LABELS.items():
                p = probs[t][i]
                j = int(np.argmax(p))
                row[t] = {
                    "label": lbls[j],
                    "confidence": round(float(p[j]), 4),
                    "probabilities": {l: round(float(v), 4) for l, v in zip(lbls, p)},
                }
            results.append(row)
        return results


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    return TicketClassifier()
