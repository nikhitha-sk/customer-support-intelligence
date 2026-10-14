"""Inference wrapper: loads the fine-tuned multi-task model once and serves predictions.

Falls back to the TF-IDF baselines if the transformer has not been trained yet,
so the API and dashboard always work.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import joblib
import numpy as np
import torch

from csi.config import settings
from csi.models.multitask import TASK_LABELS, MultiTaskClassifier

log = logging.getLogger("csi.predictor")


class TicketClassifier:
    def __init__(self, quantize: bool = True):
        model_dir = settings.paths.models / "multitask_distilbert"
        self.backend = "distilbert-multitask"
        if (model_dir / "heads.pt").exists():
            torch.set_num_threads(os.cpu_count() or 4)
            model, self.tokenizer = MultiTaskClassifier.load(model_dir)
            # INT8 dynamic quantisation of the Linear layers: ~2x faster on CPU, negligible accuracy loss.
            self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8) \
                if quantize else model
            self.baselines = None
        else:
            log.warning("Transformer not found at %s - using TF-IDF baselines", model_dir)
            self.backend = "tfidf-logreg"
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
            probs = {t: torch.softmax(logits[t], -1).numpy() for t in TASK_LABELS}

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
