"""Customer sentiment with a pretrained 3-class transformer (negative / neutral / positive)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from csi.config import settings


class SentimentScorer:
    def __init__(self, model_name: str = settings.sentiment_model):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.labels = [id2label[i] for i in range(len(id2label))]

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=256,
                                 padding=True, return_tensors="pt")
            probs = torch.softmax(self.model(**enc).logits, -1).numpy()
            for p in probs:
                d = dict(zip(self.labels, p.tolist()))
                neg, pos = d.get("negative", 0.0), d.get("positive", 0.0)
                out.append({
                    "label": self.labels[int(np.argmax(p))],
                    "score": round(pos - neg, 4),  # -1 (very negative) .. +1 (very positive)
                    "negative": round(neg, 4),
                })
        return out


@lru_cache(maxsize=1)
def get_sentiment() -> SentimentScorer:
    return SentimentScorer()
