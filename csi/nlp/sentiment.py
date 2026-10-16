"""Customer sentiment with a pretrained 3-class transformer (negative / neutral / positive).

Run:  python -m csi.nlp.sentiment     # score every ticket once and store it in SQLite (~20 min on CPU)
"""
from __future__ import annotations

import logging
import sqlite3
import time
from functools import lru_cache

import pandas as pd

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


def score_all_tickets(batch_size: int = 64) -> pd.DataFrame:
    """Pre-compute sentiment for the whole warehouse so dashboards can slice it by queue and theme."""
    log = logging.getLogger("csi.sentiment")
    p = settings.paths
    df = pd.read_parquet(p.processed / "tickets.parquet", columns=["ticket_id", "text"])
    scorer = get_sentiment()
    # Sorting by length keeps similar-sized tickets in a batch, so there is little padding to compute.
    order = df["text"].str.len().sort_values().index
    texts = df.loc[order, "text"].tolist()
    start, scores = time.time(), []
    for i in range(0, len(texts), batch_size * 20):
        scores += scorer.score(texts[i:i + batch_size * 20], batch_size=batch_size)
        log.info("scored %d/%d tickets (%.0fs)", len(scores), len(texts), time.time() - start)

    out = pd.DataFrame(scores, index=order).sort_index()
    out.insert(0, "ticket_id", df["ticket_id"])
    out = out.rename(columns={"label": "sentiment", "score": "sentiment_score", "negative": "sentiment_negative"})
    out.to_parquet(p.processed / "ticket_sentiment.parquet", index=False)

    con = sqlite3.connect(p.warehouse)
    con.execute("CREATE INDEX IF NOT EXISTS ix_tickets_id ON tickets(ticket_id)")  # makes the UPDATEs fast
    cols = {r[1] for r in con.execute("PRAGMA table_info(tickets)")}
    for col, typ in [("sentiment", "TEXT"), ("sentiment_score", "REAL"), ("sentiment_negative", "REAL")]:
        if col not in cols:
            con.execute(f"ALTER TABLE tickets ADD COLUMN {col} {typ}")
    con.executemany("UPDATE tickets SET sentiment = ?, sentiment_score = ?, sentiment_negative = ? "
                    "WHERE ticket_id = ?",
                    out[["sentiment", "sentiment_score", "sentiment_negative", "ticket_id"]].values.tolist())
    con.commit()
    con.close()
    log.info("sentiment mix: %s", out["sentiment"].value_counts(normalize=True).round(3).to_dict())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    score_all_tickets()
