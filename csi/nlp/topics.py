"""Complaint-theme discovery, BERTopic-style:
sentence embeddings -> K-Means (k picked by silhouette) -> class-based TF-IDF keywords per cluster.

Also provides emerging-theme detection: compares each theme's share in a new batch
against its historical share and flags statistically significant spikes.

Run:  python -m csi.nlp.topics
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from csi.config import settings
from csi.rag.embeddings import embed

log = logging.getLogger("csi.topics")

EXTRA_STOP = ["dear", "customer", "support", "team", "hello", "hi", "regards", "thank", "thanks",
              "please", "issue", "issues", "assistance", "help", "writing", "reaching", "inquire",
              "would", "could", "like", "appreciate", "request", "name", "best", "kindly", "hope"]


def c_tf_idf(docs_per_topic: list[str], top_n: int = 8) -> list[list[str]]:
    """Class-based TF-IDF: treat all tickets of a cluster as one document, then find
    words that are frequent in this cluster but rare across clusters."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    cv = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS.union(EXTRA_STOP)), ngram_range=(1, 2),
                         min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    counts = cv.fit_transform(docs_per_topic).toarray().astype(float)
    tf = counts / counts.sum(1, keepdims=True).clip(min=1)
    avg_words = counts.sum() / len(docs_per_topic)
    idf = np.log(1 + avg_words / counts.sum(0).clip(min=1))
    scores = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    keywords = []
    for row in scores:
        top = vocab[np.argsort(-row)]
        chosen: list[str] = []
        for w in top:  # skip unigrams already covered by a chosen bigram and vice versa
            if any(w in c or c in w for c in chosen):
                continue
            chosen.append(w)
            if len(chosen) == top_n:
                break
        keywords.append(chosen)
    return keywords


def fit(k_range=range(12, 25, 4)) -> pd.DataFrame:
    p = settings.paths
    # Reuse the vectors already stored in the resolved-ticket FAISS index (training split)
    # instead of re-embedding every ticket.
    import faiss
    index = faiss.read_index(str(p.index / "tickets.faiss"))
    vecs = index.reconstruct_n(0, index.ntotal)
    df = pd.read_parquet(p.index / "resolved_tickets.parquet")

    rng = np.random.default_rng(settings.seed)
    sample = rng.choice(len(df), size=min(4000, len(df)), replace=False)
    best_k, best_s = None, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=3, random_state=settings.seed).fit(vecs)
        s = silhouette_score(vecs[sample], km.labels_[sample], metric="cosine")
        log.info("k=%d silhouette=%.4f", k, s)
        if s > best_s:
            best_k, best_s = k, s

    km = KMeans(n_clusters=best_k, n_init=5, random_state=settings.seed).fit(vecs)
    df["topic"] = km.labels_
    docs = [" ".join(df.loc[df.topic == t, "text"]) for t in range(best_k)]
    keywords = c_tf_idf(docs)

    topics = []
    for t in range(best_k):
        sub = df[df.topic == t]
        topics.append({
            "topic": t,
            "label": " / ".join(keywords[t][:3]),
            "keywords": keywords[t],
            "size": int(len(sub)),
            "share": round(len(sub) / len(df), 4),
            "top_queue": sub["queue"].value_counts().idxmax(),
            "high_priority_rate": round(float((sub["priority"] == "high").mean()), 3),
            "examples": sub["subject"].replace("", np.nan).dropna().head(5).tolist(),
        })

    joblib.dump({"kmeans": km, "topics": topics}, p.models / "topics.joblib")
    (p.reports / "topics.json").write_text(json.dumps({"k": best_k, "silhouette": round(best_s, 4),
                                                       "topics": topics}, indent=2))
    df[["ticket_id", "topic"]].to_parquet(p.processed / "ticket_topics.parquet", index=False)
    log.info("Fitted %d themes (silhouette %.3f)", best_k, best_s)
    return df


class TopicModel:
    def __init__(self):
        bundle = joblib.load(settings.paths.models / "topics.joblib")
        self.kmeans, self.topics = bundle["kmeans"], bundle["topics"]

    def assign(self, texts: list[str] | None = None, vectors: np.ndarray | None = None) -> list[dict]:
        vectors = vectors if vectors is not None else embed(texts)
        ids = self.kmeans.predict(vectors)
        return [{"topic": int(i), "label": self.topics[int(i)]["label"]} for i in ids]

    def emerging(self, batch_topics: list[int], min_count: int = 5, z_threshold: float = 2.5) -> list[dict]:
        """Flag themes whose share in the batch is significantly above the historical share
        (one-sided z-test on proportions)."""
        n = len(batch_topics)
        counts = pd.Series(batch_topics).value_counts()
        rows = []
        for t in self.topics:
            observed = int(counts.get(t["topic"], 0))
            p0 = max(t["share"], 1e-4)
            expected = n * p0
            z = (observed - expected) / np.sqrt(n * p0 * (1 - p0)) if n else 0.0
            rows.append({
                "topic": t["topic"], "label": t["label"], "batch_count": observed,
                "batch_share": round(observed / n, 4) if n else 0.0, "baseline_share": t["share"],
                "lift": round((observed / n) / p0, 2) if n else 0.0, "z_score": round(float(z), 2),
                "emerging": bool(observed >= min_count and z >= z_threshold),
            })
        return sorted(rows, key=lambda r: -r["z_score"])


@lru_cache(maxsize=1)
def get_topic_model() -> TopicModel:
    return TopicModel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    fit()
