"""Retrieval evaluation: does searching past resolved tickets surface relevant cases?

Proxy relevance label: a retrieved ticket is relevant if it was handled by the same queue as the
query ticket. Compares BM25 (sparse), dense (FAISS) and hybrid (reciprocal-rank fusion).

Run:  python -m csi.rag.evaluate --n 300
"""
from __future__ import annotations

import argparse
import json
import logging

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from csi.config import settings
from csi.rag.embeddings import embed
from csi.rag.retriever import rrf, tokenize

log = logging.getLogger("csi.rag.eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    p = settings.paths

    corpus = pd.read_parquet(p.index / "resolved_tickets.parquet")
    index = faiss.read_index(str(p.index / "tickets.faiss"))
    test = pd.read_parquet(p.processed / "tickets.parquet").query("split == 'test'")
    test = test.sample(n=min(args.n, len(test)), random_state=settings.seed)

    bm25 = BM25Okapi([tokenize(t) for t in corpus["text"]])
    qvecs = embed(test["text"].tolist())
    _, dense_ids = index.search(qvecs, 50)
    queues = corpus["queue"].to_numpy()

    hits = {"bm25": [], "dense": [], "hybrid": []}
    mrr = {m: [] for m in hits}
    for qi, (_, row) in enumerate(test.iterrows()):
        scores = bm25.get_scores(tokenize(row["text"]))
        sparse = list(np.argsort(-scores)[:50])
        dense = [int(i) for i in dense_ids[qi]]
        fused = rrf([dense, sparse])
        ranked = {"bm25": sparse, "dense": dense, "hybrid": sorted(fused, key=fused.get, reverse=True)}
        for m, ids in ranked.items():
            rel = queues[ids[:args.k]] == row["queue"]
            hits[m].append(rel.mean())
            first = np.argmax(rel) if rel.any() else None
            mrr[m].append(1 / (first + 1) if first is not None else 0.0)

    results = {m: {f"precision@{args.k}": round(float(np.mean(hits[m])), 4),
                   f"mrr@{args.k}": round(float(np.mean(mrr[m])), 4)} for m in hits}
    results["config"] = {"queries": len(test), "corpus": len(corpus),
                         "relevance": "retrieved ticket handled by the same queue"}
    (p.reports / "metrics_retrieval.json").write_text(json.dumps(results, indent=2))
    log.info("retrieval eval: %s", json.dumps(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
