"""Hybrid retrieval over two sources:
  * knowledge-base chunks  -> dense (FAISS) + sparse (BM25), merged with reciprocal-rank fusion
  * historical resolved tickets -> dense FAISS search ("how did we solve this last time?")
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from csi.config import settings
from csi.rag.chunking import Chunk
from csi.rag.embeddings import embed

_STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being
this that these those it its i we you he she they my our your their me us them do does did have has had
please can could would should will shall may might not no so than too very just also hi hello dear
regards thank thanks team support""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion: robust way to merge rankings whose scores are not comparable."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(self):
        d = settings.paths.index
        self.kb_chunks = [Chunk(**c) for c in json.loads((d / "kb_chunks.json").read_text(encoding="utf-8"))]
        self.kb_index = faiss.read_index(str(d / "kb.faiss"))
        self.kb_bm25 = BM25Okapi([tokenize(c.text) for c in self.kb_chunks])
        self.tickets = pd.read_parquet(d / "resolved_tickets.parquet")
        self.ticket_index = faiss.read_index(str(d / "tickets.faiss"))

    def search_kb(self, query: str, k: int = settings.top_k, candidates: int = 20,
                  mode: str = "hybrid", query_vec: np.ndarray | None = None) -> list[dict]:
        qv = query_vec if query_vec is not None else embed([query])
        sims, dense_ids = self.kb_index.search(qv, min(candidates, len(self.kb_chunks)))
        dense_rank = [int(i) for i in dense_ids[0] if i >= 0]
        dense_sim = {int(i): float(s) for i, s in zip(dense_ids[0], sims[0]) if i >= 0}

        bm25_scores = self.kb_bm25.get_scores(tokenize(query))
        sparse_rank = [int(i) for i in np.argsort(-bm25_scores)[:candidates] if bm25_scores[i] > 0]

        if mode == "dense":
            fused = {i: dense_sim[i] for i in dense_rank}
        elif mode == "bm25":
            fused = {i: float(bm25_scores[i]) for i in sparse_rank}
        else:
            fused = rrf([dense_rank, sparse_rank])

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        out = []
        for i in top:
            c = self.kb_chunks[i]
            sim = dense_sim.get(i)
            if sim is None:
                sim = float(self.kb_index.reconstruct(i) @ qv[0])
            out.append({**asdict(c), "score": round(fused[i], 4), "similarity": round(sim, 4)})
        return out

    def similar_tickets(self, query: str, k: int = 3, queue: str | None = None,
                        query_vec: np.ndarray | None = None) -> list[dict]:
        qv = query_vec if query_vec is not None else embed([query])
        sims, ids = self.ticket_index.search(qv, 50 if queue else k)
        out = []
        for i, s in zip(ids[0], sims[0]):
            if i < 0:
                continue
            row = self.tickets.iloc[int(i)]
            if queue and row["queue"] != queue:
                continue
            out.append({
                "ticket_id": row["ticket_id"], "subject": row["subject"], "queue": row["queue"],
                "priority": row["priority"], "body": row["body"][:400], "answer": row["answer"],
                "similarity": round(float(s), 4),
            })
            if len(out) == k:
                break
        return out


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
