"""Knowledge-base retrieval on a hand-labelled golden set.

`data/eval/kb_golden.jsonl` holds realistic customer questions, each labelled with the one KB section
that answers it. Questions are written in everyday words on purpose (not copied from the articles),
so keyword search cannot win by simple word overlap.

A question counts as answered at k if the right section is among the top-k distinct sections.
Compares BM25, dense, hybrid (RRF), and dense or hybrid followed by cross-encoder re-ranking.

Run:  python -m csi.rag.evaluate_kb
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np

from csi.config import settings
from csi.rag.embeddings import embed
from csi.rag.retriever import get_retriever

log = logging.getLogger("csi.rag.eval_kb")
GOLDEN = settings.paths.data / "eval" / "kb_golden.jsonl"
SETUPS = {"bm25": ("bm25", False), "dense": ("dense", False), "hybrid": ("hybrid", False),
          "hybrid + rerank": ("hybrid", True), "dense + rerank": ("dense", True)}


def section_ranking(hits: list[dict]) -> list[tuple[str, str]]:
    """Several chunks can come from one section; keep each section once, at its best rank."""
    seen: list[tuple[str, str]] = []
    for h in hits:
        key = (h["doc_id"], h["section"])
        if key not in seen:
            seen.append(key)
    return seen


def main() -> dict:
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    r = get_retriever()
    vecs = embed([g["query"] for g in golden])

    results = {}
    for name, (mode, rerank) in SETUPS.items():
        ranks, doc_hit1, start = [], [], time.time()
        for g, v in zip(golden, vecs):
            hits = r.search_kb(g["query"], k=20, mode=mode, query_vec=v[None, :], rerank=rerank)
            sections = section_ranking(hits)
            target = (g["doc_id"], g["section"])
            ranks.append(sections.index(target) + 1 if target in sections else None)
            doc_hit1.append(bool(hits) and hits[0]["doc_id"] == g["doc_id"])
        results[name] = {
            **{f"recall@{k}": round(float(np.mean([x is not None and x <= k for x in ranks])), 3)
               for k in (1, 3, 5)},
            "mrr@5": round(float(np.mean([1 / x if x and x <= 5 else 0.0 for x in ranks])), 3),
            "right_article_at_1": round(float(np.mean(doc_hit1)), 3),
            "ms_per_query": round((time.time() - start) * 1000 / len(golden), 1),
        }
        log.info("%-16s %s", name, results[name])

    report = {"queries": len(golden), "unit": "KB section", "reranker": settings.reranker_model,
              "results": results}
    (settings.paths.reports / "metrics_kb_retrieval.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    main()
