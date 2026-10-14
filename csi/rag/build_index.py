"""Build the vector indexes for the RAG assistant.

Run:  python -m csi.rag.build_index
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

import faiss
import pandas as pd

from csi.config import settings
from csi.rag.chunking import chunk_knowledge_base
from csi.rag.embeddings import embed

log = logging.getLogger("csi.index")


def build_flat_ip(vectors) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def run() -> dict:
    p = settings.paths
    p.ensure()
    start = time.time()

    chunks = chunk_knowledge_base(p.knowledge_base, settings.chunk_size, settings.chunk_overlap)
    kb_vecs = embed([c.text for c in chunks])
    faiss.write_index(build_flat_ip(kb_vecs), str(p.index / "kb.faiss"))
    (p.index / "kb_chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
    log.info("Indexed %d KB chunks from %d articles", len(chunks), len({c.doc_id for c in chunks}))

    # Only the training split goes into the "past resolutions" memory, so evaluation on
    # the test split never retrieves the ticket it is being asked about.
    df = pd.read_parquet(p.processed / "tickets.parquet")
    resolved = df[(df.split == "train") & df.has_answer].reset_index(drop=True)
    resolved = resolved[["ticket_id", "subject", "body", "text", "answer", "queue", "priority", "type"]]
    t_vecs = embed(resolved["text"].tolist(), batch_size=128, show_progress=True)
    faiss.write_index(build_flat_ip(t_vecs), str(p.index / "tickets.faiss"))
    resolved.to_parquet(p.index / "resolved_tickets.parquet", index=False)
    log.info("Indexed %d resolved tickets", len(resolved))

    manifest = {
        "embedding_model": settings.embedding_model,
        "dimension": int(kb_vecs.shape[1]),
        "kb_articles": len({c.doc_id for c in chunks}),
        "kb_chunks": len(chunks),
        "chunk_size_chars": settings.chunk_size,
        "chunk_overlap_chars": settings.chunk_overlap,
        "resolved_tickets": len(resolved),
        "build_seconds": round(time.time() - start, 1),
    }
    (p.index / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Index manifest: %s", manifest)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    run()
