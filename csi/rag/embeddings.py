"""Shared sentence-embedding model (used by RAG, topic modelling and the grounding check)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from csi.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model, device="cpu")


def embed(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """L2-normalised float32 embeddings, so inner product == cosine similarity."""
    vecs = get_embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.astype(np.float32)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, device="cpu")


def rerank_scores(query: str, passages: list[str]) -> np.ndarray:
    """Cross-encoder relevance: reads query and passage together, so it is sharper (and slower) than embeddings."""
    return np.asarray(get_reranker().predict([(query, p) for p in passages]), dtype=np.float32)
