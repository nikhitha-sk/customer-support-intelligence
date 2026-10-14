"""Structure-aware chunking for knowledge-base articles.

Articles are first split on their `##` sections (so a chunk never mixes two policies),
then long sections are split into overlapping windows on sentence boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from csi.text import split_sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


def window_sentences(text: str, size: int, overlap: int) -> list[str]:
    sentences = split_sentences(text)
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > size:
            chunks.append(current.strip())
            # carry the tail of the previous chunk forward as overlap
            tail = current[-overlap:]
            current = tail[tail.find(" ") + 1:] if " " in tail else tail
        current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_article(path: Path, size: int, overlap: int) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_id = "-".join(path.stem.split("-")[:2])  # KB-001
    title_match = re.search(r"^# (.+)$", raw, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    sections = re.split(r"^## ", raw, flags=re.M)[1:]
    chunks: list[Chunk] = []
    for sec in sections:
        heading, _, body = sec.partition("\n")
        for i, piece in enumerate(window_sentences(body, size, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#{len(chunks)}",
                doc_id=doc_id,
                title=title,
                section=heading.strip(),
                text=f"{title} - {heading.strip()}: {piece}",
            ))
    return chunks


def chunk_knowledge_base(directory: Path, size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(directory.glob("KB-*.md")):
        out.extend(chunk_article(path, size, overlap))
    return out
