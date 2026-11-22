"""In-memory vector store with cosine similarity search."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import Embedder


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; returns 0.0 for zero vectors or length mismatch."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """Small, transparent store. Swap for FAISS/Qdrant when scaling out."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._items.append((chunk, self.embedder.embed(chunk.text)))

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed(query)
        scored = [
            ScoredChunk(chunk, cosine(query_vector, vector))
            for chunk, vector in self._items
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
