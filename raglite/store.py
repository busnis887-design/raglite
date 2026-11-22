"""In-memory and JSON-persisted vector stores with cosine and lexical search."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .embeddings import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


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


def lexical_score(query: str, text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25-like term score over a single document (no index required)."""
    q_tokens = _TOKEN.findall(query.lower())
    d_tokens = _TOKEN.findall(text.lower())
    if not q_tokens or not d_tokens:
        return 0.0
    counts = Counter(d_tokens)
    avg_len = len(d_tokens)
    score = 0.0
    for token in set(q_tokens):
        tf = counts.get(token, 0)
        if not tf:
            continue
        denom = tf + k1 * (1 - b + b * avg_len / max(avg_len, 1))
        score += math.log(1 + (len(d_tokens) - tf + 0.5) / (tf + 0.5)) * (tf * (k1 + 1) / denom)
    return max(0.0, score)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class VectorStore:
    """Small, transparent store. Swap for FAISS/Qdrant when scaling out."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._items.append((chunk, self.embedder.embed(chunk.text)))

    def search(
        self, query: str, top_k: int = 3, hybrid: bool = False, alpha: float = 0.5
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed(query)
        scored: list[ScoredChunk] = []
        for chunk, vector in self._items:
            dense = cosine(query_vector, vector)
            lexical = lexical_score(query, chunk.text)
            score = alpha * dense + (1 - alpha) * lexical if hybrid else dense
            scored.append(ScoredChunk(chunk, score, dense, lexical))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [
                {"text": c.text, "source": c.source, "start": c.start, "end": c.end}
                for c, _ in self._items
            ],
            "vectors": [v for _, v in self._items],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: str | Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._items = [
            (Chunk(**item), vector)
            for item, vector in zip(data["chunks"], data["vectors"])
        ]
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
