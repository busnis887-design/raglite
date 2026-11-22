"""High-level RAG pipeline: ingest -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_text
from .embeddings import Embedder, HashingEmbedder
from .llm import ExtractiveLLM, LLM
from .store import ScoredChunk, VectorStore


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    contexts: list[ScoredChunk]

    @property
    def sources(self) -> list[str]:
        return sorted({c.chunk.source for c in self.contexts})


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> None:
        self.store = VectorStore(embedder or HashingEmbedder())
        self.llm = llm or ExtractiveLLM()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_text(self, text: str, source: str = "memory") -> int:
        chunks = chunk_text(text, source, self.chunk_size, self.overlap)
        self.store.add(chunks)
        return len(chunks)

    def ingest_files(self, path: str | Path, pattern: str = "*.txt") -> int:
        path = Path(path)
        files = sorted(path.glob(pattern)) if path.is_dir() else [path]
        total = 0
        for file in files:
            total += self.ingest_text(
                file.read_text(encoding="utf-8"), source=file.name
            )
        return total

    def ask(
        self,
        question: str,
        top_k: int = 3,
        hybrid: bool = False,
        alpha: float = 0.5,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        contexts = self.store.search(
            question, top_k=top_k, hybrid=hybrid, alpha=alpha
        )
        return Answer(
            question=question,
            answer=self.llm.generate(question, contexts),
            contexts=contexts,
        )

    def save(self, path: str | Path) -> None:
        """Persist chunks and embeddings to a JSON file."""
        self.store.save(path)

    def load(self, path: str | Path) -> int:
        """Restore a previously saved index; returns the number of vectors."""
        return self.store.load(path)
