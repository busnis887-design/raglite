"""Text splitting with sentence-aware boundaries and overlap."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    start: int
    end: int


def chunk_text(
    text: str,
    source: str = "memory",
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Boundaries prefer a sentence or line ending so retrieval units stay
    semantically intact. The overlap prevents context loss across cuts.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            sentence_break = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if sentence_break != -1 and sentence_break - start > chunk_size // 2:
                end = sentence_break + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, start=start, end=end))

        if end >= length:
            break
        start = max(start + 1, end - overlap)

    return chunks
