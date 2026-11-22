"""raglite - a small, dependency-light retrieval augmented generation toolkit."""

from .chunking import Chunk, chunk_text
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .llm import ExtractiveLLM, OpenAILLM
from .pipeline import Answer, RAGPipeline
from .store import ScoredChunk, VectorStore, cosine

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Embedder",
    "ExtractiveLLM",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "OpenAILLM",
    "RAGPipeline",
    "ScoredChunk",
    "VectorStore",
    "chunk_text",
    "cosine",
]
