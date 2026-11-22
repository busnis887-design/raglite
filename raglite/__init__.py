"""raglite - a small, dependency-light retrieval augmented generation toolkit."""

from .chunking import Chunk, chunk_text
from .embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from .evaluation import EvalCase, EvalResult, evaluate
from .llm import ExtractiveLLM, OpenAILLM
from .pipeline import Answer, RAGPipeline
from .store import ScoredChunk, VectorStore, cosine, lexical_score

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "Chunk",
    "Embedder",
    "EvalCase",
    "EvalResult",
    "ExtractiveLLM",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "OpenAILLM",
    "RAGPipeline",
    "ScoredChunk",
    "VectorStore",
    "chunk_text",
    "cosine",
    "evaluate",
    "lexical_score",
]
