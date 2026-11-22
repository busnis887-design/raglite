"""Embedding backends: deterministic offline hashing or OpenAI embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]:
        """Return a numeric vector for ``text``."""
        ...


class HashingEmbedder:
    """Dependency-free, deterministic embeddings for demos and tests.

    Not semantic: similar words get different vectors. Use OpenAIEmbedder
    (or any compatible backend) for production retrieval quality.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbedder:
    """OpenAI-compatible embeddings API client using only the standard library."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = 0

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        vector = data["data"][0]["embedding"]
        self.dim = len(vector)
        return vector
