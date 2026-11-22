"""Answer generators. Extractive by default; optional OpenAI chat backend."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Protocol

from .store import ScoredChunk


class LLM(Protocol):
    def generate(self, question: str, contexts: list[ScoredChunk]) -> str:
        ...


NO_CONTEXT = "I don't have enough information in the indexed documents to answer that."


class ExtractiveLLM:
    """Grounded, extractive answer builder -- no API key, fully deterministic."""

    def generate(self, question: str, contexts: list[ScoredChunk]) -> str:
        usable = [c for c in contexts if c.score > 0]
        if not usable:
            return NO_CONTEXT

        question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        sentences: list[str] = []
        for context in usable:
            for sentence in re.split(r"(?<=[.!?])\s+", context.chunk.text):
                sentence = sentence.strip()
                if len(sentence) > 25 and sentence not in sentences:
                    sentences.append(sentence)

        if not sentences:
            return NO_CONTEXT

        # Prefer sentences that actually talk about the question.
        def relevance(sentence: str) -> int:
            tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            return len(tokens & question_tokens)

        ranked = sorted(sentences, key=relevance, reverse=True)
        body = " ".join(ranked[:3])
        sources = ", ".join(sorted({c.chunk.source for c in usable}))
        return f"{body}\n\nSources: {sources}"


class OpenAILLM:
    """Optional chat-completions backend with a strict grounding prompt."""

    SYSTEM_PROMPT = (
        "Answer only using the provided context. If the context does not contain "
        "the answer, say you don't know. Never invent facts. Cite sources."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAILLM")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, question: str, contexts: list[ScoredChunk]) -> str:
        context_block = "\n\n".join(
            f"[{i + 1}] ({c.chunk.source}, score={c.score:.3f}) {c.chunk.text}"
            for i, c in enumerate(contexts)
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                    },
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        return data["choices"][0]["message"]["content"].strip()
