"""Tiny retrieval evaluation helpers: recall@k and mean reciprocal rank."""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import RAGPipeline


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_source: str


@dataclass(frozen=True)
class EvalResult:
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: int


def evaluate(
    pipeline: RAGPipeline,
    cases: list[EvalCase],
    top_k: int = 3,
    hybrid: bool = False,
    alpha: float = 0.5,
) -> EvalResult:
    if not cases:
        raise ValueError("cases must not be empty")
    hits = 0
    reciprocal_ranks = 0.0
    for case in cases:
        answer = pipeline.ask(case.question, top_k=top_k, hybrid=hybrid, alpha=alpha)
        sources = answer.sources
        if case.expected_source in sources:
            hits += 1
            rank = sources.index(case.expected_source) + 1
            reciprocal_ranks += 1 / rank
    n = len(cases)
    return EvalResult(hits / n, reciprocal_ranks / n, n)
