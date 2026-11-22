# raglite

[![tests](https://github.com/busnis887-design/raglite/actions/workflows/tests.yml/badge.svg)](https://github.com/busnis887-design/raglite/actions/workflows/tests.yml) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A small, testable **Retrieval-Augmented Generation (RAG)** pipeline in Python. It keeps the whole `ingest → embed → retrieve → generate` loop transparent and runs offline with zero third-party dependencies, while exposing plug-in points for real embedding and chat models.

## Why this exists

Most RAG tutorials hide the important parts behind a framework. `raglite` makes each stage explicit and swappable:

| Stage | Default | Swap with |
| --- | --- | --- |
| Chunking | sentence-aware, overlapping | any `list[Chunk]` producer |
| Embeddings | deterministic hashing embedder | `OpenAIEmbedder` / your own |
| Storage | in-memory cosine search + JSON persistence | FAISS, Qdrant, pgvector |
| Retrieval | dense cosine **or hybrid** (dense + BM25-like lexical) | custom scoring function |
| Generation | grounded extractive answers | `OpenAILLM` / your own |
| Evaluation | recall@k + MRR on labelled questions | your own metrics |

## Quickstart

```bash
python -m raglite.cli demo
```

```
Indexed 2 chunks from .../examples/docs
Timeouts are set to 30 seconds and failed calls follow an exponential backoff retry policy with three attempts.

Sources: api-guide.txt, architecture.txt
```

Index your own documents and ask questions:

```bash
python -m raglite.cli ingest ./docs --pattern "*.txt"
python -m raglite.cli ask "How does the retry policy work?" --top-k 3
```

## Library usage

```python
from raglite import RAGPipeline, OpenAIEmbedder, OpenAILLM

pipeline = RAGPipeline(
    embedder=OpenAIEmbedder(),          # optional; HashingEmbedder by default
    llm=OpenAILLM(),                    # optional; ExtractiveLLM by default
    chunk_size=400,
    overlap=80,
)
pipeline.ingest_files("./docs")

answer = pipeline.ask("What is the retry policy?")
print(answer.answer)
print(answer.sources)
```

## Architecture

```mermaid
flowchart LR
    D[Documents] --> C[Chunker<br/>400 chars / 80 overlap]
    C --> E[Embedder<br/>Hashing or OpenAI]
    E --> V[(Vector store<br/>cosine similarity)]
    Q[Question] --> E2[Embed question]
    E2 --> V
    V --> G[Generator<br/>grounded prompt]
    G --> A[Answer + sources]
```

## v0.2 highlights

```bash
# hybrid retrieval (dense + lexical, weighted by alpha)
python -m raglite.cli ask "retry policy" --hybrid --alpha 0.6

# persist / restore an index
python -m raglite.cli ingest ./docs --save index.json
python -m raglite.cli ask "..." --load index.json

# measure retrieval quality with labelled questions
python -m raglite.cli eval examples/labels.json --docs examples/docs --hybrid
# recall@3=1.000 MRR=0.833 cases=3
```

```python
from raglite import EvalCase, evaluate

result = evaluate(pipeline, [EvalCase("retry policy?", "api-guide.txt")], hybrid=True)
print(result.recall_at_k, result.mean_reciprocal_rank)
```

## Design decisions

- **Grounding by default.** The extractive generator only copies sentences from retrieved chunks and refuses to answer when retrieval returns nothing (`NO_CONTEXT`).
- **No hidden network calls.** The default pipeline is fully deterministic and works in CI without API keys.
- **Protocol-based adapters.** `Embedder` and `LLM` are structural protocols, so any object with the right method shape can be injected.
- **Small surface area.** ~350 lines of core code; easy to audit and extend.

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers chunk overlap/coverage, deterministic embeddings, cosine ranking, grounded answers, the no-context path, and validation errors.

## Roadmap

- Persistent vector index (SQLite / numpy)
- Hybrid BM25 + dense retrieval
- Answer citation spans and faithfulness scoring
- Streaming generation

## License

MIT
