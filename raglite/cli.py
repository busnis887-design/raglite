"""Command line interface: python -m raglite.cli --help"""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import EvalCase, evaluate
from .pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raglite", description="Tiny RAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index text files")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--pattern", default="*.txt")
    ingest.add_argument("--save", type=Path, help="persist the index to JSON")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--hybrid", action="store_true", help="blend dense + lexical scores")
    ask.add_argument("--alpha", type=float, default=0.5, help="dense weight when hybrid")
    ask.add_argument("--load", type=Path, help="load a persisted index")

    ev = sub.add_parser("eval", help="evaluate retrieval on labelled questions")
    ev.add_argument("labels", type=Path, help="JSON list of {question, expected_source}")
    ev.add_argument("--docs", type=Path, required=True, help="document directory to index")
    ev.add_argument("--hybrid", action="store_true")
    ev.add_argument("--top-k", type=int, default=3)

    sub.add_parser("demo", help="run the bundled knowledge-base demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    import json

    args = build_parser().parse_args(argv)
    pipeline = RAGPipeline()

    if args.command == "ingest":
        count = pipeline.ingest_files(args.path, args.pattern)
        if args.save:
            pipeline.save(args.save)
        print(f"Indexed {count} chunks from {args.path}")
        return 0

    if args.command == "demo":
        docs = Path(__file__).resolve().parent.parent / "examples" / "docs"
        count = pipeline.ingest_files(docs)
        print(f"Indexed {count} chunks from {docs}")
        print(pipeline.ask("How does the retry policy work?", hybrid=True).answer)
        return 0

    if args.command == "eval":
        pipeline.ingest_files(args.docs)
        cases = [
            EvalCase(**row)
            for row in json.loads(args.labels.read_text(encoding="utf-8"))
        ]
        result = evaluate(pipeline, cases, top_k=args.top_k, hybrid=args.hybrid)
        print(
            f"recall@{args.top_k}={result.recall_at_k:.3f} "
            f"MRR={result.mean_reciprocal_rank:.3f} cases={result.cases}"
        )
        return 0

    if getattr(args, "load", None):
        pipeline.load(args.load)
    answer = pipeline.ask(
        args.question, top_k=args.top_k, hybrid=args.hybrid, alpha=args.alpha
    )
    print(answer.answer)
    print("\nSources: " + ", ".join(answer.sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
