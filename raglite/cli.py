"""Command line interface: python -m raglite.cli --help"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raglite", description="Tiny RAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index text files")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--pattern", default="*.txt")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)

    sub.add_parser("demo", help="run the bundled knowledge-base demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = RAGPipeline()

    if args.command == "ingest":
        count = pipeline.ingest_files(args.path, args.pattern)
        print(f"Indexed {count} chunks from {args.path}")
        return 0

    if args.command == "demo":
        docs = Path(__file__).resolve().parent.parent / "examples" / "docs"
        count = pipeline.ingest_files(docs)
        print(f"Indexed {count} chunks from {docs}")
        print(pipeline.ask("How does the retry policy work?").answer)
        return 0

    answer = pipeline.ask(args.question, top_k=args.top_k)
    print(answer.answer)
    print("\nSources: " + ", ".join(answer.sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
