import json
import tempfile
import unittest
from pathlib import Path

from raglite import (
    Chunk,
    EvalCase,
    ExtractiveLLM,
    HashingEmbedder,
    RAGPipeline,
    VectorStore,
    __version__,
    chunk_text,
    cosine,
    evaluate,
    lexical_score,
)
from raglite.llm import NO_CONTEXT


class ChunkingTests(unittest.TestCase):
    def test_overlap_and_coverage(self):
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, 500)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)

    def test_sentence_boundary(self):
        text = "First sentence here. Second sentence follows. Third one ends it."
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        self.assertTrue(all(c.text for c in chunks))


class EmbeddingAndStoreTests(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        embedder = HashingEmbedder(dim=128)
        a = embedder.embed("hello world")
        b = embedder.embed("hello world")
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=6)

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([1], [1, 2]), 0.0)

    def test_search_ranking(self):
        embedder = HashingEmbedder()
        store = VectorStore(embedder)
        store.add(
            [
                Chunk("retry policy backoff three attempts", "api.txt", 0, 40),
                Chunk("billing invoices and refunds", "billing.txt", 0, 30),
            ]
        )
        results = store.search("retry policy backoff", top_k=2)
        self.assertEqual(results[0].chunk.source, "api.txt")
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertEqual(len(store), 2)


class PipelineTests(unittest.TestCase):
    def test_ingest_and_grounded_answer(self):
        pipeline = RAGPipeline()
        count = pipeline.ingest_text(
            "The retry policy uses exponential backoff with three attempts. "
            "Rate limits are enforced per API key. All endpoints require an API key.",
            source="guide.txt",
        )
        self.assertGreater(count, 0)
        answer = pipeline.ask("What is the retry policy?")
        self.assertIn("backoff", answer.answer.lower())
        self.assertEqual(answer.sources, ["guide.txt"])

    def test_no_context(self):
        answer = RAGPipeline().ask("anything")
        self.assertEqual(answer.answer, NO_CONTEXT)

    def test_empty_question(self):
        with self.assertRaises(ValueError):
            RAGPipeline().ask("   ")

    def test_extractive_llm_uses_only_context(self):
        llm = ExtractiveLLM()
        contexts = [
            type(
                "Scored",
                (),
                {
                    "chunk": Chunk("Apples are grown in orchards.", "fruit.txt", 0, 30),
                    "score": 0.8,
                },
            )()
        ]
        self.assertIn("orchards", llm.generate("Where are apples grown?", contexts))


class V02Tests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.2.0")

    def test_lexical_score(self):
        self.assertGreater(lexical_score("retry policy", "the retry policy uses backoff"), 0)
        self.assertEqual(lexical_score("retry", "unrelated words here"), 0.0)

    def test_hybrid_search(self):
        pipeline = RAGPipeline()
        pipeline.ingest_text(
            "The retry policy uses exponential backoff with three attempts.",
            source="api.txt",
        )
        answer = pipeline.ask("retry policy", hybrid=True, alpha=0.5)
        self.assertEqual(answer.sources, ["api.txt"])
        self.assertGreaterEqual(answer.contexts[0].lexical_score, 0)

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            first = RAGPipeline()
            first.ingest_text("Dense vectors and embeddings are stored here.", "doc.txt")
            first.save(path)
            second = RAGPipeline()
            self.assertEqual(second.load(path), len(first.store))
            self.assertEqual(
                second.ask("embeddings stored").sources,
                first.ask("embeddings stored").sources,
            )

    def test_evaluate(self):
        pipeline = RAGPipeline()
        docs = Path(__file__).resolve().parent.parent / "examples" / "docs"
        pipeline.ingest_files(docs)
        labels = json.loads((docs.parent / "labels.json").read_text(encoding="utf-8"))
        cases = [EvalCase(**row) for row in labels]
        result = evaluate(pipeline, cases, top_k=2, hybrid=True)
        self.assertEqual(result.cases, 3)
        self.assertGreater(result.recall_at_k, 0)
        self.assertGreater(result.mean_reciprocal_rank, 0)


if __name__ == "__main__":
    unittest.main()
