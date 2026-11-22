import unittest

from raglite import (
    Chunk,
    ExtractiveLLM,
    HashingEmbedder,
    RAGPipeline,
    VectorStore,
    chunk_text,
    cosine,
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


if __name__ == "__main__":
    unittest.main()
