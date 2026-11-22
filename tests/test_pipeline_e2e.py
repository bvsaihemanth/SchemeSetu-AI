"""End-to-end: parse -> chunk -> embed -> index -> retrieve.

Uses a deterministic hashed bag-of-words embedder in place of BGE-M3, so the
whole data path is exercised with no model download. Similarity is real
(shared vocabulary scores higher), so retrieval ordering is meaningful.

    python -m tests.test_pipeline_e2e
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient

import schemes_rag.embeddings as embeddings_mod
import schemes_rag.ingest.pipeline as pipeline_mod
import schemes_rag.vectorstore as vs_mod
from schemes_rag.config import get_settings
from schemes_rag.embeddings import EmbeddingResult
from schemes_rag.models import Section
from schemes_rag.retriever import Retriever
from schemes_rag.vectorstore import VectorStore

DIM = 1024
_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the","a","an","is","are","was","were","be","been","of","to","in","on","for",
    "and","or","it","this","that","what","which","who","whom","how","do","does",
    "did","can","i","my","me","you","your","at","by","with","from","as","if",
    "any","there","their","have","has","had","will","would","should","may","not",
}



class HashEmbedder:
    """Hashed TF vector, L2-normalised. Stand-in for BGE-M3 dense+sparse."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in (w for w in _WORD.findall(text.lower()) if w not in _STOP):
            v[hash(word) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _sparse(self, text: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for word in (w for w in _WORD.findall(text.lower()) if w not in _STOP):
            out[hash(word) % 250_000] = out.get(hash(word) % 250_000, 0.0) + 1.0
        return out

    def encode_documents(self, texts):
        texts = list(texts)
        return EmbeddingResult([self._vec(t) for t in texts], [self._sparse(t) for t in texts])

    def encode_query(self, text):
        return EmbeddingResult([self._vec(text)], [self._sparse(text)])

    def encode_queries(self, texts):
        texts = list(texts)
        return EmbeddingResult([self._vec(t) for t in texts], [self._sparse(t) for t in texts])


def main() -> int:
    # PYTHONHASHSEED must be fixed for the stand-in embedder to be stable.
    if sys.flags.hash_randomization:
        import os

        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_pipeline_e2e"])

    s = get_settings()
    embedder = HashEmbedder()
    store = VectorStore(client=QdrantClient(":memory:"))

    # Inject the stand-ins into every lookup point.
    embeddings_mod.get_embedder.cache_clear()
    embeddings_mod.get_embedder = lambda: embedder
    pipeline_mod.get_embedder = lambda: embedder
    pipeline_mod.get_vectorstore = lambda: store
    vs_mod.get_vectorstore.cache_clear()
    vs_mod.get_vectorstore = lambda: store

    print("--- INGEST ---")
    stats = pipeline_mod.run_ingest(corpus_dir="data/raw", recreate=True)
    print(f"docs={stats['documents_indexed']} chunks={stats['chunks_indexed']} "
          f"count={stats.get('collection_count')} elapsed={stats['elapsed_s']}s")
    assert stats["chunks_indexed"] == 16, stats
    assert stats.get("collection_count") == 16, stats

    print("\n--- RESUMABILITY (re-run, nothing changed) ---")
    stats2 = pipeline_mod.run_ingest(corpus_dir="data/raw")
    print(f"skipped={stats2['documents_skipped']} indexed={stats2['documents_indexed']} "
          f"count={stats2.get('collection_count')}")
    assert stats2["documents_skipped"] == 1, stats2
    assert stats2["documents_indexed"] == 0, stats2
    assert stats2.get("collection_count") == 16, "re-run must not duplicate points"

    retriever = Retriever()
    retriever.embedder = embedder
    retriever.store = store

    print("\n--- RETRIEVAL ---")
    queries = [
        ("Who can apply for Advance Authorisation?", None),
        ("What documents are required?", None),
        ("export obligation value addition percentage", None),
        ("What documents are required?", [Section.DOCUMENTS_REQUIRED.value]),
    ]
    for q, sections in queries:
        res = retriever.retrieve(q, sections=sections, final_k=3, min_score=0.0)
        tag = f" [filter: {sections[0]}]" if sections else ""
        print(f"\nQ: {q}{tag}")
        print(f"   gate_score={res.top_score:.4f} candidates={res.candidates_considered} "
              f"returned={len(res.chunks)}")
        for rc in res.chunks:
            print(f"   - {rc.chunk.section.value:20} {rc.effective_score:.4f}  "
                  f"{rc.chunk.locator()[:70]}")
        assert res.chunks, f"no results for {q}"

    # The section filter must constrain results to that section only.
    res = retriever.retrieve(
        "documents", sections=[Section.DOCUMENTS_REQUIRED.value], min_score=0.0
    )
    assert all(rc.chunk.section is Section.DOCUMENTS_REQUIRED for rc in res.chunks)
    print(f"\nsection filter honoured: {len(res.chunks)} chunk(s), all documents_required")

    print("\n--- RELEVANCE GATE (off-topic query) ---")
    res = retriever.retrieve("how to bake sourdough bread", min_score=0.05)
    print(f"insufficient={res.insufficient} gate_score={res.top_score:.4f}")
    assert res.insufficient, "off-topic query should fall below the floor"

    print("\n--- CONTEXT BUDGET ---")
    res = retriever.retrieve("advance authorisation scheme", final_k=20, min_score=0.0)
    total = sum(rc.chunk.token_count for rc in res.chunks)
    print(f"chunks={len(res.chunks)} context_tokens={total} "
          f"budget={s.llm_context_budget_tokens}")
    assert total <= s.llm_context_budget_tokens

    print("\n--- PROMPT ASSEMBLY ---")
    from schemes_rag.prompts import build_answer_prompt

    res = retriever.retrieve("Who can apply?", final_k=3, min_score=0.0)
    system, user = build_answer_prompt("Who can apply?", res.chunks)
    assert "[1]" in user and "SOURCES" in user
    print(f"system={len(system)} chars, user={len(user)} chars")
    print("  first source block:")
    print("   ", user.split("\n")[1][:95])

    print("\n--- CITATION RESOLUTION ---")
    from schemes_rag.agents.evidence import resolve_citations

    text, cites = resolve_citations(
        "Manufacturer exporters may apply [1]. Merchant exporters too [2]. Bogus [9].",
        res.chunks,
    )
    print(f"  cleaned: {text}")
    for c in cites:
        print(f"  [{c.marker}] {c.locator[:60]} | {c.source_url}")
    assert len(cites) == 2, cites
    assert "[9]" not in text, "out-of-range marker must be stripped"

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
