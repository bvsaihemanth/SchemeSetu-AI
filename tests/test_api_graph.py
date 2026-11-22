"""Exercises the LangGraph wiring and the FastAPI endpoints with a scripted
LLM, so routing, citation handling, the refusal path and the verify branch
are all verified without Ollama running.

    python -m tests.test_api_graph
"""

from __future__ import annotations

import math
import os
import re
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
# The stand-in embedder is hashed bag-of-words: its cosine scale is compressed
# compared to BGE-M3 (a good hit scores ~0.1 here, ~0.7 there). The floor is
# therefore calibrated to the fixture. What this test verifies is that the gate
# MECHANISM fires on off-topic input -- not the production threshold value,
# which must be tuned on the real index against a golden set.
os.environ["RETRIEVAL_MIN_SCORE"] = "0.05"
# Verification is off by default (it doubles latency and a local 4B judges
# faithfulness poorly). This suite exists to prove the branch WORKS, so it
# opts in explicitly rather than silently depending on the default.
os.environ["VERIFY_ENABLED"] = "true"
# Hash collisions give the fixture a noise floor around 0.10, so off-topic
# probes below use vocabulary with zero corpus overlap (measured 0.0000).

from qdrant_client import QdrantClient  # noqa: E402

import schemes_rag.embeddings as embeddings_mod  # noqa: E402
import schemes_rag.ingest.pipeline as pipeline_mod  # noqa: E402
import schemes_rag.vectorstore as vs_mod  # noqa: E402
from schemes_rag.embeddings import EmbeddingResult  # noqa: E402
from schemes_rag.vectorstore import VectorStore  # noqa: E402

DIM = 1024
_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the","a","an","is","are","was","were","be","been","of","to","in","on","for",
    "and","or","it","this","that","what","which","who","whom","how","do","does",
    "did","can","i","my","me","you","your","at","by","with","from","as","if",
    "any","there","their","have","has","had","will","would","should","may","not",
}



class HashEmbedder:
    def _vec(self, text):
        v = [0.0] * DIM
        for w in (x for x in _WORD.findall(text.lower()) if x not in _STOP):
            v[hash(w) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _sp(self, text):
        out = {}
        for w in (x for x in _WORD.findall(text.lower()) if x not in _STOP):
            out[hash(w) % 250_000] = out.get(hash(w) % 250_000, 0.0) + 1.0
        return out

    def encode_documents(self, texts):
        texts = list(texts)
        return EmbeddingResult([self._vec(t) for t in texts], [self._sp(t) for t in texts])

    def encode_query(self, text):
        return EmbeddingResult([self._vec(text)], [self._sp(text)])

    def encode_queries(self, texts):
        texts = list(texts)
        return EmbeddingResult([self._vec(t) for t in texts], [self._sp(t) for t in texts])


class ScriptedLLM:
    """Stands in for Qwen. Replies based on which prompt it was handed."""

    def __init__(self):
        self.calls = []
        self.hostile_rewrite: str | None = None

    def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
        self.calls.append(system[:40])
        if "Classify the user's question" in system:
            q = user.lower()
            if "document" in q:
                intent = "documents"
            elif "eligib" in q or "who can apply" in q:
                intent = "eligibility"
            elif "benefit" in q:
                intent = "benefits"
            else:
                intent = "general"
            # Wrapped in prose + a code fence, to prove the tolerant parser works.
            # hostile_rewrite simulates a router that replaces an off-topic
            # question with a corpus-relevant phrase.
            rewrite = self.hostile_rewrite or user.split("Question:")[-1].split("\n")[0].strip()
            return (
                "Sure, here you go:\n```json\n"
                f'{{"intent": "{intent}", "schemes": [], "state": null, '
                f'"needs_clarification": false, "clarifying_question": null, '
                f'"search_query": "{rewrite}"}}\n```'
            )
        if "You check whether an ANSWER" in system:
            return '{"supported": true, "unsupported_claims": [], "confidence": "high"}'
        if "NOTHING" in user:
            return "INSUFFICIENT_EVIDENCE"
        return (
            "Manufacturer exporters or merchant exporters tied to a supporting "
            "manufacturer may apply [1]. Pharmaceutical products made through a "
            "Non-Infringing process are issued to manufacturer exporters only [2]. "
            "An invented claim [9]."
        )

    def health(self):
        return True


def main() -> int:
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_api_graph"])

    embedder = HashEmbedder()
    store = VectorStore(client=QdrantClient(":memory:"))
    llm = ScriptedLLM()

    embeddings_mod.get_embedder.cache_clear()
    embeddings_mod.get_embedder = lambda: embedder
    pipeline_mod.get_embedder = lambda: embedder
    pipeline_mod.get_vectorstore = lambda: store
    vs_mod.get_vectorstore.cache_clear()
    vs_mod.get_vectorstore = lambda: store

    import schemes_rag.llm as llm_mod

    llm_mod.get_llm.cache_clear()
    llm_mod.get_llm = lambda: llm

    import schemes_rag.generate as gen_mod
    import schemes_rag.agents.understand as understand_mod

    gen_mod.get_llm = lambda: llm
    understand_mod.get_llm = lambda: llm

    print("--- seed index ---")
    stats = pipeline_mod.run_ingest(corpus_dir="data/raw", recreate=True)
    print(f"chunks={stats['chunks_indexed']}")

    import schemes_rag.retriever as retr_mod

    retr_mod._retriever = None

    # ---------------- graph ----------------
    import schemes_rag.graph as graph_mod

    graph_mod._graph = None

    print("\n--- GRAPH: routing fans out to the specialist agents ---")
    for q, expected in [
        ("What documents are required for Advance Authorisation?", "documents"),
        ("Who can apply for Advance Authorisation?", "eligibility"),
        ("What benefits does Advance Authorisation give?", "benefits"),
        # No scheme named: "who can apply" becomes a discovery question, since
        # there is no document to assess against.
        ("Who can apply?", "discovery"),
    ]:
        ans = graph_mod.run_query(q, thread_id=f"t-{expected}")
        print(f"  {q[:48]:50} -> route={ans.route:12} cites={len(ans.citations)}")
        assert ans.route == expected, (ans.route, expected)

    print("\n--- GRAPH: citations resolved, bogus marker stripped ---")
    # A generating route, not eligibility: an eligibility verdict is rendered
    # deterministically from the criteria and carries no [n] markers by design.
    ans = graph_mod.run_query(
        "What benefits does Advance Authorisation give?", thread_id="cite"
    )
    print(f"  route={ans.route} answer: {ans.answer[:80]}...")
    assert "[9]" not in ans.answer, "invalid marker leaked into the answer"
    assert len(ans.citations) == 2, ans.citations
    for c in ans.citations:
        print(f"  [{c.marker}] {c.locator[:62]}")
        print(f"       url={c.source_url}")

    print("\n--- GRAPH: Evidence agent ran and grounded the figures ---")
    assert ans.verification is not None, "evidence agent did not run"
    print(f"  verdict={ans.verification['verdict']} "
          f"figures={ans.verification['figures_checked']}")

    print("\n--- GRAPH: clarification path ---")
    ans = graph_mod.run_query("hi", thread_id="vague")
    print(f"  route={ans.route} answer={ans.answer[:70]}...")
    assert ans.route == "clarify", ans.route

    print("\n--- GRAPH: refusal path (off-topic) ---")
    ans = graph_mod.run_query("how to bake sourdough bread", thread_id="offtopic")
    print(f"  insufficient={ans.insufficient_evidence}")
    print(f"  answer={ans.answer[:80]}...")
    assert ans.insufficient_evidence, "off-topic must refuse"
    assert not ans.citations

    print("\n--- GRAPH: rewritten query cannot bypass the gate ---")
    llm.hostile_rewrite = "advance authorisation eligibility documents"
    graph_mod._graph = None
    ans = graph_mod.run_query("how to bake sourdough bread", thread_id="hostile")
    print(f"  router rewrote query to a corpus-relevant phrase")
    print(f"  insufficient={ans.insufficient_evidence} (gate judged the original question)")
    assert ans.insufficient_evidence, "rewrite must not smuggle an off-topic query past the gate"
    llm.hostile_rewrite = None
    graph_mod._graph = None

    print("\n--- GRAPH: latency breakdown ---")
    ans = graph_mod.run_query("documents required", thread_id="timing")
    print(f"  {ans.latency_ms}")
    assert "total_ms" in ans.latency_ms

    # ---------------- API ----------------
    print("\n--- API ---")
    from fastapi.testclient import TestClient

    from schemes_rag.api.app import create_app

    with TestClient(create_app()) as client:
        r = client.get("/health")
        print(f"  GET /health  -> {r.status_code} {r.json()}")
        assert r.status_code == 200

        r = client.post("/search", json={"question": "documents required"})
        body = r.json()
        print(f"  POST /search -> {r.status_code} hits={len(body['hits'])} "
              f"top={body['top_score']}")
        assert r.status_code == 200 and body["hits"]
        print(f"     hit[0] {body['hits'][0]['locator'][:60]}")

        r = client.post("/query", json={"question": "Who can apply?"})
        body = r.json()
        print(f"  POST /query  -> {r.status_code} route={body['route']} "
              f"citations={len(body['citations'])}")
        assert r.status_code == 200
        assert body["citations"] and body["thread_id"]
        assert "[9]" not in body["answer"]

        r = client.post(
            "/query",
            json={
                "question": "documents required",
                "filters": {"sections": ["documents_required"], "final_k": 2},
            },
        )
        body = r.json()
        print(f"  POST /query (filtered) -> {r.status_code} "
              f"schemes={body['schemes_used']}")
        assert r.status_code == 200

        r = client.post("/query", json={"question": ""})
        print(f"  POST /query (empty) -> {r.status_code} (validation)")
        assert r.status_code == 422

        # ---- the UI's wire contract ----
        # The page posts paired answers, so the shape has to be accepted and
        # the pairing has to survive validation. The interview itself is
        # exercised in test_agents; this is the HTTP boundary.
        r = client.post("/query", json={
            "question": "Answered 2 eligibility question(s).",
            "thread_id": "api-elig",
            "answers": [
                {"question": "Are you a farmer?", "answer": "yes"},
                {"question": "Do you own cultivable land?", "answer": "no"},
            ],
        })
        print(f"  POST /query (paired answers) -> {r.status_code} "
              f"route={r.json().get('route')}")
        assert r.status_code == 200, r.text

        # An answer with no question cannot be paired, so it must be refused
        # at the edge rather than stored as a note with an empty question.
        r = client.post("/query", json={
            "question": "x", "answers": [{"answer": "yes"}],
        })
        print(f"  POST /query (answer, no question) -> {r.status_code} (validation)")
        assert r.status_code == 422, r.text

        # ---- feedback ----
        # With tracing off this must refuse rather than pretend. A UI that
        # believes it recorded a rating it discarded is the worse failure.
        r = client.post("/feedback", json={
            "trace_id": "abc123", "helpful": True,
        })
        print(f"  POST /feedback (tracing off) -> {r.status_code} "
              f"{str(r.json().get('detail'))[:48]}")
        assert r.status_code == 503, r.text
        assert "LANGFUSE_ENABLED" in r.json()["detail"], r.text

        r = client.post("/feedback", json={"helpful": True})
        print(f"  POST /feedback (no trace_id) -> {r.status_code} (validation)")
        assert r.status_code == 422, r.text

        # ---- the UI itself ----
        r = client.get("/")
        print(f"  GET /         -> {r.status_code} "
              f"{r.headers.get('content-type')} {len(r.text)} bytes")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        # The page is useless if it does not read the fields it drives the
        # interview from, so assert the coupling rather than just a 200.
        for field in ("awaiting_answer", "assessment", "questions", "answers",
                      "thread_id", "data-answer", "trace_id", "/feedback"):
            assert field in r.text, f"UI never references {field}"
        print("     UI served and references the interview contract")

        r = client.get("/stats")
        print(f"  GET /stats   -> {r.status_code} points={r.json().get('points')}")

        r = client.get("/metrics")
        text = r.text
        print(f"  GET /metrics -> {r.status_code}, {len(text.splitlines())} lines")
        assert "rag_queries_total" in text
        for line in text.splitlines():
            if line.startswith("rag_queries_total{"):
                print(f"     {line}")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
