"""Regression test for the reported failure.

    Query:  "Who can apply for PM-KISAN?"
    Before: top hits were PM-YASASVI (scholarships), Prime Minister's
            Scholarship Scheme, PM Kisan MAANDHAN Yojana and PM POSHAN --
            four different schemes -- while PM-KISAN's own Eligibility
            section never appeared at all.
    Cause:  the token "PM-KISAN" exists nowhere in the document's title or
            body, only in its source URL, so similarity fell back to matching
            question *phrasing* across unrelated schemes.
    Fix:    resolve the scheme from the URL slug before searching, then scope
            retrieval to that document.

Run against the 34-document confusable subset (every scheme that collided in
the original output), with a deterministic stand-in embedder.

    python -m tests.test_scheme_resolution
"""

from __future__ import annotations

import math
import os
import re
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
# Fixture scale, measured on this 34-doc subset with the stand-in embedder:
# on-topic queries gate at 0.47-0.56, off-topic at 0.16-0.20. 0.30 separates
# them. The production default (0.35) is for BGE-M3's scale, not this one.
os.environ["RETRIEVAL_MIN_SCORE"] = "0.30"

CORPUS = "/mnt/user-data/uploads/myscheme-ai/schemes_txt"
DIM = 1024
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "it", "this", "that", "what", "which", "who", "how", "do", "does", "can",
    "i", "my", "me", "you", "your", "at", "by", "with", "from", "as", "if",
    "any", "there", "their", "have", "has", "will", "would", "not", "am",
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
        from schemes_rag.embeddings import EmbeddingResult

        return EmbeddingResult([self._vec(t) for t in texts], [self._sp(t) for t in texts])

    def encode_query(self, text):
        from schemes_rag.embeddings import EmbeddingResult

        return EmbeddingResult([self._vec(text)], [self._sp(text)])

    def encode_queries(self, texts):
        texts = list(texts)
        from schemes_rag.embeddings import EmbeddingResult

        return EmbeddingResult([self._vec(t) for t in texts], [self._sp(t) for t in texts])


class ScriptedLLM:
    """Router returns the intent; it is NOT trusted for scheme identity."""

    def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
        if "Classify the user's question" in system:
            q = user.lower()
            intent = (
                "documents" if "document" in q
                else "eligibility" if ("eligib" in q or "who can apply" in q)
                else "benefits" if "benefit" in q
                else "general"
            )
            # Deliberately hallucinates a scheme name, to prove the index
            # overrides the LLM rather than trusting it.
            return (
                f'{{"intent": "{intent}", "schemes": ["Some Wrong Scheme"], '
                f'"state": null, "needs_clarification": false, '
                f'"clarifying_question": null, "search_query": '
                f'"{user.split("Question:")[-1].split(chr(10))[0].strip()}"}}'
            )
        if "You check whether an ANSWER" in system:
            return '{"supported": true, "unsupported_claims": [], "confidence": "high"}'
        return "Landholding farmers' families with cultivable land are eligible [1]."

    def health(self):
        return True


def main() -> int:
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_scheme_resolution"])

    if not os.path.isdir(CORPUS):
        print(f"SKIP: corpus not available at {CORPUS}")
        return 0

    from qdrant_client import QdrantClient

    import schemes_rag.agents.understand as understand_mod
    import schemes_rag.embeddings as embeddings_mod
    import schemes_rag.generate as gen_mod
    import schemes_rag.graph as graph_mod
    import schemes_rag.ingest.pipeline as pipeline_mod
    import schemes_rag.llm as llm_mod
    import schemes_rag.retriever as retr_mod
    import schemes_rag.vectorstore as vs_mod
    from schemes_rag.scheme_index import reset_scheme_index
    from schemes_rag.vectorstore import VectorStore

    embedder, llm = HashEmbedder(), ScriptedLLM()
    store = VectorStore(client=QdrantClient(":memory:"))

    embeddings_mod.get_embedder.cache_clear()
    embeddings_mod.get_embedder = lambda: embedder
    pipeline_mod.get_embedder = lambda: embedder
    pipeline_mod.get_vectorstore = lambda: store
    vs_mod.get_vectorstore.cache_clear()
    vs_mod.get_vectorstore = lambda: store
    # retriever.py imported these by name, so the module-level references
    # must be replaced too, not just embeddings.get_embedder.
    retr_mod.get_embedder = lambda: embedder
    retr_mod.get_vectorstore = lambda: store
    llm_mod.get_llm.cache_clear()
    llm_mod.get_llm = lambda: llm
    gen_mod.get_llm = lambda: llm
    understand_mod.get_llm = lambda: llm

    import pathlib

    pathlib.Path("data/processed/ingest_manifest.json").unlink(missing_ok=True)

    print("--- ingest 34-document confusable subset ---")
    stats = pipeline_mod.run_ingest(corpus_dir=CORPUS, recreate=True)
    print(f"docs={stats['documents_indexed']} chunks={stats['chunks_indexed']} "
          f"schemes_indexed={stats['scheme_index']['schemes']}")
    reset_scheme_index()
    retr_mod._retriever = None
    graph_mod._graph = None

    # ---------------- the reported query ----------------
    QUERY = "Who can apply for PM-KISAN?"
    TARGET = "Pradhan Mantri Kisan Samman Nidhi"
    WRONG = [
        "PM-YASASVI", "Prime Minister's Scholarship", "Maandhan", "POSHAN",
    ]

    print(f"\n--- RESOLUTION: {QUERY!r} ---")
    plan = understand_mod.understand_query(QUERY)
    print(f"  intent         : {plan.intent}")
    print(f"  resolved       : {plan.schemes}")
    print(f"  confidence     : {plan.scheme_confidence}  (via {plan.scheme_matched_on})")
    assert plan.scheme_confidence == "exact", plan.scheme_confidence
    assert plan.schemes == [TARGET], plan.schemes
    assert "Some Wrong Scheme" not in plan.schemes, "LLM guess must be overridden"
    print("  -> index overrode the LLM's hallucinated scheme name")

    print(f"\n--- RETRIEVAL ---")
    ans = graph_mod.run_query(QUERY, thread_id="pmkisan")
    import schemes_rag.graph as g

    final = g.get_graph().invoke(
        {"question": QUERY, "history": [], "filters": {}},
        config={"configurable": {"thread_id": "pmkisan2"}},
    )
    chunks = final.get("chunks") or []
    print(f"  {len(chunks)} chunks returned")
    for rc in chunks:
        print(f"    {rc.effective_score:.4f}  {rc.chunk.section.value:20} {rc.chunk.scheme_name[:44]}")

    assert chunks, "no chunks retrieved"
    schemes = {rc.chunk.scheme_name for rc in chunks}
    assert schemes == {TARGET}, f"leaked other schemes: {schemes}"
    print(f"  -> every chunk from {TARGET!r}")

    for bad in WRONG:
        assert not any(bad.lower() in s.lower() for s in schemes), f"{bad} still leaking"
    print(f"  -> none of the originally-returned wrong schemes present")

    sections = [rc.chunk.section.value for rc in chunks]
    assert "eligibility" in sections, f"Eligibility section still missing: {sections}"
    print(f"  -> Eligibility section IS retrieved (was absent before)")

    print(f"\n--- SECTION BALANCE (66% of corpus is FAQ) ---")
    faq_n = sections.count("faq")
    print(f"  sections: {sections}")
    print(f"  faq={faq_n}/{len(sections)}")
    assert faq_n <= (len(sections) + 1) // 2, f"FAQ monopolised the window: {sections}"
    print("  -> FAQ chunks capped, narrative sections retained")

    print(f"\n--- ANSWER ---")
    print(f"  route: {ans.route}")
    print(f"  answer: {ans.answer[:90]}")
    # "Who can apply for PM-KISAN?" now reaches the Eligibility agent, whose
    # verdict is rendered deterministically from the assessed criteria rather
    # than generated -- so it carries an assessment instead of [n] citations.
    if ans.route == "eligibility":
        assert ans.assessment is not None, "eligibility produced no assessment"
        print(f"  overall: {ans.assessment.overall}")
        print(f"  criteria assessed: {len(ans.assessment.criteria)}")
    else:
        for c in ans.citations:
            print(f"  [{c.marker}] {c.locator[:66]}")
        assert ans.citations, "no citations"

    # ---------------- other acronym forms ----------------
    print(f"\n--- OTHER NAME FORMS ---")
    cases = [
        ("PM KISAN eligibility", TARGET),
        ("pmkisan", TARGET),
        ("pm kisan maandhan yojana", "Pradhan Mantri Kisan Maandhan Yojana"),
        ("what is apy", "Atal Pension Yojana"),
        ("kcc", "Kisan Credit Card"),
        ("PMFBY claim process", "Pradhan Mantri Fasal Bima Yojna (PMFBY)"),
        ("awas yojana gramin", "Pradhan Mantri Awaas Yojana - Gramin"),
    ]
    for q, expect in cases:
        p = understand_mod.understand_query(q)
        got = p.schemes[0] if p.schemes else None
        ok = got == expect
        print(f"  {'OK ' if ok else 'BAD'} {q[:34]:36} -> {str(got)[:46]}")
        assert ok, f"{q}: expected {expect}, got {got}"

    # ---------------- a named scheme is never "too vague" ----------------
    print(f"\n--- CLARIFICATION SUPPRESSED WHEN SCHEME IS RESOLVED ---")

    class VagueRouterLLM(ScriptedLLM):
        """Router insists the question needs clarification."""

        def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
            if "Classify the user's question" in system:
                return (
                    '{"intent": "eligibility", "schemes": [], "state": null, '
                    '"needs_clarification": true, "clarifying_question": '
                    '"Which scheme and which state?", "search_query": "x"}'
                )
            return super().complete(system, user, temperature, max_tokens, json_mode)

    vague = VagueRouterLLM()
    understand_mod.get_llm = lambda: vague
    plan = understand_mod.understand_query("Who can apply for PM-KISAN?")
    print(f"  router asked to clarify; resolver found {plan.schemes}")
    print(f"  needs_clarification={plan.needs_clarification}")
    assert plan.scheme_confidence == "exact"
    assert not plan.needs_clarification, "a resolved scheme is not a vague question"
    assert plan.clarifying_question is None
    print("  -> clarification suppressed, question answered instead")

    # but a genuinely vague question still clarifies
    plan = understand_mod.understand_query("hi")
    assert plan.needs_clarification, "genuinely vague input must still clarify"
    print("  -> 'hi' still clarifies (suppression is scoped to resolved schemes)")
    understand_mod.get_llm = lambda: llm

    # ---------------- short-slug collisions ----------------
    print(f"\n--- SHORT SLUGS MUST NOT COLLIDE WITH ORDINARY WORDS ---")
    from schemes_rag.scheme_index import SchemeEntry, SchemeIndex, generate_aliases

    # Real entries from the corpus. 54 schemes have three-character slugs, and
    # "am i" joins to "ami" -- the slug of Agricultural Marketing
    # Infrastructure. "am i eligible for <coffee scheme>" resolved to it with
    # `exact` confidence, and the coffee scheme could not match at all because
    # its 10-token name exceeded the old max_alias_tokens of 8.
    raw = [
        ("Agricultural Marketing Infrastructure", "ami"),
        ("Coffee Development Programme in North Eastern Region Consolidation of Coffee", "cdpnercc"),
        ("Coffee Development Programme in North Eastern Region Expansion of Coffee", "cdpnerec"),
        ("Atal Pension Yojana", "apy"),
        ("Biotechnology Ignition Grant Scheme", "big"),
        ("Education Loan Scheme", "els"),
    ]
    collide = SchemeIndex([
        SchemeEntry(doc_id=f"x{i}", doc_name=n, scheme_name=n, slug=sl,
                    aliases=generate_aliases(n, sl))
        for i, (n, sl) in enumerate(raw)
    ])
    print(f"  max_alias_tokens derived from data: {collide.max_alias_tokens}")
    collision_cases = [
        ("am i eligible for Coffee Development Programme in North Eastern "
         "Region Consolidation of Coffee", "Consolidation of Coffee"),
        ("am i eligible for Agricultural Marketing Infrastructure",
         "Agricultural Marketing"),
        ("AMI eligibility", "Agricultural Marketing"),
        ("am i eligible for anything", None),
        ("do i get big money", None),
        ("BIG scheme", "Biotechnology Ignition"),
        ("what is els", "Education Loan"),
    ]
    for q, expect in collision_cases:
        ms = collide.resolve(q)
        got = ms[0].scheme_name if ms else None
        good = (expect is None and got is None) or (expect and got and expect.lower() in got.lower())
        print(f"  {'OK ' if good else 'BAD'} {q[:52]:54} -> {str(got)[:36]}")
        assert good, f"{q!r}: expected {expect}, got {got}"
    print("  -> stopword n-grams rejected; long names resolvable; real acronyms kept")

    # ---------------- off-topic still refuses ----------------
    print(f"\n--- OFF-TOPIC STILL REFUSES ---")
    for off in ["how do I bake sourdough bread", "quantum chromodynamics lagrangian",
                "my cat is sneezing"]:
        ans = graph_mod.run_query(off, thread_id=f"off-{off[:8]}")
        print(f"  {off[:34]:36} insufficient={ans.insufficient_evidence}")
        assert ans.insufficient_evidence, f"off-topic must refuse: {off}"
        assert not ans.citations

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
