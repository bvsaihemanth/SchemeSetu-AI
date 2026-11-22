"""Query-side retrieval: question -> ranked, budget-trimmed context.

Pipeline:
    embed query -> hybrid search (dense + sparse, RRF)
                -> score floor check   (hallucination guard #1)
                -> rerank              (optional)
                -> trim to final_k and token budget

The score floor is the important part. If nothing clears it we return an
empty result and the caller answers "not in my sources" WITHOUT calling the
LLM. A 4B model handed weak context will produce fluent, plausible,
unsupported answers -- the cheapest fix is not to ask it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from qdrant_client import models as qm

from .config import get_settings
from .embeddings import get_embedder
from .models import RetrievedChunk
from .tracing import observed, score
from .vectorstore import build_filter, get_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    insufficient: bool = False
    top_score: float = 0.0
    candidates_considered: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)
    # what the query was actually scoped to -- shown by scripts/query.py and
    # /search, because "why did it return that?" is usually answered here
    scoped_schemes: list[str] = field(default_factory=list)
    scoped_sections: list[str] = field(default_factory=list)

    @property
    def schemes(self) -> list[str]:
        seen: list[str] = []
        for rc in self.chunks:
            if rc.chunk.scheme_name not in seen:
                seen.append(rc.chunk.scheme_name)
        return seen


class Retriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = get_embedder()
        self.store = get_vectorstore()

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """Traced wrapper around _retrieve.

        In a RAG system most wrong answers are retrieval failures wearing a
        fluent disguise, and from the outside the two are indistinguishable.
        This records what was searched, what came back and with what scores,
        so "the model made it up" and "the model was handed the wrong
        document" stop being the same bug report.

        as_type="retriever" is not decoration: Langfuse renders retriever
        spans with their documents listed, and makes them filterable.
        """
        with observed(
            "retrieve",
            as_type="retriever",
            input={
                "query": query,
                "gate_query": kwargs.get("gate_query"),
                "scoped_schemes": kwargs.get("scheme_names"),
                "scoped_sections": kwargs.get("sections"),
            },
        ) as span:
            result = self._retrieve(query, **kwargs)

            span.update(
                # The chunks AND their scores. A score just under the floor
                # looks identical to one far below it in the logs, and the
                # difference decides whether RETRIEVAL_MIN_SCORE is wrong.
                output=[
                    {
                        "scheme": rc.chunk.scheme_name,
                        "section": rc.chunk.section,
                        "locator": rc.chunk.locator(),
                        "score": round(rc.score, 4),
                    }
                    for rc in result.chunks
                ],
                metadata={
                    "gate_score": round(result.top_score, 4),
                    "floor": self.settings.retrieval_min_score
                    if kwargs.get("min_score") is None else kwargs["min_score"],
                    "candidates_considered": result.candidates_considered,
                    "chunks_returned": len(result.chunks),
                    "insufficient": result.insufficient,
                    "hybrid": self.settings.retrieval_hybrid,
                    "reranked": self.settings.rerank_enabled,
                    "schemes": result.schemes,
                    **{k: round(v, 1) for k, v in result.latency_ms.items()},
                },
                # A refusal is a DECISION, so it is logged at WARNING and
                # shows up as such in the dashboard rather than looking like
                # a request that quietly returned nothing.
                level="WARNING" if result.insufficient else "DEFAULT",
                status_message=(
                    f"below relevance floor ({result.top_score:.4f})"
                    if result.insufficient else None
                ),
            )

            # Trended over real traffic, this is how RETRIEVAL_MIN_SCORE gets
            # calibrated instead of guessed -- an open question since the
            # floor was first set to 0.35 on intuition.
            score("retrieval_gate_score", round(result.top_score, 4))
            score("chunks_retrieved", len(result.chunks))
            return result

    def _retrieve(
        self,
        query: str,
        *,
        gate_query: str | None = None,
        sections: list[str] | None = None,
        priority_sections: list[str] | None = None,
        scheme_names: list[str] | None = None,
        doc_ids: list[str] | None = None,
        state: str | None = None,
        level: str | None = None,
        top_k: int | None = None,
        final_k: int | None = None,
        min_score: float | None = None,
        balance_sections: bool = True,
        extra_filter: qm.Filter | None = None,
    ) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.retrieval_top_k
        final_k = final_k or s.retrieval_final_k
        floor = s.retrieval_min_score if min_score is None else min_score
        timings: dict[str, float] = {}

        # The relevance gate must judge what the USER asked, not what an LLM
        # rewrote it into. A router that turns "capital of Peru" into a
        # scheme-shaped phrase would otherwise sail straight past the floor.
        # Both texts go in one batched forward pass, so this is ~free.
        needs_gate = gate_query is not None and gate_query.strip() != query.strip()

        t0 = time.perf_counter()
        if needs_gate:
            emb = self.embedder.encode_queries([query, gate_query])  # type: ignore[arg-type]
        else:
            emb = self.embedder.encode_query(query)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        gate_dense = emb.dense[1] if needs_gate else emb.dense[0]

        query_filter = extra_filter or build_filter(
            scheme_names=scheme_names,
            sections=sections,
            doc_ids=doc_ids,
            state=state,
            level=level,
        )

        t0 = time.perf_counter()
        candidates = self.store.search(
            dense_vec=emb.dense[0],
            sparse_vec=emb.sparse[0] if emb.sparse else None,
            limit=top_k,
            query_filter=query_filter,
            hybrid=s.retrieval_hybrid,
        )
        timings["search_ms"] = (time.perf_counter() - t0) * 1000

        if not candidates:
            return RetrievalResult(
                insufficient=True, candidates_considered=0, latency_ms=timings
            )

        if s.rerank_enabled:
            t0 = time.perf_counter()
            candidates = self._rerank(query, candidates)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        # --- relevance gate (hallucination guard #1) ---
        # Always a dense cosine probe, deliberately:
        #   - RRF fusion scores are rank-based (~1/61 for rank 1) and carry no
        #     absolute meaning, so they cannot be compared to a floor at all;
        #   - cross-encoder scores are sigmoid-scaled, a different scale again.
        # Gating on one consistent measure means RETRIEVAL_MIN_SCORE has a
        # single interpretation -- "best cosine similarity in the corpus to
        # what the user actually asked" -- whatever else is switched on.
        # Reranking therefore improves ranking, not gating.
        t0 = time.perf_counter()
        gate_score = self._cosine_probe(gate_dense, query_filter)
        timings["probe_ms"] = (time.perf_counter() - t0) * 1000

        if gate_score < floor:
            logger.info(
                "Below relevance floor (%.4f < %.4f) for %r", gate_score, floor, query[:80]
            )
            return RetrievalResult(
                insufficient=True,
                top_score=gate_score,
                candidates_considered=len(candidates),
                latency_ms=timings,
            )

        top_score = gate_score
        pool = (
            self._balance_sections(candidates, final_k, priority_sections)
            if balance_sections
            else candidates
        )
        selected = self._fit_budget(pool[:final_k])

        return RetrievalResult(
            chunks=selected,
            insufficient=False,
            top_score=top_score,
            candidates_considered=len(candidates),
            latency_ms=timings,
            scoped_schemes=scheme_names or [],
            scoped_sections=sections or [],
        )

    def _balance_sections(
        self,
        candidates: list[RetrievedChunk],
        final_k: int,
        priority_sections: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Stop one section monopolising the context window, and guarantee the
        authoritative section a place.

        Two problems, one pass.

        (1) FAQ DOMINANCE. 66% of this corpus is FAQ chunks: short and
        question-shaped, so a question-shaped query ranks them above
        everything else. Six FAQ snippets can fill the whole window while the
        Eligibility section -- which states the actual rules -- never gets in,
        producing answers assembled from incidental Q&A asides. Hence a cap of
        roughly half the window per section.

        (2) THE AUTHORITATIVE SECTION. For "who can apply", the Eligibility
        section is the right answer by construction, whatever its similarity
        score. Intent routing already knows this, so one chunk from each
        priority section is promoted to the front rather than left to chance.
        Promotion is conditional on the chunk already being a candidate -- we
        never inject something retrieval did not find.

        Both are soft: overflow goes to a reserve and is backfilled, so this
        never returns fewer chunks than a plain top-k would.
        """
        if final_k <= 2 or len(candidates) <= final_k:
            return candidates

        promoted: list[RetrievedChunk] = []
        used: set[int] = set()

        for section in priority_sections or []:
            for i, rc in enumerate(candidates):
                if i not in used and rc.chunk.section.value == section:
                    promoted.append(rc)
                    used.add(i)
                    break  # one per priority section is enough

        cap = max(2, (final_k + 1) // 2)
        counts: dict[str, int] = {}
        for rc in promoted:
            counts[rc.chunk.section.value] = counts.get(rc.chunk.section.value, 0) + 1

        kept: list[RetrievedChunk] = []
        reserve: list[RetrievedChunk] = []
        for i, rc in enumerate(candidates):
            if i in used:
                continue
            key = rc.chunk.section.value
            if counts.get(key, 0) < cap:
                kept.append(rc)
                counts[key] = counts.get(key, 0) + 1
            else:
                reserve.append(rc)

        return promoted + kept + reserve

    def _cosine_probe(self, dense_vec: list[float], query_filter: qm.Filter | None) -> float:
        """Best raw cosine similarity in the corpus for this query, under the
        same filter. Used only as the relevance gate, never for ranking."""
        probe = self.store.search(
            dense_vec=dense_vec, limit=1, query_filter=query_filter, hybrid=False
        )
        return probe[0].score if probe else 0.0

    def _fit_budget(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Hard-trim to the context budget.

        A quality and cost control, not a technical limit. It was sized for a
        4B that lost accuracy well before its window filled; with a hosted
        model the window is no longer the constraint, but stuffing every
        plausible chunk in still dilutes the answer and inflates the bill.
        """
        budget = self.settings.llm_context_budget_tokens
        out: list[RetrievedChunk] = []
        used = 0
        for rc in chunks:
            cost = rc.chunk.token_count + 25  # citation header overhead
            if out and used + cost > budget:
                break
            out.append(rc)
            used += cost
        return out

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        from .rerank import get_reranker

        reranker = get_reranker()
        if reranker is None:
            return candidates
        scores = reranker.score(query, [rc.chunk.embed_text for rc in candidates])
        for rc, score in zip(candidates, scores):
            rc.rerank_score = score
        return sorted(candidates, key=lambda rc: rc.rerank_score or 0.0, reverse=True)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
