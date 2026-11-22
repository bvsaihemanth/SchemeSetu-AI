"""Endpoints.

  POST /query     full RAG: understand -> retrieve -> generate -> verify
  POST /search    retrieval only (no LLM) -- the debugging endpoint
  POST /feedback  rate an answer; becomes a Langfuse score
  GET  /health  dependency status
  GET  /stats   collection stats
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..graph import run_query
from ..observability import COLLECTION_POINTS
from ..tracing import score, tracing_enabled
from ..retriever import get_retriever
from .schemas import (
    CitationOut,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ResolvedScheme,
    SearchHit,
    SearchResponse,
    SuggestedScheme,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest) -> QueryResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    filters = req.filters.model_dump(exclude_none=True) if req.filters else {}

    try:
        answer = run_query(
            req.question,
            history=req.history,
            filters=filters,
            thread_id=thread_id,
            answers=[a.model_dump() for a in req.answers],
        )
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Pipeline error: {type(exc).__name__}")

    return QueryResponse(
        answer=answer.answer,
        citations=[
            CitationOut(
                marker=c.marker,
                scheme_name=c.scheme_name,
                section=c.section,
                locator=c.locator,
                source_url=c.source_url,
                score=c.score,
            )
            for c in answer.citations
        ],
        insufficient_evidence=answer.insufficient_evidence,
        route=answer.route,
        schemes_used=answer.schemes_used,
        thread_id=thread_id,
        trace_id=answer.trace_id,
        latency_ms=answer.latency_ms,
        awaiting_answer=answer.awaiting_answer,
        profile_summary=answer.profile_summary,
        suggested_schemes=[
            SuggestedScheme(
                scheme_name=s["scheme_name"],
                scheme_acronym=s.get("scheme_acronym"),
                source_url=s.get("source_url"),
                score=s.get("score", 0.0),
            )
            for s in answer.suggested_schemes
        ],
        assessment=answer.assessment.model_dump(mode="json") if answer.assessment else None,
        verification=answer.verification,
    )


@router.post("/search", response_model=SearchResponse, tags=["rag"])
def search(req: QueryRequest) -> SearchResponse:
    """Retrieval without generation.

    Use this to separate retrieval failures from generation failures. If
    /search returns the right chunks but /query answers badly, the problem is
    the prompt or the model -- not the index.
    """
    from ..scheme_index import get_scheme_index

    filters = req.filters.model_dump(exclude_none=True) if req.filters else {}

    # Mirror what the graph does, so /search reflects real retrieval.
    index = get_scheme_index()
    matches = index.resolve(req.question) if index else []
    doc_ids = None
    if matches and matches[0].confidence in ("exact", "strong"):
        tier = matches[0].confidence
        doc_ids = [m.doc_id for m in matches if m.confidence == tier][:4]

    result = get_retriever().retrieve(
        req.question,
        sections=filters.get("sections"),
        doc_ids=doc_ids,
        scheme_names=filters.get("scheme_names"),
        state=filters.get("state"),
        level=filters.get("level"),
        final_k=filters.get("final_k"),
    )
    return SearchResponse(
        query=req.question,
        insufficient=result.insufficient,
        top_score=round(result.top_score, 4),
        resolved_schemes=[
            ResolvedScheme(
                scheme_name=m.scheme_name,
                confidence=m.confidence,
                score=m.score,
                matched_on=m.matched_on,
            )
            for m in matches[:4]
        ],
        scoped_to_schemes=bool(doc_ids),
        hits=[
            SearchHit(
                score=round(rc.effective_score, 4),
                scheme_name=rc.chunk.scheme_name,
                section=rc.chunk.section.value,
                locator=rc.chunk.locator(),
                text=rc.chunk.text,
                source_url=rc.chunk.source_url,
            )
            for rc in result.chunks
        ],
        latency_ms={k: round(v, 1) for k, v in result.latency_ms.items()},
    )


@router.post("/feedback", tags=["ops"])
def feedback(req: FeedbackRequest) -> dict:
    """Rate an answer. Becomes a Langfuse score on that answer's trace.

    Scoring by trace_id rather than thread_id, deliberately: a thread holds
    every turn of a conversation, and "that answer was wrong" is about one of
    them. Attaching it to the thread would smear the signal across turns that
    were fine.

    Returns 503 rather than silently succeeding when tracing is off. A UI that
    thinks it recorded feedback it did not record is worse than one that knows
    it cannot -- and the UI uses this to hide the buttons.
    """
    if not tracing_enabled():
        raise HTTPException(
            status_code=503,
            detail="Tracing is off, so there is nowhere to record feedback. "
                   "Set LANGFUSE_ENABLED=true and the Langfuse keys in .env.",
        )

    score(
        "user_feedback",
        req.helpful,
        comment=req.comment,
        trace_id=req.trace_id,
    )
    logger.info(
        "Feedback on %s: %s%s", req.trace_id,
        "helpful" if req.helpful else "not helpful",
        f" ({req.comment[:80]})" if req.comment else "",
    )
    return {"recorded": True, "trace_id": req.trace_id}


@router.get("/profile/{thread_id}", tags=["rag"])
def profile(thread_id: str) -> dict:
    """What the Eligibility agent knows about this conversation so far."""
    from ..graph import get_profile, pending_question

    p = get_profile(thread_id)
    return {
        "thread_id": thread_id,
        "summary": p.describe(),
        "known_slots": p.known_slots(),
        "profile": p.model_dump(mode="json", exclude_none=True),
        "mid_interview_on": pending_question(thread_id),
    }


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    s = get_settings()
    from ..llm import get_llm
    from ..vectorstore import get_vectorstore

    store = get_vectorstore()
    qdrant_ok = store.health()
    points = None
    detail = None

    if qdrant_ok:
        try:
            points = store.count()
            COLLECTION_POINTS.set(points)
            if points == 0:
                detail = "Collection is empty; run scripts/ingest.py"
        except Exception as exc:
            qdrant_ok = False
            detail = f"Collection error: {exc}"

    llm_ok = get_llm().health()
    if not llm_ok:
        detail = (detail + "; " if detail else "") + f"LLM unreachable at {s.llm_base_url}"

    return HealthResponse(
        status="ok" if (qdrant_ok and llm_ok and points) else "degraded",
        qdrant=qdrant_ok,
        llm=llm_ok,
        collection=s.qdrant_collection,
        points=points,
        tracing=tracing_enabled(),
        detail=detail,
    )


@router.get("/stats", tags=["ops"])
def stats() -> dict:
    from ..vectorstore import get_vectorstore

    s = get_settings()
    store = get_vectorstore()
    try:
        info = store.client.get_collection(s.qdrant_collection)
        return {
            "collection": s.qdrant_collection,
            "points": store.count(),
            "status": str(info.status),
            "optimizer_status": str(info.optimizer_status),
            "segments": info.segments_count,
            "embed_model": s.embed_model,
            "llm_model": s.llm_model,
            "hybrid": s.retrieval_hybrid,
            "rerank": s.rerank_enabled,
            "tracing": tracing_enabled(),
            "tracing_target": s.langfuse_base_url if tracing_enabled() else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
