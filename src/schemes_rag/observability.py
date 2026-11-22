"""Tracing, metrics, logging. Imported once at API startup.

Division of labour, so the three tools don't overlap:
  LangSmith  -> per-request LLM traces: prompts, retrieved chunks, token
                counts. For debugging *why* one answer was wrong.
  Prometheus -> aggregate counters and latency histograms. For knowing
                *how often* and *how slow*, across all requests.
  Grafana    -> dashboards over Prometheus.

LangSmith needs no code: the LangChain/LangGraph callbacks pick up the env
vars. Setting them here keeps config in one place.
"""

from __future__ import annotations

import logging
import os
import sys

from prometheus_client import Counter, Gauge, Histogram

from .config import get_settings

logger = logging.getLogger(__name__)

# --- Prometheus metrics ---------------------------------------------------
# Latency buckets are chosen for a local 4B model: a hit is hundreds of ms,
# generation is seconds. Default buckets would lump everything into +Inf.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)

QUERIES_TOTAL = Counter(
    "rag_queries_total", "Queries received", ["route", "outcome"]
)
STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds",
    "Per-stage latency",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)
RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Chunks passed to the LLM",
    buckets=(0, 1, 2, 3, 4, 6, 8, 12, 16, 24),
)
TOP_SCORE = Histogram(
    "rag_top_score",
    "Relevance gate score per query",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
INSUFFICIENT_TOTAL = Counter(
    "rag_insufficient_evidence_total",
    "Queries answered 'not in my sources'",
    ["reason"],
)
CITATIONS_PER_ANSWER = Histogram(
    "rag_citations_per_answer", "Citations emitted", buckets=(0, 1, 2, 3, 4, 5, 6, 8)
)
LLM_ERRORS = Counter("rag_llm_errors_total", "LLM call failures", ["kind"])
COLLECTION_POINTS = Gauge("rag_collection_points", "Points in the Qdrant collection")


def record_query(
    route: str,
    outcome: str,
    latency_ms: dict[str, float] | None = None,
    chunk_count: int | None = None,
    top_score: float | None = None,
    citation_count: int | None = None,
) -> None:
    """Single funnel for query metrics, so nothing is recorded in two places."""
    QUERIES_TOTAL.labels(route=route or "unknown", outcome=outcome).inc()
    for stage, ms in (latency_ms or {}).items():
        STAGE_LATENCY.labels(stage=stage.removesuffix("_ms")).observe(ms / 1000.0)
    if chunk_count is not None:
        RETRIEVED_CHUNKS.observe(chunk_count)
    if top_score is not None:
        TOP_SCORE.observe(top_score)
    if citation_count is not None:
        CITATIONS_PER_ANSWER.observe(citation_count)


def setup_logging() -> None:
    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def setup_tracing() -> None:
    """LangSmith via env vars. No-op when disabled, so the app runs offline."""
    s = get_settings()
    if not s.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return
    if not s.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING on but no API key; tracing disabled")
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    logger.info("LangSmith tracing -> project %s", s.langsmith_project)


def setup_langfuse() -> None:
    """Build the Langfuse client at startup rather than on first request.

    The SDK starts a background exporter thread and does an auth handshake;
    doing that lazily puts it on a user's first query. Eager construction also
    means a bad key shows up in the startup log instead of as a silent absence
    of traces an hour later.

    Imported here, not at module scope, so this module keeps working with
    langfuse uninstalled.
    """
    from . import tracing

    if not get_settings().langfuse_enabled:
        return
    ok, reason, detail = tracing.auth_check()
    if ok:
        logger.info("Langfuse tracing ready (%s)", detail)
    elif reason == tracing.AUTH_UNREACHABLE:
        # Named precisely, because the obvious reading of a failed auth check
        # is "my keys are wrong" and this is not that.
        logger.warning(
            "Langfuse is not reachable at %s -- the stack is probably not "
            "running (.\\ops\\langfuse\\up.ps1). Continuing untraced. [%s]",
            get_settings().langfuse_base_url, detail,
        )
    else:
        # A warning, not a failure: observability must never be the reason
        # the application will not start.
        logger.warning("Langfuse tracing unavailable (%s): %s", reason, detail)


def init_observability() -> None:
    setup_logging()
    setup_tracing()
    setup_langfuse()
