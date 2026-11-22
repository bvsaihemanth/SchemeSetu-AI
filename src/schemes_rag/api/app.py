"""FastAPI application.

Query side only. It never writes to Qdrant and never runs ingest, so it can
be killed, restarted and replicated freely.

It also serves the UI at "/" -- one static HTML file, from this same process.
That keeps the whole query side to a single thing to start, with no second
port, no CORS to configure between the page and the API, and no build step.

Startup warms the embedding model deliberately: without this, the first user
request pays a ~20 s model load. Qdrant and the LLM are checked but not
required, so the API starts and reports unhealthy rather than crash-looping
when a dependency isn't up yet -- the page shows that state in its header.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..config import get_settings
from ..observability import COLLECTION_POINTS, LLM_ERRORS, init_observability
from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    init_observability()
    logger.info("Starting API (collection=%s, llm=%s)", s.qdrant_collection, s.llm_model)

    # Warm the embedder so the first request is not a cold model load.
    try:
        from ..embeddings import get_embedder

        get_embedder().encode_query("warmup")
        logger.info("Embedding model warm")
    except Exception as exc:
        logger.error("Embedder warmup failed: %s", exc)

    try:
        from ..vectorstore import get_vectorstore

        count = get_vectorstore().count()
        COLLECTION_POINTS.set(count)
        logger.info("Qdrant reachable: %s points", count)
        if count == 0:
            logger.warning("Collection is empty -- run scripts/ingest.py first")
    except Exception as exc:
        logger.error("Qdrant unreachable at startup: %s", exc)

    try:
        from ..llm import get_llm

        if get_llm().health():
            logger.info("LLM reachable at %s", s.llm_base_url)
        else:
            logger.warning("LLM not reachable at %s", s.llm_base_url)
    except Exception as exc:
        logger.warning("LLM check failed: %s", exc)

    logger.info("UI ready at http://localhost:%d/", s.api_port)
    yield
    # Traces are batched, so whatever is still buffered when the process ends
    # is lost unless it is flushed. That is exactly the tail of traces around
    # a crash or a restart -- the ones most worth having.
    from ..tracing import shutdown as tracing_shutdown

    tracing_shutdown()
    logger.info("Shutting down")


def create_app() -> FastAPI:
    s = get_settings()

    app = FastAPI(
        title="Indian Government Schemes RAG",
        description="Grounded question answering over myScheme documents.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled error [%s]: %s", rid, exc)
        LLM_ERRORS.labels(kind=type(exc).__name__).inc()
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error", "request_id": rid},
        )

    app.include_router(router)

    # The UI: one static file, served by this same process. A FileResponse
    # rather than a StaticFiles mount because there is exactly one file, and a
    # mount at "/" would shadow the API routes above.
    index = Path(__file__).parent / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    def ui():
        if not index.is_file():
            raise HTTPException(
                status_code=404,
                detail="UI not installed; expected api/static/index.html",
            )
        # no-cache so an edit to the page shows up on reload rather than
        # leaving you debugging a stale copy.
        return FileResponse(index, headers={"Cache-Control": "no-store"})

    if s.metrics_enabled:
        from prometheus_fastapi_instrumentator import Instrumentator

        # Exposes /metrics with HTTP latency/throughput, alongside the
        # RAG-specific metrics defined in observability.py.
        Instrumentator(
            should_group_status_codes=True,
            excluded_handlers=["/metrics", "/health"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
