"""Langfuse tracing.

THE ONLY MODULE THAT IMPORTS LANGFUSE. Everything else calls the helpers
here, which are no-ops when tracing is off. That keeps langfuse an optional
dependency -- `pip uninstall langfuse` leaves a working application -- and
keeps the vendor's API surface in one file, which matters because this SDK
has had breaking changes across v2/v3/v4/v5.

Sibling of observability.py, which owns Prometheus metrics and logging.
The split follows the question each answers: metrics tell you how the
SERVICE is doing in aggregate, traces tell you how one ANSWER was built.
Mixing them in one module conflated a counter with a payload.

WHY A SHIM RATHER THAN DECORATORS EVERYWHERE

Sprinkling @observe() through the agents would read more cleanly, but it
hard-couples every module to langfuse and makes tracing non-optional. This
project's brief was explicitly to avoid unnecessary dependencies, and
observability is the first thing you want to switch off when debugging
something else.

WHAT THIS CAPTURES, AND WHY EACH PIECE EARNS ITS PLACE

  generations   Token counts and cost per call, free, via the drop-in OpenAI
                client. Six agents each making LLM calls is otherwise an
                opaque bill.
  retriever     Query, gate score, and the chunks with their scores. In a RAG
                system a wrong answer is usually a retrieval failure, and
                without this you cannot tell the two apart.
  sessions      Keyed on thread_id, so the multi-turn eligibility interview
                reads as one conversation instead of unrelated traces.
  scores        The numbers worth trending: gate score (to calibrate
                RETRIEVAL_MIN_SCORE on real traffic), refusal rate, citation
                count, scheme-resolution confidence, eligibility outcome.
  guardrails    The score floor and numeric-grounding checks, as their own
                span type, so a refusal is visibly a decision rather than a
                missing answer.

DELIBERATELY NOT USED

  Prompt management. Moving prompts out of prompts.py into Langfuse adds a
  network dependency to every request and decouples prompts from the code
  review that changes them. The prompts here encode hard-won rules (see the
  refusal-rule history in prompts.py); they belong in version control.

SDK VERSION

Written against the Python SDK v5 API: get_client(), propagate_attributes(),
start_as_current_observation(as_type=...), create_score(). v5 requires
Python 3.11+ and a self-hosted server >= 4.59.0. Note that trace attributes
(session, tags, name) can ONLY be set via propagate_attributes in v5 --
update_current_trace() and span.update_trace() do not exist, whatever older
tutorials show.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from .config import get_settings

logger = logging.getLogger(__name__)

try:  # langfuse is optional
    from langfuse import Langfuse, get_client, propagate_attributes

    TRACING_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by uninstalling langfuse
    Langfuse = None  # type: ignore[assignment]
    get_client = None  # type: ignore[assignment]
    propagate_attributes = None  # type: ignore[assignment]
    TRACING_AVAILABLE = False


# --------------------------------------------------------------------------
# null objects
# --------------------------------------------------------------------------
class _NullSpan:
    """Stands in for a span when tracing is off.

    Call sites do `with observed(...) as span: span.update(...)`, so the
    disabled path needs an object with the same shape rather than None.
    Every method returns self so chains and nesting work unchanged.
    """

    def update(self, **_: Any) -> "_NullSpan":
        return self

    def score(self, **_: Any) -> None:
        return None

    def score_trace(self, **_: Any) -> None:
        return None

    def end(self, **_: Any) -> "_NullSpan":
        return self

    @contextmanager
    def start_as_current_observation(self, **_: Any) -> Iterator["_NullSpan"]:
        yield self


_NULL = _NullSpan()


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
def tracing_enabled() -> bool:
    """Whether to trace at all.

    Requires BOTH the flag and real credentials: a public key left at its
    placeholder means someone copied .env.example and never filled it in, and
    silently buffering traces that can never be delivered is worse than not
    tracing.
    """
    s = get_settings()
    if not s.langfuse_enabled or not TRACING_AVAILABLE:
        return False
    return bool(s.langfuse_public_key) and not s.langfuse_public_key.startswith(
        "pk-lf-your"
    )


def _mask(data: Any) -> Any:
    """Redact the user's personal details before they leave the process.

    Off by default, because on a single-developer self-hosted setup the
    profile is the most useful thing in the trace. Worth turning on the
    moment this serves real people: this assistant asks for age, income,
    caste category and disability status, which is exactly the data an
    observability backend should not be the weakest link for.
    """
    if not isinstance(data, (dict, list, str)):
        return data
    if isinstance(data, list):
        return [_mask(item) for item in data]
    if isinstance(data, str):
        return data
    return {
        key: ("[redacted]" if key in _SENSITIVE_KEYS else _mask(value))
        for key, value in data.items()
    }


_SENSITIVE_KEYS = frozenset({
    "age", "annual_income", "category", "gender", "landholding_ha",
    "special_statuses", "notes", "profile", "profile_summary",
})


@lru_cache
def get_langfuse():
    """The client, or None when tracing is off.

    Cached: the SDK starts a background exporter thread, so one per process.
    """
    if not tracing_enabled():
        return None

    s = get_settings()
    try:
        client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            base_url=s.langfuse_base_url,
            # Tags every trace with the environment, so local experiments do
            # not pollute the numbers you later trend in production.
            environment=s.langfuse_environment,
            sample_rate=s.langfuse_sample_rate,
            mask=_mask if s.langfuse_mask_pii else None,
        )
    except Exception as exc:
        # Never let observability break the request path. A broken exporter
        # is an inconvenience; a 500 because of it is a bug.
        logger.warning("Langfuse init failed (%s); continuing untraced", exc)
        return None

    logger.info(
        "Langfuse tracing -> %s (environment=%s, sample_rate=%s, mask_pii=%s)",
        s.langfuse_base_url, s.langfuse_environment, s.langfuse_sample_rate,
        s.langfuse_mask_pii,
    )
    return client


#: Returned by auth_check so callers can give advice that fits the failure.
#: "unreachable" and "rejected" need completely different fixes, and
#: conflating them sends you to check your API keys when the real problem is
#: that nothing is listening on the port.
AUTH_OK = "ok"
AUTH_NOT_INSTALLED = "not_installed"
AUTH_DISABLED = "disabled"
AUTH_NO_KEYS = "no_keys"
AUTH_UNREACHABLE = "unreachable"
AUTH_REJECTED = "rejected"
AUTH_SERVER_ERROR = "server_error"
AUTH_ERROR = "error"

#: Substrings that mark a transport failure, checked ONLY when the exception
#: carries no HTTP status. An HTTP status means the server answered, which
#: settles the question without guessing from prose.
#:
#: Matched on text because the SDK wraps httpx errors in its own types, so
#: there is no single exception class to catch. Note how narrow these are:
#: an earlier version matched the bare substring "connect", which classified
#: a plain HTTP 500 as "unreachable" because the dumped response headers
#: contained `'connection': 'keep-alive'`. Substring matching on an error
#: dump will find whatever it is given, so these now carry enough context to
#: mean what they say.
_UNREACHABLE_MARKERS = (
    "connecterror", "connecttimeout", "connection refused",
    "connection reset", "connection aborted", "actively refused",
    "10061", "10060", "timed out", "readtimeout", "pooltimeout",
    "unreachable", "name or service not known", "getaddrinfo",
    "max retries exceeded", "ssl", "certificate",
)


def _http_status(exc: Exception) -> int | None:
    """The HTTP status an exception carries, if any.

    Tried before any text matching: a status code is proof the server replied,
    and no amount of pattern matching on the message improves on that.
    """
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    # Some SDK errors keep it on a nested response object instead.
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def auth_check() -> tuple[bool, str, str]:
    """(ok, reason_code, detail). For preflight, not the hot path.

    The reason code exists so the caller can tell "the server is not there"
    from "the server said no". Both surface as a failed auth_check, but one is
    fixed by starting a container and the other by fixing two keys.
    """
    if not TRACING_AVAILABLE:
        return False, AUTH_NOT_INSTALLED, "langfuse not installed"
    if not get_settings().langfuse_enabled:
        return False, AUTH_DISABLED, "LANGFUSE_ENABLED=false"
    client = get_langfuse()
    if client is None:
        return False, AUTH_NO_KEYS, "keys missing or still the placeholder in .env"
    try:
        if client.auth_check():
            return True, AUTH_OK, "credentials accepted"
        return False, AUTH_REJECTED, "server reachable, but it rejected the keys"
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"

        # Status first. If the server replied, it is reachable -- whatever the
        # reply says -- and the three outcomes need different advice:
        #   401/403  your keys are wrong
        #   5xx      the server is broken or still starting
        #   other    something unexpected, report it verbatim
        status = _http_status(exc)
        if status is not None:
            if status in (401, 403):
                return False, AUTH_REJECTED, f"HTTP {status}: keys rejected"
            if status >= 500:
                return False, AUTH_SERVER_ERROR, f"HTTP {status} from the server"
            return False, AUTH_ERROR, f"HTTP {status}: {text[:200]}"

        lowered = text.lower()
        if any(marker in lowered for marker in _UNREACHABLE_MARKERS):
            return False, AUTH_UNREACHABLE, text
        return False, AUTH_ERROR, text


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------
@contextmanager
def observed(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """A span, or a no-op when tracing is off.

    `as_type` carries real meaning to Langfuse -- "retriever", "agent",
    "guardrail", "evaluator" and "embedding" each render differently and are
    filterable -- so it is worth naming the kind of work rather than making
    everything a generic span.
    """
    client = get_langfuse()
    if client is None:
        yield _NULL
        return

    try:
        with client.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        ) as span:
            yield span
    except Exception as exc:
        # A tracing failure must not fail the work being traced.
        logger.debug("Span %r failed (%s); continuing untraced", name, exc)
        yield _NULL


@contextmanager
def trace_context(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    trace_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Set trace-level attributes for everything inside.

    session_id is the valuable one here: passing thread_id groups the turns of
    an eligibility interview into a single session in the dashboard, which is
    the only way to read "asked five questions, got five answers, reached a
    verdict" as one story.

    In SDK v5 this is the ONLY way to set these -- there is no
    update_current_trace() and no span.update_trace(), despite what older
    tutorials and GitHub threads show.
    """
    if get_langfuse() is None:
        yield
        return
    try:
        with propagate_attributes(
            session_id=session_id, user_id=user_id, tags=tags,
            trace_name=trace_name, metadata=metadata,
        ):
            yield
    except Exception as exc:
        logger.debug("trace_context failed (%s); continuing untraced", exc)
        yield


def current_trace_id() -> str | None:
    """The active trace id, for returning to a client that may score it."""
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------
def score(
    name: str,
    value: float | str | bool,
    *,
    data_type: str | None = None,
    comment: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Attach a score to the current trace, or to `trace_id` if given.

    Scores are what turn a pile of traces into something you can act on:
    they are chartable and filterable in the dashboard, so "refusal rate this
    week" or "gate score distribution" becomes a question you can answer.
    `trace_id` is for scoring after the fact, as user feedback does.
    """
    client = get_langfuse()
    if client is None:
        return

    # Langfuse has no boolean value type -- BOOLEAN scores are numeric 0/1 --
    # so convert here rather than at every call site.
    if isinstance(value, bool):
        value, data_type = float(value), data_type or "BOOLEAN"
    elif isinstance(value, str):
        data_type = data_type or "CATEGORICAL"
    else:
        data_type = data_type or "NUMERIC"

    try:
        if trace_id:
            client.create_score(
                name=name, value=value, trace_id=trace_id,
                data_type=data_type, comment=comment,
            )
        else:
            client.score_current_trace(
                name=name, value=value, data_type=data_type, comment=comment,
            )
    except Exception as exc:
        logger.debug("Score %r failed: %s", name, exc)


def flush() -> None:
    """Force delivery. Needed on shutdown and in short-lived scripts."""
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush failed: %s", exc)


def shutdown() -> None:
    client = get_langfuse()
    if client is not None:
        try:
            client.shutdown()
        except Exception as exc:
            logger.debug("Langfuse shutdown failed: %s", exc)


# --------------------------------------------------------------------------
# OpenAI drop-in
# --------------------------------------------------------------------------
def openai_classes() -> tuple[Any, Any]:
    """(OpenAI, AsyncOpenAI) -- instrumented when tracing is on.

    The drop-in is the highest-value hookup in this file for the least code:
    it records model, parameters, prompt, completion, token counts, cost and
    latency for every call, with no changes at the call sites. It works here
    only because llm.py was written against the OpenAI protocol rather than a
    vendor SDK -- the same decision that made the move to Gemini three lines
    of config.

    Falls back to the plain classes on any import problem, because a version
    skew between langfuse and openai must not take the application down.
    """
    import openai as _plain

    if get_langfuse() is None:
        return _plain.OpenAI, _plain.AsyncOpenAI

    try:
        from langfuse.openai import AsyncOpenAI, OpenAI

        return OpenAI, AsyncOpenAI
    except Exception as exc:
        logger.warning(
            "langfuse.openai unavailable (%s); LLM calls will not be traced. "
            "Check that langfuse and openai versions are compatible.", exc,
        )
        return _plain.OpenAI, _plain.AsyncOpenAI


def generation_metadata(
    *, name: str | None = None, session_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Extra kwargs for an instrumented OpenAI call.

    The drop-in reads `name` and the `langfuse_*` metadata keys and strips
    them before the request goes out. The PLAIN client would reject them, so
    this returns {} when tracing is off and the caller can splat it
    unconditionally.
    """
    if get_langfuse() is None:
        return {}
    extra: dict[str, Any] = {}
    if name:
        extra["name"] = name
    meta: dict[str, Any] = {}
    if session_id:
        meta["langfuse_session_id"] = session_id
    if tags:
        meta["langfuse_tags"] = tags
    if meta:
        extra["metadata"] = meta
    return extra
