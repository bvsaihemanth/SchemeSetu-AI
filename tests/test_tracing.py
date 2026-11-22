"""Langfuse tracing, against the real SDK.

Mocking the SDK would only prove the mock matches my assumptions. Instead the
real client is built with two test seams it already supports:

    span_exporter=InMemorySpanExporter()   -> spans, in process
    httpx_client=MockTransport(...)        -> score POSTs, in process

so these assertions run against the code paths that will run in production,
with nothing leaving the machine. This matters more than usual here: the SDK
has had breaking API changes across v2/v3/v4/v5, and the v5 surface used by
tracing.py (propagate_attributes for trace attributes, no update_current_trace)
is not what most tutorials show.

Four properties:
  1. OFF   zero spans, and generation_metadata() stays empty -- the PLAIN
           OpenAI client rejects `name`/`metadata`, so leaking them would
           break every LLM call rather than just the tracing.
  2. ON    the span tree matches the pipeline, with the right semantic types,
           and the scores the dashboard trends are actually sent.
  3. SAFE  a broken exporter must not fail the query it was watching.
  4. PII   masking redacts the profile when asked to.

    python -m tests.test_tracing
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["RETRIEVAL_MIN_SCORE"] = "0.30"
os.environ["VERIFY_ENABLED"] = "false"
os.environ["LLM_API_KEY"] = "test-key-looks-real"

from tests.test_agents import CORPUS, ScriptedLLM, build_env  # noqa: E402


def make_client(*, exporter=None, mask=False, breaking=False):
    """A real Langfuse client that talks to nothing."""
    import httpx
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    posts: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append((request.url.path, request.content))
        return httpx.Response(200, json={} if "scores" in request.url.path
                              else {"data": [], "meta": {}})

    class Breaking(InMemorySpanExporter):
        def export(self, spans):  # noqa: D401
            raise RuntimeError("exporter is down")

    exporter = exporter or (Breaking() if breaking else InMemorySpanExporter())

    from schemes_rag.config import get_settings

    s = get_settings()
    s.langfuse_enabled = True
    s.langfuse_public_key = "pk-lf-test"
    s.langfuse_secret_key = "sk-lf-test"
    s.langfuse_mask_pii = mask

    from schemes_rag import tracing

    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://localhost:3000",
        span_exporter=exporter,
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        mask=tracing._mask if mask else None,
    )
    # Every helper in tracing.py routes through get_langfuse(), so replacing
    # it instruments the whole shim at once. hasattr-guarded because a second
    # call lands on the lambda left by the first, which has no cache.
    if hasattr(tracing.get_langfuse, "cache_clear"):
        tracing.get_langfuse.cache_clear()
    tracing.get_langfuse = lambda: client  # type: ignore[assignment]
    return client, exporter, posts


def score_names(posts) -> dict[str, object]:
    """Score name -> value, from the captured POST bodies."""
    out: dict[str, object] = {}
    for path, body in posts:
        if "scores" not in path:
            continue
        for item in json.loads(body):
            out[item["name"]] = item["value"]
    return out


def restore_off():
    """Put the module back, so a later suite in the same process is untraced."""
    import importlib

    from schemes_rag import tracing
    from schemes_rag.config import get_settings

    get_settings().langfuse_enabled = False
    get_settings().langfuse_mask_pii = False
    importlib.reload(tracing)


def main() -> int:  # noqa: C901
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_tracing"])

    if not os.path.isdir(CORPUS):
        print(f"SKIP: corpus not available at {CORPUS}")
        return 0

    import importlib

    from schemes_rag import tracing

    failures: list[str] = []
    question = "What documents are required for PM-KISAN?"

    # ---------------- 1. OFF ----------------
    print("\n--- 1. Tracing off: no spans, and no stray kwargs ---")
    importlib.reload(tracing)
    from schemes_rag.config import get_settings

    get_settings().langfuse_enabled = False
    tracing.get_langfuse.cache_clear()

    assert not tracing.tracing_enabled(), "disabled flag ignored"
    # The critical one: these kwargs reach chat.completions.create(), and the
    # plain OpenAI client raises on unknown keys. An empty dict here is what
    # keeps langfuse optional rather than load-bearing.
    assert tracing.generation_metadata(name="x", session_id="s") == {}, \
        "metadata leaked to the plain client"
    with tracing.observed("nothing") as span:
        span.update(output="x").score(name="n", value=1)   # must not raise
    with tracing.trace_context(session_id="s"):
        tracing.score("n", 1.0)
    assert tracing.current_trace_id() is None
    print("  OK  helpers are safe no-ops; generation_metadata() empty")

    llm = ScriptedLLM()
    build_env(llm)
    import schemes_rag.graph as G

    ans = G.run_query(question, thread_id="off-1")
    assert ans.trace_id is None, ans.trace_id
    assert ans.citations, "pipeline broken with tracing off"
    print(f"  OK  query works untraced: route={ans.route}, "
          f"{len(ans.citations)} citations, trace_id=None")

    # ---------------- 2. ON ----------------
    print("\n--- 2. Tracing on: span tree, session, scores ---")
    client, exporter, posts = make_client()
    llm = ScriptedLLM()
    build_env(llm)
    importlib.reload(G)
    # build_env/reload rebuilt the module; re-point the seam it imported.
    G.obs.get_langfuse = lambda: client  # type: ignore[assignment]

    ans = G.run_query(question, thread_id="traced-session-1")
    client.flush()

    spans = exporter.get_finished_spans()
    found = {s.name: s.attributes.get("langfuse.observation.type") for s in spans}
    print(f"  spans: {found}")

    for name, want_type in (
        ("query", "chain"),
        ("understand", "agent"),
        ("documents", "agent"),
        ("retrieve", "retriever"),
        ("evidence", "evaluator"),
    ):
        if name not in found:
            failures.append(f"span {name!r} missing (got {sorted(found)})")
        elif found[name] != want_type:
            failures.append(
                f"span {name!r} typed {found[name]!r}, expected {want_type!r}"
            )
    if not failures:
        print("  OK  pipeline span tree present with correct semantic types")

    # The session is what groups the interview's turns into one conversation.
    # Asserted by VALUE rather than attribute key, so an SDK key rename does
    # not silently turn this into a test of nothing.
    session_seen = any(
        v == "traced-session-1"
        for s in spans for v in s.attributes.values() if isinstance(v, str)
    )
    if not session_seen:
        failures.append("session_id never reached any span attribute")
    else:
        print("  OK  session_id=thread_id propagated to the trace")

    if not ans.trace_id:
        failures.append("Answer.trace_id not set, so feedback cannot be attached")
    else:
        print(f"  OK  trace_id returned to the client: {ans.trace_id[:16]}...")

    got = score_names(posts)
    print(f"  scores: {got}")
    # These are the ones the dashboard trends; the retrieval gate score is the
    # one that finally makes RETRIEVAL_MIN_SCORE calibratable from real data.
    for name in ("route", "outcome", "insufficient_evidence", "citations",
                 "latency_ms", "retrieval_gate_score", "chunks_retrieved"):
        if name not in got:
            failures.append(f"score {name!r} never sent (got {sorted(got)})")
    if got.get("route") != "documents":
        failures.append(f"route score was {got.get('route')!r}")
    if not failures:
        print("  OK  all trended scores sent to /api/public/scores")

    # A retriever span must carry the chunks AND their scores: without the
    # scores, "retrieved the wrong thing" and "retrieved nothing good enough"
    # look identical.
    retriever = next((s for s in spans if s.name == "retrieve"), None)
    if retriever is not None:
        output = str(retriever.attributes.get("langfuse.observation.output", ""))
        meta = str(retriever.attributes.get("langfuse.observation.metadata.gate_score", ""))
        if "score" not in output:
            failures.append("retriever span output has no per-chunk scores")
        elif not meta:
            failures.append("retriever span has no gate_score metadata")
        else:
            print(f"  OK  retriever span carries chunk scores and gate_score={meta}")

    # ---------------- 3. SAFE ----------------
    print("\n--- 3. A broken exporter must not break the query ---")
    client, _, _ = make_client(breaking=True)
    llm = ScriptedLLM()
    build_env(llm)
    importlib.reload(G)
    G.obs.get_langfuse = lambda: client  # type: ignore[assignment]
    try:
        ans = G.run_query(question, thread_id="broken-1")
        client.flush()
        if not ans.citations:
            failures.append("broken exporter degraded the answer")
        else:
            print(f"  OK  answer unaffected: route={ans.route}, "
                  f"{len(ans.citations)} citations")
    except Exception as exc:
        failures.append(f"broken exporter raised into the request path: {exc!r}")

    # ---------------- 4. PII ----------------
    print("\n--- 4. PII masking ---")
    from schemes_rag import tracing as t2

    sensitive = {
        "profile": {"age": 41, "annual_income": 150000.0, "category": "obc"},
        "question": "am i eligible",
        "notes": ["Q: Are you a farmer? A: yes"],
    }
    masked = t2._mask(sensitive)
    print(f"  masked: {masked}")
    if masked["profile"] != "[redacted]":
        failures.append(f"profile not redacted: {masked['profile']}")
    if masked["notes"] != "[redacted]":
        failures.append("notes not redacted -- they hold the interview answers")
    if masked["question"] != "am i eligible":
        failures.append("masking destroyed the question, which is not PII")
    if not any("profile not redacted" in f or "notes not" in f for f in failures):
        print("  OK  profile and interview answers redacted, question kept")

    # ---------------- 5. failure classification ----------------
    #
    # THE REGRESSION THIS GUARDS. A first run hit a truncated image pull, so
    # the web container never started and the SDK raised
    #   ConnectError: [WinError 10061] ... actively refused it
    # which the preflight reported as "Credentials rejected" and then listed
    # three key-mismatch causes. The keys were fine; nothing was listening.
    # Advice that confident and that wrong costs more time than no advice.
    print("\n--- 5. 'unreachable' is not 'rejected' ---")
    from schemes_rag import tracing as t3

    # Driven through the real auth_check() with a client that raises the
    # exception shape in question. Re-implementing the matching in the test
    # would only assert that I can copy a loop -- and the bug below was in
    # the DECISION, not the loop.
    class Boom(Exception):
        def __init__(self, text, status=None):
            super().__init__(text)
            if status is not None:
                self.status_code = status

    def classify(text, status=None):
        class Raiser:
            def auth_check(self_inner):
                raise Boom(text, status)
        t3.get_langfuse = lambda: Raiser()  # type: ignore[assignment]
        return t3.auth_check()[1]

    gs0 = __import__("schemes_rag.config", fromlist=["get_settings"]).get_settings
    gs0().langfuse_enabled = True
    gs0().langfuse_public_key = "pk-lf-test"

    cases = [
        # No HTTP status -> transport. These must be unreachable.
        ("ConnectError: [WinError 10061] No connection could be made because "
         "the target machine actively refused it", None, t3.AUTH_UNREACHABLE),
        ("ConnectTimeout: timed out", None, t3.AUTH_UNREACHABLE),
        ("ConnectError: [Errno 111] Connection refused", None, t3.AUTH_UNREACHABLE),
        ("ConnectError: getaddrinfo failed", None, t3.AUTH_UNREACHABLE),
        ("SSLError: certificate verify failed", None, t3.AUTH_UNREACHABLE),

        # THE REGRESSION. A real HTTP 500 from a running server. The first
        # version matched the bare substring "connect" and this error dump
        # contains the `connection: keep-alive` response header, so a healthy
        # -but-starting server was reported as "nothing is listening" -- the
        # opposite of the truth, with advice to match.
        ("ApiError: headers: {'x-robots-tag': 'noindex', 'connection': "
         "'keep-alive', 'keep-alive': 'timeout=5'}, status_code: 500, "
         "body: Internal Server Error", 500, t3.AUTH_SERVER_ERROR),
        # Same trap without a status attribute to lean on: the markers alone
        # must not call this unreachable.
        ("ApiError: 'connection': 'keep-alive', 'keep-alive': 'timeout=5'",
         None, t3.AUTH_ERROR),

        # A server that answered. Status decides, not prose.
        ("ApiError: unauthorized", 401, t3.AUTH_REJECTED),
        ("ApiError: forbidden", 403, t3.AUTH_REJECTED),
        ("ApiError: bad gateway", 502, t3.AUTH_SERVER_ERROR),
        ("ApiError: not found", 404, t3.AUTH_ERROR),
        ("ValueError: malformed payload", None, t3.AUTH_ERROR),
    ]
    for text, status, want in cases:
        got = classify(text, status)
        marker = "  " if got == want else "<-"
        print(f"  {marker} {got:13} status={str(status):4} {text[:46]}")
        if got != want:
            failures.append(f"{text[:44]!r} (status={status}) classified "
                            f"{got!r}, expected {want!r}")

    # And the real thing, end to end: a client pointed at a dead port must
    # come back as unreachable, not rejected.
    import httpx
    from langfuse import Langfuse

    from schemes_rag.config import get_settings as gs

    gs().langfuse_enabled = True
    gs().langfuse_public_key = "pk-lf-dead"

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[WinError 10061] actively refused it")

    # A DIFFERENT public key from the clients above, deliberately: the v5 SDK
    # keys its client instances on the public key, so reusing "pk-lf-test"
    # hands back the earlier mock -- which answers 200 -- and this would test
    # nothing while appearing to pass.
    dead = Langfuse(
        public_key="pk-lf-dead", secret_key="sk-lf-dead",
        base_url="http://localhost:3000",
        httpx_client=httpx.Client(transport=httpx.MockTransport(refuse)),
    )
    t3.get_langfuse = lambda: dead  # type: ignore[assignment]
    ok, reason, detail = t3.auth_check()
    print(f"  live check -> ok={ok} reason={reason}")
    if ok or reason != t3.AUTH_UNREACHABLE:
        failures.append(f"dead port classified {reason!r}, expected unreachable")
    else:
        print("  OK  a dead port reports 'unreachable', so the advice fits")

    restore_off()

    print("\n" + "=" * 76)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("TRACING VERIFIED: spans, types, session, scores, isolation, masking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
