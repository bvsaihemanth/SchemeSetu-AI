"""Preflight: verify every dependency is reachable before blaming retrieval.

    python scripts\\check_llm.py

    python scripts\\check_llm.py --list-models   # what your API key can call

Checks, in dependency order, and tells you the exact fix for whatever fails:
  1. Qdrant reachable, collection populated, index built
  2. scheme_index.json present  (without it, acronym queries cannot resolve)
  3. chunk payloads carry scheme_acronym  (stale vectors if not)
  4. API key set, LLM reachable, model name actually served
  5. plain completion works
  6. no <think> / reasoning leakage
  7. JSON mode (router + verifier depend on parseable output)
  8. a real grounded answer, end to end
  9. GPU placement -- local models only; skipped for a hosted endpoint

The advice adapts to where the model runs: VRAM and Ollama diagnostics for a
localhost base URL, key/model/rate-limit diagnostics for a hosted one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
_results: list[tuple[str, str, str]] = []

# Qwen3-4B: 36 layers, 8 KV heads, head_dim 128 -> 2*36*8*128*2 bytes per
# token in fp16. Used only to explain a VRAM shortfall in concrete terms.
_KV_BYTES_PER_TOKEN_FP16 = 2 * 36 * 8 * 128 * 2


def gpu_memory() -> tuple[int, int] | None:
    """(total_mib, used_mib) from nvidia-smi, or None if unavailable."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        total, used = (int(x.strip()) for x in out.split(","))
        return total, used
    except Exception:
        return None


def ollama_loaded_models(base_url: str) -> list[dict] | None:
    """Ollama's /api/ps: the definitive GPU/CPU split for loaded models.

    `size_vram` < `size` means part of the model is on the CPU. This is the
    same information `ollama ps` prints, but readable right after a request
    instead of needing to beat the idle-unload timer.
    """
    import httpx

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        resp = httpx.get(f"{root}/api/ps", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models") or []
    except Exception:
        return None


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"\n         {detail}" if detail else ""))


def sample_schemes(index, n: int = 40) -> list:
    """A deterministic sample of the index, so runs are comparable."""
    entries = sorted(index.entries.values(), key=lambda e: e.doc_id)
    rng = random.Random(1234)
    return rng.sample(entries, min(n, len(entries)))


def check_resolver(index, sample_size: int = 40) -> None:
    """Resolution accuracy over a SAMPLE of the real corpus.

    A single hardcoded probe only proves that one scheme works. What matters
    is whether the alias table resolves schemes in general, so this tests both
    name forms users actually type -- the acronym and the full title -- across
    a random-but-fixed sample, and reports the failures by name so they can be
    fixed in extract.py's alias rules.
    """
    entries = sample_schemes(index, sample_size)
    by_acronym_ok = by_acronym_total = 0
    by_name_ok = 0
    skipped = 0
    failures: list[str] = []

    for e in entries:
        matches = index.resolve(e.scheme_name)
        if matches and matches[0].doc_id == e.doc_id:
            by_name_ok += 1
        elif len(failures) < 6:
            got = matches[0].scheme_name if matches else "(none)"
            failures.append(f"name   {e.scheme_name[:44]!r} -> {got[:40]!r}")

        from schemes_rag.scheme_index import MIN_ALIAS_LEN_AUTHORITATIVE, tight

        acronym = (e.slug or "").replace("-", " ").strip()
        # Slugs shorter than the alias floor are excluded from the index on
        # purpose ("ef", "ay" collide with everything and nobody types them),
        # so counting them as misses would permanently understate accuracy.
        too_short = len(tight(acronym)) < MIN_ALIAS_LEN_AUTHORITATIVE
        if too_short:
            skipped += 1
        elif acronym and acronym.lower() != e.scheme_name.lower():
            by_acronym_total += 1
            matches = index.resolve(f"Who can apply for {acronym}?")
            if matches and matches[0].doc_id == e.doc_id:
                by_acronym_ok += 1
            elif len(failures) < 6:
                got = matches[0].scheme_name if matches else "(none)"
                failures.append(f"slug   {acronym[:44]!r} -> {got[:40]!r}")

    n = len(entries)
    name_pct = 100 * by_name_ok / n if n else 0
    acr_pct = 100 * by_acronym_ok / by_acronym_total if by_acronym_total else 0
    detail = (
        f"full name: {by_name_ok}/{n} ({name_pct:.0f}%)   "
        f"slug/acronym: {by_acronym_ok}/{by_acronym_total} ({acr_pct:.0f}%)"
    )
    if skipped:
        detail += f"   (skipped {skipped} slug(s) under "
        detail += "the alias length floor)"
    if failures:
        detail += "\n         misses: " + "\n                 ".join(failures)

    worst = min(name_pct, acr_pct if by_acronym_total else 100)
    if worst >= 90:
        record(OK, f"Resolver accuracy on {n} sampled schemes", detail)
    elif worst >= 70:
        record(WARN, f"Resolver accuracy is mixed ({worst:.0f}%)", detail)
    else:
        record(FAIL, f"Resolver accuracy is poor ({worst:.0f}%)",
               detail + "\n         Review SECTION_ALIASES / generate_aliases in extract.py")


def pick_probe(store, index, collection: str) -> tuple[str, str] | None:
    """Build an end-to-end probe from whatever is actually indexed.

    Finds a real chunk that has both an acronym and an eligibility section,
    then confirms the resolver maps that acronym back to the same scheme, so
    the probe is answerable by construction -- rather than assuming any
    particular scheme exists in the collection.
    """
    from qdrant_client import models as qm

    try:
        points, _ = store.client.scroll(
            collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="section", match=qm.MatchValue(value="eligibility"))]
            ),
            limit=200,
            with_payload=True,
        )
    except Exception:
        return None

    # Prefer a SHORT acronym: slugs in this corpus range from real-world
    # abbreviations ("pm-kisan", "apy") to machine-generated initialisms like
    # "DSCTASSGCSCCM". The latter resolve fine but make a nonsense probe that
    # no user would ever type, so they are a last resort behind the full name.
    candidates: list[tuple[int, str, str]] = []
    for point in sorted(points, key=lambda p: str(p.id)):
        payload = point.payload or {}
        name = payload.get("scheme_name")
        if not name:
            continue
        acronym = payload.get("scheme_acronym") or ""
        flat = acronym.replace("-", "")

        if 3 <= len(flat) <= 8:
            candidates.append((0, f"Who can apply for {acronym}?", name))  # best
        elif len(name) <= 70:
            candidates.append((1, f"Who can apply for {name}?", name))     # natural
        elif acronym:
            candidates.append((2, f"Who can apply for {acronym}?", name))  # fallback

    for _, question, name in sorted(candidates, key=lambda c: c[0]):
        matches = index.resolve(question)
        if matches and matches[0].scheme_name == name:
            return question, name
    return None


PLACEHOLDER_KEYS = {"api-key-here", "not-needed", "", "your-api-key", "changeme"}

# Gemini's /models listing returns ids as "models/gemini-3.5-flash-lite" while
# its chat endpoint wants the bare "gemini-3.5-flash-lite". Comparing the two
# literally reports a perfectly good LLM_MODEL as "not served", which pushes
# you into the prefixed form -- the one the chat endpoint 404s on.
_MODEL_PREFIX = "models/"


def bare_model(name: str) -> str:
    return name[len(_MODEL_PREFIX):] if name.startswith(_MODEL_PREFIX) else name


def model_is_served(name: str, models: list[str]) -> bool:
    """Prefix-insensitive membership, so either spelling resolves."""
    return bare_model(name) in {bare_model(m) for m in models}


def key_is_placeholder(s) -> bool:
    """Whether LLM_API_KEY is still the shipped placeholder.

    Worth its own check because the resulting 400 says only that the key is
    invalid, which reads as "my key is wrong" rather than "my key was never
    set". Local servers don't need one, so this only matters off-localhost.
    """
    if "localhost" in s.llm_base_url or "127.0.0.1" in s.llm_base_url:
        return False
    return (s.llm_api_key or "").strip().lower() in PLACEHOLDER_KEYS


def list_models(s) -> int:
    """Print what the configured endpoint will actually serve."""
    from schemes_rag.llm import get_llm

    print(f"Endpoint : {s.llm_base_url}")
    print(f"LLM_MODEL: {s.llm_model}")
    print()

    if key_is_placeholder(s):
        print("LLM_API_KEY is still the placeholder. Paste your key into .env:")
        print("    LLM_API_KEY=<your key>")
        print("Get one at https://aistudio.google.com/apikey")
        return 1

    try:
        models = get_llm().available_models()
    except Exception as exc:
        print(f"Could not list models: {type(exc).__name__}: {exc}")
        return 1

    if not models:
        print("The endpoint returned no models. Usually a bad or unauthorised")
        print("API key, or a base URL that is not the OpenAI-compatible one.")
        print("Gemini's is: https://generativelanguage.googleapis.com/v1beta/openai/")
        return 1

    # Printed bare, which is the form the chat endpoint wants. The listing
    # returns them prefixed with "models/"; copying that verbatim into
    # LLM_MODEL is the single most likely way to misconfigure this.
    print(f"{len(models)} model(s) available — use these names as shown:")
    for name in sorted(bare_model(m) for m in models):
        print(f"  {'->' if name == bare_model(s.llm_model) else '  '} {name}")
    print()

    if model_is_served(s.llm_model, models):
        if s.llm_model.startswith(_MODEL_PREFIX):
            print(f"LLM_MODEL={s.llm_model} is served, but drop the "
                  f"{_MODEL_PREFIX!r} prefix:")
            print(f"    LLM_MODEL={bare_model(s.llm_model)}")
            print("The listing returns prefixed ids; the chat endpoint wants "
                  "the bare name.")
            return 1
        print(f"LLM_MODEL={s.llm_model} is served. Nothing to change.")
        return 0

    # Suggest the closest names rather than making them read the whole list.
    import difflib

    bare = [bare_model(m) for m in models]
    close = difflib.get_close_matches(bare_model(s.llm_model), bare, n=3, cutoff=0.3)
    print(f"LLM_MODEL={s.llm_model!r} is NOT in that list, so requests will 404.")
    if close:
        print("Closest available: " + ", ".join(close))
    print("Set LLM_MODEL in .env to one of the names above.")
    return 1


def check_tracing(s) -> int:
    """Verify Langfuse end to end: SDK, credentials, and a delivered trace.

    Separate from the main run because the failure modes are different and the
    advice is specific. "No traces in the dashboard" has four causes that look
    identical from the application side -- not installed, flag off, keys
    wrong, server unreachable -- and this names which one.
    """
    from schemes_rag import tracing

    print(f"Dashboard  : {s.langfuse_base_url}")
    print(f"Enabled    : {s.langfuse_enabled}")
    print(f"Environment: {s.langfuse_environment}")
    print(f"Public key : {(s.langfuse_public_key or '(unset)')[:12]}...")
    print(f"Mask PII   : {s.langfuse_mask_pii}")
    print()

    if not tracing.TRACING_AVAILABLE:
        print("The langfuse SDK is not installed.")
        print("  pip install 'langfuse>=5,<6'")
        print("SDK v5 needs Python 3.11+ and a self-hosted server >= 4.59.0.")
        return 1

    import langfuse

    print(f"SDK version: {langfuse.__version__}")

    if not s.langfuse_enabled:
        print("\nLANGFUSE_ENABLED=false, so nothing is traced.")
        print("Set LANGFUSE_ENABLED=true in .env.")
        return 1

    ok, reason, detail = tracing.auth_check()
    if not ok and reason == tracing.AUTH_UNREACHABLE:
        # The common first-run failure, and the one the old message got
        # wrong: it listed key-mismatch causes for what is plainly a dead
        # port. Nothing answered, so the keys have not been judged at all.
        print(f"\nNothing is listening at {s.langfuse_base_url}")
        print(f"  ({detail})")
        print("\nThe server never answered, so your keys have not been tested")
        print("yet -- this is not an authentication problem.")
        print("\n  1. Start the stack:   .\\ops\\langfuse\\up.ps1")
        print("  2. Did the image pull finish? A truncated pull, e.g.")
        print("       short read: expected 228857902 bytes but got 166033248")
        print("     leaves the web container never started. Retrying resumes")
        print("     from the layers already downloaded:")
        print("       docker compose -f <langfuse>\\docker-compose.yml \\")
        print("                      --env-file ops\\langfuse\\langfuse.env pull")
        print("     up.ps1 now retries this for you.")
        print("  3. Is it still starting? The stack runs migrations on first")
        print("     boot and the dashboard answers only after they finish:")
        print("       docker ps")
        print("       docker compose ... logs -f langfuse-web")
        print(f"  4. Is LANGFUSE_BASE_URL right? It is {s.langfuse_base_url}")
        return 1
    if not ok and reason == tracing.AUTH_SERVER_ERROR:
        # The server is there and broken, which on a fresh stack almost always
        # means it is still coming up rather than misconfigured.
        print(f"\nThe server answered, but with an error: {detail}")
        print("So it IS reachable and your keys have not been judged yet.")
        print("\nOn a freshly started stack this is normally migrations. The")
        print("worker applies the ClickHouse schema on first boot, and the web")
        print("container returns 500 for anything touching ClickHouse until it")
        print("finishes -- which can take several minutes.")
        print("\n  1. Wait a minute and run this again.")
        print("  2. Watch it happen:")
        print("       docker compose -f <langfuse>\\docker-compose.yml \\")
        print("         -f ops\\langfuse\\docker-compose.override.yml \\")
        print("         --env-file ops\\langfuse\\langfuse.env \\")
        print("         logs -f langfuse-worker langfuse-web")
        print("  3. If it persists, the logs name the cause. A bad")
        print("     LANGFUSE_INIT_* value fails the bootstrap this way too.")
        print("  4. Last resort, start clean:")
        print("       .\\ops\\langfuse\\up.ps1 -Down -Wipe")
        print("       .\\ops\\langfuse\\up.ps1")
        return 1
    if not ok and reason == tracing.AUTH_REJECTED:
        print(f"\nThe server is up but rejected the keys: {detail}")
        print("  1. Do LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env match")
        print("     LANGFUSE_INIT_PROJECT_* in ops\\langfuse\\langfuse.env?")
        print("  2. Was the stack first started WITHOUT --env-file? Then no")
        print("     project was bootstrapped and those keys do not exist.")
        print("     Recreate it:  .\\ops\\langfuse\\up.ps1 -Down -Wipe")
        print("  3. Or copy the keys from the dashboard: Settings -> API keys")
        return 1
    if not ok:
        print(f"\nTracing unavailable ({reason}): {detail}")
        return 1
    print(f"Credentials: {detail}")

    # Deliver a real trace. Credentials being accepted does not prove the
    # exporter can reach the ingestion path, which is a separate code path.
    print("\nSending a test trace...")
    try:
        with tracing.trace_context(
            session_id="check-tracing", trace_name="preflight",
            tags=["preflight"],
        ), tracing.observed(
            "preflight", as_type="span", input={"check": "check_llm --check-tracing"}
        ) as span:
            trace_id = tracing.current_trace_id()
            span.update(output={"ok": True})
            tracing.score("preflight", 1.0, comment="from check_llm")
        tracing.flush()
    except Exception as exc:
        print(f"Failed to send: {type(exc).__name__}: {exc}")
        return 1

    print(f"Sent. Trace id: {trace_id}")
    print(f"\nOpen {s.langfuse_base_url} and look for a trace named "
          '"preflight" with a score of 1.')
    print("If the dashboard is empty, the worker may still be starting:")
    print("  docker compose ... logs -f langfuse-worker")
    return 0


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Preflight checks for the RAG stack")
    parser.add_argument(
        "--query",
        help="End-to-end probe question. Omitted: one is derived from your own "
             "indexed data, and the answer is asserted to cite that scheme.",
    )
    parser.add_argument(
        "--sample", type=int, default=40,
        help="How many schemes to sample for the resolver accuracy check.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print the models your API key can call, then exit. Run this "
             "when a request 404s: the error names no alternative, so a wrong "
             "LLM_MODEL is otherwise guesswork.",
    )
    parser.add_argument(
        "--check-tracing", action="store_true",
        help="Verify Langfuse: SDK installed, keys accepted, a test trace "
             "delivered. Then exit.",
    )
    args = parser.parse_args()

    from schemes_rag.config import get_settings

    s = get_settings()

    if args.list_models:
        return list_models(s)
    if args.check_tracing:
        return check_tracing(s)
    print("=" * 72)
    print("PREFLIGHT")
    print("=" * 72)
    print(f"  qdrant     {s.qdrant_url if s.qdrant_mode == 'server' else s.qdrant_local_path}")
    print(f"  collection {s.qdrant_collection}")
    print(f"  llm        {s.llm_model} @ {s.llm_base_url}")
    print(f"  embeddings {s.embed_model} on {s.embed_device}")
    print(f"  hybrid={s.retrieval_hybrid}  rerank={s.rerank_enabled}  "
          f"verify={s.verify_enabled}  floor={s.retrieval_min_score}")
    mem = gpu_memory()
    if mem:
        total, used = mem
        print(f"  gpu        {total} MiB total, {used} MiB used, {total - used} MiB free")
    print()

    # ---------- 1. Qdrant ----------
    points = 0
    try:
        from schemes_rag.vectorstore import get_vectorstore

        store = get_vectorstore()
        if not store.health():
            record(FAIL, "Qdrant unreachable", "docker compose up -d qdrant")
        else:
            points = store.count()
            if points == 0:
                record(FAIL, "Collection is empty", "python scripts\\ingest.py --recreate")
            else:
                record(OK, f"Qdrant reachable, {points:,} points")
                try:
                    info = store.client.get_collection(s.qdrant_collection)
                    opt = str(getattr(info, "optimizer_status", "")).lower()
                    if "ok" in opt:
                        record(OK, "Vector index built", f"optimizer_status={opt}")
                    else:
                        record(WARN, "Index may still be building", f"optimizer_status={opt}")
                except Exception as exc:
                    record(WARN, "Could not read collection info", str(exc))
    except Exception as exc:
        record(FAIL, "Qdrant check failed", f"{type(exc).__name__}: {exc}")

    # ---------- 2. scheme index ----------
    idx_path = Path(s.processed_dir) / "scheme_index.json"
    index = None
    try:
        from schemes_rag.scheme_index import get_scheme_index

        index = get_scheme_index()
        if index is None:
            record(
                FAIL,
                "Scheme alias index MISSING",
                f"expected {idx_path}\n         "
                "Acronym queries ('PM-KISAN') cannot resolve without it.\n         "
                "Fix: python scripts\\ingest.py --recreate",
            )
        else:
            no_slug = sum(1 for e in index.entries.values() if not e.slug)
            record(OK, f"Scheme index: {len(index)} schemes, {len(index.alias_map)} aliases")
            if no_slug:
                record(WARN, f"{no_slug} scheme(s) have no URL slug",
                       "those lose their acronym alias")
            check_resolver(index, args.sample)
    except Exception as exc:
        record(FAIL, "Scheme index check failed", f"{type(exc).__name__}: {exc}")

    # ---------- 3. are the vectors current? ----------
    if points:
        try:
            from schemes_rag.vectorstore import get_vectorstore

            pts, _ = get_vectorstore().client.scroll(
                s.qdrant_collection, limit=20, with_payload=True
            )
            have = sum(1 for p in pts if (p.payload or {}).get("scheme_acronym"))
            if not pts:
                record(WARN, "Could not sample payloads")
            elif have == 0:
                record(
                    FAIL,
                    "Indexed chunks have no scheme_acronym -- STALE VECTORS",
                    "These were embedded before the acronym fix, so no chunk is\n         "
                    "matchable by the name users type.\n         "
                    "Fix: python scripts\\ingest.py --recreate",
                )
            else:
                record(OK, f"Payloads carry scheme_acronym ({have}/{len(pts)} sampled)")
        except Exception as exc:
            record(WARN, "Payload freshness check failed", str(exc))

    # ---------- 4. LLM reachable ----------
    llm_ok = False
    local_llm = "localhost" in s.llm_base_url or "127.0.0.1" in s.llm_base_url

    if key_is_placeholder(s):
        # Checked before reaching out, because every call below would fail
        # with an authentication error that reads like a wrong key rather
        # than an unset one.
        record(
            FAIL,
            "LLM_API_KEY is still the placeholder",
            f"Current value: {s.llm_api_key!r}\n         "
            "Paste your key into .env:  LLM_API_KEY=<your key>\n         "
            "Get one at https://aistudio.google.com/apikey",
        )
    else:
        try:
            from schemes_rag.llm import get_llm

            llm = get_llm()
            if not llm.health():
                record(
                    FAIL,
                    f"LLM unreachable at {s.llm_base_url}",
                    (
                        "Ollama: start it, then `ollama list` to confirm the model.\n"
                        "         Check LLM_BASE_URL ends with /v1 "
                        "(Ollama: http://localhost:11434/v1)"
                        if local_llm else
                        "Check network access, the API key, and that LLM_BASE_URL is\n"
                        "         the OpenAI-COMPATIBLE endpoint. Gemini's is:\n"
                        "         https://generativelanguage.googleapis.com/v1beta/openai/"
                    ),
                )
            else:
                models = llm.available_models()
                record(OK, "LLM endpoint reachable", f"{len(models)} model(s) served")
                if models and not model_is_served(s.llm_model, models):
                    import difflib

                    bare = [bare_model(m) for m in models]
                    close = difflib.get_close_matches(
                        bare_model(s.llm_model), bare, n=3, cutoff=0.3
                    )
                    record(
                        FAIL,
                        f"Model {s.llm_model!r} is NOT served",
                        (f"Closest available: {', '.join(close)}\n         " if close else "")
                        + "Full list: python scripts\\check_llm.py --list-models\n         "
                        + (
                            "Fix: ollama create qwen3-4b-16k -f unused\\Modelfile.qwen3-4b-16k\n"
                            "              or set LLM_MODEL to a served name"
                            if local_llm else
                            "Fix: set LLM_MODEL in .env to a served name"
                        ),
                    )
                elif models and s.llm_model.startswith(_MODEL_PREFIX):
                    # Served, but in the listing's spelling rather than the
                    # chat endpoint's. llm.py recovers by retrying with the
                    # prefix stripped, so this costs a wasted call per
                    # process rather than failing -- still worth fixing.
                    llm_ok = True
                    record(
                        WARN,
                        f"Drop the {_MODEL_PREFIX!r} prefix from LLM_MODEL",
                        f"LLM_MODEL={bare_model(s.llm_model)}\n         "
                        "The /models listing returns prefixed ids, but the chat\n         "
                        "endpoint wants the bare name. It works either way --\n         "
                        "llm.py retries with the prefix stripped -- but that\n         "
                        "wastes one failed request per process.",
                    )
                else:
                    llm_ok = True
                    record(OK, f"Model {s.llm_model!r} is served")
        except Exception as exc:
            record(FAIL, "LLM check failed", f"{type(exc).__name__}: {exc}")

    # ---------- 5/6/7. completion, thinking, JSON ----------
    if llm_ok:
        from schemes_rag.llm import get_llm

        llm = get_llm()
        from schemes_rag.llm import ThinkingOnlyResponse

        try:
            t0 = time.perf_counter()
            # Generous budget: a thinking model needs room to finish reasoning
            # AND answer, so a small cap would misreport the cause.
            out = llm.complete("You answer in one short word.", "Say: ready", max_tokens=512)
            ms = (time.perf_counter() - t0) * 1000
            if out:
                record(OK, "Completion works", f"{ms:.0f} ms -> {out[:60]!r}")
            else:
                record(FAIL, "Completion returned empty string")

            if "<think>" in out.lower():
                record(FAIL, "<think> leaked into output", "strip_thinking() did not catch it")
            else:
                record(OK, "No <think> leakage")

            completion_ok = bool(out)
        except ThinkingOnlyResponse as exc:
            record(
                FAIL,
                "Model is THINKING instead of answering",
                f"{exc}\n         "
                "This is why answers come back blank with no citations.",
            )
            completion_ok = False
        except Exception as exc:
            record(FAIL, "Completion failed", f"{type(exc).__name__}: {exc}")
            completion_ok = False

        # Throughput, not single-call latency, distinguishes a modest GPU from
        # CPU offload: the first call also pays model load. A 4B model on GPU
        # does 30-80 tok/s; under 10 tok/s means layers are on the CPU.
        if completion_ok:
            try:
                t0 = time.perf_counter()
                long_out, used_tokens = llm.complete_verbose(
                    "You write plain prose.",
                    "Write a single paragraph of about 120 words explaining what "
                    "a government subsidy is. Prose only, no lists.",
                    max_tokens=200,
                )
                gen_s = time.perf_counter() - t0

                # Prefer the server's own count. A chars/4 estimate is ~2x off
                # for digit- and punctuation-heavy text, which is enough to
                # turn a healthy GPU into a false "CPU-slow" verdict.
                if used_tokens:
                    n_tokens, source = used_tokens, "server usage"
                else:
                    n_tokens, source = max(1, len(long_out) // 4), "estimated"
                tps = n_tokens / gen_s if gen_s > 0 else 0.0
                detail = f"{n_tokens} tokens ({source}) in {gen_s:.1f}s = {tps:.1f} tok/s"

                if tps < 10:
                    record(
                        FAIL,
                        f"Generation is very slow (~{tps:.1f} tok/s)",
                        f"{detail}\n         " + (
                            "A 4B model on GPU does 30-80 tok/s, so layers are on\n         "
                            "the CPU.\n         "
                            "1. EMBED_DEVICE=cpu in .env (frees ~2.3 GB of VRAM)\n         "
                            "2. setx OLLAMA_FLASH_ATTENTION 1 / OLLAMA_KV_CACHE_TYPE q8_0\n         "
                            "   in PowerShell -- NOT in .env -- then restart Ollama\n         "
                            "3. lower num_ctx in unused\\Modelfile.qwen3-4b-16k and rebuild"
                            if local_llm else
                            "A hosted flash-class model should exceed 100 tok/s, so this\n         "
                            "is the network or rate limiting, not the model. Check your\n         "
                            "connection and whether the key is being throttled.\n         "
                            "If it is thinking rather than generating, cut it:\n         "
                            "LLM_REASONING_EFFORT=minimal"
                        ),
                    )
                elif tps < 25:
                    record(WARN, f"Generation is modest (~{tps:.1f} tok/s)", detail)
                else:
                    record(OK, f"Generation throughput ~{tps:.0f} tok/s", detail)
            except Exception as exc:
                record(WARN, "Throughput probe failed", f"{type(exc).__name__}: {exc}")

        try:
            raw = llm.complete(
                'Reply with JSON only: {"ok": true}',
                "Return the JSON.",
                json_mode=True,
                max_tokens=64,
            )
            json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            record(OK, "JSON mode works")
        except json.JSONDecodeError:
            record(WARN, "JSON mode output was not parseable",
                   "Tolerated: the router falls back to keyword heuristics.")
        except Exception as exc:
            record(WARN, "JSON mode unavailable", f"{type(exc).__name__} -- heuristic fallback")

    # ---------- 8. end to end ----------
    if llm_ok and points and index is not None:
        try:
            from schemes_rag.graph import run_query
            from schemes_rag.vectorstore import get_vectorstore

            expected: str | None = None
            if args.query:
                q = args.query
            else:
                probe = pick_probe(get_vectorstore(), index, s.qdrant_collection)
                if probe is None:
                    record(WARN, "Could not derive a probe from the index",
                           "no indexed chunk had both an acronym and an eligibility section")
                    q, expected = "What schemes are available for farmers?", None
                else:
                    q, expected = probe

            t0 = time.perf_counter()
            ans = run_query(q, thread_id="preflight")
            ms = (time.perf_counter() - t0) * 1000
            print()
            print("-" * 72)
            label = "END-TO-END  " + q
            if expected:
                label += f"\n            (auto-derived; expecting {expected[:48]})"
            print(label)
            print("-" * 72)
            print(ans.answer)
            print()
            for c in ans.citations:
                print(f"  [{c.marker}] {c.locator}")
                if c.source_url:
                    print(f"       {c.source_url}")
            print(f"\n  route={ans.route}  {ms:.0f} ms  {ans.latency_ms}")
            print("-" * 72)

            if ans.route == "clarify":
                record(
                    WARN,
                    "Pipeline asked for clarification instead of answering",
                    "The probe named a specific scheme, so this should not happen.\n         "
                    f"Asked: {ans.answer[:80]}",
                )
            elif ans.insufficient_evidence:
                record(FAIL, "Pipeline refused a question the corpus covers",
                       f"Lower RETRIEVAL_MIN_SCORE (now {s.retrieval_min_score}) "
                       "or check the gate score\n         "
                       "via: python scripts\\query.py \"...\" --search-only")
            elif not ans.citations:
                record(WARN, "Answer had no citations", "unverifiable; check prompts.py rule 2")
            else:
                schemes = set(ans.schemes_used)
                if expected is None:
                    record(OK, "End-to-end answer produced with citations",
                           f"grounded in: {', '.join(sorted(schemes))[:90]}")
                elif schemes == {expected}:
                    record(OK, "End-to-end answer grounded in the correct scheme only",
                           f"{expected[:70]}")
                else:
                    record(WARN, "Answer drew on other schemes",
                           f"expected only {expected[:50]!r}, got {sorted(schemes)}")
        except Exception as exc:
            record(FAIL, "End-to-end query failed", f"{type(exc).__name__}: {exc}")

    # ---------- 9. is the model actually on the GPU? ----------
    # Only meaningful for a locally served model. Against a hosted endpoint
    # there is no /api/ps and no VRAM of ours to run out of, so these checks
    # would emit warnings about a machine that is not doing the inference.
    if llm_ok and local_llm:
        models = ollama_loaded_models(s.llm_base_url)
        if models is None:
            record(WARN, "Could not read Ollama /api/ps", "not Ollama, or not reachable")
        elif not models:
            record(
                WARN,
                "No model currently loaded",
                "Ollama unloads after ~5 min idle, so the split is unknown.\n         "
                "This is also why a bare `ollama ps` can print nothing.",
            )
        else:
            for m in models:
                size = int(m.get("size") or 0)
                vram = int(m.get("size_vram") or 0)
                name = m.get("name") or m.get("model") or "?"
                pct = (100 * vram / size) if size else 0
                detail = (
                    f"{name}: {vram / 2**30:.2f} GiB on GPU of "
                    f"{size / 2**30:.2f} GiB total ({pct:.0f}% GPU)"
                )
                if size and pct < 99:
                    advice = (
                        f"{detail}\n         "
                        f"{(size - vram) / 2**30:.2f} GiB is running on the CPU, which is\n         "
                        "why generation is slow. Free VRAM or shrink the KV cache:\n         "
                        "  1. EMBED_DEVICE=cpu in .env (frees ~2.3 GB)\n         "
                        "  2. setx OLLAMA_FLASH_ATTENTION 1\n         "
                        "     setx OLLAMA_KV_CACHE_TYPE q8_0   (halves the KV cache)\n         "
                        "     then restart Ollama\n         "
                        "  3. lower num_ctx in unused\\Modelfile.qwen3-4b-16k and rebuild"
                    )
                    record(FAIL, "Model is PARTLY ON CPU", advice)
                else:
                    record(OK, "Model fully on GPU", detail)

        if mem:
            total, used = mem
            free_mib = total - used
            kv_mib = (s.llm_context_budget_tokens + 2048) * _KV_BYTES_PER_TOKEN_FP16 / 2**20
            if free_mib < 2500 + kv_mib:
                record(
                    WARN,
                    f"VRAM is tight for this context size",
                    f"{free_mib} MiB free; ~2500 MiB of q4 weights plus ~{kv_mib:.0f} MiB\n         "
                    "of fp16 KV cache for the configured budget. OLLAMA_KV_CACHE_TYPE=q8_0\n         "
                    "halves the KV figure.",
                )

    # ---------- summary ----------
    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print()
    print("=" * 72)
    print(f"{len(_results) - len(fails) - len(warns)} ok, {len(warns)} warnings, {len(fails)} failures")
    if fails:
        print("\nFix these first, in order:")
        for _, name, detail in fails:
            print(f"  - {name}")
            if detail:
                print(f"      {detail.splitlines()[0]}")
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
