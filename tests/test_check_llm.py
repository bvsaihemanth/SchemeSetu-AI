"""Runs scripts/check_llm.py end to end under mocks, through every branch.

A preflight script is only useful if it never crashes -- it runs when things
are already broken. The UnboundLocalError that shipped earlier lived in a
failure path, so the healthy path alone would never have caught it. This
executes the healthy path AND each failure path, asserting main() returns an
exit code rather than raising.

    python -m tests.test_check_llm
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["RETRIEVAL_MIN_SCORE"] = "0.05"
# The shipped default is the placeholder "api-key-here", which check_llm now
# fails on before it tries anything. These scenarios are about the checks
# downstream of the key, so they get a key that looks real; the placeholder
# path has a scenario of its own.
os.environ["LLM_API_KEY"] = "test-key-not-a-placeholder"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CORPUS = "/mnt/user-data/uploads/myscheme-ai/schemes_txt"


class ScriptedLLM:
    """Configurable stand-in: answer normally, stall, empty, or think forever."""

    def __init__(self, mode: str = "ok", delay: float = 0.0) -> None:
        self.mode = mode
        self.delay = delay

    def _text(self, system: str, user: str) -> str:
        if "Classify the user's question" in system:
            return (
                '{"intent": "eligibility", "schemes": [], "state": null, '
                '"needs_clarification": false, "clarifying_question": null, '
                '"search_query": "eligibility"}'
            )
        if "Reply with JSON only" in system:
            return '{"ok": true}'
        if "You write plain prose" in system:
            return "A government subsidy is a transfer of public funds. " * 6
        if "one short word" in system:
            return "ready"
        return "Landholding farmers' families are eligible [1]."

    def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
        if self.delay:
            time.sleep(self.delay)
        if self.mode == "empty":
            return ""
        if self.mode == "thinking":
            from schemes_rag.llm import ThinkingOnlyResponse

            raise ThinkingOnlyResponse("2000 chars of reasoning, no answer")
        if self.mode == "boom":
            raise RuntimeError("connection reset by peer")
        return self._text(system, user)

    def complete_verbose(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
        text = self.complete(system, user, temperature, max_tokens, json_mode)
        # mode "slow" reports few tokens over a long wall time -> CPU-slow path
        return text, (3 if self.mode == "slow" else 140)

    def health(self):
        return self.mode != "down"

    def available_models(self):
        from schemes_rag.config import get_settings

        return [get_settings().llm_model]


def build_env(
    llm, *, with_index: bool = True, qdrant_up: bool = True,
    placeholder_key: bool = False, hosted_llm: bool = False,
):
    """Wire check_llm's dependencies to in-memory stand-ins."""
    from qdrant_client import QdrantClient

    import check_llm as C
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
    from tests.test_scheme_resolution import HashEmbedder

    embedder = HashEmbedder()
    store = VectorStore(client=QdrantClient(":memory:"))

    # getattr-guarded: once replaced by a lambda, cache_clear is gone, so a
    # second scenario would crash the harness rather than the script.
    for mod, attr in ((embeddings_mod, "get_embedder"), (vs_mod, "get_vectorstore"),
                      (llm_mod, "get_llm")):
        fn = getattr(mod, attr, None)
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    embeddings_mod.get_embedder = lambda: embedder
    pipeline_mod.get_embedder = lambda: embedder
    pipeline_mod.get_vectorstore = lambda: store
    vs_mod.get_vectorstore = lambda: store
    retr_mod.get_embedder = lambda: embedder
    retr_mod.get_vectorstore = lambda: store
    retr_mod._retriever = None
    llm_mod.get_llm = lambda: llm
    gen_mod.get_llm = lambda: llm
    understand_mod.get_llm = lambda: llm
    graph_mod._graph = None

    Path("data/processed/ingest_manifest.json").unlink(missing_ok=True)
    pipeline_mod.run_ingest(corpus_dir=CORPUS, recreate=True)

    if not with_index:
        Path("data/processed/scheme_index.json").unlink(missing_ok=True)
    reset_scheme_index()

    if not qdrant_up:
        store.health = lambda: False

    # Deterministic hardware readings, so the VRAM branch is exercised.
    C.gpu_memory = lambda: (4094, 575)
    C.ollama_loaded_models = lambda base: [
        {"name": "qwen3-4b-16k:latest", "size": 3_489_660_928, "size_vram": 3_489_660_928}
    ]

    # The script branches on the API key and on whether the base URL is local,
    # so both are driven through the settings object it reads. Mutated on the
    # cached instance rather than via env, because get_settings is lru_cached
    # and these have to differ per scenario.
    from schemes_rag.config import get_settings

    settings = get_settings()
    settings.llm_api_key = "api-key-here" if placeholder_key else "test-key-looks-real"
    settings.llm_base_url = (
        "https://generativelanguage.googleapis.com/v1beta/openai/" if hosted_llm
        else "http://localhost:11434/v1"
    )
    settings.llm_model = "gemini-3.5-flash-lite" if hosted_llm else "qwen3-4b-16k"

    C._results.clear()
    return C


def run_scenario(name: str, llm, *, argv: list[str], **kwargs) -> int:
    import check_llm  # noqa: F401

    C = build_env(llm, **kwargs)
    old_argv = sys.argv
    sys.argv = ["check_llm.py", *argv]
    try:
        code = C.main()
    finally:
        sys.argv = old_argv

    fails = [r for r in C._results if r[0] == C.FAIL]
    warns = [r for r in C._results if r[0] == C.WARN]
    print(f"\n>>> {name}: exit={code}  ok={len(C._results) - len(fails) - len(warns)} "
          f"warn={len(warns)} fail={len(fails)}")
    for status, title, _ in C._results:
        if status in (C.FAIL, C.WARN):
            print(f"      [{status.strip()}] {title}")
    return code


def main() -> int:
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_check_llm"])

    if not os.path.isdir(CORPUS):
        print(f"SKIP: corpus not available at {CORPUS}")
        return 0

    # expected_exit of None means "any exit code, just do not crash": for the
    # auto-derived probe the outcome depends on how well the stand-in
    # bag-of-words embedder scores that particular scheme, which is a property
    # of the fixture, not of the script under test.
    scenarios = [
        # (name, llm, argv, kwargs, expected_exit)
        ("healthy, auto-derived probe", ScriptedLLM("ok"), [], {}, None),
        ("healthy, explicit --query", ScriptedLLM("ok"),
         ["--query", "Who can apply for PM-KISAN?"], {}, 0),
        ("small --sample", ScriptedLLM("ok"), ["--sample", "5"], {}, None),
        ("LLM returns empty", ScriptedLLM("empty"), [], {}, 1),
        ("LLM thinks, never answers", ScriptedLLM("thinking"), [], {}, 1),
        ("LLM raises", ScriptedLLM("boom"), [], {}, 1),
        ("LLM endpoint down", ScriptedLLM("down"), [], {}, 1),
        ("throughput is CPU-slow", ScriptedLLM("slow", delay=0.35), [], {}, 1),
        ("scheme index missing", ScriptedLLM("ok"), [], {"with_index": False}, 1),
        ("qdrant unreachable", ScriptedLLM("ok"), [], {"qdrant_up": False}, 1),
        # The most likely first-run failure: .env copied, key never pasted.
        # It must fail with that as the named cause, before anything tries to
        # call the API and reports a confusing auth error instead.
        # hosted_llm matters: a placeholder key against localhost is FINE,
        # because a local server ignores it. The check only applies when the
        # endpoint is remote and will actually authenticate.
        ("API key still the placeholder", ScriptedLLM("ok"), [],
         {"placeholder_key": True, "hosted_llm": True}, 1),
        ("placeholder key is OK for a local server", ScriptedLLM("ok"),
         ["--query", "Who can apply for PM-KISAN?"],
         {"placeholder_key": True, "hosted_llm": False}, 0),
        # Against a hosted endpoint the Ollama and VRAM checks must be skipped
        # rather than warning about hardware that is not doing the inference.
        ("hosted LLM skips GPU checks", ScriptedLLM("ok"),
         ["--query", "Who can apply for PM-KISAN?"], {"hosted_llm": True}, 0),
        ("--list-models", ScriptedLLM("ok"), ["--list-models"],
         {"hosted_llm": True}, 0),
    ]

    failures: list[str] = []
    for name, llm, argv, kwargs, expected in scenarios:
        print("\n" + "=" * 72)
        print(f"SCENARIO: {name}")
        print("=" * 72)
        try:
            code = run_scenario(name, llm, argv=argv, **kwargs)
        except SystemExit as exc:  # argparse should not exit on valid argv
            failures.append(f"{name}: SystemExit({exc.code})")
            continue
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failures.append(f"{name}: CRASHED {type(exc).__name__}: {exc}")
            continue

        if not isinstance(code, int):
            failures.append(f"{name}: returned {code!r}, not an int")
        elif expected is not None and code != expected:
            failures.append(f"{name}: exit {code}, expected {expected}")

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} SCENARIO FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {len(scenarios)} SCENARIOS RAN WITHOUT CRASHING")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
