"""Tests the Qwen3-specific handling in llm.py without a running server.

Covers thinking-block stripping (including the stream case, where a block can
straddle chunk boundaries) and the JSON-mode fallback for servers that reject
response_format.

    python -m tests.test_llm_client
"""

from __future__ import annotations

import asyncio
import sys

from schemes_rag.llm import strip_thinking


def test_strip() -> None:
    cases = [
        # (input, expected)
        ("plain answer", "plain answer"),
        ("<think>deliberating</think>The answer is X [1].", "The answer is X [1]."),
        ("<think>a</think>\n\nFarmers are eligible [1].", "Farmers are eligible [1]."),
        # multiple blocks
        ("<think>a</think>One [1]. <think>b</think>Two [2].", "One [1]. Two [2]."),
        # unclosed block from a truncated generation -> drop the tail
        ("Partial answer [1]. <think>still thinking and ran out of", "Partial answer [1]."),
        # block only -> empty, which the caller treats as a failed generation
        ("<think>only thinking</think>", ""),
        # case variations some builds emit
        ("<THINK>x</THINK>Answer.", "Answer."),
        # multiline, the realistic shape
        ("<think>\nStep 1\nStep 2\n</think>\nEligible farmers [1].", "Eligible farmers [1]."),
        # must NOT eat legitimate angle brackets
        ("Use <code> tags and a < b comparison.", "Use <code> tags and a < b comparison."),
        ("", ""),
    ]
    for raw, expected in cases:
        got = strip_thinking(raw)
        status = "OK " if got == expected else "BAD"
        print(f"  {status} {raw[:46]!r:50} -> {got[:40]!r}")
        assert got == expected, f"{raw!r}: expected {expected!r}, got {got!r}"


class FakeChoice:
    def __init__(self, content):
        self.delta = type("D", (), {"content": content})()


class FakeChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


async def collect_stream(pieces: list[str]) -> str:
    """Drive QwenClient.astream's suppression logic over scripted chunks."""
    from schemes_rag.llm import QwenClient

    class FakeStream:
        def __aiter__(self):
            async def gen():
                for p in pieces:
                    yield FakeChunk(p)

            return gen()

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeStream()

    client = QwenClient.__new__(QwenClient)  # bypass __init__ / no network
    from schemes_rag.config import get_settings

    client.settings = get_settings()
    client._json_mode_supported = True
    client._async = type(
        "A", (), {"chat": type("C", (), {"completions": FakeCompletions()})()}
    )()

    out = []
    async for piece in client.astream("sys", "user"):
        out.append(piece)
    return "".join(out)


def test_stream() -> None:
    cases = [
        (["Farmers ", "are ", "eligible [1]."], "Farmers are eligible [1]."),
        # block arrives whole in one chunk
        (["<think>hmm</think>", "Answer [1]."], "Answer [1]."),
        # block straddles chunk boundaries -- the case naive stripping leaks
        (["<thi", "nk>delib", "erating</thi", "nk>", "Answer [1]."], "Answer [1]."),
        # opener split at the very tail of a chunk
        (["Text before <", "think>hidden</think> after."], "Text before  after."),
        # thinking after content
        (["Part one. ", "<think>x</think>", "Part two."], "Part one. Part two."),
        # unclosed block -> suppressed to the end, nothing leaks
        (["Visible. ", "<think>never closes..."], "Visible. "),
    ]
    for pieces, expected in cases:
        got = asyncio.run(collect_stream(pieces))
        status = "OK " if got == expected else "BAD"
        print(f"  {status} {pieces} -> {got!r}")
        assert got == expected, f"{pieces}: expected {expected!r}, got {got!r}"


def _bad_request(message: str):
    import httpx
    from openai import BadRequestError

    return BadRequestError(
        message,
        response=httpx.Response(
            400, request=httpx.Request("POST", "http://x/v1/chat/completions")
        ),
        body=None,
    )


def _fake_client(create):
    """A client with no network, whose completions.create is `create`."""
    from schemes_rag.config import get_settings
    from schemes_rag.llm import LLMClient

    c = LLMClient.__new__(LLMClient)   # no __init__: opens no connection
    c.settings = get_settings()
    c._sync = type(
        "S", (), {"chat": type("C", (), {"completions": type(
            "P", (), {"create": staticmethod(create)})()})()}
    )()
    return c


def test_optional_field_degradation() -> None:
    """A 400 on an optional field must shed that field, not fail the request.

    A rejection does not say WHICH field offended -- providers return a
    generic 400 -- so the client sheds them one at a time in a fixed order
    (reasoning_effort, extra_body, response_format: optimisation first,
    load-bearing last) and retries. These assertions pin that ladder down:
    it must make progress, terminate, and remember.
    """
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        # Stands in for a server that knows nothing newer than plain
        # chat-completions, e.g. an older Ollama.
        for field in ("reasoning_effort", "response_format", "extra_body"):
            if field in kwargs:
                raise _bad_request(f"{field} is not supported")
        msg = type("M", (), {"content": '{"ok": true}'})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    client = _fake_client(create)
    out = client.complete("sys", "user", json_mode=True)

    assert out == '{"ok": true}', out
    # One call per field shed, plus the one that succeeds. reasoning_effort is
    # only in the first payload if the config sets it, so count from the data.
    shed = [f for f in ("reasoning_effort", "extra_body", "response_format")
            if f in calls[0]]
    assert len(calls) == len(shed) + 1, f"{len(shed)} shed, {len(calls)} calls"
    assert not any(f in calls[-1] for f in shed), calls[-1]
    print(f"  OK  shed {shed} one at a time -> {len(calls)} calls -> {out!r}")

    # Flags stay off, so the 400s are paid once per process, not per request.
    before = len(calls)
    client.complete("sys", "user", json_mode=True)
    assert len(calls) == before + 1, "a remembered rejection must not retry"
    assert not any(f in calls[-1] for f in shed)
    print("  OK  rejections remembered -> no repeat 400s on later calls")


def test_model_prefix_recovery() -> None:
    """A model name in the listing's spelling must still work.

    Gemini's /models endpoint returns ids as "models/gemini-3.5-flash-lite";
    its chat endpoint wants the bare "gemini-3.5-flash-lite" and 404s on the
    prefixed form, saying only "not found" and naming no alternative. Anyone
    who copies a name out of the listing hits that, so the client tries the
    other spelling once and remembers which worked.
    """
    import httpx
    from openai import NotFoundError

    def not_found():
        return NotFoundError(
            "model not found",
            response=httpx.Response(
                404, request=httpx.Request("POST", "http://x/chat/completions")
            ),
            body=None,
        )

    for configured, accepted in (
        ("models/gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),  # prefix shed
        ("gemini-3.5-flash-lite", "models/gemini-3.5-flash-lite"),  # prefix added
    ):
        seen: list[str] = []

        def create(_accepted=accepted, _seen=seen, **kwargs):
            _seen.append(kwargs["model"])
            if kwargs["model"] != _accepted:
                raise not_found()
            msg = type("M", (), {"content": "ready"})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        client = _fake_client(create)
        client.settings.llm_model = configured

        assert client.complete("sys", "user") == "ready"
        assert seen == [configured, accepted], seen
        # The winning name is remembered, so the 404 is paid once per process.
        assert client._model_override == accepted
        client.complete("sys", "user")
        assert seen[-1] == accepted and len(seen) == 3, seen
        print(f"  OK  {configured!r} -> retried as {accepted!r}, then remembered")

    # A name that is simply wrong must surface the 404 rather than loop.
    seen2: list[str] = []

    def always_404(**kwargs):
        seen2.append(kwargs["model"])
        raise not_found()

    client = _fake_client(always_404)
    client.settings.llm_model = "gemini-does-not-exist"
    try:
        client.complete("sys", "user")
    except NotFoundError:
        pass
    else:
        raise AssertionError("expected NotFoundError to surface")
    assert len(seen2) == 2, f"tried {len(seen2)} names, expected 2 (bounded)"
    print(f"  OK  unknown model -> raised after {len(seen2)} attempts (bounded)")


def test_preflight_model_matching() -> None:
    """check_llm must accept either spelling instead of demanding the listing's.

    This guards a real mistake: the first version compared LLM_MODEL to the
    listing literally, so a correct bare name was reported as "NOT served" --
    which is advice to switch to the prefixed form the chat endpoint rejects.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_llm as C

    served = ["models/gemini-3.5-flash-lite", "models/gemini-3.8-flash"]
    for name in ("gemini-3.5-flash-lite", "models/gemini-3.5-flash-lite"):
        assert C.model_is_served(name, served), name
        print(f"  OK  {name!r} recognised as served")
    assert not C.model_is_served("gemini-nope", served)
    assert C.bare_model("models/x") == "x" and C.bare_model("x") == "x"
    print("  OK  unknown name still rejected; bare_model is idempotent")


def test_degradation_terminates() -> None:
    """A server that 400s on everything must raise, not loop forever.

    _create retries after each shed field; once only required fields remain,
    _degrade returns None and the error has to surface. Without that guard
    this is an infinite loop against a misconfigured endpoint -- which is the
    shape a wrong model name takes, since that 400s no matter what is sent.
    """
    from openai import BadRequestError

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        raise _bad_request("model not found")

    client = _fake_client(create)
    try:
        client.complete("sys", "user", json_mode=True)
    except BadRequestError:
        pass
    else:
        raise AssertionError("expected BadRequestError to surface")

    # Bounded by the number of optional fields: at most 3 sheds + the first try.
    assert len(calls) <= 4, f"{len(calls)} calls -- not bounded"
    assert "messages" in calls[-1] and "model" in calls[-1]
    print(f"  OK  rejected everything -> raised after {len(calls)} calls (bounded)")


def test_thinking_only_raises() -> None:
    """A thinking-only response must fail loudly, not return ''."""
    from schemes_rag.config import get_settings
    from schemes_rag.llm import QwenClient, ThinkingOnlyResponse

    class FakeCompletions:
        def __init__(self, content):
            self.content = content

        def create(self, **kwargs):
            msg = type("M", (), {"content": self.content})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    def client_for(content):
        c = QwenClient.__new__(QwenClient)
        c.settings = get_settings()
        c._json_mode_supported = True
        c._sync = type(
            "S", (), {"chat": type("C", (), {"completions": FakeCompletions(content)})()}
        )()
        return c

    # unclosed thinking that ran out of budget
    try:
        client_for("<think>" + "reasoning " * 200).complete("s", "u")
        raise AssertionError("expected ThinkingOnlyResponse")
    except ThinkingOnlyResponse as exc:
        # The message has to be actionable, not just accurate: this failure
        # looks like "the model returned nothing" and the fix is always a
        # setting. Assert it names settings rather than particular prose, so
        # the wording stays free to change with the provider.
        text = str(exc)
        assert "LLM_MAX_TOKENS" in text, text
        assert "LLM_REASONING_EFFORT" in text, text
        assert "max_tokens=" in text, "must report the budget that was hit"
        print(f"  OK  unclosed thinking -> raised: {text.splitlines()[0][:66]}...")

    # closed block with nothing after it
    try:
        client_for("<think>done reasoning</think>").complete("s", "u")
        raise AssertionError("expected ThinkingOnlyResponse")
    except ThinkingOnlyResponse:
        print("  OK  closed block, no answer -> raised")

    # interleaved reasoning: every non-thinking fragment survives
    got = client_for("<think>a</think>One [1]. <think>b</think>Two [2].").complete("s", "u")
    assert got == "One [1]. Two [2].", got
    print(f"  OK  interleaved blocks keep all content -> {got!r}")

    # a genuinely empty model reply is NOT misreported as thinking
    got = client_for("").complete("s", "u")
    assert got == ""
    print("  OK  empty reply with no think tags -> '' (not misreported)")


def test_intent_routing() -> None:
    """'who can apply' is an eligibility question despite containing 'apply'."""
    from schemes_rag.agents.understand import heuristic_plan

    cases = [
        ("Who can apply for PM-KISAN?", "eligibility"),
        ("who can avail this scheme", "eligibility"),
        ("am i eligible for PMFBY", "eligibility"),
        ("can i apply if i own 3 hectares", "eligibility"),
        ("what is the age limit", "eligibility"),
        ("how do I apply for PM-KISAN", "application"),
        ("where to apply", "application"),
        ("application process for KCC", "application"),
        ("registration portal", "application"),
        ("what documents are required", "documents"),
        ("how much subsidy do I get", "benefits"),
        ("compare PMJJBY and PMSBY", "comparison"),
        ("what schemes for farmers", "discovery"),
    ]
    for q, expected in cases:
        got = heuristic_plan(q).intent
        status = "OK " if got == expected else "BAD"
        print(f"  {status} {q[:38]:40} -> {got}")
        assert got == expected, f"{q!r}: expected {expected}, got {got}"


def test_empty_generation_surfaces() -> None:
    """An empty completion with sources in hand must not become a blank answer."""
    import schemes_rag.generate as gen_mod
    from schemes_rag.llm import ThinkingOnlyResponse
    from schemes_rag.models import Chunk, RetrievedChunk, Section

    chunk = Chunk(
        doc_id="d1", chunk_index=0, text="Farmers are eligible.", token_count=8,
        doc_name="PM-KISAN", scheme_name="Pradhan Mantri Kisan Samman Nidhi",
        scheme_acronym="PM-KISAN", section=Section.ELIGIBILITY,
        section_title="Eligibility", source_url="https://example.gov.in/pm-kisan",
    )
    rc = [RetrievedChunk(chunk=chunk, score=0.8)]

    class EmptyLLM:
        def complete(self, *a, **k):
            return "   \n  "

    class ThinkingLLM:
        def complete(self, *a, **k):
            raise ThinkingOnlyResponse("spent 2000 chars reasoning; use a non-thinking model")

    original = gen_mod.get_llm
    try:
        gen_mod.get_llm = lambda: EmptyLLM()
        ans = gen_mod.generate_answer("Who can apply?", rc, route="eligibility")
        assert not ans.insufficient_evidence, "empty model output is not 'no evidence'"
        assert "empty response" in ans.answer.lower(), ans.answer
        assert "1 relevant sources" in ans.answer or "1 relevant source" in ans.answer
        print(f"  OK  empty completion -> explicit message, not blank")
        print(f"      {ans.answer[:88]}...")

        gen_mod.get_llm = lambda: ThinkingLLM()
        ans = gen_mod.generate_answer("Who can apply?", rc, route="eligibility")
        assert "internal reasoning" in ans.answer.lower(), ans.answer
        assert not ans.insufficient_evidence
        print(f"  OK  thinking-only -> explicit message, flags model not retrieval")
    finally:
        gen_mod.get_llm = original


def main() -> int:
    print("--- strip_thinking ---")
    test_strip()
    print("\n--- astream suppression ---")
    test_stream()
    print("\n--- unsupported optional fields (400 degradation) ---")
    test_optional_field_degradation()
    test_degradation_terminates()
    print("\n--- model name spelling (404 recovery) ---")
    test_model_prefix_recovery()
    test_preflight_model_matching()
    print("\n--- thinking-only responses ---")
    test_thinking_only_raises()
    print("\n--- intent routing ---")
    test_intent_routing()
    print("\n--- empty / thinking-only generation ---")
    test_empty_generation_surfaces()
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
