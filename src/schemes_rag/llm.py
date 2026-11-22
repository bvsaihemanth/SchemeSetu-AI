"""LLM client.

Deliberately an OpenAI-compatible client, not a vendor SDK. Gemini, vLLM,
Ollama (/v1), llama.cpp server, SGLang, LM Studio and OpenAI itself all speak
this protocol, so the serving backend is a config change rather than a code
change. That is why moving this project from a local Qwen 4B to Gemini cost
three lines in .env and nothing here:

  Gemini : https://generativelanguage.googleapis.com/v1beta/openai/  + API key
  Ollama : ollama serve  ->  base_url http://localhost:11434/v1
  vLLM   : vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 16384

Three hazards are handled here, because each produces a confusing downstream
failure rather than a clean error:

1. THINKING. Reasoning models deliberate before answering, and the tokens are
   billed against max_tokens. Three different shapes of the same problem:
     - the text arrives inline in `content` wrapped in <think>...</think>,
       where it lands in the user's answer and wraps the JSON the router and
       verifier expect  ->  stripped by strip_thinking()
     - the server parses it out into a SIDE CHANNEL (`reasoning`), leaving
       `content` empty with no tags anywhere  ->  found by message_parts()
     - the whole budget is spent thinking and no answer is produced
       ->  ThinkingOnlyResponse, rather than a silent blank reply
   We also ask the model to think less: `reasoning_effort` on providers that
   honour it (Gemini), Qwen3's /no_think soft switch when serving Qwen. Both
   are advisory, which is why the three defences above run regardless.

2. JSON MODE. `response_format={"type":"json_object"}` is supported by Gemini
   and vLLM guided decoding, but older Ollama rejects it with a 400. Rather
   than fail the request we retry once without it and fall back to tolerant
   parsing -- see agents/understand.py.

3. SDK RETRIES. Disabled. The default of two silently turns one timeout into
   three, so a slow backend looks like a hang instead of a timeout.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from contextlib import contextmanager
from functools import lru_cache

from openai import APIStatusError, BadRequestError, NotFoundError

from . import usage
from .config import get_settings
from .tracing import generation_metadata, openai_classes

logger = logging.getLogger(__name__)

# Non-greedy, DOTALL: strips each block, keeps everything around it. Also
# handles an unclosed <think> (truncated generation) by dropping the tail.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

# Qwen3's soft switch for disabling reasoning. Harmlessly ignored by models
# that don't implement it.
NO_THINK_SUFFIX = " /no_think"


class ThinkingOnlyResponse(RuntimeError):
    """The model spent its whole token budget thinking and never answered.

    Raised rather than returning "" because an empty answer is indistinguish-
    able from a model that had nothing to say, and silently produces a blank
    reply with no citations. The caller can see the cause and act on it.
    """


def strip_thinking(text: str) -> str:
    """Remove complete <think> blocks, and drop an unclosed trailing one.

    Block removal rather than "keep everything after the last </think>":
    a model that interleaves reasoning with output ("<think>a</think>One.
    <think>b</think>Two.") must keep BOTH sentences, and taking the tail
    would silently discard the first.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


# Servers that parse reasoning out of the text put it in a side channel
# instead of `content`. Ollama >= 0.9 does this for Qwen3, which is why a
# thinking-only response arrives as an EMPTY content string with no <think>
# tags anywhere -- the most confusing possible failure mode.
_REASONING_FIELDS = ("reasoning", "reasoning_content", "thinking")


def message_parts(message) -> tuple[str, str]:
    """(content, reasoning) from a chat message, whichever shape the server uses."""
    content = getattr(message, "content", None) or ""
    reasoning = ""
    for attr in _REASONING_FIELDS:
        val = getattr(message, attr, None)
        if val:
            reasoning = str(val)
            break
    if not reasoning:
        extra = getattr(message, "model_extra", None) or {}
        for key in _REASONING_FIELDS:
            if extra.get(key):
                reasoning = str(extra[key])
                break
    return content, reasoning


_THINKING_REMEDY = (
    "The model spent its token budget thinking. Fix, in order:\n"
    "  1. Raise the ceiling so there is room for an answer after the\n"
    "     deliberation:  LLM_MAX_TOKENS=4096 in .env\n"
    "  2. Cut the deliberation:  LLM_REASONING_EFFORT=minimal\n"
    "     (Gemini 2.5 Flash/Flash-Lite also accept 'none'; 2.5 Pro and the\n"
    "      Gemini 3 family cannot disable thinking at all, so use 'minimal')\n"
    '  3. Or set the provider budget directly:  LLM_EXTRA_BODY={"google":'
    ' {"thinking_config": {"thinking_budget": 0}}}\n'
    "  Serving Qwen locally instead? LLM_NO_THINK_SUFFIX=true, or pull the\n"
    "  non-thinking build:  ollama pull qwen3:4b-instruct"
)


def _finish(message, max_tokens: int) -> str:
    """Post-process a completion, failing loudly on a thinking-only response."""
    content, reasoning = message_parts(message)

    cleaned = strip_thinking(content)
    if cleaned:
        return cleaned

    if reasoning:
        raise ThinkingOnlyResponse(
            f"Model returned {len(reasoning)} characters of reasoning in a "
            f"separate field and an EMPTY answer (max_tokens={max_tokens}). "
            f"The reasoning never reaches `content`, which is why this looks "
            f"like an empty response rather than a visible <think> block.\n"
            + _THINKING_REMEDY
        )
    if content and "<think>" in content.lower():
        raise ThinkingOnlyResponse(
            f"Model emitted {len(content)} characters of reasoning and no "
            f"answer within max_tokens={max_tokens}.\n" + _THINKING_REMEDY
        )
    return cleaned


#: Statuses worth retrying: the server is telling you to come back, not that
#: the request is wrong. 429 is rate limiting; 5xx is the provider being
#: overloaded, which Gemini returns as 503 under load.
#:
#: Deliberately NOT retried: 400 (the payload is wrong -- see _degrade), 401
#: and 403 (the key is wrong), 404 (the model name is wrong). Retrying those
#: just takes longer to fail.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class _RateGate:
    """Client-side requests-per-minute ceiling.

    A sliding window rather than a fixed one: a fixed window lets 2x the limit
    through at a boundary (all of minute one's budget at :59, all of minute
    two's at :01), which is precisely the burst a provider rate-limits on.

    Shared across threads because the quota is per API key, not per thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._times: list[float] = []

    def wait(self, rpm: int) -> None:
        if rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._times = [t for t in self._times if t > cutoff]
                if len(self._times) < rpm:
                    self._times.append(now)
                    return
                sleep_for = max(0.01, self._times[0] - cutoff)
            logger.debug("Rate gate: %d/%d in the last minute, waiting %.1fs",
                         len(self._times), rpm, sleep_for)
            time.sleep(sleep_for)


_GATE = _RateGate()


def quota_scope(exc: Exception) -> str | None:
    """"day", "minute" or None -- which quota a 429 exhausted.

    THE DISTINCTION THAT DECIDES WHAT TO DO. Both arrive as HTTP 429 and read
    identically, but they need opposite responses:

      per-minute  transient. Back off and retry; it resets in under a minute.
      per-day     terminal for today. Retrying cannot succeed, and every
                  retry SPENDS more of the quota that is already gone -- so
                  backoff actively makes it worse.

    Gemini labels the violation in the error body, e.g. quotaId
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier". Without reading it,
    the only honest advice is "lower your request rate", which is useless for
    a daily cap and was what this told the user while they watched a run die
    61 questions in.
    """
    body = getattr(exc, "body", None)
    blobs: list[str] = []
    if isinstance(body, dict):
        error = body.get("error") or {}
        for detail in (error.get("details") or []):
            if isinstance(detail, dict):
                for violation in (detail.get("violations") or []):
                    if isinstance(violation, dict):
                        blobs.append(
                            f"{violation.get('quotaId', '')} "
                            f"{violation.get('quotaMetric', '')}"
                        )
        blobs.append(str(error.get("message") or ""))
    blobs.append(str(exc))

    text = " ".join(blobs).lower().replace("_", "").replace(" ", "")
    if "perday" in text or "daily" in text:
        return "day"
    if "perminute" in text or "perminue" in text:
        return "minute"
    return None


def _retry_after(exc: Exception) -> float | None:
    """Seconds the server asked us to wait, if it said.

    Preferred over any backoff we invent: the provider knows when its own
    quota window resets. Both the header and the JSON body are checked --
    Gemini puts a retryDelay in the body, while the OpenAI-compatible layer
    may surface a Retry-After header.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
            raw = headers.get(key) if hasattr(headers, "get") else None
            if raw:
                try:
                    return max(0.0, float(str(raw).rstrip("s")))
                except ValueError:
                    pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        details = (body.get("error") or {}).get("details")
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict) and "retryDelay" in item:
                    try:
                        return max(0.0, float(str(item["retryDelay"]).rstrip("s")))
                    except ValueError:
                        pass
    return None



class LLMClient:
    # Class-level defaults, overridden per instance by _degrade(). Declared
    # here rather than only in __init__ so every attribute the request path
    # reads exists on any instance -- including one built with __new__ to
    # avoid opening a connection, which is how the tests drive astream().
    _json_mode_supported: bool = True
    _reasoning_effort_supported: bool = True
    _extra_body_supported: bool = True
    #: Set once a model-not-found error is resolved by toggling the Gemini
    #: `models/` prefix, so the working name is reused instead of paying a
    #: failed call per request. None means "use the configured name".
    _model_override: str | None = None
    _model_retried: bool = False

    def __init__(self) -> None:
        s = get_settings()
        self.settings = s
        # Instrumented classes when Langfuse tracing is on, plain ones
        # otherwise. The drop-in speaks the same protocol, so nothing below
        # changes -- it just records model, parameters, prompt, completion,
        # token counts and cost for every call. Resolved here rather than at
        # import time so the config decides, not import order.
        sync_cls, async_cls = openai_classes()
        # max_retries=0 deliberately: the SDK retries twice by default, so a
        # server that is merely slow turns one 120 s timeout into a silent
        # ~360 s stall that looks like a hang rather than a timeout.
        self._sync = sync_cls(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            timeout=s.llm_timeout,
            max_retries=0,
        )
        self._async = async_cls(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            timeout=s.llm_timeout,
            max_retries=0,
        )
        # The flags above flip to False permanently the first time the server
        # rejects that field, so an unsupported option costs one failed call
        # per process rather than one per request.

    # ---------------- internals ----------------

    def _system(self, system: str) -> str:
        # Qwen-only text switch. Appending it for Gemini would just put a
        # stray token in every system prompt.
        if self.settings.llm_no_think_suffix:
            return system + NO_THINK_SUFFIX
        return system

    def _payload(
        self,
        system: str,
        user: str,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        name: str | None = None,
    ) -> dict:
        s = self.settings
        payload: dict = {
            "model": self._model_override or s.llm_model,
            "messages": [
                {"role": "system", "content": self._system(system)},
                {"role": "user", "content": user},
            ],
            "temperature": s.llm_temperature if temperature is None else temperature,
            # Callers budget for their OUTPUT; the headroom pays for thinking,
            # which is billed against the same ceiling. Without it a caller
            # asking for 700 tokens of JSON can spend all 700 deliberating and
            # return nothing -- see llm_thinking_headroom_tokens.
            "max_tokens": (max_tokens or s.llm_max_tokens)
            + max(0, s.llm_thinking_headroom_tokens),
        }
        if json_mode and self._json_mode_supported:
            payload["response_format"] = {"type": "json_object"}
        # Caps deliberation on providers that honour it. Gemini maps this onto
        # its thinking level; OpenAI reasoning models use it natively; local
        # servers generally 400 on it, which _degrade() handles once.
        if s.llm_reasoning_effort and self._reasoning_effort_supported:
            payload["reasoning_effort"] = s.llm_reasoning_effort
        # Provider options the OpenAI schema has no field for, e.g. Gemini's
        # thinking_config or Ollama's {"think": false}.
        if s.llm_extra_body and self._extra_body_supported:
            payload["extra_body"] = dict(s.llm_extra_body)
        # Labels this generation in Langfuse, so the dashboard shows which
        # agent spent the tokens rather than six indistinguishable calls.
        # Empty when tracing is off -- the PLAIN OpenAI client would reject
        # these keys, so they must not be present unless the instrumented
        # client is the one receiving them.
        payload.update(generation_metadata(name=name))
        return payload

    # The order a 400 is debugged in: give up the pure optimisation first and
    # the load-bearing field last. response_format going means the agents fall
    # back to tolerant JSON parsing, which works but is weaker, so it is the
    # last thing surrendered.
    _DEGRADE_ORDER = (
        ("reasoning_effort", "_reasoning_effort_supported"),
        ("extra_body", "_extra_body_supported"),
        ("response_format", "_json_mode_supported"),
    )

    def _degrade(self, payload: dict) -> str | None:
        """Drop one optional field the server may have rejected.

        Returns the field's name, or None when only required fields are left
        -- in which case the 400 is about the model name or the messages and
        retrying cannot help, so the caller re-raises.
        """
        for field, flag in self._DEGRADE_ORDER:
            if field in payload:
                payload.pop(field)
                setattr(self, flag, False)
                logger.warning(
                    "Server rejected `%s`; dropping it for the rest of this "
                    "process and retrying.", field,
                )
                return field
        return None

    # ---------------- sync ----------------

    # Gemini's /models endpoint returns ids as "models/gemini-3.5-flash-lite",
    # but its OpenAI-compatible chat endpoint wants the bare
    # "gemini-3.5-flash-lite". Anyone who copies a name out of the listing
    # therefore configures a form the chat endpoint 404s on -- and the 404
    # says only that the model was not found, naming no alternative.
    _MODEL_PREFIX = "models/"

    def _retry_other_model_form(self, payload: dict) -> str | None:
        """Toggle the `models/` prefix once. Returns the name tried, or None.

        Only ever tried once per process, and the winning form is remembered
        in _model_override so the cost is one failed call, not one per
        request. Returning None means the name is wrong in a way this cannot
        fix, so the 404 must surface.
        """
        if self._model_retried:
            return None
        self._model_retried = True

        current = str(payload.get("model") or "")
        if current.startswith(self._MODEL_PREFIX):
            alt = current[len(self._MODEL_PREFIX):]
        else:
            alt = self._MODEL_PREFIX + current
        if not alt or alt == current:
            return None

        payload["model"] = alt
        self._model_override = alt
        logger.warning(
            "Model %r was not found; retrying as %r. Set LLM_MODEL=%s in .env "
            "to skip this retry on every future run.", current, alt, alt,
        )
        return alt

    def _record_usage(
        self, payload: dict, resp, seconds: float, name: str | None
    ) -> None:
        """Log this call's tokens to the active usage ledger, if one is open.

        Here rather than at the call sites because every completion passes
        through _create/_acreate, so one hook covers all of them -- including
        any added later. Counts come from the server's `usage` block; an
        earlier version of this project estimated tokens as len/4 and was off
        by ~2x on digit-heavy output, which produced a confident wrong verdict
        about inference speed.
        """
        usage_block = getattr(resp, "usage", None)
        if usage_block is None:
            return
        usage.record(
            name=name,
            model=str(payload.get("model") or ""),
            prompt_tokens=int(getattr(usage_block, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_block, "completion_tokens", 0) or 0),
            seconds=seconds,
        )

    @contextmanager
    def using_model(self, model: str | None):
        """Temporarily answer with a different model.

        Public because evaluation needs it: a judge should not be the model it
        is judging, since they share blind spots. Done by swapping the model
        rather than building a second client so the retry, 400/404 recovery
        and token accounting all still apply -- and so the judge's spend lands
        in the same ledger.

        Restores the previous value even on error, and is a no-op when `model`
        is falsy or already current.
        """
        if not model or model == (self._model_override or self.settings.llm_model):
            yield self
            return
        previous = self._model_override
        self._model_override = model
        try:
            yield self
        finally:
            self._model_override = previous

    def _gate(self) -> None:
        """Hold back if we are over the configured requests-per-minute."""
        _GATE.wait(self.settings.llm_max_requests_per_minute)

    def _retry_delay(
        self, exc: APIStatusError, attempt: int, name: str | None
    ) -> float | None:
        """Seconds to wait before retrying, or None to give up and raise.

        Exponential backoff with FULL JITTER. The jitter is not decoration:
        several agents hitting the same quota retry in lockstep otherwise, so
        they collide again on every attempt and the backoff achieves nothing.
        Randomising the whole interval spreads them out.

        The server's own Retry-After wins when present -- it knows when its
        quota window resets -- capped so a provider asking for fifteen minutes
        surfaces as a visible failure rather than a process that looks hung.
        """
        status = getattr(exc, "status_code", None)
        if status not in _RETRYABLE_STATUS:
            return None

        s = self.settings
        if status == 429 and quota_scope(exc) == "day":
            # Do NOT retry. The daily allowance is gone; every further attempt
            # consumes more of it and cannot succeed. Failing immediately also
            # lets a batch job notice and stop, instead of spending twenty
            # minutes of backoff discovering the same thing.
            logger.error(
                "%s: DAILY quota exhausted (HTTP 429). Not retrying -- this "
                "does not reset for hours, and each retry spends more of it.\n"
                "         Options: wait for the reset, use a key with a paid "
                "tier, or run a smaller\n         slice "
                "(scripts\\evaluate_rag.py --limit N --resume).",
                name or "LLM call",
            )
            return None

        if attempt >= s.llm_max_retries:
            scope = quota_scope(exc) if status == 429 else None
            if status == 429:
                logger.error(
                    "%s: HTTP 429 after %d retries, giving up (%s quota). "
                    "Lower the rate: LLM_MAX_REQUESTS_PER_MINUTE, or "
                    "--rpm for one run.",
                    name or "LLM call", attempt, scope or "unlabelled",
                )
            else:
                logger.error(
                    "%s: HTTP %s after %d retries, giving up. The provider is "
                    "overloaded; this is not something your configuration can "
                    "fix.", name or "LLM call", status, attempt,
                )
            return None

        asked = _retry_after(exc)
        if asked is not None:
            delay = min(asked, s.llm_retry_max_delay)
            source = "server asked"
        else:
            ceiling = min(
                s.llm_retry_base_delay * (2 ** attempt), s.llm_retry_max_delay
            )
            delay = random.uniform(0, ceiling)
            source = "backoff"
        logger.warning(
            "%s: HTTP %s, retrying in %.1fs (%s, attempt %d/%d)",
            name or "LLM call", status, delay, source,
            attempt + 1, s.llm_max_retries,
        )
        return delay

    def _create(self, payload: dict, name: str | None = None):
        """POST the payload, recovering from the two fixable rejections.

        400 -> shed one unsupported optional field and retry.
        404 -> try the other spelling of the model name, once.
        Both are bounded, so an endpoint that rejects everything fails with
        its original error rather than looping.
        """
        attempt = 0
        while True:
            self._gate()
            started = time.perf_counter()
            try:
                resp = self._sync.chat.completions.create(**payload)
            except NotFoundError:
                if self._retry_other_model_form(payload) is None:
                    raise
            except BadRequestError:
                if self._degrade(payload) is None:
                    raise
            except APIStatusError as exc:
                delay = self._retry_delay(exc, attempt, name)
                if delay is None:
                    raise
                attempt += 1
                time.sleep(delay)
            else:
                self._record_usage(
                    payload, resp, time.perf_counter() - started, name
                )
                return resp

    async def _acreate(self, payload: dict, name: str | None = None):
        attempt = 0
        while True:
            self._gate()
            started = time.perf_counter()
            try:
                resp = await self._async.chat.completions.create(**payload)
            except NotFoundError:
                if self._retry_other_model_form(payload) is None:
                    raise
            except BadRequestError:
                if self._degrade(payload) is None:
                    raise
            except APIStatusError as exc:
                delay = self._retry_delay(exc, attempt, name)
                if delay is None:
                    raise
                attempt += 1
                # asyncio.sleep, not time.sleep: blocking the loop here would
                # stall every other coroutine, including the request that is
                # about to succeed.
                import asyncio

                await asyncio.sleep(delay)
            else:
                self._record_usage(
                    payload, resp, time.perf_counter() - started, name
                )
                return resp

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        name: str | None = None,
    ) -> str:
        payload = self._payload(system, user, temperature, max_tokens, json_mode, name)
        resp = self._create(payload, name)
        return _finish(resp.choices[0].message, payload.get("max_tokens", 0))

    def complete_verbose(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        name: str | None = None,
    ) -> tuple[str, int | None]:
        """(text, completion_tokens). Token count comes from the server's own
        usage block, so throughput measurements do not depend on guessing how
        the text tokenizes -- a chars/4 estimate is roughly 2x wrong for
        digit- and punctuation-heavy output."""
        payload = self._payload(system, user, temperature, max_tokens, json_mode, name)
        resp = self._create(payload, name)
        text = _finish(resp.choices[0].message, payload.get("max_tokens", 0))
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "completion_tokens", None) if usage else None
        return text, tokens

    # ---------------- async ----------------

    async def acomplete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        name: str | None = None,
    ) -> str:
        payload = self._payload(system, user, temperature, max_tokens, json_mode, name)
        resp = await self._acreate(payload, name)
        return _finish(resp.choices[0].message, payload.get("max_tokens", 0))

    async def astream(self, system: str, user: str):
        """Streaming cannot strip a <think> block that spans chunks, so we
        suppress output until the block closes rather than leaking it."""
        # Through _payload, so streaming gets the same reasoning_effort and
        # provider options as every other call rather than its own defaults.
        payload = self._payload(system, user, None, None, json_mode=False)
        payload["stream"] = True
        stream = await self._acreate(payload)
        OPEN, CLOSE = "<think>", "</think>"
        buffer = ""
        in_think = False

        def partial_tail(text: str, tag: str) -> int:
            """Length of the tail of `text` that could be a prefix of `tag`.

            A tag can straddle chunk boundaries ("<thi" + "nk>"), so that many
            characters are held back rather than emitted.
            """
            for i in range(min(len(text), len(tag) - 1), 0, -1):
                if tag.startswith(text[-i:].lower()):
                    return i
            return 0

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            buffer += delta
            emit = ""

            # Drain the buffer through the state machine until neither a
            # complete opener nor closer remains. A single chunk can contain
            # both, so this must loop rather than handle one and move on.
            while True:
                if in_think:
                    idx = buffer.lower().find(CLOSE)
                    if idx == -1:
                        # Discard thinking text, but keep any partial closer.
                        keep = partial_tail(buffer, CLOSE)
                        buffer = buffer[len(buffer) - keep :] if keep else ""
                        break
                    buffer = buffer[idx + len(CLOSE) :]
                    in_think = False
                    continue

                idx = buffer.lower().find(OPEN)
                if idx == -1:
                    break
                emit += buffer[:idx]
                buffer = buffer[idx + len(OPEN) :]
                in_think = True

            if not in_think:
                keep = partial_tail(buffer, OPEN)
                safe = len(buffer) - keep
                emit += buffer[:safe]
                buffer = buffer[safe:]

            if emit:
                yield emit

        if buffer and not in_think:
            yield buffer

    # ---------------- diagnostics ----------------

    def health(self) -> bool:
        try:
            self._sync.models.list()
            return True
        except Exception:
            return False

    def available_models(self) -> list[str]:
        try:
            return [m.id for m in self._sync.models.list().data]
        except Exception:
            return []


#: Historical name from when this client only ever spoke to a local Qwen.
#: Kept so nothing outside this module has to care what it talks to.
QwenClient = LLMClient


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
