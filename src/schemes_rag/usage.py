"""Token and cost accounting.

Records what every LLM call actually consumed, so "cost per request" is a
measurement rather than an estimate. Two pieces:

  LEDGER   a per-call record of prompt/completion tokens, labelled with the
           agent that made the call. Collected per request via a context
           manager, because one user question fans out into several calls
           (router, HyDE, assessor, answer, verifier) and the interesting
           number is their sum.

  PRICING  tokens -> currency. Rates are CONFIG, not constants: they differ
           per model, change without notice, and a hardcoded number that goes
           stale silently is worse than one you had to type. See
           llm_price_input_per_mtok in config.py.

WHY NOT READ THIS BACK FROM LANGFUSE

Langfuse already records tokens and cost per generation, and for day-to-day
observability that is the right place to look. But an eval run needs the
numbers synchronously, attached to the question that produced them, with no
network round trip and no dependency on tracing being enabled. The ledger is
~60 lines and makes the evaluator self-contained.

Counts come from the server's own `usage` block. Estimating tokens by
dividing characters by four was already wrong once in this project -- it
undercounted a digit-heavy response by about half and produced a confident,
false "inference is CPU-slow" verdict.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class CallUsage:
    """One LLM call."""

    #: The agent that made it, from the `name` passed to the client
    #: (e.g. "eligibility.assess"). None when a caller did not label itself.
    name: str | None
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Wall-clock seconds, which is not the same as tokens/second when the
    #: model spends part of the budget thinking.
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageLedger:
    """Calls made inside one collect() block."""

    calls: list[CallUsage] = field(default_factory=list)

    def add(self, call: CallUsage) -> None:
        self.calls.append(call)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def llm_seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    def by_agent(self) -> dict[str, dict[str, int]]:
        """Tokens grouped by the calling agent.

        The point of labelling calls: this answers "which agent is expensive",
        which is the question that actually changes a decision. Six agents
        reported as one undifferentiated total tells you nothing you can act
        on.
        """
        out: dict[str, dict[str, int]] = {}
        for call in self.calls:
            bucket = out.setdefault(
                call.name or "(unlabelled)",
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += call.prompt_tokens
            bucket["completion_tokens"] += call.completion_tokens
        return out

    def cost(self, price_in: float, price_out: float) -> float:
        """Cost in the currency of the rates given, which are per 1M tokens.

        Input and output are priced separately because they differ by roughly
        4x on most providers -- a single blended rate would misattribute the
        cost of a RAG system, where prompts are large (20k tokens of retrieved
        context) and completions are small.
        """
        return (
            self.prompt_tokens / 1_000_000 * price_in
            + self.completion_tokens / 1_000_000 * price_out
        )


# Thread-local, not a module global: the API serves concurrent requests, and a
# shared ledger would bill one user's question for another's tokens.
_local = threading.local()


def _current() -> UsageLedger | None:
    return getattr(_local, "ledger", None)


def record(
    name: str | None, model: str, prompt_tokens: int, completion_tokens: int,
    seconds: float = 0.0,
) -> None:
    """Add a call to the active ledger, if any.

    A no-op outside a collect() block, so instrumentation in llm.py costs one
    attribute lookup in normal operation.
    """
    ledger = _current()
    if ledger is not None:
        ledger.add(CallUsage(
            name=name, model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, seconds=seconds,
        ))


@contextmanager
def collect() -> Iterator[UsageLedger]:
    """Collect usage for everything inside the block.

    Nested blocks are independent: the inner one collects its own calls and
    the outer does not see them. That is deliberate -- nesting here would mean
    double-counting, and no caller wants a sum that includes a subtotal.
    """
    previous = _current()
    ledger = UsageLedger()
    _local.ledger = ledger
    try:
        yield ledger
    finally:
        _local.ledger = previous
