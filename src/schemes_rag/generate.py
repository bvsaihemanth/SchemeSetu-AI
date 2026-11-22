"""Answer synthesis. Generation only.

Citation resolution and verification used to live here; they now belong to the
Evidence Verification Agent (agents/evidence.py), so provenance has one owner
instead of being split between a helper here and a disabled graph node.

What remains is the part that is genuinely about generating: build the prompt,
call the model, and turn its failure modes into something the caller can act
on. An empty completion with sources in hand is a model failure, not an
absence of evidence, and must never surface as a blank answer.
"""

from __future__ import annotations

import logging
import re
import time

from .llm import ThinkingOnlyResponse, get_llm
from .models import Answer, RetrievedChunk
from .prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER_TEMPLATE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    format_context,
)

logger = logging.getLogger(__name__)

_REFUSAL = re.compile(r"INSUFFICIENT_EVIDENCE", re.IGNORECASE)


def insufficient_answer(route: str | None = None) -> Answer:
    return Answer(
        answer=INSUFFICIENT_EVIDENCE_MESSAGE,
        citations=[],
        insufficient_evidence=True,
        route=route,
    )


def _model_failure(message: str, route: str | None, gen_ms: float) -> Answer:
    return Answer(
        answer=message,
        citations=[],
        insufficient_evidence=False,
        # NOT insufficient_evidence: retrieval worked, the model call did not.
        # An evaluator must be able to exclude this rather than average it in.
        model_failed=True,
        route=route,
        latency_ms={"generate_ms": gen_ms},
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    route: str | None = None,
    system: str | None = None,
    context_text: str | None = None,
    user_template: str | None = None,
    max_tokens: int | None = None,
) -> Answer:
    """Synthesise an answer over `chunks`.

    `system`, `context_text` and `user_template` let each agent supply its own
    prompt and its own context layout -- the Comparison agent groups sources
    by scheme rather than listing them flat -- without duplicating the
    generation, failure-handling and citation plumbing.
    """
    if not chunks:
        return insufficient_answer(route)

    system = system or ANSWER_SYSTEM
    template = user_template or ANSWER_USER_TEMPLATE
    context = context_text if context_text is not None else format_context(chunks)
    user = template.format(context=context, question=question.strip())

    t0 = time.perf_counter()
    try:
        # Named per route, because every section agent funnels through this
        # one function: without it the dashboard shows one "generate.answer"
        # for comparison, application, documents and benefits alike, and the
        # cost of each is unknowable.
        raw = get_llm().complete(
            system, user, max_tokens=max_tokens,
            name=f"generate.{route or 'answer'}",
        )
    except ThinkingOnlyResponse as exc:
        gen_ms = (time.perf_counter() - t0) * 1000
        logger.error("Generation produced only reasoning: %s", exc)
        return _model_failure(
            "The language model did not produce an answer: it spent its entire "
            "token budget on internal reasoning. Retrieval worked — sources "
            "were found — so this is a model configuration issue. Run "
            "scripts/check_llm.py.",
            route, gen_ms,
        )
    except Exception as exc:
        gen_ms = (time.perf_counter() - t0) * 1000
        logger.exception("Generation failed: %s", exc)
        return _model_failure(
            f"The language model call failed ({type(exc).__name__}). Retrieval "
            f"found {len(chunks)} sources, so this is a serving problem. Run "
            f"scripts/check_llm.py.",
            route, gen_ms,
        )
    gen_ms = (time.perf_counter() - t0) * 1000

    if not raw.strip():
        logger.error(
            "LLM returned empty output for %r after %.0f ms with %d sources",
            question[:80], gen_ms, len(chunks),
        )
        return _model_failure(
            f"The language model returned an empty response. Retrieval found "
            f"{len(chunks)} relevant sources, so this is a model or serving "
            f"problem rather than a retrieval one. Run scripts/check_llm.py.",
            route, gen_ms,
        )

    if _REFUSAL.search(raw):
        answer = insufficient_answer(route)
        answer.latency_ms = {"generate_ms": gen_ms}
        return answer

    # Provenance is the Evidence agent's job.
    from .agents.evidence import resolve_citations

    text, citations = resolve_citations(raw, chunks)

    return Answer(
        answer=text,
        citations=citations,
        insufficient_evidence=False,
        route=route,
        schemes_used=list(dict.fromkeys(rc.chunk.scheme_name for rc in chunks)),
        latency_ms={"generate_ms": gen_ms},
    )
