"""LLM judges for the reference-free metrics.

Each judge follows the decomposition approach the Ragas metrics use, because
asking a model to score a whole answer 1-5 produces a number that drifts
between runs and cannot be argued with. Breaking the judgement into atomic
yes/no decisions about individual claims gives a score that is a COUNT, is
reproducible at temperature 0, and -- most usefully -- comes with the
per-claim verdicts, so a bad score tells you which sentence was unsupported
rather than just that something was.

WHY NOT RAGAS ITSELF

Ragas implements these same prompts and is the reference for the definitions
used here. It is not a dependency because it would need its own LLM and
embedding wrappers alongside the ones this project already has -- duplicating
the provider config, the retry and degradation logic in llm.py, and the token
accounting that makes cost-per-request measurable. For five metrics that are
each a prompt and a count, that trade is not worth it. The definitions are
matched deliberately so numbers remain comparable to published Ragas results.

JUDGE INDEPENDENCE

eval_judge_model defaults to the answering model, which is the weakest form of
this: a model grading its own output shares its blind spots, and will tend to
find its own hallucinations plausible. Point it at a stronger model when the
numbers matter. The report always states which model judged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

# Imported as a MODULE, not `from ..llm import get_llm`: a module-level
# binding is captured at import time, so a harness that swaps the client
# afterwards would be silently ignored here and the judges would call the real
# API with real money. Late binding makes the eval pick up whatever client the
# process has configured, whatever the import order.
from .. import llm as llm_mod

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _parse(raw: str, default):
    """Tolerant JSON parse. Judges are asked for JSON; models add prose."""
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    logger.debug("Judge returned unparseable output: %s", raw[:200])
    return default


def _ask(system: str, user: str, name: str, model: str | None) -> str:
    """One judge call, at temperature 0 and in JSON mode.

    The model swap goes through the client's own using_model(), not by poking
    at its attributes: a judge has no business knowing the client's internals,
    and a client that does not offer the hook (a test double, say) must still
    work rather than raise AttributeError mid-evaluation.
    """
    llm = llm_mod.get_llm()
    kwargs = {"temperature": 0.0, "json_mode": True, "max_tokens": 1400,
              "name": name}
    switch = getattr(llm, "using_model", None)
    if model and callable(switch):
        with switch(model):
            return llm.complete(system, user, **kwargs)
    if model:
        logger.debug(
            "LLM client has no using_model(); judging with the configured "
            "model instead of %s", model,
        )
    return llm.complete(system, user, **kwargs)


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------
DECOMPOSE_SYSTEM = """You break text apart for evaluation. Return JSON only.

Given a QUESTION and an ANSWER, return:
{
  "claims": ["...", "..."],
  "aspects": ["...", "..."]
}

claims: every factual assertion the ANSWER makes, each as one standalone
sentence that can be checked on its own. Resolve pronouns and carry context
in, so "It provides Rs 6,000" becomes "PM-KISAN provides Rs 6,000 per year".
Split sentences that assert two things. Numbers, amounts, dates, percentages
and eligibility conditions are each a separate claim -- they are what matters
most and what goes wrong most. Exclude hedges, offers to help, and citation
markers like [1]. If the answer asserts nothing, return [].

aspects: what the QUESTION asks for, one item per distinct thing it requests.
"What is the premium rate and coverage limit under PMSBY?" asks for two: the
premium rate, and the coverage limit. Most questions ask for one. Base these
ONLY on the question -- do not let the answer's shape decide what was asked.
Keep each aspect to a short noun phrase."""

DECOMPOSE_USER = """QUESTION:
{question}

ANSWER:
{answer}

JSON:"""


@dataclass
class Decomposition:
    claims: list[str] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)


def decompose(question: str, answer: str, model: str | None = None) -> Decomposition:
    """Answer claims and question aspects, in one call.

    Together because they are needed together and the prompt is cheaper once:
    at 85 questions a saved call per question is a saved minute and a saved
    fraction of a cent, and the inputs overlap entirely.
    """
    raw = _ask(
        DECOMPOSE_SYSTEM,
        DECOMPOSE_USER.format(question=question, answer=answer[:6000]),
        "eval.decompose", model,
    )
    data = _parse(raw, {})
    if not isinstance(data, dict):
        return Decomposition()
    return Decomposition(
        claims=[str(c).strip() for c in (data.get("claims") or []) if str(c).strip()],
        aspects=[str(a).strip() for a in (data.get("aspects") or []) if str(a).strip()],
    )


# --------------------------------------------------------------------------
# faithfulness / groundedness
# --------------------------------------------------------------------------
FAITHFUL_SYSTEM = """You check whether claims are supported by source text.

For each numbered CLAIM, decide whether the CONTEXT supports it. Return JSON:
{"verdicts": [{"i": 1, "supported": true, "reason": "..."}]}

supported = true only if the CONTEXT states it or it follows directly from
what the CONTEXT states. Rules that decide the hard cases:

- A number, amount, percentage, age limit or date must appear in the CONTEXT
  with the same value. Close is not supported. A claim of "Rs 6,000 per year"
  against a context saying "Rs 2,000 per instalment" is NOT supported even
  though the arithmetic works, unless the context also gives the instalment
  count.
- General knowledge you happen to hold is NOT support. Judge only against the
  CONTEXT given. A true statement absent from the context is unsupported.
- A claim that the context does not mention something IS supported if the
  context indeed does not mention it.
- Keep `reason` under 20 words, and quote the deciding phrase when unsupported.

Return exactly one verdict per claim, in order. JSON only."""

FAITHFUL_USER = """CONTEXT:
{context}

CLAIMS:
{claims}

JSON:"""


def judge_faithfulness(
    claims: list[str], context: str, model: str | None = None
) -> list[dict]:
    """Per-claim support verdicts against the retrieved context.

    This is the groundedness measurement, and the one that matters most for
    this application: the brief was that it must say "not found" rather than
    invent, and an unsupported claim is exactly that failure.
    """
    if not claims:
        return []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    raw = _ask(
        FAITHFUL_SYSTEM,
        FAITHFUL_USER.format(context=context[:24000], claims=numbered),
        "eval.faithfulness", model,
    )
    data = _parse(raw, {})
    verdicts = data.get("verdicts") if isinstance(data, dict) else data
    out: list[dict] = []
    for i, claim in enumerate(claims, 1):
        found = None
        if isinstance(verdicts, list):
            for v in verdicts:
                if isinstance(v, dict) and int(v.get("i", -1) or -1) == i:
                    found = v
                    break
        out.append({
            "claim": claim,
            # Unparsed verdicts count as UNSUPPORTED. A judge failure must not
            # inflate the score -- the safe direction for a groundedness
            # metric is pessimism.
            "supported": bool(found.get("supported")) if found else False,
            "reason": (found or {}).get("reason") or ("" if found else "no verdict returned"),
            "judged": found is not None,
        })
    return out


# --------------------------------------------------------------------------
# per-chunk relevance and usefulness
# --------------------------------------------------------------------------
CHUNKS_SYSTEM = """You rate retrieved passages. Return JSON only:
{"ratings": [{"i": 1, "relevant": true, "useful": true}]}

For each numbered PASSAGE, two independent judgements:

relevant: is the passage on the topic the QUESTION asks about? A passage about
the right scheme but the wrong section is still relevant.

useful: did this passage contain information needed to produce the ANSWER
given? A passage can be relevant but not useful -- about the right scheme,
contributing nothing the answer used. Be strict: `useful` is about whether
dropping it would have changed the answer.

Exactly one rating per passage, in order. JSON only."""

CHUNKS_USER = """QUESTION:
{question}

ANSWER:
{answer}

PASSAGES:
{passages}

JSON:"""


def judge_chunks(
    question: str, answer: str, passages: list[str], model: str | None = None
) -> list[dict]:
    """Per-chunk relevance and usefulness.

    Two verdicts from one call because they share all their input and differ
    only in the question asked of it. `relevant` feeds context relevance,
    `useful` feeds context precision -- which needs per-rank usefulness to
    compute average precision, not just a count.
    """
    if not passages:
        return []
    numbered = "\n\n".join(
        f"[{i}] {p[:2000]}" for i, p in enumerate(passages, 1)
    )
    raw = _ask(
        CHUNKS_SYSTEM,
        CHUNKS_USER.format(
            question=question, answer=answer[:3000], passages=numbered,
        ),
        "eval.chunks", model,
    )
    data = _parse(raw, {})
    ratings = data.get("ratings") if isinstance(data, dict) else data
    out: list[dict] = []
    for i in range(1, len(passages) + 1):
        found = None
        if isinstance(ratings, list):
            for r in ratings:
                if isinstance(r, dict) and int(r.get("i", -1) or -1) == i:
                    found = r
                    break
        out.append({
            "rank": i,
            "relevant": bool(found.get("relevant")) if found else False,
            "useful": bool(found.get("useful")) if found else False,
            "judged": found is not None,
        })
    return out


# --------------------------------------------------------------------------
# completeness (question-aspect coverage)
# --------------------------------------------------------------------------
ASPECTS_SYSTEM = """You check whether an answer addresses what was asked.

For each numbered ASPECT the question requested, decide whether the ANSWER
addresses it. Return JSON:
{"verdicts": [{"i": 1, "addressed": true}]}

addressed = true if the answer gives that information, OR explicitly states
that the sources do not contain it. Saying "the documents do not specify the
interest rate" IS addressing the interest rate -- an honest gap is a complete
answer to a question the corpus cannot answer, and must not be scored the same
as silence.

addressed = false if the answer simply does not mention it.

One verdict per aspect, in order. JSON only."""

ASPECTS_USER = """ASPECTS REQUESTED:
{aspects}

ANSWER:
{answer}

JSON:"""


def judge_completeness(
    aspects: list[str], answer: str, model: str | None = None
) -> list[dict]:
    """Which of the question's aspects the answer actually addresses."""
    if not aspects:
        return []
    numbered = "\n".join(f"{i}. {a}" for i, a in enumerate(aspects, 1))
    raw = _ask(
        ASPECTS_SYSTEM,
        ASPECTS_USER.format(aspects=numbered, answer=answer[:6000]),
        "eval.completeness", model,
    )
    data = _parse(raw, {})
    verdicts = data.get("verdicts") if isinstance(data, dict) else data
    out: list[dict] = []
    for i, aspect in enumerate(aspects, 1):
        found = None
        if isinstance(verdicts, list):
            for v in verdicts:
                if isinstance(v, dict) and int(v.get("i", -1) or -1) == i:
                    found = v
                    break
        out.append({
            "aspect": aspect,
            "addressed": bool(found.get("addressed")) if found else False,
            "judged": found is not None,
        })
    return out


# --------------------------------------------------------------------------
# context recall (needs a reference answer)
# --------------------------------------------------------------------------
RECALL_SYSTEM = """You check whether retrieved context covers a reference answer.

Break the REFERENCE ANSWER into its factual claims, then for each decide
whether the CONTEXT supports it. Return JSON:
{"claims": [{"claim": "...", "attributed": true}]}

attributed = true only if the CONTEXT contains that information. This measures
whether RETRIEVAL found what was needed -- not whether the system's own answer
was good -- so judge only the context, never your own knowledge. JSON only."""

RECALL_USER = """CONTEXT:
{context}

REFERENCE ANSWER:
{reference}

JSON:"""


def judge_context_recall(
    reference: str, context: str, model: str | None = None
) -> list[dict]:
    """Which reference-answer claims the retrieved context covers.

    The one metric here that genuinely needs a human-written reference. Asked
    for, so implemented -- but it reports zero coverage until reference
    answers exist in the gold file, rather than quietly substituting the
    system's own answer, which would make the metric measure nothing.
    """
    if not reference:
        return []
    raw = _ask(
        RECALL_SYSTEM,
        RECALL_USER.format(context=context[:24000], reference=reference[:4000]),
        "eval.context_recall", model,
    )
    data = _parse(raw, {})
    claims = data.get("claims") if isinstance(data, dict) else data
    out: list[dict] = []
    if isinstance(claims, list):
        for c in claims:
            if isinstance(c, dict) and c.get("claim"):
                out.append({
                    "claim": str(c["claim"]),
                    "attributed": bool(c.get("attributed")),
                })
    return out


# --------------------------------------------------------------------------
# against the reference answer (needs tests/test-ans.txt)
# --------------------------------------------------------------------------
REFERENCE_SYSTEM = """You compare an answer against a known-correct reference.

Return JSON:
{
  "coverage": [{"claim": "...", "present": true}],
  "conflicts": [{"claim": "...", "verdict": "agrees|contradicts|unverifiable",
                 "detail": "..."}]
}

coverage: break the REFERENCE into its factual claims, then for each decide
whether the ANSWER contains that information. This measures what the answer
MISSED. A claim the answer states differently is still "present" -- whether it
is right is the other list's job.

conflicts: take the ANSWER's factual claims and judge each against the
REFERENCE:
  contradicts   the reference says something incompatible. Different numbers,
                amounts, ages, percentages or dates for the same thing are
                contradictions, not variations.
  agrees        the reference supports it.
  unverifiable  the reference neither supports nor contradicts it. Extra
                detail the reference does not cover is unverifiable, NOT a
                contradiction -- the reference is not exhaustive.

Keep `detail` under 25 words, and for a contradiction state both values.
JSON only."""

REFERENCE_USER = """REFERENCE ANSWER (treat as correct):
{reference}

ANSWER UNDER TEST:
{answer}

JSON:"""


def judge_against_reference(
    reference: str, answer: str, model: str | None = None
) -> tuple[list[dict], list[dict]]:
    """(coverage, conflicts) -- the two directions of comparison.

    One call for both because they share the whole input and differ only in
    which text is the subject. At 85 questions that is 85 saved calls.

    The two directions answer different questions and neither substitutes for
    the other:

      coverage   what the answer LEFT OUT. A terse but correct answer scores
                 badly here, which is the point -- "Compare PMJJBY and PMSBY"
                 has five dimensions in its reference, and an answer giving
                 one of them is not a good answer.
      conflicts  what the answer got WRONG. This is the gap faithfulness
                 cannot see: faithfulness checks the answer against the
                 RETRIEVED CONTEXT, so an answer that faithfully reports the
                 wrong document scores a perfect 1.00 while being incorrect.
                 Only a reference catches that.
    """
    if not reference or not answer:
        return [], []
    raw = _ask(
        REFERENCE_SYSTEM,
        REFERENCE_USER.format(reference=reference[:8000], answer=answer[:8000]),
        "eval.reference", model,
    )
    data = _parse(raw, {})
    if not isinstance(data, dict):
        return [], []

    coverage: list[dict] = []
    for c in data.get("coverage") or []:
        if isinstance(c, dict) and c.get("claim"):
            coverage.append({
                "claim": str(c["claim"]),
                "present": bool(c.get("present")),
            })
    conflicts: list[dict] = []
    for c in data.get("conflicts") or []:
        if isinstance(c, dict) and c.get("claim"):
            verdict = str(c.get("verdict") or "unverifiable").lower()
            if verdict not in ("agrees", "contradicts", "unverifiable"):
                verdict = "unverifiable"
            conflicts.append({
                "claim": str(c["claim"]),
                "verdict": verdict,
                "detail": str(c.get("detail") or ""),
            })
    return coverage, conflicts
