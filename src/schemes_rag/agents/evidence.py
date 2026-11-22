"""Evidence Verification Agent — owns provenance end to end.

This agent now holds the citation logic that used to live in generate.py, and
replaces the old LLM verifier that cost 116 seconds and returned unparseable
output. Three layers, cheapest first:

1. CITATION RESOLUTION (deterministic, ~0 ms)
   Map the [n] markers the model wrote onto the chunks actually retrieved.
   Markers outside range are stripped: a citation the user cannot follow is
   worse than no citation.

2. NUMERIC GROUNDING (deterministic, ~1 ms)   <-- the important one
   Every figure in the answer must appear in a cited chunk. In this domain the
   dangerous hallucinations are numeric: "Rs 48 lakh life insurance cover",
   "2 hectares", "18 to 40 years", "15% value addition". A wrong figure here
   sends someone to the wrong scheme or sets a false expectation, and it is
   exactly the error type a fluent 4B produces most readily. Checking it is
   arithmetic, not inference, so it costs nothing and never fails to parse.

3. LLM CLAIM CHECK (optional, one call)
   Only for sentences that carry a factual claim with NO citation at all, and
   only when verify_enabled is on. Reserved for high-stakes routes.

The division matters: layers 1 and 2 run on every answer because they are
free; layer 3 is opt-in because on this hardware it doubles latency.
"""

from __future__ import annotations

import logging
import re

from ..models import Answer, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d{1,2})\]")

# Figures worth checking. Deliberately not every integer: list numbering and
# section references ("para 4.07") are not claims about the world.
_NUMBER = re.compile(
    r"""(?:
        (?:₹|rs\.?|inr)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakh|crore|thousand))?   # money
      | [\d,]+(?:\.\d+)?\s*(?:%|per\s*cent|percent)                          # percentages
      | [\d,]+(?:\.\d+)?\s*(?:hectare|hectares|ha|acre|acres)               # land
      | [\d,]+(?:\.\d+)?\s*(?:year|years|month|months|day|days)             # durations
      | \b\d{4,}\b                                                          # bare large numbers
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Sentences that assert nothing checkable.
_HEDGE = re.compile(
    r"^\s*(?:however|note that|please|you (?:should|may|can)|"
    r"i (?:could|cannot|can't)|this (?:answer|information)|"
    r"based on|according to the sources?)\b",
    re.IGNORECASE,
)


def _normalise_number(token: str) -> str:
    """Canonical form so '₹48 lakh', 'Rs. 48 Lakh' and '48 lakh' compare equal."""
    t = token.lower()
    t = re.sub(r"(?:₹|rs\.?|inr)", "", t)
    t = re.sub(r"per\s*cent|percent", "%", t)
    t = re.sub(r"hectares?|\bha\b", "ha", t)
    t = re.sub(r"acres?", "acre", t)
    t = re.sub(r"years?", "year", t)
    t = re.sub(r"months?", "month", t)
    t = re.sub(r"days?", "day", t)
    t = t.replace(",", "")
    return re.sub(r"\s+", "", t).strip()


def _numeric_values(text: str) -> set[str]:
    return {_normalise_number(m.group(0)) for m in _NUMBER.finditer(text)}


def resolve_citations(
    text: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Map [n] markers to real chunks and strip invented ones.

    A small model occasionally writes [7] when handed 6 sources. Leaving it in
    gives the user a reference they cannot check, so it is removed from the
    prose rather than silently rendered.
    """
    used: dict[int, Citation] = {}
    valid = range(1, len(chunks) + 1)

    for match in _MARKER.finditer(text):
        n = int(match.group(1))
        if n in valid and n not in used:
            rc = chunks[n - 1]
            c = rc.chunk
            used[n] = Citation(
                marker=n,
                scheme_name=c.scheme_name,
                section=c.section.value,
                locator=c.locator(),
                source_url=c.source_url,
                doc_name=c.doc_name,
                score=round(rc.effective_score, 4),
            )

    def keep_valid(m: re.Match) -> str:
        return m.group(0) if int(m.group(1)) in valid else ""

    cleaned = _MARKER.sub(keep_valid, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()

    return cleaned, [used[k] for k in sorted(used)]


def check_numeric_grounding(
    text: str, chunks: list[RetrievedChunk], user_text: str = ""
) -> tuple[list[str], list[str]]:
    """(grounded, ungrounded) figures from the answer.

    A figure is grounded when it appears in ANY retrieved chunk, not only the
    one cited beside it: models frequently cite the section a fact came from
    while the exact amount sits in the adjacent FAQ. Checking against the whole
    retrieved context is the honest test of "did we read this somewhere".

    `user_text` carries the question and the known profile, and its figures
    count as grounded too. Without it an eligibility answer that echoes the
    user's own input -- "based on: landholding 0.81 ha" -- gets flagged as a
    hallucination, because 0.81 ha appears nowhere in the scheme document. The
    user is a valid source for facts about the user.
    """
    if not chunks:
        return [], sorted(_numeric_values(text))

    corpus = " ".join(rc.chunk.text for rc in chunks)
    available = _numeric_values(corpus) | _numeric_values(user_text)

    grounded: list[str] = []
    ungrounded: list[str] = []
    for value in sorted(_numeric_values(text)):
        (grounded if value in available else ungrounded).append(value)
    return grounded, ungrounded


def find_uncited_claims(text: str) -> list[str]:
    """Sentences that assert something but carry no [n] marker."""
    out: list[str] = []
    for sentence in _SENTENCE.split(text):
        s = sentence.strip(" -•\t")
        if len(s.split()) < 6 or _HEDGE.match(s):
            continue
        if not _MARKER.search(s):
            out.append(s)
    return out


def verify(
    answer: Answer, chunks: list[RetrievedChunk], user_text: str = ""
) -> dict:
    """Run the deterministic layers and annotate the answer.

    Returns a report; mutates `answer` only to append a caveat when a figure
    cannot be traced to the retrieved sources or to the user's own input.
    """
    combined = " ".join(filter(None, [user_text, answer.profile_summary or ""]))
    grounded, ungrounded = check_numeric_grounding(answer.answer, chunks, combined)

    # An eligibility assessment is rendered deterministically from the
    # criteria, not generated, so it carries no [n] markers by design. Auditing
    # it for missing citations reports a problem that does not exist.
    uncited = [] if answer.assessment is not None else find_uncited_claims(answer.answer)

    report = {
        "figures_checked": len(grounded) + len(ungrounded),
        "figures_grounded": grounded,
        "figures_ungrounded": ungrounded,
        "uncited_claims": uncited,
        "citations": len(answer.citations),
        "verdict": "grounded",
    }

    if ungrounded:
        report["verdict"] = "ungrounded_figures"
        logger.warning(
            "Figures not found in retrieved sources: %s (question route=%s)",
            ungrounded, answer.route,
        )
        pretty = ", ".join(ungrounded[:4])
        answer.answer += (
            f"\n\n_Caution: these figures could not be traced to the source "
            f"documents ({pretty}). Verify them on the official scheme page "
            f"before relying on them._"
        )
    elif not answer.citations and uncited:
        report["verdict"] = "uncited"
        logger.warning("Answer carries %d uncited claim(s)", len(uncited))

    answer.verification = report
    return report


def verify_with_llm(answer: Answer, chunks: list[RetrievedChunk]) -> dict:
    """Layer 3. Only called when verify_enabled is on.

    Kept separate so the cheap layers always run and this one is a deliberate
    opt-in: on a local 4B it roughly doubles latency, and judging faithfulness
    is a harder task than answering in the first place.
    """
    import json

    from ..llm import get_llm
    from ..prompts import VERIFIER_SYSTEM, VERIFIER_USER_TEMPLATE, format_context

    user = VERIFIER_USER_TEMPLATE.format(
        context=format_context(chunks), answer=answer.answer
    )
    try:
        raw = get_llm().complete(VERIFIER_SYSTEM, user, json_mode=True, temperature=0.0,
                                 name="evidence.verify")
        data = json.loads(raw)
        return {
            "supported": bool(data.get("supported", False)),
            "unsupported_claims": list(data.get("unsupported_claims") or []),
            "confidence": str(data.get("confidence", "low")),
        }
    except Exception as exc:
        logger.warning("LLM verification failed: %s", exc)
        # Fail open but honest: unknown, never "verified".
        return {"supported": None, "unsupported_claims": [], "confidence": "unknown"}
