"""Agent 1: Query Understanding & Clarification.

Turns a raw question into a routing decision. Runs before retrieval, so it
must be cheap and must never block: if the LLM returns junk, we fall back to
heuristics and let retrieval do its job rather than failing the request.

A 4B model is not reliable at JSON, so there are three layers:
  1. json_mode / guided decoding where the server supports it
  2. a tolerant extractor for JSON embedded in prose or fenced in markdown
  3. keyword heuristics as the final fallback

Layer 3 is not a formality -- on a local 4B it fires regularly, and the
system stays usable because of it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..llm import get_llm
from ..prompts import ROUTER_SYSTEM, ROUTER_USER_TEMPLATE
from ..scheme_index import get_scheme_index

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "discovery",
    "eligibility",
    "benefits",
    "comparison",
    "application",
    "documents",
    "general",
}

# Ordered: the first match wins, so more specific patterns come first.
#
# ELIGIBILITY MUST PRECEDE APPLICATION. "Who can apply for PM-KISAN?" contains
# the word "apply", so an application-first ordering routes it to the
# application-process section -- which is what happened in practice, giving
# route=application for a question that is purely about eligibility. The
# application patterns below therefore require process-shaped phrasing
# ("how to apply", "where to apply") rather than the bare verb.
_HEURISTICS: list[tuple[str, re.Pattern]] = [
    ("eligibility", re.compile(
        r"\b(eligib\w*|qualif\w*|who can (apply|avail|get|benefit|claim)"
        r"|who is eligible|am i (eligible|entitled)|can i (apply|avail|get|claim)"
        r"|criteria|age limit|income limit)\b", re.I)),
    ("documents", re.compile(r"\b(document|papers?|certificate|proof|attach)\w*\b", re.I)),
    ("comparison", re.compile(r"\b(compare|versus|vs\.?|difference between|better)\b", re.I)),
    ("application", re.compile(
        r"\b(how (do|to|can) i apply|how to apply|where (do|to) i? ?apply"
        r"|application process|apply online|registration|register|portal"
        r"|submit|steps to apply)\b", re.I)),
    ("benefits", re.compile(r"\b(benefit|subsidy|amount|how much|get paid|interest rate|pension)\w*\b", re.I)),
    ("discovery", re.compile(r"\b(what schemes?|which schemes?|list .*schemes?|schemes? for)\b", re.I)),
]

_STATE_WORDS = re.compile(
    r"\b(andhra pradesh|arunachal pradesh|assam|bihar|chhattisgarh|goa|gujarat|haryana|"
    r"himachal pradesh|jharkhand|karnataka|kerala|madhya pradesh|maharashtra|manipur|"
    r"meghalaya|mizoram|nagaland|odisha|punjab|rajasthan|sikkim|tamil nadu|telangana|"
    r"tripura|uttar pradesh|uttarakhand|west bengal|delhi|puducherry|chandigarh|ladakh|"
    r"jammu and kashmir|lakshadweep)\b",
    re.I,
)

_VAGUE = re.compile(r"^\s*(hi|hello|hey|help|info|details|schemes?|yes|no|ok)\W*$", re.I)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class QueryPlan:
    intent: str = "general"
    search_query: str = ""
    schemes: list[str] = field(default_factory=list)
    state: str | None = None
    needs_clarification: bool = False
    clarifying_question: str | None = None
    source: str = "llm"  # llm | heuristic -- surfaced in traces
    # Resolved by SchemeIndex, never by the LLM. doc_ids are exact identifiers;
    # a 4B model asked to map "PM-KISAN" to a document title will invent one.
    scheme_doc_ids: list[str] = field(default_factory=list)
    scheme_confidence: str = "none"  # exact | strong | weak | none
    scheme_matched_on: str | None = None


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    if m := _JSON_BLOCK.search(raw):
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def heuristic_plan(question: str) -> QueryPlan:
    q = question.strip()
    plan = QueryPlan(search_query=q, source="heuristic")

    if len(q) < 3 or _VAGUE.match(q):
        plan.needs_clarification = True
        plan.clarifying_question = (
            "What would you like to know? You can ask about a specific scheme, "
            "or tell me your situation (for example: farmer in Telangana "
            "looking for crop support)."
        )
        return plan

    for intent, pattern in _HEURISTICS:
        if pattern.search(q):
            plan.intent = intent
            break

    if m := _STATE_WORDS.search(q):
        plan.state = m.group(0).title()

    return plan


def resolve_schemes(question: str, plan: QueryPlan) -> QueryPlan:
    """Attach scheme identity to a plan, from the alias index.

    Run against the user's ORIGINAL wording, never a rewritten search query:
    the acronym they typed is the evidence, and a rewrite can lose or mangle
    it. Only "exact" and "strong" matches are allowed to scope retrieval --
    a wrong hard filter hides the right answer completely, so an uncertain
    match is recorded and left for ranking to use.
    """
    index = get_scheme_index()
    if index is None:
        return plan

    matches = index.resolve(question)
    if not matches:
        return plan

    plan.scheme_confidence = matches[0].confidence
    plan.scheme_matched_on = matches[0].matched_on

    # A question naming a specific scheme is not vague, whatever the LLM
    # thought. Observed: "Who can apply for DSCTASSGCSCCM?" resolved exactly
    # to one document, and the router still asked the user what the scheme was
    # and which state it covered -- a clarification request for information
    # the system had already determined. Resolution overrides it.
    if matches[0].confidence in ("exact", "strong") and plan.needs_clarification:
        logger.info(
            "Suppressing clarification: %r resolved to %s",
            question[:50], matches[0].scheme_name[:50],
        )
        plan.needs_clarification = False
        plan.clarifying_question = None
    # Keep every match at the top confidence tier: "compare PMJJBY and PMSBY"
    # legitimately resolves to two schemes.
    tier = matches[0].confidence
    plan.scheme_doc_ids = [m.doc_id for m in matches if m.confidence == tier][:4]
    plan.schemes = [m.scheme_name for m in matches if m.confidence == tier][:4]
    logger.info(
        "Resolved %r -> %s (%s via %s)",
        question[:60],
        plan.schemes,
        tier,
        matches[0].matched_on,
    )
    return plan


def understand_query(question: str, history: list[dict[str, str]] | None = None) -> QueryPlan:
    fallback = heuristic_plan(question)

    # Don't spend an LLM call deciding that "hi" is vague.
    if fallback.needs_clarification:
        return fallback

    try:
        raw = get_llm().complete(
            ROUTER_SYSTEM,
            ROUTER_USER_TEMPLATE.format(question=question.strip()),
            temperature=0.0,
            max_tokens=256,
            json_mode=True,
            name="understand.router",
        )
    except Exception as exc:
        logger.warning("Router LLM call failed (%s); using heuristics", exc)
        return resolve_schemes(question, fallback)

    data = _extract_json(raw)
    if not isinstance(data, dict):
        logger.info("Router returned unparseable output; using heuristics")
        return resolve_schemes(question, fallback)

    intent = str(data.get("intent") or "").strip().lower()
    if intent not in VALID_INTENTS:
        intent = fallback.intent

    schemes = data.get("schemes") or []
    if not isinstance(schemes, list):
        schemes = []
    schemes = [str(x).strip() for x in schemes if str(x).strip()][:5]

    state = data.get("state")
    state = str(state).strip().title() if state else fallback.state

    search_query = str(data.get("search_query") or "").strip() or question.strip()

    needs_clarification = bool(data.get("needs_clarification"))
    clarifying = data.get("clarifying_question")
    clarifying = str(clarifying).strip() if clarifying else None
    # A clarification request with no question to ask is useless; ignore it.
    if needs_clarification and not clarifying:
        needs_clarification = False

    plan = QueryPlan(
        intent=intent,
        search_query=search_query,
        schemes=schemes,
        state=state,
        needs_clarification=needs_clarification,
        clarifying_question=clarifying,
        source="llm",
    )
    # Index resolution overwrites whatever the LLM guessed about scheme names.
    return resolve_schemes(question, plan)
