"""Scheme Discovery Agent — "what can someone like me get?"

Three retrieval strategies, chosen by what we actually know. Picking the
right one is most of the agent:

  PROFILE FILTER   we know occupation / age / income / category
                   -> Qdrant filter on the enrichment facets. Precise and
                      fast: it EXCLUDES schemes the person cannot qualify
                      for, which semantic search never does.

  HyDE + SEMANTIC  a descriptive query but no usable profile
                   -> generate a pseudo-document, embed that for the dense
                      side, keep the user's words for the sparse side.

  PLAIN SEMANTIC   short or keyword-ish query
                   -> no point paying for HyDE on "crop insurance".

Results are grouped BY SCHEME, not returned as chunks. A discovery answer is
a shortlist of schemes with one line each; returning six chunks from two
schemes would read as a muddle and waste the context budget.
"""

from __future__ import annotations

import logging

from ..config import get_settings
from ..models import RetrievedChunk, UserProfile
from ..retriever import RetrievalResult, get_retriever
from ..vectorstore import build_profile_filter
from .hyde import generate_hypothetical

logger = logging.getLogger(__name__)

# Sections worth searching for discovery: what the scheme IS and who it is
# FOR. Application steps and document lists say nothing about fit.
DISCOVERY_SECTIONS = ["details", "benefits", "eligibility"]


def _group_by_scheme(chunks: list[RetrievedChunk], limit: int) -> list[dict]:
    """Collapse chunks into ranked schemes, keeping the best evidence for each."""
    order: list[str] = []
    best: dict[str, dict] = {}

    for rc in chunks:
        name = rc.chunk.scheme_name
        if name not in best:
            order.append(name)
            best[name] = {
                "scheme_name": name,
                "scheme_acronym": rc.chunk.scheme_acronym,
                "source_url": rc.chunk.source_url,
                "score": round(rc.effective_score, 4),
                "evidence": [],
                "sections": [],
            }
        entry = best[name]
        entry["score"] = max(entry["score"], round(rc.effective_score, 4))
        if len(entry["evidence"]) < 2:
            entry["evidence"].append(rc.chunk.text[:400])
            entry["sections"].append(rc.chunk.section.value)

    ranked = sorted((best[n] for n in order), key=lambda e: -e["score"])
    return ranked[:limit]


def discover(
    query: str,
    profile: UserProfile | None = None,
    *,
    state: str | None = None,
    scheme_resolved: bool = False,
    limit: int | None = None,
) -> tuple[list[dict], RetrievalResult, str]:
    """(schemes, retrieval_result, strategy_used)."""
    settings = get_settings()
    limit = limit or settings.discovery_max_schemes
    retriever = get_retriever()
    profile = profile or UserProfile()
    known = profile.known_slots()

    # ---- strategy 1: filter on the enrichment facets ----
    if len(known) >= settings.discovery_min_slots:
        query_filter = build_profile_filter(
            profile, sections=DISCOVERY_SECTIONS, state=state, strict=True
        )
        # The PROFILE is the query here, not the words. "what schemes can I
        # get" reduces to almost nothing after stopwords, so searching on it
        # alone finds nothing; the person's occupation, status and state carry
        # the signal. Combine both.
        search_text = f"{query.strip()}\n{profile.describe()}".strip()
        result = retriever.retrieve(
            search_text,
            # No relevance floor on the filtered path. The filter already
            # guarantees topicality -- we are inside the eligibility and
            # benefits sections of schemes that match this person's facets --
            # and the gate would reject a vague-but-valid "what can I get"
            # because its literal wording resembles nothing in the corpus.
            min_score=0.0,
            extra_filter=query_filter,
            final_k=limit * 2,
            balance_sections=False,
        )
        if result.chunks:
            logger.info("Discovery via profile filter: %d chunks", len(result.chunks))
            return _group_by_scheme(result.chunks, limit), result, "profile_filter"

        # Empty under a strict filter usually means facets have not been
        # extracted yet (facets=None on every point), not that nothing fits.
        logger.info("Strict profile filter matched nothing; relaxing to numeric only")
        relaxed = build_profile_filter(
            profile, sections=DISCOVERY_SECTIONS, state=state, strict=False
        )
        result = retriever.retrieve(
            search_text,
            min_score=0.0,
            extra_filter=relaxed,
            final_k=limit * 2,
            balance_sections=False,
        )
        if result.chunks:
            return _group_by_scheme(result.chunks, limit), result, "profile_filter_relaxed"

    # ---- strategy 2: HyDE, only when there is no entity to resolve ----
    strategy = "semantic"
    search_text = query
    if not scheme_resolved and len(query.split()) >= 4:
        hypothetical = generate_hypothetical(query)
        if hypothetical:
            search_text = hypothetical
            strategy = "hyde"
            logger.info("Discovery via HyDE (%d chars)", len(hypothetical))

    result = retriever.retrieve(
        search_text,
        # Gate on the user's real question, never the hypothetical: that is
        # what stops a fluent invention sailing past the relevance floor.
        gate_query=query,
        sections=DISCOVERY_SECTIONS,
        state=state or profile.state,
        final_k=limit * 2,
        balance_sections=False,
    )
    return _group_by_scheme(result.chunks, limit), result, strategy


def render(schemes: list[dict], profile: UserProfile, strategy: str) -> str:
    """Terminal-friendly shortlist."""
    if not schemes:
        return (
            "I could not find schemes matching that in the documents I have. "
            "Try naming a sector (farming, education, housing, business) or "
            "telling me more about your situation."
        )

    header = f"{len(schemes)} scheme(s) that may fit"
    if profile.known_slots():
        header += f" — based on: {profile.describe()}"
    lines = [header, ""]
    for i, s in enumerate(schemes, 1):
        name = s["scheme_name"]
        if s.get("scheme_acronym"):
            name += f" ({s['scheme_acronym']})"
        lines.append(f"{i}. {name}")
        if s["evidence"]:
            snippet = " ".join(s["evidence"][0].split())[:180]
            lines.append(f"   {snippet}...")
        if s.get("source_url"):
            lines.append(f"   {s['source_url']}")
        lines.append("")

    if strategy.startswith("profile_filter_relaxed"):
        lines.append(
            "Note: matched on numeric criteria only. Run scripts/enrich.py to "
            "enable filtering by occupation and category."
        )
    elif strategy == "hyde":
        lines.append("Tell me your occupation, age or income to narrow this down.")

    lines.append("Ask \"am I eligible for <scheme>\" to check one properly.")
    return "\n".join(lines)
