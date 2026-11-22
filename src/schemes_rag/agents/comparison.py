"""Benefits & Scheme Comparison Agent.

THE ONE THING THAT MATTERS HERE: PER-SCHEME RETRIEVAL.

The naive build runs a single search for "compare PMJJBY and PMSBY" and hands
the top chunks to the model. That fails in a specific, hard-to-spot way: the
two schemes are near-identical insurance products, so their chunks are
semantically adjacent, retrieval returns four from one and one from the other,
and the model confidently attributes the first scheme's premium to the second.
The answer reads perfectly and is wrong.

So each scheme gets its own scoped retrieval with a guaranteed budget, and the
join happens in the prompt where the model can see which text belongs to
which scheme. Scoping is free because resolution already returned the doc_ids.
"""

from __future__ import annotations

import logging

from ..models import RetrievedChunk
from ..retriever import get_retriever

logger = logging.getLogger(__name__)

# What a comparison is actually about. Application steps rarely differ in ways
# anyone cares about when choosing between schemes.
COMPARE_SECTIONS = ["benefits", "eligibility", "details"]


def retrieve_per_scheme(
    query: str,
    doc_ids: list[str],
    scheme_names: list[str],
    *,
    per_scheme_k: int = 4,
) -> tuple[dict[str, list[RetrievedChunk]], float]:
    """One scoped search per scheme. Returns {scheme_name: chunks}, top score."""
    retriever = get_retriever()
    by_scheme: dict[str, list[RetrievedChunk]] = {}
    best = 0.0

    for doc_id, name in zip(doc_ids, scheme_names):
        result = retriever.retrieve(
            query,
            gate_query=query,
            doc_ids=[doc_id],
            priority_sections=COMPARE_SECTIONS,
            final_k=per_scheme_k,
            # Each scheme gets its own floor check; a scheme with nothing
            # relevant simply contributes nothing rather than dragging the
            # whole comparison below the gate.
            min_score=0.0,
            balance_sections=True,
        )
        if result.chunks:
            by_scheme[name] = result.chunks
            best = max(best, result.top_score)
        else:
            logger.info("No chunks for %s in comparison", name)

    return by_scheme, best


def format_context(by_scheme: dict[str, list[RetrievedChunk]]) -> tuple[str, list[RetrievedChunk]]:
    """Render as clearly separated per-scheme blocks.

    Returns the prompt text and a flat chunk list whose order matches the [n]
    numbering, so the Evidence agent can resolve citations against it.
    """
    blocks: list[str] = []
    flat: list[RetrievedChunk] = []

    for name, chunks in by_scheme.items():
        blocks.append(f"===== SCHEME: {name} =====")
        for rc in chunks:
            flat.append(rc)
            blocks.append(
                f"[{len(flat)}] ({name} | {rc.chunk.section_title})\n{rc.chunk.text}"
            )
        blocks.append("")

    return "\n".join(blocks), flat


def render_sources(by_scheme: dict[str, list[RetrievedChunk]]) -> str:
    lines = []
    for name, chunks in by_scheme.items():
        url = next((c.chunk.source_url for c in chunks if c.chunk.source_url), None)
        lines.append(f"  {name}" + (f"\n    {url}" if url else ""))
    return "\n".join(lines)
