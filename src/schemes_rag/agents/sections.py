"""Application Guidance Agent and Document Checklist Agent.

Both answer from one known section of one known scheme, so they share a
retrieval shape and differ only in which section and which prompt. They live
together because splitting near-identical code across two files invites the
two copies to drift.

WHY THESE ARE MOSTLY EXTRACTIVE

`Application Process` and `Documents Required` are already lists in the source
documents -- portal URLs, step sequences, "IEC Copy", "Valid RCMC". The right
behaviour is to reproduce them faithfully in order, not to summarise. So the
prompts forbid paraphrase, and the retrieval guarantees the whole section
rather than its highest-scoring fragment: a document checklist missing its
last four items is worse than useless, because the user believes it complete.
"""

from __future__ import annotations

import logging

from ..models import RetrievedChunk
from ..retriever import get_retriever
from ..vectorstore import build_filter

logger = logging.getLogger(__name__)


def _whole_section(
    doc_ids: list[str] | None,
    sections: list[str],
    query: str,
    *,
    max_chunks: int = 8,
) -> list[RetrievedChunk]:
    """Every chunk of the named section(s), in document order.

    Retrieval ranks by similarity, which is wrong for a checklist: the parts
    of a document list that happen to match the query wording would come back
    and the rest would be dropped. When the scheme is known we therefore take
    the section whole and restore its original order.
    """
    retriever = get_retriever()
    result = retriever.retrieve(
        query,
        gate_query=query,
        doc_ids=doc_ids,
        sections=sections,
        final_k=max_chunks,
        min_score=0.0 if doc_ids else None,
        balance_sections=False,
    )
    chunks = result.chunks
    if doc_ids:
        # chunk_index preserves the order the section appeared in the file,
        # which for a numbered application process is the meaning.
        chunks = sorted(chunks, key=lambda rc: rc.chunk.chunk_index)
    return chunks


def retrieve_application(
    query: str, doc_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    """Application Guidance: how and where to apply.

    FAQ is included deliberately -- in this corpus the portal path, the
    prerequisite registrations and the deadlines are as often in an FAQ answer
    as in the Application Process section itself.
    """
    chunks = _whole_section(doc_ids, ["application_process"], query)
    if len(chunks) < 4:
        extra = _whole_section(doc_ids, ["faq"], query, max_chunks=4)
        seen = {(rc.chunk.doc_id, rc.chunk.chunk_index) for rc in chunks}
        chunks += [
            rc for rc in extra
            if (rc.chunk.doc_id, rc.chunk.chunk_index) not in seen
        ]
    return chunks


def retrieve_documents(
    query: str, doc_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    """Document Checklist: which papers and proofs are needed."""
    chunks = _whole_section(doc_ids, ["documents_required"], query)
    if len(chunks) < 2:
        extra = _whole_section(doc_ids, ["faq", "application_process"], query, max_chunks=4)
        seen = {(rc.chunk.doc_id, rc.chunk.chunk_index) for rc in chunks}
        chunks += [
            rc for rc in extra
            if (rc.chunk.doc_id, rc.chunk.chunk_index) not in seen
        ]
    return chunks


def render_verbatim(kind: str, chunks: list[RetrievedChunk]) -> str:
    """The section, reproduced as-is, with citations.

    The safety net for these two routes. Both sections are ALREADY structured
    lists in the source documents, so if generation refuses, returns empty, or
    the LLM is simply unreachable, the honest fallback is to show the real
    text rather than an apology. The user came for the steps; the model was
    only ever formatting them.
    """
    heading = {
        "application": "Application process",
        "documents": "Documents required",
    }.get(kind, "Details")

    scheme = chunks[0].chunk.scheme_name if chunks else ""
    lines = [f"{heading} for {scheme}:".strip(), ""]
    for i, rc in enumerate(chunks, 1):
        body = rc.chunk.text.strip()
        if rc.chunk.faq_question:
            lines.append(f"[{i}] {body}")
        else:
            lines.append(body + f" [{i}]")
        lines.append("")
    url = next((rc.chunk.source_url for rc in chunks if rc.chunk.source_url), None)
    if url:
        lines.append(f"Official page: {url}")
    return "\n".join(lines).strip()


def missing_section_notice(kind: str, scheme_name: str | None) -> str:
    """Honest message when the scheme genuinely lacks the section.

    Six documents in this corpus have no documents_required section at all.
    Saying so beats letting the model improvise a plausible checklist.
    """
    what = {
        "application": "an application-process section",
        "documents": "a documents-required section",
    }.get(kind, "that section")
    where = f"The myScheme page for {scheme_name}" if scheme_name else "This scheme's page"
    return (
        f"{where} does not include {what}, so I have nothing to quote. "
        f"Check the official scheme page or the implementing department "
        f"directly rather than relying on a guess."
    )
