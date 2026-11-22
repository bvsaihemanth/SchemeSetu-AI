"""Stage 2 of ingest: SchemeDoc -> list[Chunk].

Chunking strategy, and why:

1. SECTION IS THE UNIT. Your documents are already semantically segmented by
   their headings, so a section boundary is a real topic boundary. Splitting
   on a fixed character window would cut eligibility rules in half and glue
   the tail of "Benefits" onto the head of "Eligibility" -- which is exactly
   the failure that makes the Eligibility agent unbuildable.

2. ONE FAQ = ONE CHUNK. FAQ pairs are already atomic, self-contained
   question/answer units. They are the single highest-precision retrieval
   target in this corpus, because user questions resemble FAQ questions far
   more than they resemble policy prose.

3. ONLY OVERSIZED SECTIONS GET SPLIT, on line boundaries, with overlap.
   These files are list-heavy (one document requirement per line), so lines
   are safe cut points; character windows are not.

4. NO CROSS-SECTION CHUNKS, ever. A chunk must have exactly one `section`
   tag for metadata filtering to mean anything.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from .config import get_settings
from .models import Chunk, Section, SchemeDoc

logger = logging.getLogger(__name__)


@lru_cache
def _tokenizer():
    """BGE-M3's own tokenizer when available, so chunk sizes match what the
    model actually sees. Falls back to a chars/4 estimate -- good enough for
    sizing decisions, and avoids making chunking depend on a 2 GB download."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(get_settings().embed_model)
    except Exception as exc:
        logger.info("Tokenizer unavailable (%s); using length estimate", type(exc).__name__)
        return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text, add_special_tokens=False))


_SENTENCE = re.compile(r"(?<=[.!?;])\s+")


def _split_long_line(line: str, target: int) -> list[str]:
    """Last resort for a single line exceeding the target: sentences, then
    hard character cuts."""
    if count_tokens(line) <= target:
        return [line]

    out: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(line):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > target:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)

    # Still oversized (no sentence punctuation) -> cut by characters.
    final: list[str] = []
    budget = target * 4
    for piece in out:
        if count_tokens(piece) <= target:
            final.append(piece)
        else:
            final.extend(piece[i : i + budget] for i in range(0, len(piece), budget))
    return final


def split_section_text(text: str, target: int, overlap: int, minimum: int) -> list[str]:
    """Group lines into windows of ~target tokens with ~overlap carryover."""
    lines = [ln for ln in (l.rstrip() for l in text.splitlines()) if ln.strip()]
    if not lines:
        return []

    units: list[tuple[str, int]] = []
    for line in lines:
        for piece in _split_long_line(line, target):
            units.append((piece, count_tokens(piece)))

    total = sum(n for _, n in units)
    if total <= target:
        return ["\n".join(p for p, _ in units)]

    chunks: list[str] = []
    buf: list[tuple[str, int]] = []
    size = 0

    for unit, n in units:
        if buf and size + n > target:
            chunks.append("\n".join(p for p, _ in buf))
            # Carry the tail back as overlap so a rule split across the
            # boundary still appears whole in one of the two chunks.
            carry: list[tuple[str, int]] = []
            carried = 0
            for prev, prev_n in reversed(buf):
                if carried + prev_n > overlap:
                    break
                carry.insert(0, (prev, prev_n))
                carried += prev_n
            buf = carry
            size = carried
        buf.append((unit, n))
        size += n

    if buf:
        tail = "\n".join(p for p, _ in buf)
        # Don't emit a runt chunk: fold it into the previous one instead.
        if chunks and count_tokens(tail) < minimum:
            chunks[-1] = f"{chunks[-1]}\n{tail}"
        else:
            chunks.append(tail)

    return chunks


def chunk_document(doc: SchemeDoc) -> list[Chunk]:
    s = get_settings()
    target = s.chunk_target_tokens
    overlap = s.chunk_overlap_tokens
    minimum = s.chunk_min_tokens

    chunks: list[Chunk] = []
    index = 0

    def base_kwargs() -> dict:
        return dict(
            doc_id=doc.doc_id,
            doc_name=doc.doc_name,
            scheme_name=doc.scheme_name,
            scheme_slug=doc.scheme_slug,
            scheme_acronym=doc.scheme_acronym,
            ministry=doc.ministry,
            level=doc.level,
            state=doc.state,
            tags=doc.tags,
            source_url=doc.source_url,
            source_path=doc.source_path,
        )

    # --- narrative sections ---
    for section in doc.sections:
        if section.section is Section.FAQ:
            continue  # handled below as Q/A pairs
        parts = split_section_text(section.text, target, overlap, minimum)
        for i, part_text in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    chunk_index=index,
                    text=part_text,
                    token_count=count_tokens(part_text),
                    section=section.section,
                    section_title=section.section_title,
                    part=i,
                    part_total=len(parts),
                    **base_kwargs(),
                )
            )
            index += 1

    # --- FAQ pairs: one chunk each ---
    for faq in doc.faqs:
        body = f"Q: {faq.question}\nA: {faq.answer}"
        # A single FAQ answer over budget is rare; split rather than truncate.
        parts = split_section_text(body, target, overlap, minimum) or [body]
        for i, part_text in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    chunk_index=index,
                    text=part_text,
                    token_count=count_tokens(part_text),
                    section=Section.FAQ,
                    section_title="Frequently Asked Questions",
                    faq_question=faq.question,
                    part=i,
                    part_total=len(parts),
                    **base_kwargs(),
                )
            )
            index += 1

    if not chunks:
        logger.warning("No chunks produced for %s", doc.doc_name)
    return chunks
