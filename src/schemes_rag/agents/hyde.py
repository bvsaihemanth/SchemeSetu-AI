"""HyDE — Hypothetical Document Embeddings. Used by the Discovery agent.

WHAT IT DOES

Instead of embedding the user's question, generate a short passage that *looks
like* the document which would answer it, and embed that. Query and corpus
then sit in the same register, which is the whole point: users write "I'm a
farmer below poverty line, what can I get?" while the corpus writes "All
landholding farmers' families, which have cultivable land holding in their
names are eligible to get benefit under the scheme."

WHY IT IS CONDITIONAL HERE

HyDE is applied only when scheme resolution finds nothing confident, and that
restriction matters.

For "Who can apply for PM-KISAN?" the alias index already returns the exact
document. A hypothetical passage cannot improve on a correct document id, and
a 4B inventing scheme details can actively drag retrieval toward a different
scheme. It would cost ~5 seconds to make the answer worse.

For "schemes for women entrepreneurs in Telangana" there is no entity to
resolve and the vocabulary gap is real. That is where HyDE pays.

TWO SAFEGUARDS

1. The hypothetical drives the DENSE vector only; the user's original words
   drive the SPARSE vector. Invented terminology must not pollute lexical
   matching, which is what catches real acronyms and amounts.
2. The relevance gate still scores the user's ORIGINAL question. A fluent
   hallucination therefore cannot carry an off-topic query past the floor.
"""

from __future__ import annotations

import logging

from ..config import get_settings
from ..llm import get_llm

logger = logging.getLogger(__name__)

#: Roughly the length of one scheme-document paragraph, which is the unit this
#: text is meant to imitate and be compared against.
_MAX_HYDE_WORDS = 150

HYDE_SYSTEM = """You write a short passage in the style of an Indian \
government scheme document, as if answering the user's situation.

Rules:
1. Write 60-100 words of plain prose. No headings, no lists, no preamble.
2. Use the register of official scheme documents: "beneficiaries", "eligible",
   "applicants", "financial assistance", "subject to", "as notified".
3. Describe the KIND of scheme and the KIND of eligibility that would fit.
4. Do NOT invent scheme names, rupee amounts, dates or percentages. Those would
   be retrieved as if real. Describe categories, not specifics.
5. Output the passage only."""

HYDE_USER = """User's situation or question: {query}

Passage:"""


def generate_hypothetical(query: str) -> str | None:
    """A pseudo-document for `query`, or None if generation is unusable.

    Returns None rather than raising: HyDE is an enhancement, and falling back
    to embedding the raw query is always a valid path.
    """
    settings = get_settings()
    if not settings.hyde_enabled:
        return None

    try:
        text = get_llm().complete(
            HYDE_SYSTEM,
            HYDE_USER.format(query=query.strip()),
            temperature=0.3,  # a little variety helps coverage; not 0
            max_tokens=settings.hyde_max_tokens,
            name="hyde.hypothetical",
        )
    except Exception as exc:
        logger.warning("HyDE generation failed (%s); using the raw query", exc)
        return None

    text = text.strip()
    words = text.split()
    if len(words) < 12:
        logger.info("HyDE output too short (%d words); using the raw query", len(words))
        return None

    # An explicit cap, because hyde_max_tokens no longer bounds the prose: the
    # client adds thinking headroom to every ceiling, so a verbose model could
    # return many hundreds of words. Length is a RETRIEVAL concern here -- this
    # text becomes a dense vector, and a long hypothetical dilutes the user's
    # own terms until the embedding describes the genre rather than the
    # question. The prompt asks for a short passage; this enforces it.
    if len(words) > _MAX_HYDE_WORDS:
        logger.info(
            "HyDE output was %d words; truncating to %d to keep the dense "
            "vector focused", len(words), _MAX_HYDE_WORDS,
        )
        text = " ".join(words[:_MAX_HYDE_WORDS])

    # Prepending the real question keeps the user's own terms in the dense
    # vector too, so a hypothetical that drifts off-topic cannot erase them.
    return f"{query.strip()}\n\n{text}"
