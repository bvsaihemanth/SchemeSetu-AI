"""Stage 1 of ingest: file on disk -> SchemeDoc.

Parsing only. No chunking, no embedding, no Qdrant. Keeping this stage pure
means you can re-chunk the whole corpus without re-reading 725 files, and
unit-test parsing without any service running.

Deliberately a hand-written line parser rather than a markdown library: the
layout is fixed and known, the files are small, and a 60-line parser that
fails loudly beats a dependency that silently reinterprets headings.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator

from .models import FaqPair, Section, SchemeDoc, SchemeLevel, SectionContent

logger = logging.getLogger(__name__)

# Raw heading -> canonical section. Lowercased, punctuation-insensitive lookup.
SECTION_ALIASES: dict[str, Section] = {
    "details": Section.DETAILS,
    "detail": Section.DETAILS,
    "scheme details": Section.DETAILS,
    "about": Section.DETAILS,
    "overview": Section.DETAILS,
    "objective": Section.DETAILS,
    "scheme objective": Section.DETAILS,
    "benefits": Section.BENEFITS,
    "benefit": Section.BENEFITS,
    "financial benefits": Section.BENEFITS,
    "eligibility": Section.ELIGIBILITY,
    "eligibility criteria": Section.ELIGIBILITY,
    "who can apply": Section.ELIGIBILITY,
    "application process": Section.APPLICATION_PROCESS,
    "how to apply": Section.APPLICATION_PROCESS,
    "process": Section.APPLICATION_PROCESS,
    "documents required": Section.DOCUMENTS_REQUIRED,
    "required documents": Section.DOCUMENTS_REQUIRED,
    "documents": Section.DOCUMENTS_REQUIRED,
    "frequently asked questions": Section.FAQ,
    "faq": Section.FAQ,
    "faqs": Section.FAQ,
    "exclusions": Section.EXCLUSIONS,
    "exclusion": Section.EXCLUSIONS,
    "references": Section.REFERENCES,
    "reference": Section.REFERENCES,
    "sources": Section.REFERENCES,
    "contact": Section.CONTACT,
    "contact us": Section.CONTACT,
    "contact details": Section.CONTACT,
}

_H1 = re.compile(r"^#\s+(?P<title>.+?)\s*$")
_H2 = re.compile(r"^##\s+(?P<title>.+?)\s*$")
_H3 = re.compile(r"^###\s+(?P<title>.+?)\s*$")
_META = re.compile(r"^\*\*(?P<key>[^:*]+):\*\*\s*(?P<value>.*?)\s*$")

_NOT_SPECIFIED = {"", "not specified", "n/a", "na", "none", "-", "nil"}

# Union territories are tagged separately from states: eligibility rules and
# implementing agencies differ, and users ask about them by name.
_UTS = {
    "andaman and nicobar islands", "chandigarh", "dadra and nagar haveli and daman and diu",
    "delhi", "nct of delhi", "jammu and kashmir", "ladakh", "lakshadweep", "puducherry",
}
_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal",
}


def normalise_heading(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", title.strip().lower()).strip()


def classify_section(title: str) -> Section:
    key = normalise_heading(title)
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key]
    # Tolerate decorated headings like "Eligibility Criteria (Updated)".
    for alias, section in SECTION_ALIASES.items():
        if key.startswith(alias) or alias in key:
            return section
    logger.debug("Unmapped section heading: %r", title)
    return Section.OTHER


def _clean_meta(value: str) -> str | None:
    v = value.strip()
    return None if v.lower() in _NOT_SPECIFIED else v


def _infer_level(ministry: str | None, state: str | None, text: str) -> SchemeLevel:
    if state:
        return SchemeLevel.UT if state.lower() in _UTS else SchemeLevel.STATE
    if ministry:
        return SchemeLevel.CENTRAL
    return SchemeLevel.UNKNOWN


def _detect_state(scheme_name: str, ministry: str | None, head_text: str) -> str | None:
    """myScheme state schemes usually name the state in the scheme title or
    carry a state department as 'Ministry'. Checked against a closed list so
    a stray mention inside body text cannot mislabel a central scheme."""
    haystack = f"{scheme_name} {ministry or ''}".lower()
    for name in sorted(_STATES | _UTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", haystack):
            return name.title()
    return None


def parse_text(text: str, doc_name: str, source_path: str) -> SchemeDoc:
    """Parse the fixed myScheme layout into a SchemeDoc."""
    lines = text.splitlines()

    scheme_name = doc_name
    meta: dict[str, str] = {}

    current_section: Section | None = None
    current_title = ""
    section_order = 0
    buffer: list[str] = []
    sections: list[SectionContent] = []

    faqs: list[FaqPair] = []
    faq_question: str | None = None
    faq_buffer: list[str] = []
    in_faq = False

    def flush_section() -> None:
        nonlocal buffer
        if current_section is not None:
            body = "\n".join(buffer).strip()
            if body:
                sections.append(
                    SectionContent(
                        section=current_section,
                        section_title=current_title,
                        text=body,
                        order=section_order,
                    )
                )
        buffer = []

    def flush_faq() -> None:
        nonlocal faq_question, faq_buffer
        if faq_question:
            answer = "\n".join(faq_buffer).strip()
            if answer:
                faqs.append(
                    FaqPair(question=faq_question, answer=answer, order=len(faqs))
                )
        faq_question, faq_buffer = None, []

    seen_h1 = False
    for line in lines:
        if (m := _H1.match(line)) and not seen_h1:
            scheme_name = m.group("title").strip()
            seen_h1 = True
            continue

        if m := _H2.match(line):
            flush_faq()
            flush_section()
            current_title = m.group("title").strip()
            current_section = classify_section(current_title)
            section_order += 1
            in_faq = current_section is Section.FAQ
            continue

        if in_faq and (m := _H3.match(line)):
            flush_faq()
            faq_question = m.group("title").strip()
            continue

        # Metadata lines appear before the first ## heading.
        if current_section is None and (m := _META.match(line)):
            meta[normalise_heading(m.group("key"))] = m.group("value")
            continue

        if in_faq and faq_question is not None:
            faq_buffer.append(line)
        elif current_section is not None:
            buffer.append(line)

    flush_faq()
    flush_section()

    ministry = _clean_meta(meta.get("ministry", ""))
    source_url = _clean_meta(meta.get("source", ""))
    scheme_slug = None
    scheme_acronym = None
    if source_url:
        scheme_slug = source_url.rstrip("/").rsplit("/", 1)[-1] or None
    if scheme_slug:
        # "pm-kisan" -> "PM-KISAN". Skipped when the slug is just the name
        # squashed together, which would add nothing but noise.
        flat_slug = re.sub(r"[^a-z0-9]", "", scheme_slug.lower())
        flat_name = re.sub(r"[^a-z0-9]", "", scheme_name.lower())
        if flat_slug and flat_slug != flat_name and len(flat_slug) <= 24:
            scheme_acronym = scheme_slug.upper()

    head_text = sections[0].text if sections else ""
    state = _detect_state(scheme_name, ministry, head_text)

    return SchemeDoc(
        doc_id=hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:16],
        doc_name=doc_name,
        source_path=source_path,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        scheme_name=scheme_name,
        scheme_slug=scheme_slug,
        scheme_acronym=scheme_acronym,
        source_url=source_url,
        ministry=ministry,
        level=_infer_level(ministry, state, head_text),
        state=state,
        sections=sections,
        faqs=faqs,
        raw_text=text,
    )


def parse_file(path: Path, root: Path | None = None) -> SchemeDoc:
    # utf-8-sig strips the BOM Windows editors like to add; errors="replace"
    # means one bad byte cannot abort a 725-file run.
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rel = str(path.relative_to(root)) if root else str(path)
    return parse_text(text, doc_name=path.stem, source_path=rel)


def iter_documents(root: str | Path, pattern: str = "*.txt") -> Iterator[SchemeDoc]:
    """Walk the corpus. Yields parse failures as log warnings, never raises,
    so a single malformed file cannot kill the ingest run."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root_path}")

    for path in sorted(root_path.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            yield parse_file(path, root=root_path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path.name, exc)
