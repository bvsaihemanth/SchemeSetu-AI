"""Offline facet extraction: eligibility prose -> structured, filterable fields.

WHY THIS EXISTS

"I'm a farmer below poverty line, suggest schemes for me" cannot be answered
by similarity search. There is no embedding of "age under 40" that reliably
excludes a scheme for 18-25 year olds, and no amount of reranking turns prose
into a numeric comparison. The question needs a FILTER, and a filter needs
structured fields.

So once, offline, the LLM reads each scheme's eligibility text and maps it
into the closed vocabulary in facets.py. 725 calls, then the question becomes
a Qdrant query that runs in milliseconds and cannot hallucinate.

WHY IT IS CHEAP TO APPLY

Facets live in the point PAYLOAD, not the vector. So applying them uses
`set_payload` against existing points -- no re-embedding, no re-ingest. The
only cost is the one-time extraction pass; updating 12,811 points afterwards
takes seconds.

WHY IT IS CACHED

A 4B at ~20 tok/s needs roughly an hour for 725 schemes. The cache is keyed by
content hash, so an interrupted run resumes, and only edited documents are
re-extracted later.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

from .facets import (
    AGE_MAX_UNSET,
    AGE_MIN_UNSET,
    ANY,
    Category,
    EducationLevel,
    Gender,
    INCOME_MAX_UNSET,
    LAND_MAX_UNSET,
    Occupation,
    SpecialStatus,
    coerce,
    vocabulary_block,
)
from .models import SchemeDoc, SchemeFacets, Section

logger = logging.getLogger(__name__)

CACHE_NAME = "scheme_facets.json"

EXTRACT_SYSTEM = """You read an Indian government scheme and extract WHO is \
eligible, as JSON only. No prose, no explanation.

Use ONLY these exact values:

{vocabulary}

Output schema:
{{
  "occupations": [...],          // who the scheme is for; ["any"] if open to all occupations
  "categories": [...],           // social category required; ["any"] if no requirement
  "genders": [...],              // ["any"] unless the scheme is gender-specific
  "education_levels": [...],     // [] unless a specific level is required
  "special_statuses": [...],     // ["any"] if no special status is required
  "age_min": null,               // number, or null if no minimum is stated
  "age_max": null,               // number, or null if no maximum is stated
  "income_max": null,            // max ANNUAL income in rupees, or null
  "land_max_ha": null,           // max landholding in HECTARES, or null
  "beneficiary_type": "individual" // or "institution" or "both"
}}

Rules:
1. Extract only what the text states. Never infer an age limit that is absent.
2. NEVER copy the whole list of allowed values. The lists above are a
   vocabulary to CHOOSE FROM, not an answer.
3. If MORE THAN THREE values of a list would apply, output ["any"] instead of
   listing them. "any" means the scheme places no restriction on that axis.
   Most schemes restrict only one or two axes, so "any" is the common answer.
   Example: a scheme open to all workers -> "occupations": ["any"], NOT a list
   of every occupation.
4. beneficiary_type is "institution" when the applicant is a college, NGO,
   company, state government or similar, not a person.
5. Convert units: 1 acre = 0.405 hectares. "2 lakh" = 200000. "5 lakh" = 500000.
6. If the text says small/marginal farmer, add small_or_marginal_farmer to
   special_statuses AND farmer to occupations.
7. If the text mentions BPL or below poverty line, add below_poverty_line.
8. Output JSON only."""

EXTRACT_USER = """Scheme: {scheme_name}

{text}

JSON:"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)

# A list this long no longer discriminates, so it carries no filtering value
# and is collapsed to the ANY sentinel.
#
# This guard exists because the model does not select, it COPIES. Handed a
# 38-value occupation vocabulary and an open-ended eligibility text, Qwen
# returned all 38 for "Agricultural Extension" -- and all 9 categories, and
# all 16 special statuses. Such a scheme then matches every user while
# LOOKING constrained, which is worse than an honest "any": it defeats the
# filter and hides the failure from the unconstrained count.
#
# A model cannot be trusted to choose, but "it chose everything" is trivial
# to detect, so the guard is deterministic rather than another prompt rule.
# It is kept for the same reason now that generation is hosted: copying the
# vocabulary is a failure a better model makes less often, not never, and the
# cost of the guard being unnecessary is zero.
_COLLAPSE_AT = {
    "occupations": 6,        # of 39 -- genuine multi-occupation schemes name 2-4
    "categories": 5,         # of 9  -- a majority of categories means no bar
    "genders": 2,            # of 3  -- gender-specific schemes name exactly one
    "education_levels": 5,   # of 11
    "special_statuses": 5,   # of 16
}


def _as_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip().lower().replace(",", "")
        if not txt or txt in {"null", "none", "n/a", "na", "-"}:
            return None
        # "2 lakh", "1.5 crore" -- the forms this corpus actually uses
        multiplier = 1.0
        if "lakh" in txt:
            multiplier, txt = 100_000.0, txt.replace("lakh", "").replace("lakhs", "")
        elif "crore" in txt:
            multiplier, txt = 10_000_000.0, txt.replace("crore", "").replace("crores", "")
        m = re.search(r"-?\d+(?:\.\d+)?", txt)
        return float(m.group(0)) * multiplier if m else None
    return None


def eligibility_text(doc: SchemeDoc, max_chars: int = 4000) -> str:
    """The text the extractor reads: eligibility first, then details for
    context, then FAQ answers that often carry the real age/income rules."""
    parts: list[str] = []
    for wanted in (Section.ELIGIBILITY, Section.DETAILS, Section.EXCLUSIONS):
        for section in doc.sections:
            if section.section is wanted:
                parts.append(f"## {section.section_title}\n{section.text}")
    for faq in doc.faqs[:6]:
        if re.search(r"eligib|who can|age|income|apply", faq.question, re.I):
            parts.append(f"## FAQ\nQ: {faq.question}\nA: {faq.answer}")
    joined = "\n\n".join(parts).strip()
    return joined[:max_chars] if joined else (doc.raw_text or "")[:max_chars]


def _collapse(axis: str, items: list[str]) -> tuple[list[str], bool]:
    """(values, was_collapsed). Over-long lists become uninformative."""
    threshold = _COLLAPSE_AT.get(axis)
    if threshold and len(items) >= threshold:
        logger.debug("Collapsing %s (%d values) to ANY", axis, len(items))
        # education_levels has no ANY sentinel: an empty list already means
        # "no level required", which is what an all-levels list amounts to.
        return ([] if axis == "education_levels" else [ANY]), True
    return items, False


def parse_facets(raw: str, model_name: str | None = None) -> SchemeFacets:
    """Coerce model output into the vocabulary. Never raises: an unparseable
    response yields an unconstrained, low-confidence facet set, which is
    strictly better than dropping the scheme out of every filter."""
    data = None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except Exception:
        if m := _JSON.search(text):
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        logger.warning("Unparseable facet JSON; storing unconstrained facets")
        return SchemeFacets(
            occupations=[ANY], categories=[ANY], genders=[ANY],
            special_statuses=[ANY], extraction_confidence="low",
            extraction_model=model_name,
        )

    occupations, c1 = _collapse("occupations", coerce(data.get("occupations"), Occupation))
    categories, c2 = _collapse("categories", coerce(data.get("categories"), Category))
    genders, c3 = _collapse("genders", coerce(data.get("genders"), Gender))
    education, c4 = _collapse(
        "education_levels", coerce(data.get("education_levels"), EducationLevel)
    )
    statuses, c5 = _collapse(
        "special_statuses", coerce(data.get("special_statuses"), SpecialStatus)
    )
    collapsed_axes = sum([c1, c2, c3, c4, c5])

    # An empty list after coercion means "no restriction stated" -- store the
    # sentinel so MatchAny([user_value, ANY]) still matches this scheme.
    def with_any(items: list[str], raw_value) -> list[str]:
        if items:
            if isinstance(raw_value, (list, tuple)) and any(
                str(x).strip().lower() == ANY for x in raw_value
            ):
                return [*items, ANY] if ANY not in items else items
            return items
        return [ANY]

    age_min = _as_number(data.get("age_min"))
    age_max = _as_number(data.get("age_max"))
    income_max = _as_number(data.get("income_max"))
    land_max = _as_number(data.get("land_max_ha"))

    btype = str(data.get("beneficiary_type") or "individual").strip().lower()
    if btype not in {"individual", "institution", "both"}:
        btype = "individual"

    # A collapsed axis is not evidence of extraction quality, so confidence is
    # judged on what SURVIVED the collapse.
    real_categorical = [
        axis for axis in (occupations, categories, genders, education, statuses)
        if axis and axis != [ANY]
    ]
    constrained = bool(real_categorical) or any([age_min, age_max, income_max, land_max])

    return SchemeFacets(
        occupations=with_any(occupations, data.get("occupations")),
        categories=with_any(categories, data.get("categories")),
        genders=with_any(genders, data.get("genders")),
        education_levels=education,  # [] is meaningful: no level required
        special_statuses=with_any(statuses, data.get("special_statuses")),
        age_min=int(age_min) if age_min is not None else AGE_MIN_UNSET,
        age_max=int(age_max) if age_max is not None else AGE_MAX_UNSET,
        income_max=income_max if income_max is not None else INCOME_MAX_UNSET,
        land_max_ha=land_max if land_max is not None else LAND_MAX_UNSET,
        beneficiary_type=btype,
        extraction_confidence=(
            "low" if collapsed_axes >= 3 and not constrained
            else "medium" if (collapsed_axes >= 3 or not constrained)
            else "high"
        ),
        extraction_model=model_name,
    )


def extract_facets(doc: SchemeDoc) -> SchemeFacets:
    from .config import get_settings
    from .llm import get_llm

    settings = get_settings()
    text = eligibility_text(doc)
    if not text.strip():
        logger.info("No eligibility text for %s", doc.doc_name)
        return SchemeFacets(
            occupations=[ANY], categories=[ANY], genders=[ANY],
            special_statuses=[ANY], extraction_confidence="low",
        )

    try:
        raw = get_llm().complete(
            EXTRACT_SYSTEM.format(vocabulary=vocabulary_block()),
            EXTRACT_USER.format(scheme_name=doc.scheme_name, text=text),
            temperature=0.0,
            max_tokens=512,
            json_mode=True,
            name="enrich.facets",
        )
    except Exception as exc:
        logger.warning("Facet extraction failed for %s: %s", doc.doc_name, exc)
        return SchemeFacets(
            occupations=[ANY], categories=[ANY], genders=[ANY],
            special_statuses=[ANY], extraction_confidence="low",
        )

    return parse_facets(raw, model_name=settings.llm_model)


class FacetCache:
    """doc_id -> {content_hash, facets}. Makes the pass resumable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Ignoring unreadable facet cache: %s", exc)

    def get(self, doc: SchemeDoc) -> SchemeFacets | None:
        entry = self.data.get(doc.doc_id)
        if not entry or entry.get("content_hash") != doc.content_hash:
            return None
        try:
            return SchemeFacets(**entry["facets"])
        except Exception:
            return None

    def put(self, doc: SchemeDoc, facets: SchemeFacets) -> None:
        self.data[doc.doc_id] = {
            "content_hash": doc.content_hash,
            "scheme_name": doc.scheme_name,
            "facets": facets.model_dump(mode="json"),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def apply_facets(store, doc_id: str, facets: SchemeFacets) -> None:
    """Write facets onto every existing chunk of a document.

    set_payload, not upsert: the vectors are untouched, so this costs no
    embedding time and cannot disturb retrieval quality.
    """
    from qdrant_client import models as qm

    store.client.set_payload(
        collection_name=store.collection,
        payload={"facets": facets.model_dump(mode="json")},
        points=qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        ),
        wait=False,
    )


def run_enrichment(
    docs: Iterable[SchemeDoc],
    store=None,
    *,
    cache_path: str | Path | None = None,
    force: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    from .config import get_settings

    settings = get_settings()
    cache = FacetCache(cache_path or Path(settings.processed_dir) / CACHE_NAME)

    stats = {
        "seen": 0, "extracted": 0, "cached": 0, "applied": 0,
        "confidence": {}, "beneficiary_type": {},
        "unconstrained": 0, "occupation_coverage": {},
        # Surfaced because a bare occupation histogram hid the copy-the-whole-
        # vocabulary failure: the counts looked plausible in aggregate.
        "axis_sizes": {"occupations": [], "categories": [], "special_statuses": []},
        "open_occupations": 0,
    }

    for doc in docs:
        stats["seen"] += 1
        if limit and stats["seen"] > limit:
            stats["seen"] -= 1
            break

        facets = None if force else cache.get(doc)
        if facets is None:
            facets = extract_facets(doc)
            cache.put(doc, facets)
            stats["extracted"] += 1
            if stats["extracted"] % 10 == 0:
                cache.save()  # checkpoint, so a crash loses at most 10
                logger.info("extracted %s (%s seen)", stats["extracted"], stats["seen"])
        else:
            stats["cached"] += 1

        conf = facets.extraction_confidence
        stats["confidence"][conf] = stats["confidence"].get(conf, 0) + 1
        bt = facets.beneficiary_type
        stats["beneficiary_type"][bt] = stats["beneficiary_type"].get(bt, 0) + 1
        if facets.is_unconstrained():
            stats["unconstrained"] += 1
        for occ in facets.occupations:
            stats["occupation_coverage"][occ] = stats["occupation_coverage"].get(occ, 0) + 1
        stats["axis_sizes"]["occupations"].append(len(facets.occupations))
        stats["axis_sizes"]["categories"].append(len(facets.categories))
        stats["axis_sizes"]["special_statuses"].append(len(facets.special_statuses))
        if facets.occupations == [ANY]:
            stats["open_occupations"] += 1

        if store is not None and not dry_run:
            apply_facets(store, doc.doc_id, facets)
            stats["applied"] += 1

    cache.save()
    return stats
