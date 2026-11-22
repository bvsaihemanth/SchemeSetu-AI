"""Shared data contracts.

The chunk payload schema here is the spine of the system: ingest writes it,
Qdrant indexes it, agents filter on it. Changing it later means a re-ingest.

Schema is derived from the actual myScheme .txt layout:

    # <Scheme Name>
    **Ministry:** <name | "Not specified">
    **Source:** <myscheme.gov.in URL>
    ## Details
    ## Benefits
    ## Eligibility
    ## Application Process
    ## Documents Required
    ## Frequently Asked Questions
    ### <question>
    <answer>
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Section(str, Enum):
    """Canonical sections. `section_title` keeps the raw heading so an
    unrecognised one degrades to OTHER instead of losing information."""

    DETAILS = "details"
    BENEFITS = "benefits"
    ELIGIBILITY = "eligibility"
    APPLICATION_PROCESS = "application_process"
    DOCUMENTS_REQUIRED = "documents_required"
    FAQ = "faq"
    EXCLUSIONS = "exclusions"
    REFERENCES = "references"
    CONTACT = "contact"
    OTHER = "other"


class SchemeLevel(str, Enum):
    CENTRAL = "central"
    STATE = "state"
    UT = "ut"
    UNKNOWN = "unknown"


class SectionContent(BaseModel):
    section: Section
    section_title: str  # raw heading text, e.g. "Application Process"
    text: str
    order: int = 0


class FaqPair(BaseModel):
    question: str
    answer: str
    order: int = 0


class SchemeDoc(BaseModel):
    """One parsed source file, pre-chunking."""

    doc_id: str  # sha1 of relative path -> stable across re-ingest
    doc_name: str  # file stem, e.g. "Advance Authorisation(AA)"
    source_path: str
    content_hash: str  # sha256 of file bytes -> skip unchanged docs

    scheme_name: str
    scheme_slug: str | None = None  # trailing segment of the source URL
    scheme_acronym: str | None = None  # slug upper-cased, e.g. "PM-KISAN"
    source_url: str | None = None
    ministry: str | None = None  # None when "Not specified"
    level: SchemeLevel = SchemeLevel.UNKNOWN
    state: str | None = None
    tags: list[str] = Field(default_factory=list)

    sections: list[SectionContent] = Field(default_factory=list)
    faqs: list[FaqPair] = Field(default_factory=list)
    raw_text: str = ""

    def section_titles(self) -> list[str]:
        return [s.section_title for s in self.sections]


class Chunk(BaseModel):
    """One Qdrant point."""

    doc_id: str
    chunk_index: int
    text: str  # raw text shown to the user / LLM
    token_count: int

    # --- filterable metadata (payload-indexed) ---
    doc_name: str
    scheme_name: str
    scheme_slug: str | None = None
    scheme_acronym: str | None = None
    ministry: str | None = None
    level: SchemeLevel = SchemeLevel.UNKNOWN
    state: str | None = None
    tags: list[str] = Field(default_factory=list)
    section: Section = Section.OTHER
    section_title: str = ""

    # Denormalised onto every chunk: Qdrant filters on point payloads, so the
    # scheme-level facets must live on each point to be filterable.
    facets: SchemeFacets | None = None

    # --- citation data ---
    source_url: str | None = None
    source_path: str = ""
    faq_question: str | None = None  # set when section == FAQ
    part: int = 1  # 1-of-N when a long section is split
    part_total: int = 1

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = 1

    @property
    def embed_text(self) -> str:
        """What actually gets embedded.

        Two things are prepended, for two different reasons.

        The SECTION, because chunks are not self-describing: "Minimum value
        addition of 15%" is unretrievable without knowing which scheme's
        eligibility rules it belongs to.

        The ACRONYM, because it is otherwise absent from the text entirely.
        "Pradhan Mantri Kisan Samman Nidhi" never contains the string
        "PM-KISAN" -- that lives only in the source URL. Without it here, no
        chunk of this document can be matched lexically or semantically by the
        name users actually type, which is precisely how a PM-KISAN query ends
        up returning scholarship schemes.
        """
        name = self.scheme_name
        if self.scheme_acronym:
            name = f"{name} ({self.scheme_acronym})"
        header = f"Scheme: {name}\nSection: {self.section_title}"
        if self.faq_question:
            header += f"\nQuestion: {self.faq_question}"
        return f"{header}\n\n{self.text}"

    def locator(self) -> str:
        """Human-readable citation. These documents have no pages, so the
        stable locator is scheme + section (+ FAQ question)."""
        base = f"{self.scheme_name} - {self.section_title}"
        if self.faq_question:
            base += f" - “{self.faq_question}”"
        if self.part_total > 1:
            base += f" ({self.part}/{self.part_total})"
        return base


class SchemeFacets(BaseModel):
    """Structured eligibility, extracted once per scheme by the offline pass.

    Stored on every chunk of that scheme so a single Qdrant filter can answer
    "which schemes fit this person". Unconstrained axes use sentinels (see
    facets.py) so range filters need no null handling.
    """

    occupations: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    genders: list[str] = Field(default_factory=list)
    education_levels: list[str] = Field(default_factory=list)
    special_statuses: list[str] = Field(default_factory=list)

    age_min: int = 0
    age_max: int = 200
    income_max: float = 1_000_000_000_000.0
    land_max_ha: float = 1_000_000.0

    # Who the beneficiary is: an individual, or an organisation. Roughly a
    # third of this corpus funds institutions, and surfacing those to a farmer
    # asking about himself is noise.
    beneficiary_type: str = "individual"  # individual | institution | both
    extraction_confidence: str = "low"  # high | medium | low
    extraction_model: str | None = None

    def is_unconstrained(self) -> bool:
        """True when nothing here narrows the audience.

        ["any"] counts as unconstrained, not as a value. Treating it as a real
        restriction is how a scheme that matches everyone reports itself as
        constrained -- which is precisely what hid the copy-the-vocabulary
        failure behind an "unconstrained 0" line.
        """
        from .facets import AGE_MAX_UNSET, AGE_MIN_UNSET, ANY
        from .facets import INCOME_MAX_UNSET, LAND_MAX_UNSET

        categorical = any(
            axis and axis != [ANY]
            for axis in (self.occupations, self.categories, self.genders,
                         self.education_levels, self.special_statuses)
        )
        numeric = (
            self.age_min != AGE_MIN_UNSET
            or self.age_max != AGE_MAX_UNSET
            or self.income_max != INCOME_MAX_UNSET
            or self.land_max_ha != LAND_MAX_UNSET
        )
        return not (categorical or numeric)


class UserProfile(BaseModel):
    """What the Eligibility agent has learned about the person so far.

    Every field is optional: the agent asks only for what the retrieved
    eligibility criteria actually require, rather than interrogating the user
    about axes the scheme does not mention.
    """

    occupation: str | None = None
    category: str | None = None
    gender: str | None = None
    education_level: str | None = None
    special_statuses: list[str] = Field(default_factory=list)

    age: int | None = None
    annual_income: float | None = None
    landholding_ha: float | None = None
    state: str | None = None

    # Free-text the user volunteered, kept for the assessment prompt so
    # nuance is not lost to the slot schema.
    notes: list[str] = Field(default_factory=list)

    def known_slots(self) -> list[str]:
        out = [
            k for k in ("occupation", "category", "gender", "education_level",
                        "age", "annual_income", "landholding_ha", "state")
            if getattr(self, k) is not None
        ]
        if self.special_statuses:
            out.append("special_statuses")
        return out

    def describe(self) -> str:
        """Human-readable summary, used in prompts and shown back to the user
        so they can see what the system believes about them."""
        bits: list[str] = []
        if self.occupation:
            bits.append(f"occupation: {self.occupation}")
        if self.age is not None:
            bits.append(f"age: {self.age}")
        if self.gender:
            bits.append(f"gender: {self.gender}")
        if self.category:
            bits.append(f"category: {self.category}")
        if self.education_level:
            bits.append(f"education: {self.education_level}")
        if self.annual_income is not None:
            bits.append(f"annual income: Rs {self.annual_income:,.0f}")
        if self.landholding_ha is not None:
            bits.append(f"landholding: {self.landholding_ha} ha")
        if self.state:
            bits.append(f"state: {self.state}")
        if self.special_statuses:
            bits.append("status: " + ", ".join(self.special_statuses))
        return "; ".join(bits) if bits else "(nothing known yet)"


class CriterionVerdict(BaseModel):
    """One eligibility rule, assessed against the profile."""

    criterion: str
    verdict: str  # met | not_met | unknown
    reason: str = ""
    needs_slot: str | None = None  # which profile field would settle it


class EligibilityAssessment(BaseModel):
    scheme_name: str
    overall: str  # likely_eligible | likely_ineligible | insufficient_info
    criteria: list[CriterionVerdict] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    # One question per UNKNOWN criterion, asked together.
    #
    # Asking one at a time was the original design and it was wrong for this
    # hardware: at ~20 s per model call, a scheme with four unresolved
    # criteria cost four round trips and over a minute before the user saw a
    # verdict. Every question is already known after a single assessment, so
    # there is nothing to gain by drip-feeding them.
    questions: list[str] = Field(default_factory=list)

    #: The assessment could not be produced -- the model call raised, or its
    #: output would not parse. Distinct from a genuine insufficient_info, and
    #: the distinction matters to the user: after they have answered every
    #: question, "not enough information to decide" describes THEM, and is a
    #: lie when the truth is that our request failed. Rendering says so.
    failed: bool = False

    @property
    def next_question(self) -> str | None:
        """Back-compat: the first outstanding question."""
        return self.questions[0] if self.questions else None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rerank_score: float | None = None

    @property
    def effective_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


class Citation(BaseModel):
    marker: int  # the [1] the model writes
    scheme_name: str
    section: str
    locator: str
    source_url: str | None = None
    doc_name: str = ""
    score: float = 0.0


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    #: The MODEL CALL failed -- rate limited, overloaded, or it produced only
    #: reasoning. Distinct from insufficient_evidence, which means retrieval
    #: genuinely found nothing: this means we never got to ask.
    #:
    #: Needed because the two are indistinguishable from the outside -- both
    #: return prose and no citations -- and conflating them corrupts any
    #: measurement. During an evaluation a burst of 429s would otherwise be
    #: scored as 85 bad answers, and the report would blame the retriever for
    #: a quota problem.
    model_failed: bool = False
    route: str | None = None
    schemes_used: list[str] = Field(default_factory=list)
    #: Langfuse trace id for this turn, when tracing is on. Returned to the
    #: client so a thumbs-up can be attached to THIS answer after the fact --
    #: user feedback needs something to point at.
    trace_id: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    verification: dict | None = None
    # Discovery / eligibility results, when the route produced them
    suggested_schemes: list[dict] = Field(default_factory=list)
    assessment: "EligibilityAssessment | None" = None
    profile_summary: str | None = None
    awaiting_answer: bool = False  # the agent asked a question and wants a reply
