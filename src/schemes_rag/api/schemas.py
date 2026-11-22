"""API request/response models. Separate from internal models on purpose:
the wire contract should not change just because an internal field is
renamed."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    sections: list[str] | None = None
    scheme_names: list[str] | None = None
    state: str | None = None
    level: str | None = None
    final_k: int | None = Field(default=None, ge=1, le=20)


class AnsweredQuestion(BaseModel):
    """One eligibility question with the user's answer already attached.

    Pairing happens in the client, where a button press unambiguously belongs
    to the question on screen. The alternative -- posting "1)yes,2)yes,3)no"
    as free text and asking the model to re-associate the answers with its own
    criteria -- is what made the interview unusable: it would re-ask with an
    empty question list and then report "not enough information" about a
    scheme it had just been given six facts about.
    """

    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=100)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(
        default=None, description="Pass the same id across turns to keep clarification context"
    )
    history: list[dict[str, str]] = Field(default_factory=list)
    filters: QueryFilters | None = None
    answers: list[AnsweredQuestion] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Answers to the questions the previous response returned in "
            "assessment.questions. Requires the same thread_id."
        ),
    )


class CitationOut(BaseModel):
    marker: int
    scheme_name: str
    section: str
    locator: str
    source_url: str | None = None
    score: float = 0.0


class SuggestedScheme(BaseModel):
    scheme_name: str
    scheme_acronym: str | None = None
    source_url: str | None = None
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    insufficient_evidence: bool = False
    route: str | None = None
    schemes_used: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    #: Langfuse trace id, when tracing is on. The client sends it back to
    #: POST /feedback so a rating lands on the answer it was about. Null when
    #: tracing is off, which is how the UI knows to hide the buttons.
    trace_id: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)

    # True when the agent wants a reply. The client must answer on the SAME
    # thread_id or the interview restarts from nothing. Two shapes:
    #   assessment.questions non-empty -> yes/no questions; walk the user
    #       through them and post back `answers`
    #   otherwise (route="clarify")    -> free text; post back `question`
    awaiting_answer: bool = False
    profile_summary: str | None = None
    suggested_schemes: list[SuggestedScheme] = Field(default_factory=list)
    assessment: dict | None = None
    verification: dict | None = None


class FeedbackRequest(BaseModel):
    """A rating for one answer, written to Langfuse as a score.

    The one signal no automated check produces. The deterministic scores say
    whether an answer was grounded and cited; only a person can say whether it
    was any USE -- and in this domain, whether it was right about a scheme
    they actually qualify for.
    """

    trace_id: str = Field(min_length=1, max_length=64)
    # 1 = helpful, 0 = not. Kept binary deliberately: a thumbs pair gets
    # used, a 1-5 star rating gets ignored or answered inconsistently.
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


class SearchHit(BaseModel):
    score: float
    scheme_name: str
    section: str
    locator: str
    text: str
    source_url: str | None = None


class ResolvedScheme(BaseModel):
    scheme_name: str
    confidence: str
    score: float
    matched_on: str


class SearchResponse(BaseModel):
    """Retrieval without generation. The endpoint you debug with: it shows
    exactly what the LLM would have been given, and what the search was
    scoped to."""

    query: str
    insufficient: bool
    top_score: float
    resolved_schemes: list[ResolvedScheme] = Field(default_factory=list)
    scoped_to_schemes: bool = False
    hits: list[SearchHit] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    llm: bool
    collection: str
    points: int | None = None
    #: Whether Langfuse tracing is live. The UI uses it to say so in the
    #: header, because "I turned tracing on but see no traces" is otherwise
    #: indistinguishable from "the app never picked up my keys".
    tracing: bool = False
    detail: str | None = None
