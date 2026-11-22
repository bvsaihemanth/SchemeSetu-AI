"""LangGraph orchestration across the six agents.

    START
      │
      ▼
   UNDERSTAND ─────────────────────────────────┐
   resolver (0 ms) → heuristics → LLM if needed│
   merges volunteered facts into the profile   │
      │                                        │
      ├─ vague, nothing resolved ──────────► CLARIFY ──► END
      │
      ├─ resumed interview ──────────────────┐
      ▼                                      ▼
   DISCOVERY   ELIGIBILITY   COMPARISON   APPLICATION   DOCUMENTS   GENERAL
   (HyDE /     (slots +      (per-scheme  (process +    (checklist  (plain
    facets)     assess)       fan-out)     FAQ)          whole)      RAG)
      └────────────┴─────────────┴────────────┴─────────────┴──────────┘
                                 │
                                 ▼
                             EVIDENCE            ← citations + numeric grounding
                                 │                 (always; cheap and deterministic)
                                 ▼
                                END

ROUTING IS DETERMINISTIC FIRST. The alias resolver and keyword heuristics
settle most queries with no LLM call at all, and the router call is a fallback.
On a 4B at ~20 tok/s that choice is worth about five seconds on every query,
which is the difference between the 15-second budget and missing it.

STATE SURVIVES TURNS. The eligibility interview asks one question at a time,
so the profile and the scheme under assessment live in graph state behind a
checkpointer. Each terminal invocation passes the same thread_id, and the
answer carries awaiting_answer=True when the agent wants a reply.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import comparison as comparison_agent
from .agents import discovery as discovery_agent
from .agents import eligibility as eligibility_agent
from .agents import evidence as evidence_agent
from .agents import sections as sections_agent
from .agents.understand import understand_query
from .config import get_settings
from . import tracing as obs
from .generate import generate_answer, insufficient_answer
from .models import Answer, EligibilityAssessment, RetrievedChunk, UserProfile
from .observability import record_query
from .prompts import (
    APPLICATION_SYSTEM,
    COMPARISON_SYSTEM,
    COMPARISON_USER_TEMPLATE,
    DISCOVERY_SYSTEM,
    DOCUMENTS_SYSTEM,
)
from .retriever import get_retriever

logger = logging.getLogger(__name__)


def _keep_last(_: Any, new: Any) -> Any:
    return new


class RagState(TypedDict, total=False):
    # input
    question: str
    history: list[dict[str, str]]
    filters: dict[str, Any]
    # Eligibility answers already paired to their questions by the client:
    # [{"question": "Are you a farmer?", "answer": "yes"}, ...]. Present only
    # on a turn that answers the agent's own questions. See node_understand.
    answers: list[dict[str, str]]

    # understand
    intent: str
    search_query: str
    schemes: list[str]
    scheme_doc_ids: list[str]
    scheme_confidence: str
    state_name: str | None
    needs_clarification: bool
    clarifying_question: str | None

    # Persisted across turns, as a PLAIN DICT. LangGraph's checkpointer warns
    # on deserialising unregistered pydantic types and will refuse them in a
    # future version, so the one value that must survive restarts travels as
    # a dict and is rehydrated on read.
    profile: Annotated[dict | None, _keep_last]
    pending_scheme: str | None
    pending_doc_ids: list[str]
    pending_question: str | None
    questions_asked: int

    # retrieval / answer
    chunks: Annotated[list[RetrievedChunk], _keep_last]
    top_score: float
    insufficient: bool
    answer: Annotated[Answer | None, _keep_last]
    suggested_schemes: list[dict]
    # Dict, not the model: the checkpointer warns on unregistered pydantic
    # types and will refuse them in a future version.
    assessment: Annotated[dict | None, _keep_last]

    timings: dict[str, float]


# Per-intent retrieval policy for the GENERAL path. The specialist agents own
# their own retrieval, because section filters alone do not express what they
# need (per-scheme fan-out, whole-section ordering, facet filters).
RETRIEVAL_POLICY: dict[str, dict[str, Any]] = {
    "benefits": {"sections": ["benefits", "faq"], "final_k": 6},
    "general": {"sections": None, "final_k": 6},
}


# --------------------------------------------------------------------------
# understand
# --------------------------------------------------------------------------


def _profile_of(state: RagState) -> UserProfile:
    """Rehydrate the profile from state, whichever shape it is in."""
    raw = state.get("profile")
    if isinstance(raw, UserProfile):
        return raw
    if isinstance(raw, dict):
        try:
            return UserProfile(**raw)
        except Exception:
            logger.warning("Stored profile could not be parsed; starting fresh")
    return UserProfile()


def node_understand(state: RagState) -> dict:
    t0 = time.perf_counter()
    question = state["question"]
    profile = _profile_of(state)

    answers = state.get("answers") or []

    # A turn that answers the agent's own question is not a new query: merge
    # it into the profile and resume the interview rather than re-routing.
    #
    # Structured answers are proof on their own -- the client only has them
    # because we asked -- so they skip the word-count heuristic, which exists
    # only to tell a short free-text reply from a fresh question.
    resuming = bool(state.get("pending_scheme")) and (
        bool(answers) or len(question.split()) <= 25
    )

    # Structured answers need no routing at all. We know the intent (we asked
    # the questions), we know the scheme (it is pending), and the "question"
    # text is a client-side placeholder like "Answered: yes, no, yes" that
    # carries no information. Calling the router on it would cost an LLM round
    # trip to rediscover what state already holds -- and risk it deciding that
    # a string of yes/no is too vague to search, which is the shape of the bug
    # that killed the interview on turn two.
    if resuming and answers:
        profile = eligibility_agent.record_answers(profile, answers)
        return {
            "intent": "eligibility",
            "search_query": state.get("pending_scheme") or question,
            "schemes": [state["pending_scheme"]],
            "scheme_doc_ids": state.get("pending_doc_ids") or [],
            "scheme_confidence": "exact",
            "state_name": profile.state,
            "needs_clarification": False,
            "clarifying_question": None,
            "profile": profile.model_dump(mode="json"),
            "answers": [],
            "timings": {"understand_ms": (time.perf_counter() - t0) * 1000},
        }

    plan = understand_query(question, history=state.get("history") or [])

    # Anything the user volunteered about themselves is worth keeping, on any
    # route -- "I'm a farmer" said during a benefits question still narrows a
    # later eligibility check.
    #
    # But the LLM slot call is only worth its ~5 seconds when there is
    # plausibly something to extract. The free heuristics run first, and the
    # model is consulted only for eligibility/discovery (where the profile is
    # the point), when resuming an interview, or when the heuristics already
    # found a fact and might have missed neighbouring ones. "How do I apply
    # for PM-KISAN?" states nothing about the user, so it skips the call.
    #
    # (Structured answers returned above, so everything from here is prose.)
    wants_profile = resuming or plan.intent in {"eligibility", "discovery"}
    if wants_profile or eligibility_agent.heuristic_slots(question):
        profile = eligibility_agent.update_profile(
            profile, question, use_llm=wants_profile
        )

    # Pair the reply with the question we asked. "yes" on its own is
    # meaningless to the next assessment; "Is the institution AICTE approved?
    # -> yes" is the fact we were missing. Without this the interview asks,
    # the user answers, and the answer is thrown away.
    #
    # This is the free-text path, used by the terminal client. `asked` may
    # hold several questions joined by " | ", so a reply like "1 yes, 2 no"
    # still has to be re-associated by the model -- which is exactly what the
    # web UI's paired answers avoid.
    if resuming and (asked := state.get("pending_question")):
        pair = f"Q: {asked} A: {question.strip()}"
        if pair not in profile.notes:
            profile.notes.append(pair[:300])

    intent = plan.intent
    doc_ids = plan.scheme_doc_ids
    schemes = plan.schemes
    needs_clarification = plan.needs_clarification
    clarifying = plan.clarifying_question

    if resuming and not doc_ids:
        intent = "eligibility"
        doc_ids = state.get("pending_doc_ids") or []
        schemes = [state["pending_scheme"]] if state.get("pending_scheme") else []
        # A reply to the agent's OWN question is never "too vague to search".
        # "yes" matches the vague-input pattern in heuristic_plan, which
        # returns early with needs_clarification=True before resolution ever
        # runs -- so the router sent a perfectly good interview answer to
        # CLARIFY and the interview died on turn two.
        needs_clarification = False
        clarifying = None

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "intent": intent,
        "search_query": plan.search_query or question,
        "schemes": schemes,
        "scheme_doc_ids": doc_ids,
        "scheme_confidence": plan.scheme_confidence,
        "state_name": plan.state or profile.state,
        "needs_clarification": needs_clarification,
        "clarifying_question": clarifying,
        "profile": profile.model_dump(mode="json"),
        # Consumed. Cleared so a checkpointed copy cannot be folded into the
        # profile a second time on the next turn.
        "answers": [],
        # Fresh dict, NOT merged with state: timings persist in the
        # checkpointer, so merging showed last turn's application_ms beside
        # this turn's eligibility_ms and made the numbers unreadable.
        "timings": {"understand_ms": elapsed},
    }


def node_clarify(state: RagState) -> dict:
    question = state.get("clarifying_question") or (
        "What would you like to know? You can name a scheme, or tell me about "
        "your situation — for example \"I'm a farmer in Telangana with 2 "
        "hectares\" — and I'll suggest schemes that may fit."
    )
    return {
        "answer": Answer(
            answer=question, route="clarify", awaiting_answer=True,
            profile_summary=_profile_of(state).describe(),
        )
    }


# --------------------------------------------------------------------------
# specialists
# --------------------------------------------------------------------------


def node_discovery(state: RagState) -> dict:
    t0 = time.perf_counter()
    profile = _profile_of(state)
    schemes, result, strategy = discovery_agent.discover(
        state["question"],
        profile,
        state=state.get("state_name"),
        scheme_resolved=state.get("scheme_confidence") in ("exact", "strong"),
    )
    timings = {
        **(state.get("timings") or {}), **result.latency_ms,
        "discovery_ms": (time.perf_counter() - t0) * 1000,
    }

    if not schemes:
        answer = insufficient_answer("discovery")
        answer.profile_summary = profile.describe()
        return {"answer": answer, "chunks": [], "timings": timings}

    # The shortlist is rendered deterministically; the LLM then adds the
    # "why this fits you" prose over the same evidence. If generation fails
    # the user still gets the list, which is the part they need most.
    answer = generate_answer(
        state["question"], result.chunks, route="discovery", system=DISCOVERY_SYSTEM
    )
    answer.suggested_schemes = schemes
    answer.profile_summary = profile.describe()
    if answer.insufficient_evidence or not answer.answer.strip():
        answer = Answer(
            answer=discovery_agent.render(schemes, profile, strategy),
            route="discovery", suggested_schemes=schemes,
            profile_summary=profile.describe(),
        )
    else:
        answer.answer += "\n\n" + discovery_agent.render(schemes, profile, strategy)

    return {"answer": answer, "chunks": result.chunks, "timings": timings,
            "suggested_schemes": schemes, "top_score": result.top_score}


def node_eligibility(state: RagState) -> dict:
    t0 = time.perf_counter()
    settings = get_settings()
    profile = _profile_of(state)
    doc_ids = state.get("scheme_doc_ids") or []
    schemes = state.get("schemes") or []

    # No scheme named: this is really "what am I eligible for", which is
    # discovery over the facets, not an assessment of one scheme.
    if not doc_ids:
        logger.info("Eligibility without a named scheme -> discovery")
        return node_discovery(state)

    scheme_name = schemes[0] if schemes else "this scheme"
    retriever = get_retriever()
    result = retriever.retrieve(
        state.get("search_query") or state["question"],
        gate_query=state["question"],
        doc_ids=doc_ids[:1],
        priority_sections=["eligibility", "faq"],
        final_k=6,
        min_score=0.0,  # the scheme is known; the floor would only exclude
    )

    assessment = eligibility_agent.assess(scheme_name, result.chunks, profile)
    timings = {
        **(state.get("timings") or {}), **result.latency_ms,
        "eligibility_ms": (time.perf_counter() - t0) * 1000,
    }

    rounds = state.get("questions_asked") or 0
    # Rounds, not questions: every outstanding question goes out together, so
    # a scheme with four unknowns takes one round trip, not four.
    if assessment.questions and rounds < settings.eligibility_max_rounds:
        return {
            "answer": Answer(
                answer=eligibility_agent.render_questions(assessment, profile),
                route="eligibility",
                awaiting_answer=True,
                assessment=assessment,
                profile_summary=profile.describe(),
            ),
            "chunks": result.chunks,
            "assessment": assessment.model_dump(mode="json"),
            "pending_scheme": scheme_name,
            "pending_doc_ids": doc_ids[:1],
            "pending_question": " | ".join(assessment.questions),
            "questions_asked": rounds + 1,
            "timings": timings,
        }

    # Out of questions, or nothing left to ask: commit to a verdict and say
    # plainly what is still unknown rather than asking a fifth question.
    return {
        "answer": Answer(
            answer=eligibility_agent.render(assessment, profile),
            route="eligibility",
            assessment=assessment,
            profile_summary=profile.describe(),
            schemes_used=[scheme_name],
        ),
        "chunks": result.chunks,
        "assessment": assessment.model_dump(mode="json"),
        "pending_scheme": None,
        "pending_doc_ids": [],
        "pending_question": None,
        "questions_asked": 0,
        "timings": timings,
    }


def node_comparison(state: RagState) -> dict:
    t0 = time.perf_counter()
    doc_ids = state.get("scheme_doc_ids") or []
    schemes = state.get("schemes") or []

    if len(doc_ids) < 2:
        named = schemes[0] if schemes else "one scheme"
        return {
            "answer": Answer(
                answer=(
                    f"I could only identify {named} in that question. To "
                    f"compare, name both schemes — for example \"compare "
                    f"PMJJBY and PMSBY\"."
                ),
                route="comparison", awaiting_answer=True,
            ),
            "chunks": [],
        }

    by_scheme, top = comparison_agent.retrieve_per_scheme(
        state.get("search_query") or state["question"], doc_ids, schemes
    )
    context, flat = comparison_agent.format_context(by_scheme)
    timings = {
        **(state.get("timings") or {}),
        "comparison_ms": (time.perf_counter() - t0) * 1000,
    }

    if not flat:
        return {"answer": insufficient_answer("comparison"), "chunks": [], "timings": timings}

    answer = generate_answer(
        state["question"], flat, route="comparison",
        system=COMPARISON_SYSTEM, context_text=context,
        user_template=COMPARISON_USER_TEMPLATE, max_tokens=1200,
    )
    answer.schemes_used = list(by_scheme)
    return {"answer": answer, "chunks": flat, "top_score": top, "timings": timings}


def _section_node(state: RagState, kind: str) -> dict:
    t0 = time.perf_counter()
    doc_ids = state.get("scheme_doc_ids") or None
    schemes = state.get("schemes") or []
    question = state["question"]

    if kind == "application":
        chunks = sections_agent.retrieve_application(question, doc_ids)
        system = APPLICATION_SYSTEM
    else:
        chunks = sections_agent.retrieve_documents(question, doc_ids)
        system = DOCUMENTS_SYSTEM

    timings = {**(state.get("timings") or {}), f"{kind}_ms": (time.perf_counter() - t0) * 1000}

    if not chunks:
        return {
            "answer": Answer(
                answer=sections_agent.missing_section_notice(
                    kind, schemes[0] if schemes else None
                ),
                route=kind,
                schemes_used=schemes,
            ),
            "chunks": [], "timings": timings,
        }

    answer = generate_answer(question, chunks, route=kind, system=system)

    # If the model refused or produced nothing, show the real section instead
    # of an apology: it is already a list, and the user came for its contents.
    if answer.insufficient_evidence or not answer.answer.strip():
        logger.info("Falling back to verbatim %s section for %s", kind, schemes)
        from .agents.evidence import resolve_citations

        text, citations = resolve_citations(
            sections_agent.render_verbatim(kind, chunks), chunks
        )
        answer = Answer(
            answer=text, citations=citations, route=kind,
            schemes_used=list(dict.fromkeys(rc.chunk.scheme_name for rc in chunks)),
            latency_ms=answer.latency_ms,
        )

    return {"answer": answer, "chunks": chunks, "timings": timings}


def node_application(state: RagState) -> dict:
    return _section_node(state, "application")


def node_documents(state: RagState) -> dict:
    return _section_node(state, "documents")


def node_general(state: RagState) -> dict:
    t0 = time.perf_counter()
    intent = state.get("intent", "general")
    policy = RETRIEVAL_POLICY.get(intent, RETRIEVAL_POLICY["general"])
    overrides = state.get("filters") or {}

    doc_ids = None
    if state.get("scheme_confidence") in ("exact", "strong"):
        doc_ids = state.get("scheme_doc_ids") or None

    sections = overrides.get("sections", policy["sections"])
    priority = None
    if doc_ids and len(doc_ids) == 1 and "sections" not in overrides:
        priority, sections = sections, None

    result = get_retriever().retrieve(
        state.get("search_query") or state["question"],
        gate_query=state["question"],
        sections=sections,
        priority_sections=priority,
        doc_ids=doc_ids,
        state=overrides.get("state") or state.get("state_name"),
        level=overrides.get("level"),
        final_k=overrides.get("final_k", policy["final_k"]),
    )
    timings = {
        **(state.get("timings") or {}), **result.latency_ms,
        "general_ms": (time.perf_counter() - t0) * 1000,
    }

    if result.insufficient or not result.chunks:
        return {
            "answer": insufficient_answer(intent), "chunks": [],
            "insufficient": True, "top_score": result.top_score, "timings": timings,
        }

    return {
        "answer": generate_answer(state["question"], result.chunks, route=intent),
        "chunks": result.chunks, "top_score": result.top_score, "timings": timings,
    }


# --------------------------------------------------------------------------
# evidence (terminal)
# --------------------------------------------------------------------------


def node_evidence(state: RagState) -> dict:
    """Citations and numeric grounding. Runs on every answer: both layers are
    deterministic and cost about a millisecond."""
    answer = state.get("answer")
    chunks = state.get("chunks") or []

    if answer is None:
        return {"answer": insufficient_answer(state.get("intent"))}
    # Nothing to verify in a question we asked, or a refusal.
    if answer.awaiting_answer or answer.insufficient_evidence or not chunks:
        return {}

    t0 = time.perf_counter()
    # The question and the known profile are legitimate sources for figures
    # the user themselves supplied.
    evidence_agent.verify(answer, chunks, user_text=state.get("question", ""))

    if get_settings().verify_enabled and answer.route in {"eligibility", "comparison"}:
        extra = evidence_agent.verify_with_llm(answer, chunks)
        if answer.verification is not None:
            answer.verification["llm"] = extra
        if extra.get("supported") is False:
            answer.answer += (
                "\n\n_Note: parts of this answer could not be verified against "
                "the source documents. Confirm on the official scheme page._"
            )

    return {
        "answer": answer,
        "timings": {
            **(state.get("timings") or {}),
            "evidence_ms": (time.perf_counter() - t0) * 1000,
        },
    }


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

Route = Literal[
    "clarify", "discovery", "eligibility", "comparison",
    "application", "documents", "general",
]


def route_after_understand(state: RagState) -> Route:
    intent = state.get("intent", "general")
    resolved = state.get("scheme_confidence") in ("exact", "strong")
    doc_ids = state.get("scheme_doc_ids") or []
    profile = _profile_of(state)

    # Clarify only when there is genuinely nothing to work with: no scheme, no
    # profile, and the router asked. A resolved scheme already overrides this
    # inside understand_query.
    if state.get("needs_clarification") and not resolved and not profile.known_slots():
        return "clarify"

    if intent == "comparison":
        # The node itself handles "only one scheme named" -- it can tell the
        # user which one it found, which a reroute to `general` could not.
        return "comparison"
    if intent == "eligibility":
        return "eligibility"
    if intent == "discovery":
        return "discovery"
    if intent == "application":
        return "application"
    if intent == "documents":
        return "documents"

    # An unnamed scheme plus a known profile is a discovery question however
    # the router labelled it: "what help can I get" arrives as `general`.
    if not resolved and profile.known_slots() and len(state["question"].split()) >= 4:
        return "discovery"

    return "general"


def _traced(node_fn, name: str, as_type: str = "agent"):
    """Bracket a graph node in a span.

    Wrapping at graph-construction time rather than decorating each node keeps
    the node functions plain -- the tests call them directly -- and keeps the
    span naming in one place. as_type="agent" because that is what these are:
    Langfuse renders agent spans as the steps of a pipeline, which is exactly
    how the six agents relate to a query.
    """
    def wrapped(state: RagState) -> dict:
        with obs.observed(name, as_type=as_type) as span:
            result = node_fn(state)
            # Only the node's own additions, not the whole accumulated state:
            # a full RagState per node would repeat the chunks and the profile
            # at every step and bury the one thing that changed.
            span.update(
                output={
                    k: v for k, v in result.items()
                    if k in ("intent", "route", "scheme_confidence", "schemes",
                             "needs_clarification", "questions_asked",
                             "awaiting_answer", "top_score")
                },
                metadata={k: round(v, 1) for k, v in
                          (result.get("timings") or {}).items()},
            )
            return result
    wrapped.__name__ = getattr(node_fn, "__name__", name)
    return wrapped


def build_graph(checkpointer=None):
    g = StateGraph(RagState)

    # Each node is traced as an agent span, except evidence -- it judges the
    # answer rather than producing one, and "evaluator" is the span type
    # Langfuse has for exactly that.
    g.add_node("understand", _traced(node_understand, "understand"))
    g.add_node("clarify", _traced(node_clarify, "clarify"))
    g.add_node("discovery", _traced(node_discovery, "discovery"))
    g.add_node("eligibility", _traced(node_eligibility, "eligibility"))
    g.add_node("comparison", _traced(node_comparison, "comparison"))
    g.add_node("application", _traced(node_application, "application"))
    g.add_node("documents", _traced(node_documents, "documents"))
    g.add_node("general", _traced(node_general, "general"))
    g.add_node("evidence", _traced(node_evidence, "evidence", as_type="evaluator"))

    g.add_edge(START, "understand")
    g.add_conditional_edges(
        "understand",
        route_after_understand,
        {
            "clarify": "clarify",
            "discovery": "discovery",
            "eligibility": "eligibility",
            "comparison": "comparison",
            "application": "application",
            "documents": "documents",
            "general": "general",
        },
    )
    g.add_edge("clarify", END)
    for node in ("discovery", "eligibility", "comparison", "application",
                 "documents", "general"):
        g.add_edge(node, "evidence")
    g.add_edge("evidence", END)

    return g.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    """Compiled once per process, behind a persistent checkpointer.

    SQLite rather than MemorySaver: the eligibility interview spans turns, and
    losing a half-finished interview to an API restart is a real failure. One
    file, no service. Swap for Postgres when you run more than one replica.
    """
    global _graph
    if _graph is None:
        checkpointer = None
        settings = get_settings()
        try:
            from pathlib import Path

            from langgraph.checkpoint.sqlite import SqliteSaver

            path = Path(settings.processed_dir) / "conversations.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            import sqlite3

            conn = sqlite3.connect(str(path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info("Conversation state -> %s", path)
        except Exception as exc:
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning(
                "SQLite checkpointer unavailable (%s); using in-memory state. "
                "Eligibility interviews will not survive a restart. "
                "Fix: pip install langgraph-checkpoint-sqlite",
                type(exc).__name__,
            )
            checkpointer = MemorySaver()
        _graph = build_graph(checkpointer=checkpointer)
    return _graph


def get_profile(thread_id: str = "default") -> UserProfile:
    """Read the stored profile without running the graph.

    Uses the checkpointer's snapshot, so `/profile` costs nothing and shows
    exactly what the agents will use on the next turn -- rather than a second
    copy kept in the CLI that could drift from it.
    """
    try:
        snapshot = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        logger.warning("Could not read conversation state: %s", exc)
        return UserProfile()
    values = getattr(snapshot, "values", None) or {}
    profile = values.get("profile")
    if isinstance(profile, UserProfile):
        return profile
    if isinstance(profile, dict):
        try:
            return UserProfile(**profile)
        except Exception:
            pass
    return UserProfile()


def pending_question(thread_id: str = "default") -> str | None:
    """The scheme an interview is mid-way through, if any."""
    try:
        snapshot = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return None
    return (getattr(snapshot, "values", None) or {}).get("pending_scheme")


def run_query(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    filters: dict[str, Any] | None = None,
    thread_id: str = "default",
    answers: list[dict[str, str]] | None = None,
) -> Answer:
    """Run one turn.

    `answers` carries eligibility replies already paired to their questions,
    as the web UI's Yes/No buttons produce them. Always passed (empty when
    there are none) so a checkpointed set from the previous turn is overwritten
    rather than silently re-applied.
    """
    started = time.perf_counter()
    graph = get_graph()

    # One trace per turn, grouped into a session by thread_id.
    #
    # The session is what makes the eligibility interview legible: three turns
    # that ask five questions, receive five answers and reach a verdict are
    # one story, and as separate traces they read as three unrelated requests.
    # Tags are set here because in SDK v5 trace attributes can only be set on
    # entry -- so the route, which is not known until the router has run,
    # becomes a score below rather than a tag.
    with obs.trace_context(
        session_id=thread_id,
        trace_name="query",
        tags=["answers" if answers else "question"],
    ), obs.observed(
        "query",
        as_type="chain",
        input={"question": question, "answers": answers or []},
    ) as root:
        trace_id = obs.current_trace_id()

        final: RagState = graph.invoke(
            {
                "question": question,
                "history": history or [],
                "filters": filters or {},
                "answers": answers or [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        answer = final.get("answer") or insufficient_answer(final.get("intent"))
        timings = final.get("timings") or {}
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        answer.latency_ms = {k: round(v, 1) for k, v in timings.items()}
        answer.route = answer.route or final.get("intent")
        # Handed back to the client so a thumbs-up can be attached to THIS
        # answer later. Without it, user feedback has nothing to point at.
        answer.trace_id = trace_id

        outcome = (
            "clarify" if answer.route == "clarify"
            else "question" if answer.awaiting_answer
            else "insufficient" if answer.insufficient_evidence
            else "answered"
        )

        root.update(
            output={
                "answer": answer.answer,
                "citations": len(answer.citations),
                "route": answer.route,
                "outcome": outcome,
            },
            metadata={
                "thread_id": thread_id,
                "schemes_used": answer.schemes_used,
                "scheme_confidence": final.get("scheme_confidence"),
                "chunks": len(final.get("chunks") or []),
                "awaiting_answer": answer.awaiting_answer,
                **{k: round(v, 1) for k, v in timings.items()},
            },
        )

        # Scores, not metadata, for anything worth TRENDING or filtering on:
        # metadata is visible per-trace, scores are chartable across traces.
        obs.score("route", answer.route or "unknown")
        obs.score("outcome", outcome)
        obs.score("insufficient_evidence", answer.insufficient_evidence)
        obs.score("citations", len(answer.citations))
        obs.score("latency_ms", round(timings["total_ms"], 1))
        if final.get("scheme_confidence"):
            # The entity resolver is the component that caused the original
            # "PM-KISAN returns four wrong schemes" bug, so its confidence
            # distribution is worth watching rather than trusting.
            obs.score("scheme_resolution", str(final["scheme_confidence"]))
        if answer.assessment is not None:
            obs.score("eligibility_outcome", answer.assessment.overall)
            # Was the verdict a real verdict, or a failed call wearing one?
            obs.score("assessment_failed", answer.assessment.failed)
        if answer.verification:
            # A dict, not a model -- see Answer.verification. getattr() here
            # would silently never fire.
            verdict = answer.verification.get("verdict")
            if verdict:
                obs.score("verification", str(verdict))

    record_query(
        route=answer.route or "unknown",
        outcome=outcome,
        latency_ms=timings,
        chunk_count=len(final.get("chunks") or []),
        top_score=final.get("top_score"),
        citation_count=len(answer.citations),
    )
    return answer
