"""Exercises all six agents through the graph, with a scripted LLM.

What this proves without Ollama running:
  - routing reaches the right agent for each question shape
  - the Eligibility interview spans turns and accumulates a profile
  - Comparison retrieves PER SCHEME and never blends two schemes' sources
  - Document/Application agents return their whole section in order
  - Discovery groups results by scheme and uses HyDE only when unresolved
  - Evidence grounds figures and strips invented citation markers

    python -m tests.test_agents
"""

from __future__ import annotations

import math
import os
import re
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["RETRIEVAL_MIN_SCORE"] = "0.30"
os.environ["VERIFY_ENABLED"] = "false"

CORPUS = "/mnt/user-data/uploads/myscheme-ai/schemes_txt"
DIM = 1024
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "it", "this", "that", "what", "which", "who", "how", "do", "does", "can",
    "i", "my", "me", "you", "your", "at", "by", "with", "from", "as", "if",
    "any", "there", "their", "have", "has", "will", "would", "not", "am",
}


class HashEmbedder:
    def _vec(self, text):
        v = [0.0] * DIM
        for w in (x for x in _WORD.findall(text.lower()) if x not in _STOP):
            v[hash(w) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _sp(self, text):
        out = {}
        for w in (x for x in _WORD.findall(text.lower()) if x not in _STOP):
            out[hash(w) % 250_000] = out.get(hash(w) % 250_000, 0.0) + 1.0
        return out

    def encode_documents(self, texts):
        from schemes_rag.embeddings import EmbeddingResult

        texts = list(texts)
        return EmbeddingResult([self._vec(t) for t in texts], [self._sp(t) for t in texts])

    def encode_query(self, text):
        from schemes_rag.embeddings import EmbeddingResult

        return EmbeddingResult([self._vec(text)], [self._sp(text)])

    def encode_queries(self, texts):
        from schemes_rag.embeddings import EmbeddingResult

        texts = list(texts)
        return EmbeddingResult([self._vec(t) for t in texts], [self._sp(t) for t in texts])


class ScriptedLLM:
    """Stands in for Qwen. Records which prompts it was asked, so a test can
    assert that an agent did or did not spend an LLM call."""

    def __init__(self):
        self.calls: list[str] = []
        self.slot_reply: str | None = None
        self.assess_reply: str | None = None

    def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
        if "Classify the user's question" in system:
            self.calls.append("router")
            q = user.lower()
            intent = (
                "documents" if "document" in q or "papers" in q
                else "comparison" if (" vs " in q or "compare" in q)
                else "application" if ("how to apply" in q or "how do i apply" in q)
                else "eligibility" if ("eligib" in q or "who can apply" in q or "qualify" in q)
                else "discovery" if ("schemes for" in q or "what can i get" in q or "suggest" in q)
                else "benefits" if "benefit" in q
                else "general"
            )
            asked = user.split("Question:")[-1].split("\n")[0].strip()
            return (
                f'{{"intent": "{intent}", "schemes": [], "state": null, '
                f'"needs_clarification": false, "clarifying_question": null, '
                f'"search_query": "{asked}"}}'
            )

        if "Extract facts about the USER" in system:
            self.calls.append("slots")
            return self.slot_reply or "{}"

        if "You assess whether a person qualifies" in system:
            self.calls.append("assess")
            if self.assess_reply is not None:
                return self.assess_reply
            return (
                '{"criteria": ['
                '{"criterion": "Applicant must be a landholding farmer",'
                ' "verdict": "met", "reason": "user is a farmer", "needs_slot": null},'
                '{"criterion": "Cultivable landholding in their own name",'
                ' "verdict": "unknown", "reason": "landholding not stated",'
                ' "needs_slot": "landholding_ha"}],'
                '"overall": "insufficient_info",'
                '"questions": ["How much cultivable land do you hold, in acres or hectares?"]}'
            )

        if "You write a short passage" in system:
            self.calls.append("hyde")
            return (
                "Eligible beneficiaries under this scheme include small and "
                "marginal farmers holding cultivable land, who receive financial "
                "assistance subject to conditions as notified by the implementing "
                "department. Applicants from rural areas below the poverty line "
                "are given priority for enrolment and income support."
            )

        if "You compare Indian government schemes" in system:
            self.calls.append("compare")
            # Cite the FIRST marker of each scheme, read out of the context we
            # were actually handed. Hardcoding [1] and [3] silently tested the
            # wrong thing: with 4 chunks per scheme, [3] is still scheme one.
            # This simulates a model that attributes correctly per scheme,
            # which is exactly what per-scheme retrieval is meant to enable.
            first: dict[str, int] = {}
            for m in re.finditer(r"\[(\d+)\] \(([^|]+)\|", user):
                name = m.group(2).strip()
                first.setdefault(name, int(m.group(1)))
            parts = [f"{name} provides cover [{n}]." for name, n in first.items()]
            markers = "".join(f"[{n}]" for n in first.values())
            return " ".join(parts) + f"\n\nKey differences:\n- Different risks {markers}."

        if "documents required for this scheme" in system:
            self.calls.append("documents")
            return "- Aadhaar card [1]\n- Bank account details [1]"

        if "official application process" in system:
            self.calls.append("application")
            return "1. Visit the portal [1].\n2. Complete registration [1]."

        if "You help someone find Indian government schemes" in system:
            self.calls.append("discovery")
            return "This scheme supports landholding farmers with income support [1]."

        self.calls.append("answer")
        return "Landholding farmers' families are eligible [1]."

    def health(self):
        return True

    def available_models(self):
        return ["scripted"]


def build_env(llm):
    from qdrant_client import QdrantClient

    import schemes_rag.agents.eligibility as elig_mod
    import schemes_rag.agents.hyde as hyde_mod
    import schemes_rag.agents.understand as understand_mod
    import schemes_rag.embeddings as embeddings_mod
    import schemes_rag.generate as gen_mod
    import schemes_rag.graph as graph_mod
    import schemes_rag.ingest.pipeline as pipeline_mod
    import schemes_rag.llm as llm_mod
    import schemes_rag.retriever as retr_mod
    import schemes_rag.vectorstore as vs_mod
    from schemes_rag.scheme_index import reset_scheme_index
    from schemes_rag.vectorstore import VectorStore

    embedder = HashEmbedder()
    store = VectorStore(client=QdrantClient(":memory:"))

    for mod, attr in ((embeddings_mod, "get_embedder"), (vs_mod, "get_vectorstore"),
                      (llm_mod, "get_llm")):
        fn = getattr(mod, attr, None)
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()

    embeddings_mod.get_embedder = lambda: embedder
    pipeline_mod.get_embedder = lambda: embedder
    pipeline_mod.get_vectorstore = lambda: store
    vs_mod.get_vectorstore = lambda: store
    retr_mod.get_embedder = lambda: embedder
    retr_mod.get_vectorstore = lambda: store
    retr_mod._retriever = None
    llm_mod.get_llm = lambda: llm
    gen_mod.get_llm = lambda: llm
    understand_mod.get_llm = lambda: llm
    elig_mod.get_llm = lambda: llm
    hyde_mod.get_llm = lambda: llm
    graph_mod._graph = None

    from pathlib import Path

    Path("data/processed/ingest_manifest.json").unlink(missing_ok=True)
    Path("data/processed/conversations.sqlite").unlink(missing_ok=True)
    pipeline_mod.run_ingest(corpus_dir=CORPUS, recreate=True)
    reset_scheme_index()
    return store


def main() -> int:  # noqa: C901
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_agents"])

    if not os.path.isdir(CORPUS):
        print(f"SKIP: corpus not available at {CORPUS}")
        return 0

    llm = ScriptedLLM()
    build_env(llm)

    import schemes_rag.graph as G
    from schemes_rag.models import UserProfile

    # ---------------- 1. Document Checklist ----------------
    print("--- AGENT 5: Document Checklist ---")
    llm.calls.clear()
    ans = G.run_query("What documents are required for PMFBY?", thread_id="t-doc")
    print(f"  route={ans.route}  citations={len(ans.citations)}  llm_calls={llm.calls}")
    assert ans.route == "documents", ans.route
    assert "documents" in llm.calls, llm.calls
    assert ans.schemes_used == ["Pradhan Mantri Fasal Bima Yojna (PMFBY)"], ans.schemes_used
    print(f"  scheme scoped to: {ans.schemes_used[0]}")

    # ---------------- 2. Application Guidance ----------------
    print("\n--- AGENT 4: Application Guidance ---")
    llm.calls.clear()
    ans = G.run_query("How do I apply for PM-KISAN?", thread_id="t-app")
    print(f"  route={ans.route}  llm_calls={llm.calls}")
    assert ans.route == "application", ans.route
    assert "application" in llm.calls, llm.calls
    assert ans.schemes_used == ["Pradhan Mantri Kisan Samman Nidhi"], ans.schemes_used

    # ---------------- 3. Comparison, per-scheme ----------------
    print("\n--- AGENT 3: Benefits & Comparison ---")
    llm.calls.clear()
    ans = G.run_query(
        "compare Pradhan Mantri Jeevan Jyoti Bima Yojana and Pradhan Mantri "
        "Suraksha Bima Yojana", thread_id="t-cmp",
    )
    print(f"  route={ans.route}  schemes={ans.schemes_used}  llm_calls={llm.calls}")
    assert ans.route == "comparison", ans.route
    assert len(ans.schemes_used) == 2, ans.schemes_used
    # Both schemes must contribute sources, or the model cannot compare.
    cited = {c.scheme_name for c in ans.citations}
    print(f"  cited schemes: {sorted(cited)}")
    assert len(cited) == 2, f"comparison cited only {cited}"
    print("  -> per-scheme retrieval gave both schemes their own sources")

    # ---------------- 4. Eligibility interview, multi-turn ----------------
    print("\n--- AGENT 2: Eligibility Assessment (multi-turn) ---")
    thread = "t-elig"
    llm.calls.clear()
    ans = G.run_query("Am I eligible for PM-KISAN?", thread_id=thread)
    print(f"  turn 1: route={ans.route} awaiting={ans.awaiting_answer}")
    print(f"          asked: {ans.answer[:72]}")
    assert ans.route == "eligibility", ans.route
    assert ans.awaiting_answer, "should have asked a question"
    assert G.pending_question(thread) == "Pradhan Mantri Kisan Samman Nidhi"

    # answer the question; the interview must resume, not re-route
    llm.assess_reply = (
        '{"criteria": ['
        '{"criterion": "Applicant must be a landholding farmer",'
        ' "verdict": "met", "reason": "user is a farmer", "needs_slot": null},'
        '{"criterion": "Cultivable landholding in their own name",'
        ' "verdict": "met", "reason": "holds 0.81 ha", "needs_slot": null}],'
        '"overall": "likely_eligible", "questions": []}'
    )
    ans = G.run_query("I am a farmer with 2 acres in Telangana", thread_id=thread)
    profile = G.get_profile(thread)
    v2 = ans.verification or {}
    assert not v2.get("figures_ungrounded"), (
        f"the user's own figures were flagged as hallucinated: "
        f"{v2.get('figures_ungrounded')}"
    )
    print(f"  turn 2: route={ans.route} awaiting={ans.awaiting_answer}")
    print(f"          profile: {profile.describe()}")
    print(f"          verdict: {ans.answer.splitlines()[0]}")
    assert ans.route == "eligibility", ans.route
    assert not ans.awaiting_answer, "should have concluded"
    assert profile.occupation == "farmer", profile
    assert profile.landholding_ha is not None, profile
    assert profile.state == "Telangana", profile
    assert ans.assessment and ans.assessment.overall == "likely_eligible"
    assert "ELIGIBLE" in ans.answer
    print("  -> interview spanned turns and accumulated the profile")
    llm.assess_reply = None

    # ---------------- 4b. a one-word reply must resume the interview ---------
    print("\n--- One-word answers resume the interview (not CLARIFY) ---")
    thread2 = "t-yesno"
    llm.assess_reply = (
        '{"criteria": [{"criterion": "Institution must be AICTE approved",'
        ' "verdict": "unknown", "reason": "not stated", "needs_slot": "occupation"}],'
        '"overall": "insufficient_info",'
        '"questions": ["Is the institution AICTE approved?"]}'
    )
    ans = G.run_query("am i eligible for PM-KISAN", thread_id=thread2)
    print(f"  turn 1: route={ans.route} awaiting={ans.awaiting_answer}")
    print(f"          asked: {ans.answer[:60]}")
    assert ans.awaiting_answer, "should have asked"

    # "yes" matches the vague-input pattern, which previously routed to CLARIFY
    # and killed the interview on turn two.
    llm.assess_reply = (
        '{"criteria": [{"criterion": "Institution must be AICTE approved",'
        ' "verdict": "met", "reason": "user confirmed", "needs_slot": null}],'
        '"overall": "likely_eligible", "questions": []}'
    )
    ans = G.run_query("yes", thread_id=thread2)
    print(f"  turn 2 ('yes'): route={ans.route} awaiting={ans.awaiting_answer}")
    assert ans.route == "eligibility", f"a reply must not route to {ans.route}"
    assert not ans.awaiting_answer

    # the answer must have reached the assessment, not vanished into notes
    prof = G.get_profile(thread2)
    pairs = [n for n in prof.notes if n.startswith("Q: ")]
    print(f"  recorded Q/A: {pairs}")
    assert pairs, "the user's answer was not paired with the question"
    assert "yes" in pairs[0].lower()
    print("  -> interview resumed and the answer was recorded")
    llm.assess_reply = None

    print("\n--- Timings are per-turn, not cumulative ---")
    keys = set(ans.latency_ms)
    print(f"  keys: {sorted(keys)}")
    assert "application_ms" not in keys, f"stale timing leaked across turns: {keys}"

    # ---------------- 4c. all unknown criteria asked in one round -----------
    print("\n--- Every unresolved criterion is asked in a single round ---")
    llm.assess_reply = (
        '{"criteria": ['
        '{"criterion": "Applicant must be a farmer", "verdict": "met",'
        ' "reason": "stated", "needs_slot": null},'
        '{"criterion": "Must have insurable interest in a notified crop",'
        ' "verdict": "unknown", "reason": "", "needs_slot": "occupation"},'
        '{"criterion": "Tenant farmers must furnish land records",'
        ' "verdict": "unknown", "reason": "", "needs_slot": "landholding_ha"},'
        '{"criterion": "Must apply before the season cut-off",'
        ' "verdict": "unknown", "reason": "", "needs_slot": "state"}],'
        '"overall": "insufficient_info",'
        '"questions": ["Do you have an insurable interest in a notified crop?",'
        ' "How much land do you hold?", "Are you within the season cut-off?"]}'
    )
    llm.calls.clear()
    ans = G.run_query("am i eligible for PMFBY", thread_id="t-batch")
    print(f"  route={ans.route} awaiting={ans.awaiting_answer} llm_calls={llm.calls}")
    for line in ans.answer.splitlines():
        print(f"    {line}")
    assert ans.awaiting_answer
    # all three questions in ONE message, not one per turn
    for q in ("insurable interest", "How much land", "season cut-off"):
        assert q in ans.answer, f"missing question: {q}"
    # and the criteria are shown, including the one already met
    assert "Applicant must be a farmer" in ans.answer
    assert ans.answer.count("[?]") == 3, ans.answer
    print("  -> 3 questions + 4 criteria in one round trip, not 3 round trips")

    # ------- 4d. structured Yes/No answers settle the interview -------------
    #
    # THE REGRESSION THIS GUARDS. Previously the user answered six questions
    # as one string -- "1)yes,2)yes,3)no, 4)yes, 5)yes, 6)yes" -- and the
    # model had to work out which answer belonged to which of its criteria.
    # It could not: it re-rendered with no criteria and no questions, and then
    # concluded "Not enough information to decide" about a scheme it had just
    # been told six facts about.
    #
    # The web UI now pairs each answer with its question client-side, so the
    # association is made by a button press rather than inferred. This asserts
    # the pairing survives the round trip, costs no router call, and reaches a
    # verdict instead of re-asking.
    print("\n--- Structured Yes/No answers reach a verdict (no re-ask) ---")
    questions = list(ans.assessment.questions)
    assert len(questions) == 3, questions

    # The assessor now sees the answers; it resolves what it asked about.
    llm.assess_reply = (
        '{"criteria": ['
        '{"criterion": "Applicant must be a farmer", "verdict": "met",'
        ' "reason": "stated", "needs_slot": null},'
        '{"criterion": "Must have insurable interest in a notified crop",'
        ' "verdict": "met", "reason": "user answered yes", "needs_slot": null},'
        '{"criterion": "Tenant farmers must furnish land records",'
        ' "verdict": "met", "reason": "user answered yes", "needs_slot": null},'
        '{"criterion": "Must apply before the season cut-off",'
        ' "verdict": "met", "reason": "user answered yes", "needs_slot": null}],'
        '"overall": "likely_eligible", "questions": []}'
    )
    llm.calls.clear()
    paired = [{"question": q, "answer": "yes"} for q in questions]
    ans = G.run_query(
        "Answered 3 eligibility question(s).",
        thread_id="t-batch",
        answers=paired,
    )
    print(f"  route={ans.route} awaiting={ans.awaiting_answer} llm_calls={llm.calls}")
    for line in ans.answer.splitlines():
        print(f"    {line}")

    assert ans.route == "eligibility", ans.route
    assert not ans.awaiting_answer, "answered questions must not be re-asked"
    assert ans.assessment.overall == "likely_eligible", ans.assessment.overall
    assert "ELIGIBLE" in ans.answer, ans.answer

    # No router call: the intent and the scheme were already in state, and the
    # placeholder question text carries nothing worth classifying.
    assert "router" not in llm.calls, llm.calls
    # No slot-extraction call either: "yes" is not prose to mine.
    assert "slots" not in llm.calls, llm.calls

    # Each answer is stored against ITS OWN question, one note apiece -- not
    # joined into a single "q1 | q2 | q3" note with one lump of answers.
    profile = G.get_profile("t-batch")
    notes = [n for n in profile.notes if n.startswith("Q: ")]
    for q in questions:
        assert any(n == f"Q: {q} A: yes" for n in notes), f"unpaired: {q}\n{notes}"
    assert not any(" | " in n for n in notes), f"answers were lumped: {notes}"
    print(f"  -> verdict reached; {len(questions)} answers stored paired, 0 router calls")

    # And the Yes/No answers fed the profile for free, with no LLM call: the
    # occupation came out of "Do you have an insurable interest..." -> no, so
    # only genuinely implied slots are taken. Numeric thresholds in a question
    # must NEVER be read as the user's own figure.
    print("\n--- A Yes implies profile slots; thresholds are not facts ---")
    from schemes_rag.agents.eligibility import slots_from_yes

    cases = [
        ("Are you a farmer?", "yes", {"occupation": "farmer"}),
        ("Are you a farmer?", "no", {}),
        ("Are you a woman?", "yes", {"gender": "female"}),
        ("Are you below the poverty line?", "yes",
         {"special_statuses": ["below_poverty_line"]}),
        # The 2 hectares is the scheme's LIMIT, not the user's holding, so
        # nothing numeric may be taken from it.
        ("Do you hold less than 2 hectares of land?", "yes", {}),
        ("Is your annual income below Rs 2 lakh?", "yes", {}),
    ]
    for question, answer, expected in cases:
        got = slots_from_yes(question, answer)
        print(f"  {question[:44]:46} {answer:3} -> {got}")
        assert got == expected, f"{question!r}/{answer}: expected {expected}, got {got}"
    print("  -> categorical slots inferred, numeric thresholds correctly ignored")

    # A free-text answer is the user's own statement, so numbers in IT are
    # facts -- the opposite of numbers in the question.
    print("\n--- Free-text answers may contribute numeric slots ---")
    from schemes_rag.agents.eligibility import record_answers
    from schemes_rag.models import UserProfile

    p = record_answers(UserProfile(), [
        {"question": "How much land do you hold?", "answer": "2 hectares"},
        {"question": "What is your annual income?", "answer": "income 1.5 lakh"},
    ])
    print(f"  landholding_ha={p.landholding_ha}  annual_income={p.annual_income}")
    assert p.landholding_ha == 2.0, p.landholding_ha
    assert p.annual_income == 150000.0, p.annual_income
    assert len([n for n in p.notes if n.startswith("Q: ")]) == 2
    print("  -> numbers the USER gave are recorded; both answers stored paired")

    # ------- 4e. a FAILED assessment must not read as the user's fault ------
    #
    # Observed with real Gemini: the user answered all five PMEGP questions
    # and got "Not enough information to decide". The cause was not missing
    # information -- it was the assessor's JSON being truncated, because
    # thinking tokens are billed against max_tokens and 700 did not cover
    # both. The old code mapped that to overall="insufficient_info", which is
    # indistinguishable from "you did not tell me enough" and invites the user
    # to answer again, which fails identically.
    print("\n--- A failed assessment is reported as a failure, not a verdict ---")
    for label, reply in (
        ("truncated JSON", '{"criteria": [{"criterion": "Must be above 18'),
        ("empty output", ""),
        ("prose, not JSON", "I am unable to determine eligibility."),
    ):
        llm.assess_reply = reply
        ans = G.run_query("am i eligible for PMFBY", thread_id=f"t-fail-{label[:5]}")
        text = ans.answer
        print(f"  {label:16} -> {text.splitlines()[0][:64]}")
        assert ans.assessment is not None, label
        assert ans.assessment.failed, f"{label}: failed flag not set"
        assert "could not complete" in text, f"{label}: {text[:120]}"
        assert "Not enough information" not in text, (
            f"{label}: a broken call still blames the user\n{text}"
        )
        assert "failure on my side" in text, label
    print("  -> all three failure shapes reported honestly, none as a verdict")

    # And the opposite: a genuine insufficient_info is still a verdict, and it
    # credits the answers rather than implying none were given. Needs its own
    # thread with a LIVE interview -- t-batch's was already completed above,
    # so posting answers there resumes nothing.
    print("\n--- A genuine unknown stays a verdict, and credits the answers ---")
    llm.assess_reply = (
        '{"criteria": ['
        '{"criterion": "Applicant must be a farmer", "verdict": "unknown",'
        ' "reason": "", "needs_slot": "occupation"}],'
        '"overall": "insufficient_info",'
        '"questions": ["Are you a cultivator of land?"]}'
    )
    thread_unknown = "t-genuine-unknown"
    ans = G.run_query("am i eligible for PMFBY", thread_id=thread_unknown)
    asked = list(ans.assessment.questions)
    assert ans.awaiting_answer and asked, ans.answer

    # Now answer it, and have the assessor STILL fail to resolve the criterion
    # -- a real possibility, and it must not read as a missing answer.
    llm.assess_reply = (
        '{"criteria": ['
        '{"criterion": "Must hold a valid trade licence", "verdict": "unknown",'
        ' "reason": "not stated in the answers", "needs_slot": null}],'
        '"overall": "insufficient_info", "questions": []}'
    )
    ans = G.run_query(
        f"Answered {len(asked)} eligibility question(s).", thread_id=thread_unknown,
        answers=[{"question": q, "answer": "yes"} for q in asked],
    )
    print(f"  -> {ans.answer.splitlines()[0][:64]}")
    assert ans.assessment is not None and not ans.assessment.failed
    assert "Not enough information" in ans.answer, ans.answer
    assert "were recorded" in ans.answer, ans.answer
    assert "not something missing from your answers" in ans.answer, ans.answer
    print("  -> verdict given, answers credited, blame not shifted to the user")

    llm.assess_reply = None

    # ---------------- 5. Discovery with a profile ----------------
    print("\n--- AGENT 1b: Scheme Discovery (profile known) ---")
    llm.calls.clear()
    ans = G.run_query("what schemes can I get", thread_id=thread)
    print(f"  route={ans.route}  suggested={len(ans.suggested_schemes)}  llm_calls={llm.calls}")
    print(f"  profile used: {ans.profile_summary}")
    assert ans.route == "discovery", ans.route
    assert ans.suggested_schemes, "no schemes suggested"
    for s in ans.suggested_schemes[:3]:
        print(f"    - {s['scheme_name'][:58]}")
    # With a profile present, HyDE must NOT fire: the profile drives retrieval
    # (via facets once enriched, via the relaxed section filter before that).
    assert "hyde" not in llm.calls, f"HyDE should not run with a profile: {llm.calls}"
    print("  -> used profile-driven retrieval, skipped HyDE")

    # ---------------- 6. Discovery with HyDE (no profile, no scheme) ----------------
    print("\n--- AGENT 1a: Query Understanding + HyDE ---")
    llm.calls.clear()
    ans = G.run_query(
        "suggest schemes for people who weave cloth on handlooms", thread_id="t-hyde"
    )
    print(f"  route={ans.route}  llm_calls={llm.calls}")
    assert ans.route == "discovery", ans.route
    assert "hyde" in llm.calls, f"HyDE should fire with no scheme and no profile: {llm.calls}"
    print(f"  suggested: {[s['scheme_name'][:40] for s in ans.suggested_schemes[:3]]}")
    print("  -> HyDE fired for an unresolved descriptive query")

    # ---------------- 7. HyDE must NOT fire for a named scheme ----------------
    print("\n--- HyDE suppressed when the scheme is resolved ---")
    llm.calls.clear()
    ans = G.run_query("what are the benefits of PM-KISAN", thread_id="t-named")
    print(f"  route={ans.route}  llm_calls={llm.calls}")
    assert "hyde" not in llm.calls, "HyDE wasted a call on a resolved scheme"
    print("  -> skipped HyDE, saving a generation on a query it cannot improve")

    # ---------------- 8. Evidence agent ----------------
    print("\n--- AGENT 6: Evidence Verification ---")
    from schemes_rag.agents import evidence as E

    llm.calls.clear()
    ans = G.run_query("What are the benefits of Agnipath Yojana?", thread_id="t-ev")
    v = ans.verification or {}
    print(f"  route={ans.route}  figures_checked={v.get('figures_checked')} "
          f"ungrounded={v.get('figures_ungrounded')} verdict={v.get('verdict')}")
    assert ans.verification is not None, "evidence agent did not run"

    # invented figure must be caught
    from schemes_rag.models import Answer

    fake = Answer(answer="Agniveers get Rs 999 crore cover [1].", route="benefits")
    chunks = []
    import schemes_rag.graph as GG

    snapshot = GG.get_graph().get_state({"configurable": {"thread_id": "t-ev"}})
    chunks = (getattr(snapshot, "values", None) or {}).get("chunks") or []
    rep = E.verify(fake, chunks)
    print(f"  invented figure -> verdict={rep['verdict']} ungrounded={rep['figures_ungrounded']}")
    assert rep["verdict"] == "ungrounded_figures", rep
    assert "Caution" in fake.answer
    print("  -> numeric grounding caught the invention and caveated the answer")

    # ---------------- 8b. refusal falls back to the real section ----------------
    print("\n--- Section agents survive a refusing model ---")

    class RefusingLLM(ScriptedLLM):
        def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
            # A 4B handed a strict prompt refuses even good context; the user
            # must still get the steps.
            if "official application process" in system or "documents required for this scheme" in system:
                self.calls.append("refused")
                return "INSUFFICIENT_EVIDENCE"
            return super().complete(system, user, temperature, max_tokens, json_mode)

    import schemes_rag.generate as gen_mod

    refusing = RefusingLLM()
    gen_mod.get_llm = lambda: refusing
    import schemes_rag.graph as GR

    GR._graph = None
    ans = GR.run_query("How do I apply for PM-KISAN?", thread_id="t-fallback")
    print(f"  model said: INSUFFICIENT_EVIDENCE")
    print(f"  route={ans.route} insufficient={ans.insufficient_evidence} "
          f"citations={len(ans.citations)}")
    print(f"  answer starts: {ans.answer.splitlines()[0][:66]}")
    assert not ans.insufficient_evidence, "refusal should fall back, not surface"
    assert "Step 1" in ans.answer or "Application process" in ans.answer, ans.answer[:200]
    assert ans.citations, "verbatim fallback lost its citations"
    print("  -> real application steps shown instead of an apology")
    gen_mod.get_llm = lambda: llm
    GR._graph = None

    # ---------------- 8c. slot extraction skipped when nothing to extract ----
    print("\n--- Slot-extraction LLM call skipped when the query states no facts ---")
    llm.calls.clear()
    G.run_query("How do I apply for PM-KISAN?", thread_id="t-noslots")
    print(f"  'how do I apply for PM-KISAN' -> {llm.calls}")
    assert "slots" not in llm.calls, f"wasted a slot call: {llm.calls}"
    llm.calls.clear()
    G.run_query("I am a farmer, am I eligible for PM-KISAN?", thread_id="t-slots")
    print(f"  'I am a farmer, am I eligible...'  -> {llm.calls}")
    assert "slots" in llm.calls, f"should extract slots here: {llm.calls}"
    print("  -> ~5s saved on queries that say nothing about the user")

    # ---------------- 9. Off-topic still refuses ----------------
    print("\n--- Off-topic still refuses ---")
    for off in ["how do I bake sourdough bread", "quantum chromodynamics lagrangian"]:
        ans = G.run_query(off, thread_id=f"t-off-{off[:6]}")
        print(f"  {off[:34]:36} insufficient={ans.insufficient_evidence} route={ans.route}")
        assert ans.insufficient_evidence, off

    # ---------------- 10. Router stays cheap ----------------
    print("\n--- Deterministic routing keeps LLM calls down ---")
    llm.calls.clear()
    G.run_query("What documents are required for PMFBY?", thread_id="t-cost")
    doc_calls = list(llm.calls)
    print(f"  documents path: {len(doc_calls)} LLM call(s) -> {doc_calls}")
    assert len(doc_calls) <= 3, f"too many calls for a simple path: {doc_calls}"

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
