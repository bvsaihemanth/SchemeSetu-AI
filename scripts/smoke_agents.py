"""End-to-end smoke test: every agent, against the REAL stack.

    python scripts\\smoke_agents.py

Unlike tests\\test_agents.py -- which drives the graph with a scripted LLM and
an in-memory index -- this makes real Gemini calls against your real Qdrant
collection. It answers the one question the unit tests cannot: does the whole
thing actually work with this provider, this model name, this key and this
index.

Run it once after changing LLM_MODEL, the API key, the provider, or anything
in the retrieval path. It costs a dozen or so Gemini calls.

WHAT EACH CHECK PROVES

  1 Discovery     a situation with no scheme named produces a shortlist
                  (exercises HyDE, facet filtering, profile extraction)
  2 Eligibility   questions come back, AND answering them reaches a verdict.
                  Two turns, because the second is the one that used to fail:
                  the model could not map "1)yes,2)yes,3)no" onto its own
                  criteria, re-asked with an empty question list, then said
                  "not enough information" about a scheme it had just been
                  told six facts about. This now posts answers already PAIRED
                  to their questions, the way the web UI does.
  3 Comparison    two named schemes retrieve per-scheme and do not blend
  4 Application   the application section is used, with citations
  5 Documents     the checklist section is used, with citations
  6 Evidence      every citation marker in the prose resolves to a real
                  source, and nothing was invented
  7 Refusal       an off-corpus question is refused rather than answered

Scheme names are taken from YOUR index, not hardcoded, so this works on any
corpus. Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
# Also this folder, so `from check_llm import ...` works when this module is
# imported rather than run as a script (python -m puts only the cwd on the
# path, so the test harness would otherwise fail on that import).
sys.path.insert(0, str(_HERE))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_MARKER = re.compile(r"\[(\d+)\]")

_results: list[tuple[str, str, str, float]] = []


def record(status: str, agent: str, detail: str, seconds: float = 0.0) -> None:
    _results.append((status, agent, detail, seconds))
    stamp = f"{seconds:5.1f}s" if seconds else "      "
    print(f"  [{status}] {stamp}  {agent}: {detail}")


def show(answer, limit: int = 3) -> None:
    """Print the first few lines of an answer, so a pass is inspectable too.

    A check that only prints PASS hides the thing you actually want to see on
    a first real run: whether the prose is any good.
    """
    for line in [ln for ln in answer.answer.splitlines() if ln.strip()][:limit]:
        print(f"           | {line[:92]}")
    if answer.citations:
        c = answer.citations[0]
        print(f"           | [1] {c.scheme_name} — {c.locator}")


def citations_resolve(answer) -> tuple[bool, str]:
    """Every [n] in the prose must map to a citation we actually returned.

    This is the Evidence agent's whole job. A marker with no matching source
    is an invented citation, which is the most dangerous failure mode here:
    it looks grounded and is not.
    """
    markers = {int(m) for m in _MARKER.findall(answer.answer)}
    available = {c.marker for c in answer.citations}
    orphans = sorted(markers - available)
    if orphans:
        return False, f"invented markers {orphans}; have {sorted(available)}"
    if not answer.citations:
        return False, "no citations at all"
    return True, f"{len(answer.citations)} citations, {len(markers)} markers, all resolve"


def pick_schemes(index, wanted: int = 2) -> list[str]:
    """Scheme names from the user's own index, preferring short ones.

    Short names resolve unambiguously and read better in the output. Taken
    from the index rather than hardcoded so this script is not tied to one
    corpus -- and sorted, so repeated runs are comparable.
    """
    entries = sorted(index.entries.values(), key=lambda e: (len(e.scheme_name), e.doc_id))
    names: list[str] = []
    for entry in entries:
        name = entry.scheme_name.strip()
        # Skip names so short they are probably an acronym fragment, and
        # anything that would resolve to a scheme already chosen.
        if len(name) < 8 or any(name.lower() in n.lower() for n in names):
            continue
        names.append(name)
        if len(names) == wanted:
            break
    return names


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scheme",
        help="Scheme to use for the single-scheme checks. Default: taken from "
             "your index.",
    )
    parser.add_argument(
        "--situation",
        default="I am a farmer in Telangana with 2 hectares of land and my "
                "annual income is 1.5 lakh",
        help="Free-text situation for the Discovery check.",
    )
    args = parser.parse_args()

    from schemes_rag.config import get_settings

    s = get_settings()
    print("=" * 76)
    print("AGENT SMOKE TEST — real Gemini, real Qdrant")
    print("=" * 76)
    print(f"  model      {s.llm_model}")
    print(f"  endpoint   {s.llm_base_url}")
    print(f"  collection {s.qdrant_collection}")
    print(f"  effort     {s.llm_reasoning_effort}   max_tokens={s.llm_max_tokens}")
    print()

    # ---- preconditions, so a config mistake is not reported as an agent bug
    from check_llm import key_is_placeholder  # same folder

    if key_is_placeholder(s):
        print("LLM_API_KEY is still the placeholder. Paste your key into .env.")
        return 1

    from schemes_rag.vectorstore import get_vectorstore

    try:
        points = get_vectorstore().count()
    except Exception as exc:
        print(f"Qdrant unreachable: {exc}")
        print("  docker compose up -d qdrant")
        return 1
    if not points:
        print("Collection is empty. Run: python scripts\\ingest.py")
        return 1
    print(f"  {points:,} chunks indexed")

    from schemes_rag.scheme_index import get_scheme_index

    index = get_scheme_index()
    if index is None:
        print("scheme_index.json missing. Run: python scripts\\ingest.py")
        return 1

    picked = pick_schemes(index, wanted=2)
    scheme = args.scheme or (picked[0] if picked else None)
    other = next((n for n in picked if n != scheme), None)
    if not scheme:
        print("Could not pick a scheme from the index.")
        return 1
    print(f"  using     {scheme!r}" + (f" and {other!r}" if other else ""))
    print()

    from schemes_rag.graph import run_query

    def ask(question: str, thread: str, **kw):
        t0 = time.perf_counter()
        answer = run_query(question, thread_id=thread, **kw)
        return answer, time.perf_counter() - t0

    # ---------------- 1. Discovery ----------------
    print("1. Query Understanding + Scheme Discovery")
    try:
        a, dt = ask(f"{args.situation}. What schemes can I get?", "smoke-discovery")
        if a.suggested_schemes:
            record(PASS, "discovery",
                   f"{len(a.suggested_schemes)} schemes shortlisted; "
                   f"profile={a.profile_summary}", dt)
        elif a.insufficient_evidence:
            record(WARN, "discovery",
                   "refused — no scheme matched this profile. Not necessarily "
                   "wrong, but check the situation is covered by your corpus", dt)
        else:
            record(FAIL, "discovery", f"no shortlist; route={a.route}", dt)
        show(a)
    except Exception as exc:
        record(FAIL, "discovery", f"{type(exc).__name__}: {exc}")

    # ---------------- 2. Eligibility, both turns ----------------
    print("\n2. Eligibility Assessment (the two-turn interview)")
    try:
        a, dt = ask(f"Am I eligible for {scheme}?", "smoke-eligibility")
        questions = list(a.assessment.questions) if a.assessment else []

        if not a.awaiting_answer and a.assessment and a.assessment.overall != "insufficient_info":
            # Legitimately decidable with nothing known: rare, but not a bug.
            record(WARN, "eligibility",
                   f"decided with no questions: {a.assessment.overall}", dt)
            show(a)
        elif not questions:
            record(FAIL, "eligibility",
                   f"asked nothing and reached no verdict; route={a.route} "
                   f"awaiting={a.awaiting_answer}", dt)
            show(a)
        else:
            record(PASS, "eligibility.ask",
                   f"{len(questions)} question(s) from "
                   f"{len(a.assessment.criteria)} criteria", dt)
            for i, q in enumerate(questions, 1):
                print(f"           | Q{i}: {q[:86]}")

            # THE CHECK THAT MATTERS. Answer yes to everything, paired the way
            # the UI pairs it, and require a verdict rather than a re-ask.
            paired = [{"question": q, "answer": "yes"} for q in questions]
            a2, dt2 = ask(
                f"Answered {len(paired)} eligibility question(s).",
                "smoke-eligibility", answers=paired,
            )
            verdict = a2.assessment.overall if a2.assessment else "(none)"
            if a2.awaiting_answer:
                still = list(a2.assessment.questions) if a2.assessment else []
                repeats = [q for q in still if q in questions]
                record(
                    FAIL, "eligibility.answer",
                    f"re-asked instead of deciding: {len(still)} question(s)"
                    + (f", {len(repeats)} of them REPEATS" if repeats else ""),
                    dt2,
                )
            elif verdict == "insufficient_info":
                record(FAIL, "eligibility.answer",
                       "answered every question and still 'insufficient_info' "
                       "— the answers are not reaching the assessor", dt2)
            else:
                record(PASS, "eligibility.answer",
                       f"paired answers accepted -> {verdict}", dt2)
            show(a2, limit=4)
    except Exception as exc:
        record(FAIL, "eligibility", f"{type(exc).__name__}: {exc}")

    # ---------------- 3. Comparison ----------------
    print("\n3. Benefits & Scheme Comparison")
    if not other:
        record(WARN, "comparison", "index has only one usable scheme; skipped")
    else:
        try:
            a, dt = ask(f"Compare {scheme} and {other}", "smoke-comparison")
            if a.route == "comparison" and len(a.schemes_used) >= 2:
                record(PASS, "comparison",
                       f"both retrieved separately: {a.schemes_used}", dt)
            elif a.awaiting_answer:
                record(FAIL, "comparison",
                       "could not resolve both scheme names from the question", dt)
            else:
                record(FAIL, "comparison",
                       f"route={a.route} schemes_used={a.schemes_used}", dt)
            show(a)
        except Exception as exc:
            record(FAIL, "comparison", f"{type(exc).__name__}: {exc}")

    # ---------------- 4 & 5. Application and Documents ----------------
    for label, question, expect in (
        ("application", f"How do I apply for {scheme}?", "application"),
        ("documents", f"What documents are required for {scheme}?", "documents"),
    ):
        print(f"\n{'4' if label == 'application' else '5'}. "
              f"{label.title()} {'Guidance' if label == 'application' else 'Checklist'}")
        try:
            a, dt = ask(question, f"smoke-{label}")
            if a.insufficient_evidence:
                # Worth distinguishing: the scheme's document may genuinely
                # lack the section, which is a corpus fact, not a bug.
                record(WARN, label,
                       f"refused — {scheme!r} may have no {label} section. "
                       "Check the source document before treating this as a bug", dt)
            elif a.route != expect:
                record(FAIL, label, f"routed to {a.route}, expected {expect}", dt)
            elif not a.citations:
                record(FAIL, label, "answered with no citations", dt)
            else:
                record(PASS, label, f"{len(a.citations)} citations", dt)
            show(a)
        except Exception as exc:
            record(FAIL, label, f"{type(exc).__name__}: {exc}")

    # ---------------- 6. Evidence ----------------
    print("\n6. Evidence Verification (citation integrity)")
    try:
        a, dt = ask(f"What are the benefits of {scheme}?", "smoke-evidence")
        ok, detail = citations_resolve(a)
        record(PASS if ok else FAIL, "evidence", detail, dt)
        show(a)
    except Exception as exc:
        record(FAIL, "evidence", f"{type(exc).__name__}: {exc}")

    # ---------------- 7. Refusal ----------------
    print("\n7. Refusal path (the hallucination guard)")
    try:
        a, dt = ask(
            "What is the warranty period for a Yamaha outboard motor?",
            "smoke-refusal",
        )
        if a.insufficient_evidence or a.awaiting_answer:
            record(PASS, "refusal", "off-corpus question refused, not answered", dt)
        else:
            record(FAIL, "refusal",
                   "answered a question the corpus cannot support — "
                   f"raise RETRIEVAL_MIN_SCORE (now {s.retrieval_min_score})", dt)
            show(a)
    except Exception as exc:
        record(FAIL, "refusal", f"{type(exc).__name__}: {exc}")

    # ---------------- summary ----------------
    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    passes = [r for r in _results if r[0] == PASS]
    total_s = sum(r[3] for r in _results)

    print()
    print("=" * 76)
    print(f"{len(passes)} passed, {len(warns)} warnings, {len(fails)} failed "
          f"({total_s:.0f}s of Gemini calls)")
    if passes:
        slowest = max(passes, key=lambda r: r[3])
        print(f"slowest check: {slowest[1]} at {slowest[3]:.1f}s")
    if warns:
        print("\nWarnings — probably your corpus, not the code:")
        for _, agent, detail, _ in warns:
            print(f"  - {agent}: {detail}")
    if fails:
        print("\nFAILED:")
        for _, agent, detail, _ in fails:
            print(f"  - {agent}: {detail}")
    print("=" * 76)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
