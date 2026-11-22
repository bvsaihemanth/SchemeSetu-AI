"""Query CLI, with the eligibility interview running in the terminal.

    python scripts\\query.py                       # interactive (recommended)
    python scripts\\query.py "Who can apply for PKVY?"
    python scripts\\query.py "what documents for PMFBY" --search-only
    python scripts\\query.py --profile              # show what it knows about you

Interactive mode is the one to use. Two reasons:

1. BGE-M3 loads once instead of on every invocation, which is ~10 seconds
   saved per question after the first.
2. The Eligibility agent asks one question at a time and remembers the
   answers. That only works across turns, so a single-shot invocation can
   start an interview but never finish one.

When the agent asks something, answer it in plain words -- "34", "yes",
"2 acres", "I'm SC category". Type /profile to see what it has inferred,
/reset to start over, /help for the rest.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemes_rag.config import get_settings  # noqa: E402
from schemes_rag.observability import init_observability  # noqa: E402

BANNER_COMMANDS = """Commands:
  /profile   show what the system has inferred about you
  /reset     forget the profile and start a new conversation
  /thread    show the conversation id
  /help      this list
  (blank)    quit"""


def show_resolution(question: str) -> list[str] | None:
    """Print which scheme the query resolved to, and why.

    First thing to check when an answer is about the wrong scheme: either
    resolution picked the wrong document, or it found nothing and the query
    fell back to unscoped search across the whole corpus.
    """
    from schemes_rag.scheme_index import get_scheme_index

    index = get_scheme_index()
    if index is None:
        print("  scheme index: NOT BUILT -- run scripts/ingest.py")
        return None

    matches = index.resolve(question)
    if not matches:
        print(f"  scheme: none identified (searching all {len(index)} schemes)")
        return None

    print("  scheme resolution:")
    for m in matches:
        print(f"    {m.confidence:7} {m.score:.3f}  {m.scheme_name[:52]:54} via {m.matched_on}")
    if matches[0].confidence in ("exact", "strong"):
        ids = [m.doc_id for m in matches if m.confidence == matches[0].confidence]
        print(f"  -> scoping retrieval to {len(ids)} document(s)")
        return ids
    print("  -> confidence too low to scope; searching all schemes")
    return None


def show_search(question: str, sections: list[str] | None) -> None:
    from schemes_rag.retriever import get_retriever

    doc_ids = show_resolution(question)
    result = get_retriever().retrieve(question, sections=sections, doc_ids=doc_ids)

    print(f"\ngate score: {result.top_score:.4f}   candidates: {result.candidates_considered}")
    print(f"timings   : {({k: round(v, 1) for k, v in result.latency_ms.items()})}")
    if result.insufficient:
        print("\n  BELOW RELEVANCE FLOOR -- would answer 'not in my sources'")
        return
    for i, rc in enumerate(result.chunks, 1):
        print(f"\n[{i}] {rc.effective_score:.4f}  {rc.chunk.locator()}")
        body = " ".join(rc.chunk.text.split())
        print(f"    {body[:240]}{'...' if len(body) > 240 else ''}")


def print_answer(answer, *, show_route: bool = True) -> None:
    if show_route:
        tag = f"route: {answer.route}"
        if answer.awaiting_answer:
            tag += "  (waiting for your reply)"
        print(f"\n{tag}")
    print("-" * 70)
    print(answer.answer)
    print("-" * 70)

    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            print(f"  [{c.marker}] {c.locator}")
            if c.source_url:
                print(f"       {c.source_url}")
    elif answer.assessment is not None:
        # An eligibility verdict is rendered from the assessed criteria, not
        # generated, so it has no [n] markers by design. Warning about them
        # here told the user to distrust the most carefully grounded answer
        # the system produces.
        pass
    elif not (answer.insufficient_evidence or answer.awaiting_answer):
        print("\n(no citations emitted -- answer is unverifiable, treat with suspicion)")

    v = answer.verification or {}
    if v.get("figures_ungrounded"):
        print(f"\n  ! figures not found in sources: {v['figures_ungrounded']}")
    if v.get("uncited_claims"):
        print(f"  ! uncited claim(s): {len(v['uncited_claims'])}")

    if answer.profile_summary and answer.route in {"eligibility", "discovery", "clarify"}:
        print(f"\nprofile: {answer.profile_summary}")
    print(f"timings: {answer.latency_ms}")


def interactive() -> int:
    from schemes_rag.graph import run_query

    thread_id = str(uuid.uuid4())[:8]
    print("Interactive mode. Blank line or Ctrl-C to exit. /help for commands.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            return 0
        if raw in {"/help", "/?"}:
            print(BANNER_COMMANDS)
            continue
        if raw == "/thread":
            print(f"  thread_id = {thread_id}")
            continue
        if raw == "/reset":
            thread_id = str(uuid.uuid4())[:8]
            print(f"  new conversation ({thread_id}); profile cleared")
            continue
        if raw == "/profile":
            # Read the checkpointer's snapshot rather than keeping a second
            # copy here, so this is exactly what the agents will use next turn.
            from schemes_rag.graph import get_profile, pending_question

            print(f"  {get_profile(thread_id).describe()}")
            if scheme := pending_question(thread_id):
                print(f"  mid-interview on: {scheme}")
            continue

        try:
            answer = run_query(raw, thread_id=thread_id)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")
            continue

        print_answer(answer)
        print()


def main() -> int:
    p = argparse.ArgumentParser(description="Query the schemes RAG pipeline")
    p.add_argument("question", nargs="?", help="Question; omit for interactive mode")
    p.add_argument("--search-only", action="store_true", help="Retrieval only, no LLM")
    p.add_argument("--section", action="append", dest="sections", help="Restrict to section(s)")
    p.add_argument("--thread", help="Resume a conversation by id")
    p.add_argument("--profile", action="store_true", help="Show the inferred profile and exit")
    args = p.parse_args()

    init_observability()
    logging.getLogger("schemes_rag").setLevel(logging.WARNING)

    s = get_settings()
    print(f"collection={s.qdrant_collection}  llm={s.llm_model}  "
          f"hybrid={s.retrieval_hybrid}  rerank={s.rerank_enabled}  hyde={s.hyde_enabled}")

    if args.profile:
        from schemes_rag.graph import get_profile

        print(f"profile: {get_profile(args.thread or 'default').describe()}")
        return 0

    if args.search_only:
        if not args.question:
            print("--search-only needs a question")
            return 1
        show_search(args.question, args.sections)
        return 0

    if args.question:
        from schemes_rag.graph import run_query

        show_resolution(args.question)
        answer = run_query(args.question, thread_id=args.thread or str(uuid.uuid4())[:8])
        print_answer(answer)
        if answer.awaiting_answer:
            print(
                "\nThis needs a follow-up answer. Run `python scripts/query.py` "
                "without arguments for the interactive interview."
            )
        return 0

    print("note: the first query loads BGE-M3 (~10 s); later ones are fast.\n")
    return interactive()


if __name__ == "__main__":
    sys.exit(main())
