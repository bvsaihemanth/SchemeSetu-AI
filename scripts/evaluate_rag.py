"""RAG evaluation over the 85-question test set.

    python scripts\\evaluate_rag.py --build-gold     # once, then review
    python scripts\\evaluate_rag.py                  # the evaluation
    python scripts\\evaluate_rag.py --limit 5        # a quick smoke run

WHAT IS MEASURED, AND WHAT EACH THING NEEDS

Retrieval, from gold labels. Exact arithmetic, no LLM, free:

    recall@k, precision@k, hit@k, MRR      at document AND chunk level
                                           swept across several k

Generation, by LLM judge, reference-free:

    faithfulness (groundedness)   claims in the answer supported by the
                                  retrieved context
    context relevance             retrieved chunks on-topic for the question
    context precision             rank-weighted usefulness of the chunks
                                  (average precision -- order matters)
    completeness                  aspects the question asked for that the
                                  answer addresses

Generation, needing a human-written reference answer in the gold file:

    context recall                reference claims the retrieved context covers

Cost, measured from the provider's own token counts:

    cost per request              mean / p50 / p95, broken down per agent

TWO ITEMS FROM THE REQUEST ARE NOT METRICS

top-k is a retrieval PARAMETER. There is no "top-k score" to report, so it is
treated the way it should be: swept, with recall/precision reported at every k,
so the value can be chosen from the data rather than guessed. The current
configured value is marked in the sweep table.

top-p is an LLM SAMPLING parameter (nucleus sampling) and not an evaluation
metric at all. This project does not set it -- it uses temperature 0.1 -- so
there is nothing to measure. Reported in the run header as configuration, next
to temperature and reasoning effort, because it belongs with the settings a
result has to be interpreted against. If the question is "does sampling config
change quality", that is an A/B comparison: run this script twice with
different settings and compare, which is what --tag and --out are for.

HONESTY ABOUT COVERAGE

Every metric reports how many questions it was computed over. A mean has no
meaning without its denominator, and the denominators here genuinely differ:
retrieval metrics need gold labels, context recall needs reference answers,
and faithfulness skips refusals because an answer that asserts nothing cannot
be unfaithful -- scoring refusals 1.00 would let a system that answers nothing
report perfect groundedness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemes_rag import usage  # noqa: E402
from schemes_rag.evaluation import gold as gold_mod  # noqa: E402
from schemes_rag.evaluation import judges, metrics  # noqa: E402

DEFAULT_QUESTIONS = "tests/test-data.txt"
DEFAULT_GOLD = "eval/gold.jsonl"
DEFAULT_ANSWERS = "tests/test-ans.txt"
DEFAULT_KS = "1,3,5,8,10"


# --------------------------------------------------------------------------
@dataclass
class QuestionResult:
    qid: int
    question: str
    answer: str = ""
    route: str = ""
    refused: bool = False
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_schemes: list[str] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    citations: int = 0
    gold_doc_ids: list[str] = field(default_factory=list)
    labelled: bool = False
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    by_agent: dict = field(default_factory=dict)
    # judge outputs
    claims: list[dict] = field(default_factory=list)
    chunk_ratings: list[dict] = field(default_factory=list)
    aspect_verdicts: list[dict] = field(default_factory=list)
    recall_claims: list[dict] = field(default_factory=list)
    ref_coverage: list[dict] = field(default_factory=list)
    ref_conflicts: list[dict] = field(default_factory=list)
    error: str | None = None

    # scores
    faithfulness: float | None = None
    context_relevance: float | None = None
    context_precision: float | None = None
    completeness: float | None = None
    context_recall: float | None = None
    answer_coverage: float | None = None
    answer_correctness: float | None = None


def answer_question(question: str, qid: int, settings) -> QuestionResult:
    """Run one question through the real pipeline, with usage collected.

    Uses run_query, not a hand-assembled retrieve-then-generate: the thing
    being evaluated is the system the user talks to, including routing, entity
    resolution and the evidence pass. A bespoke path would score a pipeline
    that does not exist.
    """
    from schemes_rag.graph import run_query

    result = QuestionResult(qid=qid, question=question)
    started = time.perf_counter()
    # A fresh thread per question: a shared one would carry the previous
    # question's profile and pending scheme into this one, and the eligibility
    # questions would resume an interview instead of starting fresh.
    thread = f"eval-q{qid}"
    try:
        with usage.collect() as ledger:
            answer = run_query(question, thread_id=thread)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - started
        return result

    result.seconds = time.perf_counter() - started
    result.answer = answer.answer or ""
    result.route = answer.route or ""
    result.refused = bool(answer.insufficient_evidence)
    if getattr(answer, "model_failed", False):
        # The model call failed -- rate limited, overloaded, or it returned
        # only reasoning. Recorded as an ERROR, not an answer.
        #
        # This is the difference between measuring a system and measuring a
        # quota. A burst of 429s produces 85 prose-but-uncited "answers", and
        # averaging those in reports a groundedness collapse and blames the
        # retriever for a billing problem.
        result.error = "model call failed (see the log above)"
        result.refused = False
    result.citations = len(answer.citations)
    result.prompt_tokens = ledger.prompt_tokens
    result.completion_tokens = ledger.completion_tokens
    result.cost = ledger.cost(
        settings.llm_price_input_per_mtok, settings.llm_price_output_per_mtok
    )
    result.by_agent = ledger.by_agent()

    # The chunks the answer was actually built from, in rank order. Taken from
    # the citations' own chunks so the judged context is the context the model
    # saw -- re-retrieving here would judge a different set.
    for citation in sorted(answer.citations, key=lambda c: c.marker):
        result.retrieved_schemes.append(citation.scheme_name)
    return result


def retrieve_for_metrics(question: str, max_k: int):
    """Ranked chunks for the retrieval metrics.

    A separate call from answer_question because the metrics need the ranked
    list at the LARGEST k in the sweep, while the answer uses the configured
    final_k. Done once per question and truncated per k, rather than once per
    k, so the sweep costs one retrieval rather than five.
    """
    from schemes_rag.retriever import get_retriever

    return get_retriever().retrieve(
        question, gate_query=question, final_k=max_k, top_k=max(48, max_k * 4),
    )


# --------------------------------------------------------------------------
def build_gold(args) -> int:
    from schemes_rag.scheme_index import get_scheme_index

    index = get_scheme_index()
    if index is None:
        print("scheme_index.json not found. Run: python scripts\\ingest.py")
        return 1

    questions = gold_mod.parse_questions(args.questions)
    existing = gold_mod.load(args.gold)

    qa: dict[int, tuple[str, str]] = {}
    answers_path = Path(args.answers)
    if answers_path.is_file():
        try:
            qa = gold_mod.parse_qa(answers_path)
            print(f"Reference answers: {len(qa)} from {answers_path}")
        except ValueError as exc:
            print(f"Could not read {answers_path}: {exc}")
            return 1
    else:
        print(f"No reference answers at {answers_path} -- context_recall, "
              "answer_coverage\nand answer_correctness will be unavailable.")

    drafted = gold_mod.build(questions, index, existing, qa=qa)
    gold_mod.save(drafted, args.gold)

    cov = gold_mod.coverage(drafted)
    print(f"Drafted {cov['questions']} questions -> {args.gold}")
    print(f"  {len(index.entries)} schemes in the index")
    print(f"  labelled from question : {cov['labelled_from_question']}")
    print(f"  proposed from answer   : {cov['labelled_from_answer']}"
          + ("   <-- confirm these" if cov["proposed_unconfirmed"] else ""))
    print(f"  with reference answer  : {cov['with_reference_answer']}")
    print(f"  human-reviewed (kept)  : {cov['human_reviewed']}")
    print(f"  unlabelled             : {cov['unlabelled']}")

    proposed = [g for g in drafted
                if g.is_labelled and g.label_origin == "answer"
                and g.source != "human"]
    if proposed:
        print(f"\n{len(proposed)} label(s) PROPOSED from the reference answer, "
              f"because the\nquestion names no scheme this index contains:\n")
        for g in proposed:
            print(f"  {g.qid:3}. {g.question[:70]}")
            print(f"       -> {', '.join(s[:52] for s in g.relevant_schemes)}")
        print("\nCheck each is a scheme the answer RECOMMENDS, not one it")
        print("mentions in passing. A scheme named as a required DOCUMENT is")
        print("not a relevant document: the answer to \"documents for")
        print("PMAY-Gramin\" lists an MGNREGA job card, which would propose")
        print("MGNREGA as the gold answer to a question about PMAY-G and score")
        print("correct retrieval as a failure. Then set \"source\": \"human\".")

    unresolved = [g for g in drafted if not g.is_labelled]
    if unresolved:
        print(f"\n{len(unresolved)} question(s) name no scheme the index "
              f"recognises:\n")
        for g in unresolved:
            print(f"  {g.qid:3}. {g.question[:88]}")
        print("\nMost of these describe a person rather than naming a scheme")
        print('("I am a farmer in Telangana..."), which has no single correct')
        print("document. Two options per question, in eval/gold.jsonl:")
        print('  - fill relevant_doc_ids and set "source": "human"')
        print("  - leave it, and it is EXCLUDED from the retrieval metrics")
        print("    (excluded, not scored zero -- and the exclusion is")
        print("     reported alongside every result)")
    print(f"\nReview {args.gold}, then run:  python scripts\\evaluate_rag.py")
    return 0


# --------------------------------------------------------------------------
def run(args) -> int:  # noqa: C901
    from schemes_rag.config import get_settings

    s = get_settings()
    if args.rpm:
        s.llm_max_requests_per_minute = args.rpm
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    max_k = max(ks)

    questions = gold_mod.parse_questions(args.questions)
    gold = {g.qid: g for g in gold_mod.load(args.gold)}
    if not gold:
        print(f"No gold set at {args.gold}.")
        print("Build one first:  python scripts\\evaluate_rag.py --build-gold")
        print("\nWithout labels, recall@k and precision@k have no denominator.")
        print("The judge metrics would still work -- rerun with --no-retrieval")
        print("to skip the retrieval half and score generation only.")
        if not args.no_retrieval:
            return 1

    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        questions = [(q, t) for q, t in questions if q in wanted]
    if args.limit:
        questions = questions[: args.limit]

    judge_model = args.judge_model or s.llm_model

    # Resume: anything already scored in the report stays scored.
    done: dict[int, dict] = {}
    if args.resume and args.out and Path(args.out).is_file():
        try:
            previous = json.loads(Path(args.out).read_text(encoding="utf-8"))
            done = {
                q["qid"]: q for q in previous.get("questions", [])
                if not q.get("error")
            }
            print(f"Resuming: {len(done)} question(s) already scored in "
                  f"{args.out}; errored ones will be retried.\n")
        except (OSError, ValueError) as exc:
            print(f"Could not read {args.out} to resume ({exc}); "
                  f"starting fresh.\n")

    print("=" * 78)
    print("RAG EVALUATION")
    print("=" * 78)
    print(f"  questions        {len(questions)} from {args.questions}")
    print(f"  gold set         {args.gold} ({len(gold)} entries, "
          f"{sum(1 for g in gold.values() if g.reference_answer)} with references)")
    print(f"  answering model  {s.llm_model}")
    print(f"  judge model      {judge_model}"
          + ("   (SAME as the answering model -- a model grading its own "
             "output\n                   shares its blind spots; pass "
             "--judge-model for a stronger one)"
             if judge_model == s.llm_model else ""))
    print(f"  k sweep          {ks}   (configured final_k={s.retrieval_final_k})")
    print(f"  retrieval        hybrid={s.retrieval_hybrid} "
          f"rerank={s.rerank_enabled} min_score={s.retrieval_min_score}")
    print(f"  sampling         temperature={s.llm_temperature} "
          f"top_p=(not set) reasoning_effort={s.llm_reasoning_effort}")
    print(f"  pricing          ${s.llm_price_input_per_mtok}/Mtok in, "
          f"${s.llm_price_output_per_mtok}/Mtok out")
    rpm = s.llm_max_requests_per_minute
    calls = 5 if args.no_judge else 10
    print(f"  rate limit       "
          + (f"{rpm} req/min" if rpm else "NONE -- see below"))
    budget = len(questions) * calls
    if rpm:
        print(f"  estimated        ~{budget} calls, "
              f"~{budget / max(1, rpm):.0f} min at this rate")
    else:
        print(f"                   ~{budget} LLM calls will be made back to "
              f"back. A free-tier key\n                   will return 429 "
              f"within the first few questions. Pass --rpm 10.")
    print()

    run_config = {
        "answering_model": s.llm_model,
        "judge_model": judge_model,
        "retrieval_final_k": s.retrieval_final_k,
        "retrieval_hybrid": s.retrieval_hybrid,
        "rerank_enabled": s.rerank_enabled,
        "retrieval_min_score": s.retrieval_min_score,
        "context_budget_tokens": s.llm_context_budget_tokens,
        "temperature": s.llm_temperature,
        "top_p": None,
        "reasoning_effort": s.llm_reasoning_effort,
        "price_input_per_mtok": s.llm_price_input_per_mtok,
        "price_output_per_mtok": s.llm_price_output_per_mtok,
        "max_requests_per_minute": s.llm_max_requests_per_minute,
        "tag": args.tag,
    }

    results: list[QuestionResult] = []
    retrieval_by_k: dict[int, list[metrics.RetrievalScores]] = {k: [] for k in ks}
    # Consecutive quota failures. A per-DAY cap cannot be waited out inside a
    # run, so grinding through the remaining questions produces nothing but
    # more errors and burns the quota that a retry might otherwise have used
    # tomorrow.
    quota_strikes = 0

    for n, (qid, question) in enumerate(questions, 1):
        g = gold.get(qid)
        if qid in done:
            # Rehydrate enough of the previous result for the aggregates. The
            # retrieval sweep is replayed from the stored doc ids rather than
            # re-retrieved, so resuming costs no API calls at all.
            prev = done[qid]
            restored = QuestionResult(
                qid=qid, question=question, answer=prev.get("answer") or "",
                route=prev.get("route") or "",
                refused=bool(prev.get("refused")),
                labelled=bool(prev.get("labelled")),
                gold_doc_ids=prev.get("gold_doc_ids") or [],
                retrieved_doc_ids=prev.get("retrieved_doc_ids") or [],
                citations=int(prev.get("citations") or 0),
                seconds=float(prev.get("seconds") or 0.0),
                prompt_tokens=int(prev.get("prompt_tokens") or 0),
                completion_tokens=int(prev.get("completion_tokens") or 0),
                cost=float(prev.get("cost") or 0.0),
                faithfulness=prev.get("faithfulness"),
                context_relevance=prev.get("context_relevance"),
                context_precision=prev.get("context_precision"),
                completeness=prev.get("completeness"),
                context_recall=prev.get("context_recall"),
                answer_coverage=prev.get("answer_coverage"),
                answer_correctness=prev.get("answer_correctness"),
                claims=prev.get("unsupported_claims") or [],
                ref_conflicts=prev.get("contradictions") or [],
                ref_coverage=[
                    {"claim": c, "present": False}
                    for c in (prev.get("missed_reference_claims") or [])
                ],
            )
            if restored.labelled and restored.retrieved_doc_ids:
                for k in ks:
                    retrieval_by_k[k].append(metrics.score_retrieval(
                        restored.retrieved_doc_ids, restored.gold_doc_ids, k))
            results.append(restored)
            print(f"[{n}/{len(questions)}] q{qid}: (from the previous run)")
            continue
        print(f"[{n}/{len(questions)}] q{qid}: {question[:62]}")

        result = answer_question(question, qid, s)
        result.gold_doc_ids = list(g.relevant_doc_ids) if g else []
        result.labelled = bool(result.gold_doc_ids)

        if result.error:
            print(f"      ERROR {result.error}")
            results.append(result)
            continue

        # ---- retrieval metrics ----
        if not args.no_retrieval and result.labelled:
            try:
                ranked = retrieve_for_metrics(question, max_k)
                result.retrieved_doc_ids = [rc.chunk.doc_id for rc in ranked.chunks]
                result.chunk_texts = [rc.chunk.text for rc in ranked.chunks]
                for k in ks:
                    retrieval_by_k[k].append(metrics.score_retrieval(
                        result.retrieved_doc_ids, result.gold_doc_ids, k,
                    ))
            except Exception as exc:
                print(f"      retrieval failed: {type(exc).__name__}: {exc}")

        # ---- judges ----
        if not args.no_judge and result.answer and not result.refused:
            context = "\n\n".join(result.chunk_texts[: s.retrieval_final_k])
            try:
                with usage.collect() as judge_ledger:
                    parts = judges.decompose(question, result.answer, judge_model)
                    aspects = (g.required_aspects if g and g.required_aspects
                               else parts.aspects)

                    if context:
                        result.claims = judges.judge_faithfulness(
                            parts.claims, context, judge_model)
                        result.chunk_ratings = judges.judge_chunks(
                            question, result.answer,
                            result.chunk_texts[: s.retrieval_final_k], judge_model)
                    result.aspect_verdicts = judges.judge_completeness(
                        aspects, result.answer, judge_model)
                    if g and g.reference_answer:
                        if context:
                            result.recall_claims = judges.judge_context_recall(
                                g.reference_answer, context, judge_model)
                        # Against the reference: what the answer missed, and
                        # what it got wrong. The second is the gap
                        # faithfulness cannot see -- an answer that
                        # faithfully reports the wrong document is faithful
                        # and incorrect.
                        result.ref_coverage, result.ref_conflicts = (
                            judges.judge_against_reference(
                                g.reference_answer, result.answer, judge_model)
                        )

                result.faithfulness = metrics.faithfulness(result.claims)
                result.context_relevance = metrics.context_relevance(result.chunk_ratings)
                result.context_precision = metrics.context_precision(result.chunk_ratings)
                result.completeness = metrics.completeness(result.aspect_verdicts)
                result.context_recall = metrics.context_recall(result.recall_claims)
                result.answer_coverage = metrics.answer_coverage(result.ref_coverage)
                result.answer_correctness = metrics.answer_correctness(
                    result.ref_conflicts)
                # Reference-based completeness supersedes the aspect-based
                # guess when a reference exists: the reference STATES what a
                # complete answer contains instead of asking a model to infer
                # it from the question.
                if result.answer_coverage is not None:
                    result.completeness = result.answer_coverage

                # Judging costs tokens too. Kept OUT of the request cost --
                # mixing them would inflate "cost per request" with spend that
                # only happens during evaluation, and the whole point of that
                # number is what serving a user costs.
                result.by_agent.update({
                    f"[judge] {k}": v for k, v in judge_ledger.by_agent().items()
                })
            except Exception as exc:
                print(f"      judging failed: {type(exc).__name__}: {exc}")

        if result.error and "quota" in (result.error or "").lower():
            quota_strikes += 1
        elif not result.error:
            quota_strikes = 0

        bits = [f"{result.seconds:.1f}s", f"${result.cost:.5f}",
                f"{result.citations} cites", result.route or "?"]
        if result.refused:
            bits.append("REFUSED")
        if result.faithfulness is not None:
            bits.append(f"faith={result.faithfulness:.2f}")
        if not result.labelled:
            bits.append("unlabelled")
        print(f"      {'  '.join(bits)}")
        results.append(result)
        # After EVERY question, so a run killed at q61 keeps its 61 results
        # and --resume picks up from there instead of repeating them.
        save_progress(args.out, results, run_config)

        if quota_strikes >= args.quota_strikes:
            print(f"\n{'=' * 78}")
            print(f"STOPPING: {quota_strikes} consecutive quota failures.")
            print("=" * 78)
            print("A per-day quota does not reset by waiting a few seconds, so")
            print("continuing would only turn the remaining questions into")
            print(f"errors. {len(results)} question(s) are saved in {args.out}.")
            print("\nWhen the quota resets:")
            print("  python scripts\\evaluate_rag.py --rpm 10 --resume")
            print("Nothing already scored will be re-run, and resuming costs")
            print("no API calls for those questions.")
            break

    report(results, retrieval_by_k, ks, gold, s, judge_model, args)
    return 0


# --------------------------------------------------------------------------
def question_payload(r: QuestionResult) -> dict:
    """One question's record, as it goes into the JSON report."""
    return {
        "qid": r.qid, "question": r.question, "route": r.route,
        "refused": r.refused, "labelled": r.labelled,
        "gold_doc_ids": r.gold_doc_ids,
        # Persisted so --resume can replay the k sweep without re-retrieving,
        # which is what makes resuming free.
        "retrieved_doc_ids": r.retrieved_doc_ids,
        "retrieved_schemes": r.retrieved_schemes,
        "citations": r.citations, "seconds": round(r.seconds, 2),
        "cost": r.cost, "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "faithfulness": r.faithfulness,
        "context_relevance": r.context_relevance,
        "context_precision": r.context_precision,
        "completeness": r.completeness,
        "context_recall": r.context_recall,
        "answer_coverage": r.answer_coverage,
        "answer_correctness": r.answer_correctness,
        "unsupported_claims": [v for v in r.claims if not v["supported"]],
        "contradictions": [
            c for c in r.ref_conflicts if c["verdict"] == "contradicts"
        ],
        "missed_reference_claims": [
            c["claim"] for c in r.ref_coverage if not c["present"]
        ],
        "answer": r.answer,
        "error": r.error,
    }


def save_progress(path: str, results: list[QuestionResult], config: dict) -> None:
    """Write what has been scored so far, after every question.

    WHY NOT JUST AT THE END

    Because --resume had nothing to resume from. An 85-question run takes over
    an hour, and the first version wrote the report only on completion -- so a
    run killed by quota exhaustion at q61 discarded all 61 results, and
    --resume then started from q1 and hit the same wall. The expensive thing
    here is the API calls, and they were being thrown away.

    Written in full each time rather than appended: 85 records is a few
    hundred KB, and a whole-file rewrite cannot leave a half-written record
    that breaks the next --resume.
    """
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = {
        "config": config,
        "complete": False,
        "questions": [question_payload(r) for r in results],
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(partial, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    # Atomic replace: a crash mid-write leaves the previous good report, not a
    # truncated one.
    tmp.replace(out)


def report(results, retrieval_by_k, ks, gold, s, judge_model, args) -> None:
    ok = [r for r in results if not r.error]
    answered = [r for r in ok if not r.refused]

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    errored = [r for r in results if r.error]
    print(f"\nRun: {len(results)} questions, {len(errored)} errored, "
          f"{sum(1 for r in ok if r.refused)} refused, "
          f"{sum(1 for r in ok if r.labelled)} with gold labels")

    if errored:
        print(f"\n  {len(errored)} question(s) EXCLUDED -- the call failed, so")
        print("  there is no answer to score. Not counted as bad answers:")
        for r in errored[:8]:
            print(f"    q{r.qid}: {r.error[:62]}")
        if len(errored) > 8:
            print(f"    ... and {len(errored) - 8} more")
        rate_limited = sum(
            1 for r in errored
            if "429" in (r.error or "") or "model call failed" in (r.error or "")
        )
        if rate_limited:
            print(f"\n  These are quota failures, not quality failures. Rerun")
            print("  the excluded ones with pacing:")
            print("    python scripts\\evaluate_rag.py --rpm 10 --resume")
            print("  Every number below is computed over the questions that")
            print(f"  DID complete ({len(ok)} of {len(results)}), so it is")
            print("  valid but partial -- read the coverage counts.")

    # ---- retrieval sweep ----
    if any(retrieval_by_k.values()):
        print("\n" + "-" * 78)
        print("RETRIEVAL  (gold labels, no LLM -- exact)")
        print("-" * 78)
        print("Document level: did the right scheme get retrieved at all.")
        print("Chunk level:    how much of the model's context was on target.")
        print()
        print(f"{'k':>3} {'hit@k':>7} {'doc_rec':>8} {'doc_prec':>9} "
              f"{'mrr':>6} {'chunk_prec':>11} {'chunks':>7}   n")
        for k in ks:
            agg = metrics.aggregate_retrieval(retrieval_by_k[k])
            if not agg:
                continue
            marker = " *" if k == s.retrieval_final_k else "  "
            print(f"{k:>3}{marker}{agg['hit@k']:>6.3f} "
                  f"{agg['doc_recall@k']:>8.3f} {agg['doc_precision@k']:>9.3f} "
                  f"{agg['mrr@k']:>6.3f} {agg['chunk_precision@k']:>11.3f} "
                  f"{agg['avg_chunks_returned']:>7.1f}  {agg['questions']}")
        print("\n  * the configured final_k")
        print("  Macro means (each question counts once). hit@k is the most")
        print("  interpretable: the share of questions where the correct")
        print("  document appeared in the top k at all. doc_prec falls as k")
        print("  rises by construction -- more chunks pull in more documents.")

    # ---- generation ----
    print("\n" + "-" * 78)
    print(f"GENERATION  (LLM judge: {judge_model})")
    print("-" * 78)

    aggregates = [
        ("faithfulness", "faithfulness", "claims supported by the context"),
        ("context_relevance", "context_relevance", "chunks on-topic"),
        ("context_precision", "context_precision", "rank-weighted usefulness (AP)"),
        ("completeness", "completeness",
         "reference claims the ANSWER covers"),
        ("context_recall", "context_recall", "reference claims in the CONTEXT"),
        ("answer_correctness", "answer_correctness",
         "answer claims the reference agrees with"),
    ]
    rows = []
    for name, attr, blurb in aggregates:
        agg = metrics.Aggregate(name)
        for r in answered:
            agg.add(getattr(r, attr))
        rows.append((agg, blurb))
        print(f"  {agg.format()}   {blurb}")

    recall_agg = next(a for a, _ in rows if a.name == "context_recall")
    with_refs = sum(1 for g in gold.values() if g.reference_answer)
    if recall_agg.n == 0 and not with_refs:
        print("\n  context_recall, completeness and answer_correctness are n/a:")
        print("  all three compare against a reference answer and the gold file")
        print("  has none. Point --answers at the Qn./An. file, or add")
        print('  "reference_answer" to entries in eval/gold.jsonl. Substituting')
        print("  the system's own answer would make them measure nothing, so")
        print("  they report n/a instead.")
    elif with_refs:
        print(f"\n  {with_refs} reference answer(s) in use. faithfulness and")
        print("  answer_correctness are NOT the same check: faithfulness asks")
        print("  whether the answer matches the RETRIEVED CONTEXT, so an answer")
        print("  that faithfully reports the wrong document scores 1.00 and is")
        print("  still wrong. answer_correctness is what catches that.")

    # Contradictions: the failures only a reference can reveal.
    contradictions = [
        (r, c) for r in answered for c in r.ref_conflicts
        if c["verdict"] == "contradicts"
    ]
    if contradictions:
        print(f"\n  {len(contradictions)} CONTRADICTION(S) against the reference")
        print("  -- claims the reference says are wrong:")
        for r, c in contradictions[: args.show_claims]:
            print(f"    q{r.qid}: {c['claim'][:68]}")
            if c["detail"]:
                print(f"           ref: {c['detail'][:64]}")
        if len(contradictions) > args.show_claims:
            print(f"    ... {len(contradictions) - args.show_claims} more "
                  f"(see the JSON report)")

    # What the answers consistently leave out.
    missed = [
        (r, c) for r in answered for c in r.ref_coverage if not c["present"]
    ]
    if missed:
        print(f"\n  {len(missed)} reference claim(s) MISSED across "
              f"{len({r.qid for r, _ in missed})} question(s).")
        print("  Low completeness with high faithfulness means retrieval is")
        print("  finding the right document but not enough of it -- look at")
        print("  final_k and the section balancing, not the prompts.")

    refused = [r for r in ok if r.refused]
    if refused:
        print(f"\n  {len(refused)} refusal(s), excluded from the judge metrics:")
        for r in refused[:6]:
            print(f"    q{r.qid}: {r.question[:66]}")
        print("  An answer that asserts nothing cannot be unfaithful. Scoring")
        print("  refusals 1.00 would let a system that answers nothing report")
        print("  perfect groundedness, so they are counted here instead.")

    # ---- the unfaithful claims, which are the actionable output ----
    unfaithful = [
        (r, v) for r in answered for v in r.claims if not v["supported"]
    ]
    if unfaithful:
        print(f"\n  {len(unfaithful)} UNSUPPORTED claim(s) -- the groundedness")
        print("  failures, which is what to actually go and look at:")
        for r, v in unfaithful[: args.show_claims]:
            print(f"    q{r.qid}: {v['claim'][:70]}")
            if v["reason"]:
                print(f"           why: {v['reason'][:66]}")
        if len(unfaithful) > args.show_claims:
            print(f"    ... {len(unfaithful) - args.show_claims} more "
                  f"(--show-claims N, or see the JSON report)")

    # ---- cost and latency ----
    print("\n" + "-" * 78)
    print("COST AND LATENCY  (per request, serving only -- judge spend excluded)")
    print("-" * 78)
    costs = [r.cost for r in ok]
    times = [r.seconds for r in ok]
    if costs:
        print(f"  cost/request   mean ${sum(costs)/len(costs):.5f}   "
              f"p50 ${metrics.percentile(costs, 50):.5f}   "
              f"p95 ${metrics.percentile(costs, 95):.5f}   "
              f"total ${sum(costs):.4f}")
        print(f"  latency        mean {sum(times)/len(times):.1f}s   "
              f"p50 {metrics.percentile(times, 50):.1f}s   "
              f"p95 {metrics.percentile(times, 95):.1f}s")
        pt = sum(r.prompt_tokens for r in ok)
        ct = sum(r.completion_tokens for r in ok)
        print(f"  tokens         {pt:,} in + {ct:,} out = {pt + ct:,}"
              f"   ({pt/max(1,len(ok)):.0f} in/request)")
        print("\n  Prompt tokens dominate, which is what a RAG system is: the")
        print("  lever on cost is LLM_CONTEXT_BUDGET_TOKENS and final_k, not")
        print("  answer length.")

        # Per agent -- the breakdown that changes a decision.
        agents: dict[str, dict] = {}
        for r in ok:
            for name, v in r.by_agent.items():
                if name.startswith("[judge]"):
                    continue
                b = agents.setdefault(name, {"calls": 0, "in": 0, "out": 0})
                b["calls"] += v["calls"]
                b["in"] += v["prompt_tokens"]
                b["out"] += v["completion_tokens"]
        if agents:
            print(f"\n  {'agent':<26} {'calls':>6} {'tok in':>10} {'tok out':>9} "
                  f"{'cost':>10}")
            for name, b in sorted(agents.items(), key=lambda kv: -kv[1]["in"]):
                cost = (b["in"] / 1e6 * s.llm_price_input_per_mtok
                        + b["out"] / 1e6 * s.llm_price_output_per_mtok)
                print(f"  {name:<26} {b['calls']:>6} {b['in']:>10,} "
                      f"{b['out']:>9,} ${cost:>9.4f}")

    # ---- written reports ----
    if args.out:
        payload = {
            "complete": True,
            "config": {
                "answering_model": s.llm_model,
                "judge_model": judge_model,
                "retrieval_final_k": s.retrieval_final_k,
                "retrieval_hybrid": s.retrieval_hybrid,
                "rerank_enabled": s.rerank_enabled,
                "retrieval_min_score": s.retrieval_min_score,
                "context_budget_tokens": s.llm_context_budget_tokens,
                "temperature": s.llm_temperature,
                "top_p": None,
                "reasoning_effort": s.llm_reasoning_effort,
                "price_input_per_mtok": s.llm_price_input_per_mtok,
                "price_output_per_mtok": s.llm_price_output_per_mtok,
                "tag": args.tag,
            },
            "gold_coverage": gold_mod.coverage(list(gold.values())),
            "retrieval": [
                metrics.aggregate_retrieval(retrieval_by_k[k]) for k in ks
                if retrieval_by_k[k]
            ],
            "generation": [a.as_dict() for a, _ in rows],
            "cost": {
                "mean": sum(costs) / len(costs) if costs else None,
                "p50": metrics.percentile(costs, 50),
                "p95": metrics.percentile(costs, 95),
                "total": sum(costs),
                "prompt_tokens": sum(r.prompt_tokens for r in ok),
                "completion_tokens": sum(r.completion_tokens for r in ok),
            },
            "questions": [question_payload(r) for r in results],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nFull per-question detail -> {out}")

    print("=" * 78)


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", default=DEFAULT_QUESTIONS)
    p.add_argument("--gold", default=DEFAULT_GOLD)
    p.add_argument("--answers", default=DEFAULT_ANSWERS,
                   help="Reference answers in Qn./An. form. Unlocks "
                        "context_recall, answer_coverage and "
                        "answer_correctness, and lets --build-gold label "
                        "questions that name no scheme themselves.")
    p.add_argument("--build-gold", action="store_true",
                   help="Draft the gold label file from the questions, then exit.")
    p.add_argument("--ks", default=DEFAULT_KS,
                   help=f"k values to sweep (default {DEFAULT_KS}).")
    p.add_argument("--limit", type=int, default=0, help="First N questions only.")
    p.add_argument("--only", default="", help="Specific qids, e.g. 1,4,22")
    p.add_argument("--judge-model", default="",
                   help="Model for the judges. Default: the answering model, "
                        "which is the weakest option -- a model grading itself "
                        "shares its blind spots.")
    p.add_argument("--no-judge", action="store_true",
                   help="Retrieval and cost only. No judge calls, so no cost.")
    p.add_argument("--no-retrieval", action="store_true",
                   help="Generation metrics only. Needs no gold labels.")
    p.add_argument("--show-claims", type=int, default=12,
                   help="How many unsupported claims to print (default 12).")
    p.add_argument("--out", default="eval/report.json",
                   help="Per-question JSON report. '' to skip.")
    p.add_argument("--rpm", type=int, default=0,
                   help="Client-side cap on requests per minute across ALL "
                        "calls. A full run is ~10 calls per question, so a "
                        "free-tier key needs this (try 10-15) or it dies on "
                        "429s. 0 uses LLM_MAX_REQUESTS_PER_MINUTE from .env.")
    p.add_argument("--quota-strikes", type=int, default=3,
                   help="Stop after this many consecutive quota failures "
                        "(default 3). A per-day cap cannot be waited out, so "
                        "continuing just burns quota and produces errors. "
                        "Progress is saved either way.")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions already present in the --out report. "
                        "An 85-question run takes a while; losing it at q60 "
                        "to a transient error should not mean starting over.")
    p.add_argument("--tag", default="",
                   help="Label this run in the report, for A/B comparisons.")
    args = p.parse_args()

    if args.build_gold:
        return build_gold(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
