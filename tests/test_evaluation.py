"""The evaluation harness.

An evaluator is the one piece of code nobody checks, because its output IS the
check. A silently wrong metric does not fail -- it reports a number, and you
act on it. So the properties asserted here are the ones whose failure would be
invisible:

  1. the gold matcher labels what it should and ABSTAINS rather than guessing
  2. retrieval maths is right at document and chunk level, and at each k
  3. context_precision is rank-weighted, not a fraction of useful chunks
  4. a refusal scores None, not 1.00 -- otherwise a system that answers
     nothing reports perfect groundedness
  5. an unparsed judge verdict counts as UNSUPPORTED, so a judge failure
     cannot inflate the score
  6. the whole program runs and its numbers match hand-computed values

    python -m tests.test_evaluation
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["RETRIEVAL_MIN_SCORE"] = "0.30"
os.environ["VERIFY_ENABLED"] = "false"
os.environ["LLM_API_KEY"] = "test-key-looks-real"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from schemes_rag.evaluation import gold as gold_mod  # noqa: E402
from schemes_rag.evaluation import judges, metrics  # noqa: E402
from tests.test_agents import CORPUS, build_env  # noqa: E402
from tests.test_smoke_agents import SmokeLLM  # noqa: E402

CORPUS_DIR = Path(CORPUS)
QUESTIONS = Path(__file__).resolve().parent / "test-data.txt"
ANSWERS = Path(__file__).resolve().parent / "test-ans.txt"


def judging_llm(llm):
    """Wrap a scripted LLM so it also answers the judge prompts.

    Scripted on the SYSTEM prompt, so it breaks loudly if a judge's prompt is
    reworded rather than silently returning the generic answer -- which would
    make every judge metric quietly measure nothing.
    """
    inner = llm.complete

    def complete(system, user, **kw):
        if "You break text apart" in system:
            return ('{"claims": ["PM-KISAN pays Rs 6,000 per year.",'
                    ' "The applicant must own land."],'
                    ' "aspects": ["benefit amount", "eligibility"]}')
        if "You check whether claims are supported" in system:
            # One supported, one not -> faithfulness exactly 0.5.
            return ('{"verdicts": [{"i":1,"supported":true,"reason":"stated"},'
                    '{"i":2,"supported":false,"reason":"not in context"}]}')
        if "You rate retrieved passages" in system:
            idx = sorted({int(m) for m in re.findall(r"^\[(\d+)\]", user, re.M)})
            return '{"ratings": [' + ",".join(
                f'{{"i":{i},"relevant":{"true" if i <= 3 else "false"},'
                f'"useful":{"true" if i <= 1 else "false"}}}' for i in idx
            ) + "]}"
        if "You check whether an answer addresses" in system:
            return ('{"verdicts": [{"i":1,"addressed":true},'
                    '{"i":2,"addressed":false}]}')
        if "You compare an answer against a known-correct" in system:
            # One reference claim present, one missing -> coverage 0.5.
            # One answer claim agreeing, one contradicting -> correctness 0.5.
            return ('{"coverage": [{"claim":"Rs 6,000 per year","present":true},'
                    '{"claim":"paid in three instalments","present":false}],'
                    ' "conflicts": [{"claim":"pays Rs 6,000","verdict":"agrees",'
                    '"detail":""},'
                    '{"claim":"entry age is 25","verdict":"contradicts",'
                    '"detail":"reference says 18-40"},'
                    '{"claim":"applications open year round",'
                    '"verdict":"unverifiable","detail":""}]}')
        if "You check whether retrieved context covers" in system:
            return ('{"claims": [{"claim":"Rs 6,000 per year","attributed":true},'
                    '{"claim":"paid in three instalments","attributed":false}]}')
        return inner(system, user, **kw)

    llm.complete = complete
    return llm


def main() -> int:  # noqa: C901
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_evaluation"])

    if not CORPUS_DIR.is_dir():
        print(f"SKIP: corpus not available at {CORPUS}")
        return 0
    if not QUESTIONS.is_file():
        print(f"SKIP: question set not found at {QUESTIONS}")
        return 0

    failures: list[str] = []

    # ---------------- 1. gold matcher ----------------
    print("\n--- 1. Gold labels: match what is named, abstain otherwise ---")
    llm = judging_llm(SmokeLLM())
    build_env(llm)

    from schemes_rag.scheme_index import get_scheme_index

    index = get_scheme_index()
    questions = gold_mod.parse_questions(QUESTIONS)
    print(f"  {len(questions)} questions, {len(index.entries)} schemes indexed")
    if len(questions) != 85:
        failures.append(f"expected 85 questions, parsed {len(questions)}")

    alias_map, max_span = gold_mod.build_alias_map(index)

    # THE TRAP. "Am I" collapses to "ami", which is the slug of Agricultural
    # Marketing Infrastructure -- a substring matcher labels every "Am I
    # eligible" question with a scheme about warehouses. The n-gram matcher
    # cannot, because it only ever compares whole words.
    ids, names = gold_mod.draft_labels(
        "Am I eligible to obtain a Kisan Credit Card?", index, alias_map, max_span)
    print(f"  'Am I eligible ... Kisan Credit Card' -> {names}")
    if any("Marketing Infrastructure" in n for n in names):
        failures.append('"Am I" matched Agricultural Marketing Infrastructure '
                        "via the collapsed 'ami' -- the substring trap is back")
    else:
        print("    OK  no 'ami' false positive")

    # A comparison question must label BOTH schemes, or recall@k is measured
    # against half the answer key and a correct retrieval scores 0.5.
    ids, names = gold_mod.draft_labels(
        "Compare PMJJBY and PMSBY.", index, alias_map, max_span)
    print(f"  'Compare PMJJBY and PMSBY' -> {len(ids)} docs: {names}")
    if len(ids) != 2:
        failures.append(f"comparison labelled {len(ids)} docs, expected 2")
    else:
        print("    OK  both schemes labelled")

    # An alias shared by two schemes must abstain. A wrong label is worse than
    # a missing one: it scores correct retrieval as a failure.
    shared = [a for a, hits in alias_map.items() if len(hits) > 1]
    print(f"  {len(shared)} ambiguous alias(es) in the index -> abstained on")

    # A question naming nothing must be needs_review, not silently empty.
    drafted = gold_mod.build(questions, index)
    profile_q = next(d for d in drafted if d.question.startswith("I'm a farmer"))
    print(f"  q{profile_q.qid} (profile, names no scheme) -> "
          f"source={profile_q.source!r}")
    if profile_q.source != "needs_review" or profile_q.note is None:
        failures.append("a question naming no scheme was not flagged for review")
    else:
        print("    OK  flagged for review with an explanation")

    cov = gold_mod.coverage(drafted)
    print(f"  coverage: {cov}")

    # ---------------- 1b. reference answers ----------------
    print("\n--- 1b. Reference answers: parsing, and the labels they unlock ---")
    if not ANSWERS.is_file():
        print(f"  SKIP: no answer file at {ANSWERS}")
        qa = {}
    else:
        qa = gold_mod.parse_qa(ANSWERS)
        print(f"  parsed {len(qa)} Q/A pairs")
        if len(qa) != 85:
            failures.append(f"parsed {len(qa)} Q/A pairs, expected 85")
        if set(qa) != set(range(1, 86)):
            failures.append("Q/A ids are not contiguous 1..85")
        # Multi-line answers carry bullet structure that is part of the
        # content; flattening it would change what the judges are comparing.
        if "  - Primary (Classes 1" not in qa[85][1]:
            failures.append("nested bullet indentation lost in parsing")
        else:
            print("    OK  multi-line answers and nested bullets preserved")

        # The span bug: aliases are stored tightened, so measuring THEM in
        # tokens gives 1, and a span limit of 1 silently finds only
        # single-word acronyms. "Who can apply for PM-KISAN?" is the 2-gram
        # "pm"+"kisan", which is exactly the slug -- it must match.
        _, span = gold_mod.build_alias_map(index)
        ids, names = gold_mod.draft_labels(
            "Who can apply for PM-KISAN?", index, alias_map, max_span)
        print(f"  max_span={span}; 'Who can apply for PM-KISAN?' -> {names}")
        if span < 2:
            failures.append(f"max_span is {span} -- multi-word scheme names "
                            "can never match")
        if not ids:
            failures.append("'Who can apply for PM-KISAN?' matched nothing, "
                            "though pmkisan is the slug")

        with_refs = gold_mod.build(questions, index, qa=qa)
        no_refs = gold_mod.build(questions, index)
        c_with = gold_mod.coverage(with_refs)
        c_without = gold_mod.coverage(no_refs)
        print(f"  labelled without references: {c_without['labelled']}")
        print(f"  labelled with references:    {c_with['labelled']} "
              f"({c_with['labelled_from_answer']} proposed from answers)")
        if c_with["labelled"] <= c_without["labelled"]:
            failures.append("reference answers did not improve label coverage")
        if c_with["with_reference_answer"] != len(questions):
            failures.append("not every question got its reference answer")
        else:
            print("    OK  references attached and coverage improved")

        # Answer-derived labels must be PROPOSED, not accepted: the matcher
        # cannot tell a recommended scheme from one mentioned in passing.
        proposed = [g for g in with_refs if g.label_origin == "answer"]
        if proposed and any(g.source == "auto" for g in proposed):
            failures.append("answer-derived labels were accepted as 'auto' "
                            "rather than flagged for review")
        elif proposed:
            print(f"    OK  {len(proposed)} answer-derived label(s) flagged "
                  f"needs_review")

    # ---------------- 2. retrieval maths ----------------
    print("\n--- 2. Retrieval maths, document and chunk level ---")
    # 8 chunks from docs A,A,B,A,C,B,A,D; gold = {A}
    retrieved = ["A", "A", "B", "A", "C", "B", "A", "D"]
    expected = {
        1: dict(doc_recall=1.0, doc_precision=1.0, chunk_precision=1.0, mrr=1.0),
        3: dict(doc_recall=1.0, doc_precision=0.5, chunk_precision=2 / 3, mrr=1.0),
        8: dict(doc_recall=1.0, doc_precision=0.25, chunk_precision=0.5, mrr=1.0),
    }
    for k, want in expected.items():
        s = metrics.score_retrieval(retrieved, ["A"], k)
        print(f"  k={k}: doc_rec={s.doc_recall:.3f} doc_prec={s.doc_precision:.3f} "
              f"chunk_prec={s.chunk_precision:.3f} mrr={s.doc_mrr:.3f}")
        for field, value in want.items():
            got = getattr(s, "doc_mrr" if field == "mrr" else field)
            if abs(got - value) > 1e-9:
                failures.append(f"k={k} {field}: got {got}, expected {value}")
    print("    OK  chunk duplicates counted for chunk precision, deduped for docs")

    # Partial recall on a two-document question.
    s = metrics.score_retrieval(["C", "A", "C", "B"], ["A", "B"], 2)
    if abs(s.doc_recall - 0.5) > 1e-9 or abs(s.doc_mrr - 0.5) > 1e-9:
        failures.append(f"partial recall wrong: {s.doc_recall}, mrr {s.doc_mrr}")
    else:
        print("    OK  partial recall and MRR at a truncated k")

    # ---------------- 3. context precision is rank-weighted ----------------
    print("\n--- 3. context_precision rewards ORDER, not just count ---")
    first = [{"rank": 1, "useful": True, "relevant": True}] + [
        {"rank": i, "useful": False, "relevant": True} for i in range(2, 9)]
    last = [{"rank": i, "useful": False, "relevant": True} for i in range(1, 8)] + [
        {"rank": 8, "useful": True, "relevant": True}]
    a, b = metrics.context_precision(first), metrics.context_precision(last)
    print(f"  one useful chunk at rank 1 -> {a:.3f}")
    print(f"  one useful chunk at rank 8 -> {b:.3f}")
    if abs(a - 1.0) > 1e-9 or abs(b - 0.125) > 1e-9:
        failures.append(f"context_precision not rank-weighted: {a}, {b}")
    else:
        print("    OK  same useful count, 8x the score for putting it first")
    # Both identical as a plain fraction, which is what this must NOT be.
    if abs(metrics.context_relevance(first) - metrics.context_relevance(last)) > 1e-9:
        failures.append("context_relevance should be order-independent")

    # ---------------- 4. refusals are not perfect ----------------
    print("\n--- 4. A refusal scores None, never 1.00 ---")
    print(f"  faithfulness([]) = {metrics.faithfulness([])}")
    if metrics.faithfulness([]) is not None:
        failures.append("an answer with no claims did not score None -- a system "
                        "that refuses everything would report perfect groundedness")
    else:
        print("    OK  excluded rather than credited")

    # ---------------- 5. a judge failure cannot inflate ----------------
    print("\n--- 5. An unparsed verdict counts as UNSUPPORTED ---")
    broken = judges.judge_faithfulness  # exercised through a broken judge below
    saved = llm.complete

    def garbage(system, user, **kw):
        if "You check whether claims are supported" in system:
            return "I cannot produce JSON for this."
        return saved(system, user, **kw)

    llm.complete = garbage
    verdicts = broken(["A pays Rs 6,000.", "B is free."], "some context")
    llm.complete = saved
    print(f"  verdicts: {[(v['supported'], v['judged']) for v in verdicts]}")
    print(f"  faithfulness = {metrics.faithfulness(verdicts)}")
    if metrics.faithfulness(verdicts) != 0.0 or any(v["judged"] for v in verdicts):
        failures.append("an unparseable judge response did not score 0 -- a "
                        "judge outage would silently read as perfect grounding")
    else:
        print("    OK  pessimistic on judge failure, and marked unjudged")

    # ---------------- 6. the program end to end ----------------
    print("\n--- 6. The program runs and its numbers are hand-checkable ---")
    import json
    import tempfile

    import evaluate_rag
    from schemes_rag import usage as U

    tmp = Path(tempfile.mkdtemp())
    gold_path, report_path = tmp / "gold.jsonl", tmp / "report.json"

    # The scripted LLM reports no token usage, so stand some in to exercise
    # the cost path: 900+80 and 7400+210 tokens per question.
    import schemes_rag.graph as G

    real_run = G.run_query

    def run_query(q, **kw):
        U.record(name="understand.router", model="fake",
                 prompt_tokens=900, completion_tokens=80)
        U.record(name="generate.documents", model="fake",
                 prompt_tokens=7400, completion_tokens=210)
        return real_run(q, **kw)

    G.run_query = run_query

    argv = sys.argv
    try:
        sys.argv = ["evaluate_rag.py", "--build-gold",
                    "--questions", str(QUESTIONS), "--gold", str(gold_path),
                    "--answers", str(ANSWERS)]
        if evaluate_rag.main() != 0:
            failures.append("--build-gold returned non-zero")

        labelled = [g for g in gold_mod.load(gold_path) if g.is_labelled]
        print(f"  gold built: {len(labelled)} labelled of {len(questions)}")
        if not labelled:
            failures.append("gold build produced no labels at all")

        qids = ",".join(str(g.qid) for g in labelled[:6])
        sys.argv = ["evaluate_rag.py", "--questions", str(QUESTIONS),
                    "--gold", str(gold_path), "--answers", str(ANSWERS),
                    "--ks", "1,3,8",
                    "--only", qids, "--out", str(report_path)]
        if evaluate_rag.main() != 0:
            failures.append("the evaluation returned non-zero")
    finally:
        sys.argv = argv
        G.run_query = real_run

    payload = json.loads(report_path.read_text())
    gen = {g["metric"]: g for g in payload["generation"]}
    print(f"  faithfulness {gen['faithfulness']['mean']} "
          f"({gen['faithfulness']['scored']}/{gen['faithfulness']['eligible']})")
    print(f"  completeness {gen['completeness']['mean']}")
    print(f"  cost/request ${payload['cost']['mean']:.5f}")
    print(f"  retrieval at k: {[r['k'] for r in payload['retrieval']]}")

    # The scripted judge says 1 of 2 claims supported, and 1 of 2 aspects
    # addressed, for every question -- so both means must be exactly 0.5.
    if gen["faithfulness"]["mean"] not in (None,) and abs(
        gen["faithfulness"]["mean"] - 0.5
    ) > 1e-9:
        failures.append(f"faithfulness {gen['faithfulness']['mean']}, expected 0.5")
    if gen["completeness"]["mean"] is not None and abs(
        gen["completeness"]["mean"] - 0.5
    ) > 1e-9:
        failures.append(f"completeness {gen['completeness']['mean']}, expected 0.5")
    # 8300 prompt + 290 completion tokens at the configured rates.
    from schemes_rag.config import get_settings

    st = get_settings()
    want_cost = (8300 / 1e6 * st.llm_price_input_per_mtok
                 + 290 / 1e6 * st.llm_price_output_per_mtok)
    if abs(payload["cost"]["mean"] - want_cost) > 1e-9:
        failures.append(f"cost/request {payload['cost']['mean']}, "
                        f"expected {want_cost}")
    else:
        print("    OK  cost matches the token counts and configured rates")
    for name, want in (("answer_correctness", 0.5), ("completeness", 0.5)):
        got = gen.get(name, {}).get("mean")
        print(f"  {name} {got}")
        if got is None or abs(got - want) > 1e-9:
            failures.append(f"{name} {got}, expected {want}")
    # completeness must come from the REFERENCE when one exists, not from the
    # aspect-based guess -- the scripted aspects give 0.5 too, so this is
    # checked by the coverage count instead.
    if gen.get("answer_correctness", {}).get("scored", 0) == 0:
        failures.append("answer_correctness never scored, though references exist")
    q = payload["questions"][0]
    if "contradictions" not in q or "missed_reference_claims" not in q:
        failures.append("per-question report lacks the reference detail")
    elif q["contradictions"]:
        print(f"  contradictions captured: {q['contradictions'][0]['claim'][:46]}")

    if payload["config"]["top_p"] is not None:
        failures.append("top_p reported as set; this project does not set it")
    if not payload["questions"]:
        failures.append("report has no per-question detail")

    # ---------------- 7. failures are not scored as answers ----------------
    #
    # THE REGRESSION THIS GUARDS. A full run against a free-tier key returned
    # 429 and 503 part way through. Generation caught them and returned prose
    # with no citations, which the evaluator counted as real answers -- so a
    # quota problem would have been reported as a groundedness collapse, and
    # the retriever blamed for a billing issue.
    print("\n--- 7. A failed model call is excluded, not scored ---")
    from schemes_rag.config import get_settings
    import httpx
    from openai import APIStatusError, RateLimitError

    from schemes_rag.generate import generate_answer
    from schemes_rag.models import Answer

    def rate_limited(*a, **kw):
        req = httpx.Request("POST", "http://x/chat/completions")
        raise RateLimitError(
            "quota", response=httpx.Response(429, request=req), body=None)

    saved_complete = llm.complete
    llm.complete = rate_limited
    try:
        from schemes_rag.retriever import get_retriever

        chunks = get_retriever().retrieve(
            "Who can apply for PM-KISAN?",
            gate_query="Who can apply for PM-KISAN?").chunks
        failed = generate_answer("Who can apply for PM-KISAN?", chunks,
                                 route="benefits")
    finally:
        llm.complete = saved_complete

    print(f"  model_failed={failed.model_failed} "
          f"insufficient_evidence={failed.insufficient_evidence} "
          f"citations={len(failed.citations)}")
    if not failed.model_failed:
        failures.append("a rate-limited generation did not set model_failed, so "
                        "an evaluator cannot tell it from a real answer")
    elif failed.insufficient_evidence:
        failures.append("a model failure was marked insufficient_evidence -- "
                        "that means retrieval found nothing, which is untrue")
    else:
        print("    OK  flagged as a model failure, not a refusal")

    # And the evaluator must turn that flag into an exclusion.
    probe = evaluate_rag.QuestionResult(qid=999, question="q")
    import schemes_rag.graph as GG

    saved_run = GG.run_query
    GG.run_query = lambda q, **kw: Answer(
        answer="The language model call failed (RateLimitError).",
        citations=[], insufficient_evidence=False, model_failed=True,
        route="benefits")
    try:
        probe = evaluate_rag.answer_question("q", 999, get_settings())
    finally:
        GG.run_query = saved_run
    print(f"  evaluator: error={probe.error!r} refused={probe.refused}")
    if not probe.error:
        failures.append("the evaluator scored a model failure as an answer")
    elif probe.refused:
        failures.append("a model failure was counted as a refusal")
    else:
        print("    OK  excluded from the metrics")

    # ---------------- 8. retries cover 429/503 ----------------
    print("\n--- 8. Transient statuses retry; permanent ones do not ---")
    from schemes_rag.llm import LLMClient

    st = get_settings()
    st.llm_max_retries, st.llm_retry_base_delay = 2, 0.001
    probe_client = LLMClient.__new__(LLMClient)
    probe_client.settings = st

    def status_error(code):
        req = httpx.Request("POST", "http://x/chat/completions")
        return APIStatusError(
            "e", response=httpx.Response(code, request=req), body=None)

    for code, should_retry in ((429, True), (503, True), (500, True),
                               (401, False), (404, False), (400, False)):
        got = probe_client._retry_delay(status_error(code), 0, "t") is not None
        if got != should_retry:
            failures.append(f"HTTP {code}: retry={got}, expected {should_retry}")
    print("  429/500/503 retry; 400/401/404 raise immediately")
    # Bounded, or a quota outage becomes an infinite loop.
    if probe_client._retry_delay(status_error(429), st.llm_max_retries, "t") is not None:
        failures.append("retries are not bounded by llm_max_retries")
    else:
        print(f"    OK  gives up after {st.llm_max_retries} retries")

    # ---------------- 9. daily quota vs per-minute ----------------
    #
    # THE REGRESSION THIS GUARDS. A paced run died 61 questions in to
    # sustained 429s. They were a per-DAY cap, but every one was retried four
    # times with backoff -- spending more of the exhausted allowance to
    # discover it was exhausted -- and the advice printed was "lower the
    # request rate", which cannot help a daily cap.
    print("\n--- 9. A per-day 429 fails fast; per-minute still retries ---")
    from schemes_rag.llm import quota_scope

    daily_body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{
                 "quotaMetric": "generativelanguage.googleapis.com/"
                                "generate_content_free_tier_requests",
                 "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
             }]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo",
             "retryDelay": "31s"}]}}
    minute_body = {"error": {"code": 429, "details": [{"violations": [
        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}

    def api_error(code, body=None, msg="e"):
        req = httpx.Request("POST", "http://x/chat/completions")
        return APIStatusError(
            msg, response=httpx.Response(code, request=req), body=body)

    for label, body, want in (("per-day", daily_body, "day"),
                              ("per-minute", minute_body, "minute"),
                              ("unlabelled", None, None)):
        got = quota_scope(api_error(429, body))
        print(f"  {label:12} -> scope={got}")
        if got != want:
            failures.append(f"{label} 429 classified {got!r}, expected {want!r}")

    # The daily one must NOT be retried -- a retry spends quota that is gone.
    if probe_client._retry_delay(api_error(429, daily_body), 0, "t") is not None:
        failures.append("a per-day 429 was retried; every retry consumes more "
                        "of an allowance that cannot succeed")
    else:
        print("    OK  per-day fails immediately")
    # Per-minute and 503 must still retry, or a transient blip kills the run.
    if probe_client._retry_delay(api_error(429, minute_body), 0, "t") is None:
        failures.append("a per-minute 429 was not retried")
    elif probe_client._retry_delay(api_error(503), 0, "t") is None:
        failures.append("a 503 was not retried")
    else:
        print("    OK  per-minute and 503 still back off and retry")

    # ---------------- 10. progress survives an abort ----------------
    print("\n--- 10. Progress is saved per question, so --resume has data ---")
    part = tmp / "partial.json"
    rows = [
        evaluate_rag.QuestionResult(qid=1, question="q1", answer="a",
                                    faithfulness=1.0, cost=0.001),
        evaluate_rag.QuestionResult(qid=2, question="q2",
                                    error="model call failed"),
    ]
    evaluate_rag.save_progress(str(part), rows, {"tag": "t"})
    saved = json.loads(part.read_text())
    print(f"  wrote {len(saved['questions'])} record(s), complete="
          f"{saved['complete']}")
    if len(saved["questions"]) != 2 or saved["complete"]:
        failures.append("save_progress did not write a partial report")
    # --resume must keep the good one and retry the errored one.
    resumable = {q["qid"]: q for q in saved["questions"] if not q.get("error")}
    print(f"  resumable: {sorted(resumable)} (errored q2 will be retried)")
    if sorted(resumable) != [1]:
        failures.append(f"resume set is {sorted(resumable)}, expected [1]")
    else:
        print("    OK  scored questions kept, failed ones retried")

    print("\n" + "=" * 76)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("EVALUATION HARNESS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
