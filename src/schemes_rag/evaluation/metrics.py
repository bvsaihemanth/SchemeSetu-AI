"""Metric computation.

Retrieval metrics are arithmetic over the gold labels -- no LLM, so they are
exact and free. The generation metrics are counts over the judge verdicts in
judges.py.

TWO GRANULARITIES, DELIBERATELY

Gold labels are DOCUMENTS ("this question is about PM-KISAN"); retrieval
returns CHUNKS. Collapsing one into the other loses something either way, so
both are reported:

  document level  did we find the right scheme at all? This is the retrieval
                  question -- whether entity resolution and hybrid search put
                  the correct document in front of the model.
  chunk level     how much of the model's context came from the right
                  document? This is the context question -- it drives
                  groundedness and it drives cost, since every irrelevant
                  chunk is prompt tokens paid for and attention spent.

A run can score 1.00 on document recall and 0.40 on chunk precision: the right
scheme was found, and more than half the context was still noise. Reporting
one number would hide that, and they call for different fixes -- the first for
the resolver, the second for final_k and reranking.

WHY precision@k IS NOT MEAN PRECISION

Averaging precision across questions with different numbers of relevant
documents weights a 1-relevant question the same as a 3-relevant comparison
question. Both are reported: the macro mean (every question counts once, which
is what you want for "how does this behave for a user") and the micro total
(pooled hits over pooled retrievals, which is what you want for "how much of
the retrieved context was on target overall").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
@dataclass
class RetrievalScores:
    """One question, at one k."""

    k: int
    # document level
    doc_recall: float = 0.0
    doc_precision: float = 0.0
    doc_hit: bool = False
    doc_mrr: float = 0.0
    # chunk level
    chunk_precision: float = 0.0
    # raw counts, kept so a micro average can be pooled correctly rather than
    # reconstructed from ratios
    relevant_docs_found: int = 0
    gold_docs: int = 0
    distinct_docs_retrieved: int = 0
    relevant_chunks: int = 0
    chunks_retrieved: int = 0


def score_retrieval(
    retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int
) -> RetrievalScores:
    """Retrieval metrics for the top-k chunks.

    `retrieved_doc_ids` is one entry per retrieved CHUNK, in rank order, each
    being the id of the document that chunk came from -- so the same id
    repeats when several chunks come from one document. That repetition is the
    signal for chunk-level precision and must not be deduplicated away before
    counting.
    """
    gold = set(gold_doc_ids)
    top = retrieved_doc_ids[:k]

    # Chunk level: every chunk in the context counts, duplicates included.
    relevant_chunks = sum(1 for d in top if d in gold)

    # Document level: distinct documents, order preserved for MRR.
    seen: list[str] = []
    for d in top:
        if d not in seen:
            seen.append(d)
    found = [d for d in seen if d in gold]

    # Reciprocal rank of the FIRST relevant document, by chunk rank -- the
    # rank the model actually sees it at.
    mrr = 0.0
    for rank, d in enumerate(top, 1):
        if d in gold:
            mrr = 1.0 / rank
            break

    return RetrievalScores(
        k=k,
        doc_recall=len(found) / len(gold) if gold else 0.0,
        doc_precision=len(found) / len(seen) if seen else 0.0,
        doc_hit=bool(found),
        doc_mrr=mrr,
        chunk_precision=relevant_chunks / len(top) if top else 0.0,
        relevant_docs_found=len(found),
        gold_docs=len(gold),
        distinct_docs_retrieved=len(seen),
        relevant_chunks=relevant_chunks,
        chunks_retrieved=len(top),
    )


def aggregate_retrieval(scores: list[RetrievalScores]) -> dict:
    """Macro and micro aggregates over questions, at one k."""
    if not scores:
        return {}
    gold_total = sum(s.gold_docs for s in scores)
    found_total = sum(s.relevant_docs_found for s in scores)
    docs_total = sum(s.distinct_docs_retrieved for s in scores)
    chunks_total = sum(s.chunks_retrieved for s in scores)
    rel_chunks_total = sum(s.relevant_chunks for s in scores)
    return {
        "k": scores[0].k,
        "questions": len(scores),
        # Macro: each question contributes equally. The user-facing view.
        "doc_recall@k": mean(s.doc_recall for s in scores),
        "doc_precision@k": mean(s.doc_precision for s in scores),
        "hit@k": mean(1.0 if s.doc_hit else 0.0 for s in scores),
        "mrr@k": mean(s.doc_mrr for s in scores),
        "chunk_precision@k": mean(s.chunk_precision for s in scores),
        # Micro: pooled. The corpus-wide view, and the one that reflects how
        # many prompt tokens were wasted.
        "doc_recall@k_micro": found_total / gold_total if gold_total else 0.0,
        "doc_precision@k_micro": found_total / docs_total if docs_total else 0.0,
        "chunk_precision@k_micro": (
            rel_chunks_total / chunks_total if chunks_total else 0.0
        ),
        "avg_chunks_returned": chunks_total / len(scores),
    }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def faithfulness(verdicts: list[dict]) -> float | None:
    """Supported claims / total claims. None when the answer asserts nothing.

    None rather than 1.0 for an empty answer. A refusal makes no claims, so it
    cannot be unfaithful -- but scoring it a perfect 1.0 would mean a system
    that refuses everything reports flawless groundedness, which inverts the
    metric. Refusals are counted separately and excluded here.
    """
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v["supported"]) / len(verdicts)


def context_relevance(ratings: list[dict]) -> float | None:
    """Relevant chunks / retrieved chunks, by LLM judgement."""
    if not ratings:
        return None
    return sum(1 for r in ratings if r["relevant"]) / len(ratings)


def context_precision(ratings: list[dict]) -> float | None:
    """Average precision@k over chunk usefulness -- rank-weighted.

    This is the Ragas definition, and the rank weighting is the whole point:
    it is NOT the fraction of useful chunks. Precision@i is computed at every
    rank where a useful chunk appears, and those are averaged.

    So a system that puts its one useful chunk first scores 1.00, and one that
    buries it at rank 8 scores 0.125, even though both retrieved exactly one
    useful chunk out of eight. That difference is real: it is the difference
    between a model that reads the answer immediately and one that has to find
    it past seven distractors.
    """
    if not ratings:
        return None
    useful_so_far = 0
    precisions: list[float] = []
    for i, r in enumerate(sorted(ratings, key=lambda x: x["rank"]), 1):
        if r["useful"]:
            useful_so_far += 1
            precisions.append(useful_so_far / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def completeness(verdicts: list[dict]) -> float | None:
    """Addressed aspects / requested aspects."""
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v["addressed"]) / len(verdicts)


def context_recall(claims: list[dict]) -> float | None:
    """Reference claims the context covers / reference claims."""
    if not claims:
        return None
    return sum(1 for c in claims if c["attributed"]) / len(claims)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
@dataclass
class Aggregate:
    """A metric's mean, with the coverage it was computed over.

    Coverage travels WITH the value, never beside it. "faithfulness 0.94" over
    12 of 85 questions is a different claim from the same number over all 85,
    and the difference must not be something the reader has to go and look up.
    """

    name: str
    values: list[float] = field(default_factory=list)
    eligible: int = 0
    note: str = ""

    def add(self, value: float | None) -> None:
        self.eligible += 1
        if value is not None:
            self.values.append(value)

    @property
    def mean(self) -> float | None:
        return mean(self.values) if self.values else None

    @property
    def n(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict:
        return {
            "metric": self.name,
            "mean": self.mean,
            "scored": self.n,
            "eligible": self.eligible,
            "note": self.note,
        }

    def format(self) -> str:
        value = "   n/a" if self.mean is None else f"{self.mean:6.3f}"
        return f"{self.name:<24} {value}   ({self.n}/{self.eligible} scored)"


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. For latency and cost, where the tail is the
    number that matters -- a mean cost hides the question that cost 10x."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[idx]


# --------------------------------------------------------------------------
# against the reference answer
# --------------------------------------------------------------------------
def answer_coverage(coverage: list[dict]) -> float | None:
    """Reference claims the answer contains / reference claims.

    Reference-based completeness: what the answer LEFT OUT. Preferred over the
    aspect-based version when a reference exists, because the reference states
    exactly what a complete answer contains rather than asking a model to
    guess what the question implied.
    """
    if not coverage:
        return None
    return sum(1 for c in coverage if c["present"]) / len(coverage)


def answer_correctness(conflicts: list[dict]) -> float | None:
    """Non-contradicting claims / verifiable claims.

    THE METRIC FAITHFULNESS CANNOT PROVIDE. Faithfulness asks whether the
    answer matches the retrieved context, so an answer that faithfully
    reports the WRONG document scores a perfect 1.00 while being wrong. Only
    comparison against a reference catches that.

    `unverifiable` claims are excluded from the denominator rather than
    counted as errors: a reference answer is not exhaustive, and extra true
    detail is not a mistake. That makes this a measure of how much of what the
    answer asserted is CONTRADICTED -- not how much of it the reference
    happened to mention.
    """
    verifiable = [c for c in conflicts if c["verdict"] in ("agrees", "contradicts")]
    if not verifiable:
        return None
    agrees = sum(1 for c in verifiable if c["verdict"] == "agrees")
    return agrees / len(verifiable)
