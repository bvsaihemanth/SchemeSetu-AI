"""The gold set: which documents SHOULD a question retrieve.

WHY THIS FILE EXISTS AT ALL

recall@k and precision@k are ratios against a known-correct set. Without
labels there is nothing to divide by, and a harness that reports them anyway
is reporting its own opinion of itself. So the labels come first, and they are
a reviewable artifact the project owns -- eval/gold.jsonl -- not something
computed at run time.

HOW THE DRAFT IS MADE, AND WHY IT IS NOT CIRCULAR

Most of these 85 questions name their scheme outright ("Who can apply for
PM-KISAN?"), so the relevant document is not a matter of opinion and can be
found by matching the question text against the corpus's own scheme names.

The matcher here is deliberately NOT scheme_index.resolve(), the production
resolver, even though that would match more. Grading retrieval with the same
component that performs it measures agreement, not correctness: a resolver bug
would move the gold labels to match the bug and the score would stay high.
This is a separate, dumber matcher -- case-folded substring on full names and
parenthesised acronyms, longest match wins -- whose failures look like
"unresolved", not like a wrong label.

WHAT IT CANNOT DO

About a fifth of the set describes a person rather than naming a scheme:

    "I'm a farmer in Telangana with 2 hectares, which schemes can I benefit
     from?"

There is no single correct document for that, and inventing one would be
fabricating an answer key. Those are written out as `needs_review` with an
empty label set, excluded from the retrieval metrics, and REPORTED as
excluded, so the coverage of the eval is always visible next to its results.

A label the user has edited is marked `source: "human"` and is never
overwritten by a later re-draft.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Alias kinds the answer key may use. `name` is the document's own title and
#: `slug`/`paren` come from its URL and title -- all three are statements of
#: what the document IS. `gen` is deliberately excluded: those are the
#: resolver's invented abbreviations (pmksn, pmkisansamman), and grading
#: retrieval against the resolver's guesses would measure agreement with a
#: component under test rather than correctness.
_TRUSTED_ALIAS_KINDS = frozenset({"name", "slug", "paren"})

#: Shortest alias worth matching. Two-character slugs exist ("ay" for Agnipath
#: Yojana) and are too ambiguous to label an answer key with.
_MIN_ALIAS_LEN = 3

#: Aliases that collide with ordinary question words once separators are
#: stripped. Short, generic, and each one would mislabel many questions.
_STOPALIASES = frozenset({"ami", "for", "the", "and", "any", "can", "who",
                          "has", "are", "was", "pmo", "new", "all", "one"})


@dataclass
class GoldQuestion:
    """One question and the documents that should answer it."""

    qid: int
    question: str
    #: doc_ids that genuinely answer this question. Empty means unlabelled,
    #: NOT "nothing is relevant".
    relevant_doc_ids: list[str] = field(default_factory=list)
    #: Human-readable mirror of the above, so the file can be reviewed and
    #: corrected without looking up doc_ids.
    relevant_schemes: list[str] = field(default_factory=list)
    #: "auto" (drafted by the matcher), "human" (reviewed or written by hand,
    #: never overwritten), "needs_review" (the matcher abstained).
    source: str = "auto"
    #: Where the labels came from: "question" (the question named the scheme),
    #: "answer" (only the reference answer named it), "" (unlabelled).
    #:
    #: Worth recording because the two are not equally safe. A scheme named in
    #: the QUESTION is unambiguously what was asked about. A scheme named in
    #: the ANSWER is what a human decided was relevant -- authoritative, but
    #: broader: the answer to "I am a farmer with 2 hectares" lists six
    #: schemes, and a system that finds four of them is not 33% wrong. Keeping
    #: the origin visible means a low recall@k on answer-derived labels can be
    #: read for what it is rather than mistaken for a retrieval failure.
    label_origin: str = ""
    #: A reference answer, for the metrics that need one (context recall
    #: against a reference, and reference-based completeness). Optional --
    #: those metrics report their coverage rather than assuming it.
    reference_answer: str | None = None
    #: What the question actually asks for, one aspect per item. Optional:
    #: when absent, completeness decomposes the question with an LLM instead.
    required_aspects: list[str] = field(default_factory=list)
    note: str | None = None

    @property
    def is_labelled(self) -> bool:
        return bool(self.relevant_doc_ids)

    @property
    def is_locked(self) -> bool:
        """Human-reviewed labels are never overwritten by a re-draft."""
        return self.source == "human"


def parse_questions(path: str | Path) -> list[tuple[int, str]]:
    """Read the numbered question file.

    Tolerates "1. ", "1) " and bare lines, and uses the file's own numbering
    when present so that qids stay stable if a line is added or removed --
    a gold label pointing at the wrong question is worse than no label.
    """
    out: list[tuple[int, str]] = []
    fallback = 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*[.)]\s*(.+)$", line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
        else:
            fallback += 1
            out.append((fallback, line))
    return out


#: "Q12." / "A12." at the start of a line, which is how the answer file marks
#: its pairs. Anchored to line starts so a "Q1." inside prose cannot split an
#: answer in half.
_Q_MARKER = re.compile(r"^Q(\d+)\.\s*(.*)$")
_A_MARKER = re.compile(r"^A(\d+)\.\s*(.*)$")
#: The file's own rules: 80 dashes between pairs, equals signs around the
#: header. Both are layout, not content.
_RULE = re.compile(r"^[-=]{10,}$")


def parse_qa(path: str | Path) -> dict[int, tuple[str, str]]:
    """Read the reference answer file: {qid: (question, answer)}.

    The format is

        Q1. <question>
        A1. <answer, which may run to many lines of bullets>
        --------------------------------------------------------------
        Q2. ...

    Answers are multi-line and carry bullet structure that matters -- the
    nesting in Q85's nutritional norms is part of the content -- so lines are
    accumulated verbatim until the next marker or rule, and only the
    surrounding blank lines are stripped.

    Mismatched Q/A numbering is an error worth raising rather than papering
    over: a reference answer attached to the wrong question would score a
    correct answer as wrong, and the eval would send you debugging a system
    that works.
    """
    questions: dict[int, str] = {}
    answers: dict[int, list[str]] = {}
    current: int | None = None

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if _RULE.match(line.strip()):
            current = None
            continue
        m = _Q_MARKER.match(line)
        if m:
            qid = int(m.group(1))
            questions[qid] = m.group(2).strip()
            current = None
            continue
        m = _A_MARKER.match(line)
        if m:
            qid = int(m.group(1))
            if qid not in questions:
                raise ValueError(
                    f"A{qid} has no matching Q{qid} in {path} -- the answer "
                    "file and the question numbering disagree"
                )
            current = qid
            first = m.group(2).strip()
            answers[qid] = [first] if first else []
            continue
        if current is not None:
            answers[current].append(line)

    missing = sorted(set(questions) - set(answers))
    if missing:
        raise ValueError(f"No answer for Q{missing} in {path}")

    out: dict[int, tuple[str, str]] = {}
    for qid, question in sorted(questions.items()):
        body = "\n".join(answers.get(qid, [])).strip()
        out[qid] = (question, body)
    return out


def build_alias_map(index) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """(alias -> [(doc_id, scheme_name)], longest alias in tokens).

    Keyed on the separator-insensitive form the index already stores, which is
    what makes "PM-KISAN", "PM KISAN" and "pmkisan" the same key. Note the
    acronym for a scheme lives in its SLUG, not its title: the entry for
    "Pradhan Mantri Kisan Samman Nidhi" carries `pmkisan` from its URL, and
    the string "PM-KISAN" appears nowhere in the document's name. Matching
    titles alone finds almost nothing -- which is the same trap that made
    retrieval return four wrong schemes for this exact question once before.
    """
    from ..scheme_index import tight, tokens as split_tokens

    alias_map: dict[str, list[tuple[str, str]]] = {}
    longest = 1
    for entry in index.entries.values():
        pairs = list(entry.aliases or [])
        # The title, in case an index predates alias generation.
        pairs.append([tight(entry.scheme_name), "name"])
        for alias, kind in pairs:
            if kind not in _TRUSTED_ALIAS_KINDS:
                continue
            key = tight(alias)
            if len(key) < _MIN_ALIAS_LEN or key in _STOPALIASES:
                continue
            bucket = alias_map.setdefault(key, [])
            if (entry.doc_id, entry.scheme_name) not in bucket:
                bucket.append((entry.doc_id, entry.scheme_name))
        # The span is measured on the UNTIGHTENED name, not the alias.
        #
        # Aliases are stored already collapsed -- "Pradhan Mantri Kisan Samman
        # Nidhi" is held as "pradhanmantrikisansammannidhi" -- so measuring an
        # alias in tokens always gives 1, and a span limit of 1 means only
        # single-word acronyms can ever match. That silently labelled 16 of 85
        # questions and left every multi-word scheme name unfound, including
        # "Who can apply for PM-KISAN?", whose 2-gram "pm"+"kisan" is exactly
        # the slug.
        longest = max(longest, len(split_tokens(entry.scheme_name)))
    return alias_map, min(longest, 24)


def draft_labels(question: str, index, alias_map=None, max_span: int = 0):
    """(doc_ids, scheme_names) for the schemes the question names outright.

    Exact n-gram lookup, not substring search. The question is split into
    words and every run of up to `max_span` consecutive words is collapsed to
    the index's separator-insensitive form and looked up. "PM-KISAN" becomes
    the 2-gram ("pm", "kisan") -> "pmkisan", which hits the slug exactly.

    Exactness is the point. A substring search over the collapsed question
    finds aliases inside unrelated words -- this project has already been bitten
    by "am i" collapsing to "ami" and matching the slug of Agricultural
    Marketing Infrastructure. An n-gram lookup cannot do that, because it only
    ever compares whole words.

    Longest span wins and claims its words, so in "Compare PMAY-U and PMAY-G"
    each acronym is matched once and neither is swallowed by a longer span
    covering both.
    """
    from ..scheme_index import tight, tokens as split_tokens

    if alias_map is None:
        alias_map, max_span = build_alias_map(index)
    max_span = max_span or 24

    words = split_tokens(question)
    claimed: set[int] = set()
    doc_ids: list[str] = []
    names: list[str] = []

    # Longest spans first: a 5-word title beats the 2-word acronym inside it.
    for span in range(min(max_span, len(words)), 0, -1):
        for start in range(0, len(words) - span + 1):
            if any(i in claimed for i in range(start, start + span)):
                continue
            key = "".join(words[start:start + span])
            if len(key) < _MIN_ALIAS_LEN:
                continue
            hits = alias_map.get(key)
            # Abstain on an ambiguous alias rather than guess. A WRONG label is
            # worse than a missing one: it scores correct retrieval as a
            # failure and sends you debugging a component that works.
            if not hits or len(hits) > 1:
                continue
            claimed.update(range(start, start + span))
            doc_id, name = hits[0]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
                names.append(name)
    return doc_ids, names


def build(
    questions: list[tuple[int, str]],
    index,
    existing: list[GoldQuestion] | None = None,
    qa: dict[int, tuple[str, str]] | None = None,
) -> list[GoldQuestion]:
    """Draft a gold set, preserving anything a human has edited.

    `qa` supplies reference answers, and changes what is possible. Before them,
    a question like

        "I'm a farmer in Telangana with 2 hectares, which schemes can I
         benefit from?"

    had no derivable answer key at all -- it names no scheme -- and was
    excluded from the retrieval metrics. Its reference answer names six
    (PM-KISAN, PMFBY, KCC, PM-KMDY, PMKSY, Soil Health Card), so the key is
    now a matter of reading rather than guessing.

    Question-derived labels take precedence where they exist, and the answer is
    only consulted when the question named nothing. That ordering is
    deliberate: for "Who can apply for PM-KISAN?" the question IS the
    specification, and pulling in every scheme the answer happens to mention
    in passing would inflate the gold set and score correct retrieval as
    partial recall.
    """
    locked = {
        g.qid: g for g in (existing or [])
        if g.is_locked or g.required_aspects
    }
    # Built once, not per question: 725 schemes times 85 questions times a
    # span sweep is enough work to notice.
    alias_map, max_span = build_alias_map(index)
    qa = qa or {}
    out: list[GoldQuestion] = []

    for qid, question in questions:
        reference = (qa.get(qid) or (None, None))[1]

        if qid in locked:
            kept = locked[qid]
            kept.question = question
            # A human may have corrected the labels; the reference answer is
            # still refreshed from the file, which is its source of truth.
            if reference:
                kept.reference_answer = reference
            out.append(kept)
            continue

        doc_ids, names = draft_labels(question, index, alias_map, max_span)
        origin = "question" if doc_ids else ""
        note = None
        source = "auto"

        if not doc_ids and reference:
            doc_ids, names = draft_labels(reference, index, alias_map, max_span)
            if doc_ids:
                origin = "answer"
                # PROPOSED, not accepted. The matcher cannot tell a scheme the
                # answer RECOMMENDS from one it mentions in passing, and the
                # difference decides whether the label is right.
                #
                # Observed: "What documents do I need for PMAY-Gramin?" has an
                # answer listing "MGNREGA Job Card number" among the required
                # documents, so MGNREGA gets proposed as a relevant DOCUMENT
                # for a question about PMAY-G. That label would score correct
                # retrieval as a failure and send you debugging a working
                # retriever.
                #
                # So these are filled in and flagged: confirming or deleting a
                # handful of proposals is a minute of reading, and the eval
                # reports how many of its labels came this way.
                source = "needs_review"
                note = (
                    "PROPOSED from the reference answer -- the question names "
                    "no scheme this index contains. Check these are schemes "
                    "the answer RECOMMENDS, not ones it mentions in passing "
                    "(a scheme named as a required document is not a relevant "
                    "document). Then set source=human."
                )

        if not doc_ids:
            source = "needs_review"
            note = (
                "Neither the question nor the reference answer names a scheme "
                "this index contains. Add the doc_ids you consider correct and "
                "set source=human, or leave it to be excluded from the "
                "retrieval metrics."
            )

        out.append(GoldQuestion(
            qid=qid,
            question=question,
            relevant_doc_ids=doc_ids,
            relevant_schemes=names,
            source=source,
            label_origin=origin,
            reference_answer=reference,
            note=note,
        ))
    return out


def save(gold: list[GoldQuestion], path: str | Path) -> None:
    """One JSON object per line: diffable, and reviewable question by question."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for g in gold:
            fh.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[GoldQuestion]:
    path = Path(path)
    if not path.is_file():
        return []
    out: list[GoldQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(GoldQuestion(**json.loads(line)))
    return out


def coverage(gold: list[GoldQuestion]) -> dict[str, int]:
    """How much of the set is usable, per metric family.

    Reported next to every result. A mean over 68 of 85 questions is a
    different claim from a mean over all of them, and the difference must not
    be something the reader has to work out.
    """
    return {
        "questions": len(gold),
        "labelled": sum(1 for g in gold if g.is_labelled),
        "human_reviewed": sum(1 for g in gold if g.source == "human"),
        "needs_review": sum(1 for g in gold if g.source == "needs_review"),
        "unlabelled": sum(1 for g in gold if not g.is_labelled),
        "proposed_unconfirmed": sum(
            1 for g in gold
            if g.is_labelled and g.label_origin == "answer" and g.source != "human"
        ),
        "labelled_from_question": sum(1 for g in gold if g.label_origin == "question"),
        "labelled_from_answer": sum(1 for g in gold if g.label_origin == "answer"),
        "with_reference_answer": sum(1 for g in gold if g.reference_answer),
        "with_required_aspects": sum(1 for g in gold if g.required_aspects),
    }
