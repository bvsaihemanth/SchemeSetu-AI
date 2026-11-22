"""Scheme entity resolution — the fix for the PM-KISAN failure.

THE PROBLEM THIS SOLVES

This corpus is not 12,811 passages of prose; it is 725 *named entities*, and
users address them by name. "Who can apply for PM-KISAN?" is a question about
one specific document. But the document is titled "Pradhan Mantri Kisan Samman
Nidhi" and the token "PM-KISAN" appears nowhere in its title — only in its
source URL (/schemes/pm-kisan) and, by chance, in one FAQ answer.

So dense retrieval cannot match the name, and falls back on what it CAN
match: question *phrasing*. "Who can apply for PM-KISAN?" scores highest
against "Who can apply for the scholarships?" from an unrelated scholarship
scheme. No amount of embedding tuning fixes that, because the information
needed to disambiguate is not in the chunk text at all.

The fix is to resolve the entity BEFORE searching, using the one place the
acronym is recorded — the URL slug — plus generated aliases. Once the scheme
is identified, retrieval runs scoped to that scheme's chunks, where picking
the right section is easy.

WHY NOT JUST ASK THE LLM

The router could extract "PM-KISAN", but it cannot map it to "Pradhan Mantri
Kisan Samman Nidhi" without the corpus in front of it — a 4B model will
hallucinate a plausible-looking scheme name. Resolution has to be a lookup
against real data. This module is that lookup: deterministic, instant, and
auditable.

Stdlib only, by design: 725 entries is small enough that pure Python is
faster than the import cost of a fuzzy-matching library.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

# Aliases this generic are not discriminative: they appear in hundreds of
# scheme names and would match almost any query.
_STOPALIAS = {
    "scheme", "schemes", "yojana", "yojna", "mission", "programme", "program",
    "fund", "india", "national", "pm", "pmy", "fellowship", "scholarship",
    "award", "awards", "internship", "grant", "loan", "pradhan", "mantri",
    "prime", "minister", "ministers", "central", "sector", "sponsored",
    "development", "assistance", "support", "welfare", "ns", "the", "for",
}

# Dropped when forming initials, so "Scheme for Promotion of X" -> "spx".
_INITIAL_SKIP = {"for", "of", "and", "the", "to", "in", "on", "a", "an", "under"}

_PAREN = re.compile(r"\(([^)]{2,24})\)")
_PM_PREFIX = re.compile(r"^(?:pradhan\s+mantri|prime\s+ministers?|pm)\b", re.I)

# Minimum alias length, by provenance. Authoritative slugs and declared
# acronyms are allowed to be short because users really do type "APY", "KCC",
# "PPF" and "NSC"; our own generated contractions must be longer, since a
# 3-char guess collides with everything.
MIN_ALIAS_LEN = 4
MIN_ALIAS_LEN_AUTHORITATIVE = 3

# A MULTI-TOKEN query n-gram must be at least this long to count as an alias.
#
# The joiner collapses separators, which is what makes "PM KISAN", "pm-kisan"
# and "pmkisan" one key. But it also collapses ordinary words: "am i" becomes
# "ami", which is the authoritative slug of "Agricultural Marketing
# Infrastructure". So "am i eligible for <coffee scheme>" resolved, with
# `exact` confidence, to a marketing-infrastructure scheme.
#
# Real multi-token aliases are long ("pmkisan" is 7). Three-letter acronyms
# are typed as ONE token. Requiring 5 characters across 2+ tokens keeps the
# former and rejects the latter.
MIN_MULTITOKEN_ALIAS_LEN = 5

# Query words that must never form an alias on their own. 54 schemes in this
# corpus have three-character slugs -- "ami", "ati", "els", "eps", "big",
# "cas" -- and English is full of three-letter fragments that collide.
_QUERY_NOISE = {
    "am", "is", "are", "was", "be", "i", "me", "my", "we", "us", "you", "your",
    "it", "its", "he", "she", "they", "them", "this", "that", "these", "those",
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "if", "do", "does", "did", "can", "could", "will", "would", "should",
    "what", "which", "who", "whom", "how", "when", "where", "why", "any",
    "all", "get", "got", "have", "has", "had", "want", "need", "eligible",
    "apply", "scheme", "schemes", "about", "tell", "give", "show", "there",
    "from", "with", "under", "not", "no", "yes", "please", "help",
}
MAX_DOCS_PER_ALIAS = 3  # above this an alias is ambiguous, not identifying

# Alias provenance decides who wins a collision, and this matters concretely:
# "Pradhan Mantri Kisan Samman Nidhi" has the SLUG pm-kisan, while "Pradhan
# Mantri Kisan Maandhan Yojana" merely *generates* the contraction "pmkisan"
# from its first content word. Both claim the alias; only one is authoritative.
# Without this weighting the resolver reproduces the very ambiguity it exists
# to remove.
ALIAS_WEIGHTS = {
    "slug": 1.00,   # assigned by myScheme in the source URL -- ground truth
    "paren": 0.95,  # acronym the title itself declares, e.g. "(PMFBY)"
    "name": 0.92,   # the full canonical name
    "gen": 0.55,    # our own heuristic contraction or initials -- a guess
}


def strip_accents(text: str) -> str:
    # Also folds the curly quotes and en-dashes these filenames are full of.
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", strip_accents(text).lower())


def tight(text: str) -> str:
    """Separator-insensitive key: 'PM-KISAN', 'PM KISAN' and 'pmkisan' all
    collapse to 'pmkisan', which is what makes slug lookup work."""
    return "".join(tokens(text))


@dataclass
class SchemeEntry:
    doc_id: str
    doc_name: str
    scheme_name: str
    slug: str | None = None
    source_url: str | None = None
    # [alias, kind] pairs; kind is a key of ALIAS_WEIGHTS
    aliases: list[list[str]] = field(default_factory=list)

    @property
    def name_tight(self) -> str:
        return tight(self.scheme_name)


@dataclass
class SchemeMatch:
    doc_id: str
    scheme_name: str
    score: float
    matched_on: str  # the alias or name that matched
    confidence: str  # "exact" | "strong" | "weak"
    tokens_matched: int = 0  # query tokens this match spans (coverage)
    span: tuple[int, int] = (0, 0)  # [start, end) token offsets in the query


def generate_aliases(scheme_name: str, slug: str | None = None) -> list[list[str]]:
    """Every string a user might plausibly type for this scheme.

    Derived from real naming habits in this corpus:
      slug                 pm-kisan            -> pmkisan   (the key one)
      parenthetical        ...Yojna (PMFBY)    -> pmfby
      word initials        Atal Pension Yojana -> apy
      PM-prefix contraction
                           Pradhan Mantri Kisan Samman Nidhi -> pmkisan
      full name            pradhanmantrikisansammannidhi
    """
    out: list[tuple[str, str]] = []
    name_tokens = tokens(scheme_name)

    # 1. full canonical name
    if name_tokens:
        out.append(("".join(name_tokens), "name"))

    # 2. slug from the source URL — authoritative, since myScheme assigns it
    if slug:
        out.append((tight(slug), "slug"))

    # 3. parenthetical acronyms, e.g. "(PM-DAKSH)", "(PMFBY)"
    for inner in _PAREN.findall(scheme_name):
        cand = tight(inner)
        # Keep genuine acronyms/short codes, not parenthetical prose.
        if cand and len(cand) <= 14 and len(tokens(inner)) <= 3:
            out.append((cand, "paren"))

    # 4. initials of significant words: "Atal Pension Yojana" -> "apy"
    significant = [t for t in name_tokens if t not in _INITIAL_SKIP]
    if len(significant) >= 2:
        out.append(("".join(t[0] for t in significant), "gen")) 

    # 5. "Pradhan Mantri X ..." / "Prime Minister's X ..." -> "pmx"
    #    This is how these schemes are colloquially written, and it is what
    #    myScheme itself does for the slug (pm-kisan, pm-svanidhi).
    if _PM_PREFIX.match(" ".join(name_tokens)):
        rest = [t for t in name_tokens if t not in {"pradhan", "mantri", "prime", "minister", "ministers", "pm"}]
        rest = [t for t in rest if t not in _INITIAL_SKIP]
        if rest:
            out.append(("pm" + rest[0], "gen"))
            if len(rest) >= 2:
                out.append(("pm" + rest[0] + rest[1], "gen"))
                out.append(("pm" + "".join(t[0] for t in rest), "gen"))

    # Dedupe keeping the strongest provenance for each alias, drop generic
    # and too-short forms.
    best: dict[str, str] = {}
    for alias, kind in out:
        floor = (
            MIN_ALIAS_LEN_AUTHORITATIVE if kind in ("slug", "paren") else MIN_ALIAS_LEN
        )
        if len(alias) < floor or alias in _STOPALIAS:
            continue
        if alias not in best or ALIAS_WEIGHTS[kind] > ALIAS_WEIGHTS[best[alias]]:
            best[alias] = kind
    return [[a, k] for a, k in best.items()]


class SchemeIndex:
    """Alias -> scheme lookup, with a fuzzy fallback."""

    def __init__(self, entries: Sequence[SchemeEntry]) -> None:
        self.entries: dict[str, SchemeEntry] = {e.doc_id: e for e in entries}
        # alias -> [(doc_id, kind)]
        self.alias_map: dict[str, list[tuple[str, str]]] = {}
        for e in entries:
            for alias, kind in e.aliases:
                self.alias_map.setdefault(alias, []).append((e.doc_id, kind))

        # An alias shared by too many schemes identifies nothing. Keep it for
        # ranking but never let it drive a hard filter.
        self.ambiguous = {
            a for a, ids in self.alias_map.items() if len(ids) > MAX_DOCS_PER_ALIAS
        }
        if self.ambiguous:
            logger.debug("%d ambiguous aliases dropped from exact matching", len(self.ambiguous))

        # Derived, not guessed. Scheme names in this corpus run to 15+ tokens
        # ("Coffee Development Programme in North Eastern Region Consolidation
        # of Coffee" is 10), and a fixed cap of 8 meant their full-name alias
        # could never be formed from the query -- so the only thing that COULD
        # match was a short spurious n-gram elsewhere in the sentence.
        self.max_alias_tokens = max(
            (len(tokens(e.scheme_name)) for e in entries), default=8
        )
        self.max_alias_tokens = max(4, min(self.max_alias_tokens, 24))

    # ---------------- persistence ----------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"version": 1, "entries": [asdict(e) for e in self.entries.values()]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SchemeIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([SchemeEntry(**e) for e in data["entries"]])

    @classmethod
    def build(cls, docs: Iterable) -> "SchemeIndex":
        """From parsed SchemeDocs (anything with the right attributes)."""
        entries = [
            SchemeEntry(
                doc_id=d.doc_id,
                doc_name=d.doc_name,
                scheme_name=d.scheme_name,
                slug=getattr(d, "scheme_slug", None),
                source_url=getattr(d, "source_url", None),
                aliases=generate_aliases(d.scheme_name, getattr(d, "scheme_slug", None)),
            )
            for d in docs
        ]
        return cls(entries)

    # ---------------- resolution ----------------

    def resolve(self, query: str, limit: int = 4) -> list[SchemeMatch]:
        """Identify which scheme(s) a query is about.

        Exact alias hits win outright. Only if there are none do we fall back
        to fuzzy name similarity, which is deliberately conservative: a wrong
        hard filter hides the right answer entirely, so an uncertain match is
        reported as "weak" and the caller boosts rather than filters.
        """
        q_tokens = tokens(query)
        if not q_tokens:
            return []
        # Original casing, to tell an acronym ("APY") from a word ("big").
        raw_tokens = re.findall(r"[A-Za-z0-9]+", strip_accents(query))

        exact: dict[str, SchemeMatch] = {}

        # Longest n-gram first, so "pm kisan samman nidhi" beats "pm kisan".
        max_n = min(self.max_alias_tokens, len(q_tokens))
        for n in range(max_n, 0, -1):
            for i in range(len(q_tokens) - n + 1):
                window = q_tokens[i : i + n]
                gram = "".join(window)
                if len(gram) < MIN_ALIAS_LEN_AUTHORITATIVE or gram in _STOPALIAS:
                    continue
                if n >= 2 and len(gram) < MIN_MULTITOKEN_ALIAS_LEN:
                    continue
                if all(t in _QUERY_NOISE for t in window):
                    continue
                # A three-letter alias is only credible as an acronym the user
                # typed deliberately: "APY" in caps, or a query short enough
                # that it is plainly the subject ("kcc", "apy eligibility").
                if len(gram) <= 3:
                    original = raw_tokens[i] if i < len(raw_tokens) else ""
                    if not (original.isupper() or len(q_tokens) <= 3):
                        continue
                hits = self.alias_map.get(gram)
                if not hits or gram in self.ambiguous:
                    continue
                for doc_id, kind in hits:
                    entry = self.entries[doc_id]
                    weight = ALIAS_WEIGHTS[kind]
                    score = round(min(1.0, weight * (0.85 + min(0.15, len(gram) / 120))), 4)
                    prev = exact.get(doc_id)
                    # Ranked by query COVERAGE first, provenance second -- see
                    # the sort below for why.
                    if prev is None or (n, score) > (prev.tokens_matched, prev.score):
                        exact[doc_id] = SchemeMatch(
                            doc_id=doc_id,
                            scheme_name=entry.scheme_name,
                            score=score,
                            matched_on=f"{gram} ({kind})",
                            confidence="exact" if kind != "gen" else "weak",
                            tokens_matched=n,
                            span=(i, i + n),
                        )

        if exact:
            # Coverage outranks provenance. "pm kisan maandhan yojana" matches
            # PM-KISAN's authoritative slug on 2 tokens AND PM-Kisan-Maandhan's
            # generated alias on 3 -- and the user plainly means the latter.
            # Spanning more of the query is stronger evidence of intent than a
            # shorter match from a better source, so sort on (coverage, score).
            ranked = sorted(exact.values(), key=lambda m: (-m.tokens_matched, -m.score))

            # Keep matches on DISJOINT spans; drop only those overlapping a
            # better match.
            #
            # Comparing by token count alone was wrong: "compare <scheme A>
            # and <scheme B>" matches A on 6 tokens and B on 5, so a
            # count-based filter silently discarded the second scheme and the
            # Comparison agent received only one. Overlap is the real test --
            # "pmkisan" inside "pm kisan maandhan" is an artefact because it
            # covers the same words, whereas two schemes named side by side
            # occupy different words and are both genuine.
            kept: list[SchemeMatch] = []
            claimed: list[tuple[int, int]] = []
            for match in ranked:
                start, end = match.span
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue
                kept.append(match)
                claimed.append((start, end))
            return kept[:limit]

        # ---- fuzzy fallback ----
        q_tight = "".join(q_tokens)
        q_set = set(q_tokens)
        scored: list[SchemeMatch] = []

        for entry in self.entries.values():
            e_tokens = tokens(entry.scheme_name)
            e_set = set(e_tokens)
            if not e_set:
                continue
            # Token containment: how much of the scheme name the query covers.
            overlap = len(q_set & e_set) / len(e_set)
            # Character-level similarity catches typos and word order.
            ratio = SequenceMatcher(None, q_tight, entry.name_tight).ratio()
            score = 0.6 * overlap + 0.4 * ratio
            if score >= 0.45:
                scored.append(
                    SchemeMatch(
                        doc_id=entry.doc_id,
                        scheme_name=entry.scheme_name,
                        score=round(score, 4),
                        matched_on=entry.scheme_name,
                        confidence="weak",
                        tokens_matched=len(q_set & e_set),
                    )
                )

        scored.sort(key=lambda m: -m.score)
        if not scored:
            return []

        # Promote to "strong" only when clearly ahead of the runner-up:
        # ambiguity here means two schemes with similar names, and filtering
        # to the wrong one of those is the worst available outcome.
        top = scored[0]
        runner_up = scored[1].score if len(scored) > 1 else 0.0
        if top.score >= 0.78 and (top.score - runner_up) >= 0.08:
            top.confidence = "strong"

        return scored[:limit]

    def scheme_names(self, doc_ids: Iterable[str]) -> list[str]:
        return [self.entries[d].scheme_name for d in doc_ids if d in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


_index: SchemeIndex | None = None
INDEX_FILENAME = "scheme_index.json"


def get_scheme_index(path: str | Path | None = None) -> SchemeIndex | None:
    """Loaded once per process. Returns None when the index has not been
    built yet, so retrieval degrades to plain hybrid search rather than
    failing — but logs loudly, because that is the degraded path."""
    global _index
    if _index is not None:
        return _index

    from .config import get_settings

    p = Path(path) if path else Path(get_settings().processed_dir) / INDEX_FILENAME
    if not p.exists():
        logger.warning(
            "Scheme index missing at %s — scheme-name queries will be unreliable. "
            "Run scripts/ingest.py to build it.",
            p,
        )
        return None
    _index = SchemeIndex.load(p)
    logger.info("Loaded scheme index: %d schemes, %d aliases", len(_index), len(_index.alias_map))
    return _index


def reset_scheme_index() -> None:
    global _index
    _index = None
