"""Eligibility Assessment Agent.

Two jobs, and they run in opposite directions:

  FORWARD   "Am I eligible for PM-KISAN?"
            -> retrieve that scheme's criteria, assess against what we know,
               ask about the single most decisive missing fact.

  REVERSE   "I'm a farmer below poverty line, what can I get?"
            -> handled by the Discovery agent, which filters on the facets
               this agent helps populate. See agents/discovery.py.

DESIGN FOR A 15-SECOND BUDGET

The obvious build is three LLM calls: extract the criteria, assess each one,
then phrase a question. At ~20 tok/s that is 30-45 seconds.

Instead, ONE call returns criteria, verdicts and the next question together.
The model is already reading the eligibility text to judge it; asking it to
emit the criteria list in the same breath costs a few output tokens, not a
round trip. Slot extraction is a second call, and only when the user actually
volunteered free text.

WHY THE QUESTIONS COME FROM THE RETRIEVED TEXT

A fixed questionnaire would ask every user for age, income, caste, land and
education. Most schemes restrict on one or two axes, so most of those
questions are noise. The criteria that come back from the scheme's own
eligibility section decide what is worth asking -- which is why `needs_slot`
is part of the model's output rather than a lookup on our side.
"""

from __future__ import annotations

import json
import logging
import re

from ..facets import Category, EducationLevel, Gender, Occupation, SpecialStatus, coerce
from ..llm import get_llm
from ..models import (
    CriterionVerdict,
    EligibilityAssessment,
    RetrievedChunk,
    UserProfile,
)

logger = logging.getLogger(__name__)

_JSON = re.compile(r"\{.*\}", re.DOTALL)

SLOT_SYSTEM = """Extract facts about the USER from their message. JSON only.

Use ONLY these values where a list is given:
occupation: {occupations}
category: {categories}
gender: {genders}
education_level: {education_levels}
special_statuses: {special_statuses}

Schema (omit anything the user did not state -- never guess):
{{
  "occupation": null,
  "category": null,
  "gender": null,
  "education_level": null,
  "age": null,
  "annual_income": null,      // rupees per year, as a number
  "landholding_ha": null,     // hectares, as a number
  "state": null,
  "special_statuses": []
}}

Rules:
1. Only what the user stated about THEMSELVES. Ignore scheme names.
2. "below poverty line", "BPL", "poor" -> special_statuses ["below_poverty_line"]
3. "small farmer", "marginal farmer" -> occupation farmer AND
   special_statuses ["small_or_marginal_farmer"]
4. Convert: "2 lakh" -> 200000, "3 acres" -> 1.21 (hectares).
5. JSON only."""

SLOT_USER = """User message: {message}

JSON:"""


ASSESS_SYSTEM = """You assess whether a person qualifies for one government \
scheme, using ONLY the eligibility text provided. JSON only.

Schema:
{
  "criteria": [
    {
      "criterion": "short restatement of one eligibility rule",
      "verdict": "met" | "not_met" | "unknown",
      "reason": "one short clause citing the rule or the user's fact",
      "needs_slot": null
    }
  ],
  "overall": "likely_eligible" | "likely_ineligible" | "insufficient_info",
  "questions": []
}

Rules:
1. List every distinct rule in the eligibility text as its own criterion.
2. "met" only when the user's stated facts satisfy it. "not_met" only when
   they clearly violate it. Otherwise "unknown" -- never assume.
3. For "unknown", set needs_slot to the ONE field that would settle it:
   occupation, category, gender, education_level, age, annual_income,
   landholding_ha, state, or special_statuses.
4. overall is "likely_ineligible" if ANY criterion is not_met.
   It is "likely_eligible" only if every criterion is met.
   Otherwise "insufficient_info".
5. List EVERY distinct rule you can find. Do not stop at the first one.
6. questions: one short question for EACH criterion whose verdict is
   "unknown", in the same order. Ask nothing for criteria already met or not
   met. Empty list if nothing is unknown.
   Each question MUST be answerable with yes or no, because the user answers
   by pressing a Yes or No button and cannot type anything else. Turn an open
   question into a closed one: not "What is your age?" but "Are you between
   18 and 40 years old?"; not "What is your income?" but "Is your annual
   family income below Rs 2 lakh?". Take the threshold from the eligibility
   text so the answer settles the criterion outright.
7. ANSWERED QUESTIONS ARE FACTS. Anything under "THE USER HAS ALREADY
   ANSWERED" was confirmed by the user directly. Treat it as given: set that
   criterion to "met" or "not_met" accordingly and never ask about it again.
   A criterion covered by an answer is NOT "unknown".
8. Never invent rules absent from the text. JSON only."""

ASSESS_USER = """SCHEME: {scheme_name}

ELIGIBILITY TEXT:
{criteria_text}

WHAT WE KNOW ABOUT THE USER:
{profile}
{answers}
JSON:"""


# --------------------------------------------------------------------------
# Deterministic slot extraction — the fallback that keeps this working when
# a 4B returns unusable JSON, which it does regularly.
# --------------------------------------------------------------------------

_AGE = re.compile(r"\b(?:i\s*(?:a|')?m|aged?|age\s*(?:is)?)\s*(\d{1,2})\b", re.I)
_AGE_YEARS = re.compile(r"\b(\d{1,2})\s*(?:years?\s*old|yrs?\s*old|year\s*old)\b", re.I)
_INCOME = re.compile(
    r"\b(?:income|earn(?:ing)?s?|salary)\D{0,20}?([\d,.]+)\s*(lakh|crore|thousand|k)?\b", re.I
)
_LAND = re.compile(r"\b([\d.]+)\s*(hectare|hectares|ha|acre|acres)\b", re.I)

_KEYWORDS: list[tuple[str, str, re.Pattern]] = [
    ("special_statuses", SpecialStatus.BPL.value,
     re.compile(r"\b(below poverty line|bpl|very poor|poverty)\b", re.I)),
    ("special_statuses", SpecialStatus.SMALL_MARGINAL_FARMER.value,
     re.compile(r"\b(small farmer|marginal farmer|small or marginal)\b", re.I)),
    ("special_statuses", SpecialStatus.DISABLED.value,
     re.compile(r"\b(disabled|disability|divyang|handicapped|pwd)\b", re.I)),
    ("special_statuses", SpecialStatus.WIDOW.value, re.compile(r"\bwidow\b", re.I)),
    ("special_statuses", SpecialStatus.RURAL.value,
     re.compile(r"\b(rural|village|gram)\b", re.I)),
    ("occupation", Occupation.FARMER.value,
     re.compile(r"\b(farmer|farming|cultivator|agricultur)\w*\b", re.I)),
    ("occupation", Occupation.STUDENT.value,
     re.compile(r"\b(student|studying|school|college)\b", re.I)),
    ("occupation", Occupation.SOFTWARE_PROFESSIONAL.value,
     re.compile(r"\b(software|developer|programmer|it professional)\b", re.I)),
    ("occupation", Occupation.STREET_VENDOR.value,
     re.compile(r"\b(street vendor|hawker|thela)\b", re.I)),
    ("occupation", Occupation.ARTISAN.value,
     re.compile(r"\b(artisan|craftsman|handicraft)\b", re.I)),
    ("occupation", Occupation.WEAVER.value, re.compile(r"\b(weaver|handloom)\b", re.I)),
    ("occupation", Occupation.UNEMPLOYED.value,
     re.compile(r"\b(unemployed|jobless|no job)\b", re.I)),
    ("category", Category.SC.value, re.compile(r"\b(sc|scheduled caste|dalit)\b", re.I)),
    ("category", Category.ST.value, re.compile(r"\b(st|scheduled tribe|adivasi|tribal)\b", re.I)),
    ("category", Category.OBC.value, re.compile(r"\b(obc|other backward)\b", re.I)),
    ("gender", Gender.FEMALE.value, re.compile(r"\b(woman|women|female|girl|she|her)\b", re.I)),
    ("gender", Gender.MALE.value, re.compile(r"\b(\bman\b|male|boy)\b", re.I)),
]

_MULTIPLIERS = {"lakh": 100_000, "crore": 10_000_000, "thousand": 1_000, "k": 1_000}

# Reuse the router's closed state list rather than a second copy: a state name
# matched here must be the same string the payload filter compares against.
from .understand import _STATE_WORDS  # noqa: E402


def heuristic_slots(message: str) -> dict:
    """Slots readable without an LLM. Cheap, and it never fails to parse."""
    found: dict = {}

    if m := (_AGE.search(message) or _AGE_YEARS.search(message)):
        age = int(m.group(1))
        if 1 <= age <= 120:
            found["age"] = age

    if m := _INCOME.search(message):
        try:
            value = float(m.group(1).replace(",", ""))
            value *= _MULTIPLIERS.get((m.group(2) or "").lower(), 1)
            if value > 0:
                found["annual_income"] = value
            # A bare "income 2" almost certainly meant lakh.
            if not m.group(2) and value < 1000:
                found["annual_income"] = value * 100_000
        except ValueError:
            pass

    if m := _LAND.search(message):
        try:
            value = float(m.group(1))
            if m.group(2).lower().startswith("acre"):
                value *= 0.405
            found["landholding_ha"] = round(value, 3)
        except ValueError:
            pass

    if m := _STATE_WORDS.search(message):
        found["state"] = m.group(0).title()

    statuses: list[str] = []
    for field, value, pattern in _KEYWORDS:
        if not pattern.search(message):
            continue
        if field == "special_statuses":
            if value not in statuses:
                statuses.append(value)
        elif field not in found:
            found[field] = value
    if statuses:
        found["special_statuses"] = statuses

    return found


#: Yes/No answers that mean "yes". Anything else is treated as not-yes, which
#: is the safe direction: a missed slot costs a question, a wrong one costs a
#: wrong verdict.
_AFFIRMATIVE = {"yes", "y", "true", "yeah", "yep", "haan", "ha"}


def slots_from_yes(question: str, answer: str) -> dict:
    """Profile slots implied by answering YES to a question.

    "Are you a farmer?" -> yes  gives occupation=farmer for free, with no LLM
    call and nothing to misparse. The point is that the UI's Yes/No buttons
    carry real profile information, not just a criterion verdict, so a later
    Discovery query benefits from an interview the user already sat through.

    NUMERIC SLOTS ARE DELIBERATELY EXCLUDED. The numbers in these questions
    are the scheme's thresholds, not the user's facts: "Do you hold less than
    2 hectares?" -> yes would otherwise record landholding_ha=2, which is the
    limit the user is UNDER. Only the categorical keyword patterns are safe to
    read this way.
    """
    if answer.strip().lower().rstrip(".!") not in _AFFIRMATIVE:
        return {}

    found: dict = {}
    statuses: list[str] = []
    for field, value, pattern in _KEYWORDS:
        if not pattern.search(question):
            continue
        if field == "special_statuses":
            if value not in statuses:
                statuses.append(value)
        elif field not in found:
            found[field] = value
    if statuses:
        found["special_statuses"] = statuses
    return found


def apply_slots(profile: UserProfile, data: dict) -> UserProfile:
    """Merge a slot dict into a profile without consulting an LLM.

    Existing values win: the interview should never overwrite something the
    user stated outright with something inferred from a yes/no.
    """
    merged = profile.model_copy(deep=True)
    for field in ("occupation", "category", "gender", "education_level", "state"):
        value = data.get(field)
        if value and getattr(merged, field) is None:
            merged.__setattr__(field, str(value))
    value = data.get("age")
    if isinstance(value, (int, float)) and merged.age is None:
        merged.age = int(value)
    for field in ("annual_income", "landholding_ha"):
        value = data.get(field)
        if isinstance(value, (int, float)) and getattr(merged, field) is None:
            merged.__setattr__(field, float(value))
    for status in coerce(data.get("special_statuses"), SpecialStatus):
        if status not in merged.special_statuses:
            merged.special_statuses.append(status)
    return merged


def record_answers(
    profile: UserProfile, answers: list[dict[str, str]]
) -> UserProfile:
    """Fold client-paired Q/A into the profile. No LLM, no parsing.

    THIS IS THE FIX for the failure that made free-text interviews unusable.
    Asked six questions, the user replied "1)yes,2)yes,3)no, 4)yes, 5)yes,
    6)yes" -- one string the model had to re-associate with its own criteria.
    It could not, re-asked with an empty question list, and then declared
    "Not enough information to decide" about a scheme it had just been told
    six facts about.

    Pairing in the UI removes the step that failed. Each question travels back
    attached to its own answer, and the pairing is done by a button press
    rather than inferred from a numbered list.
    """
    merged = profile
    for item in answers:
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue

        # Two directions, and the distinction matters. A YES makes the
        # QUESTION's subject true of the user, but only categorically --
        # numbers in a question are the scheme's thresholds. A free-text
        # ANSWER is the user's own statement, so numbers in it are facts.
        merged = apply_slots(merged, slots_from_yes(question, answer))
        if answer.strip().lower().rstrip(".!") not in _AFFIRMATIVE:
            merged = apply_slots(merged, heuristic_slots(answer))

        pair = f"Q: {question} A: {answer}"
        if pair not in merged.notes:
            merged.notes.append(pair[:300])
    return merged


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if m := _JSON.search(text):
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def update_profile(profile: UserProfile, message: str, use_llm: bool = True) -> UserProfile:
    """Merge facts from `message` into `profile`.

    Heuristics run first and always; the LLM then fills gaps they missed. The
    order matters: a regex that found "age 34" is more trustworthy than a 4B
    paraphrasing it, and this way an unparseable LLM response costs nothing.
    """
    merged = apply_slots(profile, heuristic_slots(message))

    if use_llm:
        try:
            raw = get_llm().complete(
                SLOT_SYSTEM.format(
                    occupations=", ".join(m.value for m in Occupation),
                    categories=", ".join(m.value for m in Category),
                    genders=", ".join(m.value for m in Gender),
                    education_levels=", ".join(m.value for m in EducationLevel),
                    special_statuses=", ".join(m.value for m in SpecialStatus),
                ),
                SLOT_USER.format(message=message.strip()),
                temperature=0.0,
                max_tokens=256,
            name="eligibility.slots",
                json_mode=True,
            )
            data = _extract_json(raw)
            if isinstance(data, dict):
                # Normalise through the vocabulary before merging.
                for field, enum_cls in (
                    ("occupation", Occupation), ("category", Category),
                    ("gender", Gender), ("education_level", EducationLevel),
                ):
                    if data.get(field):
                        mapped = coerce([data[field]], enum_cls)
                        data[field] = mapped[0] if mapped else None
                merged = apply_slots(merged, data)
            else:
                logger.info("Slot extraction returned unusable JSON; heuristics stand")
        except Exception as exc:
            logger.warning("Slot extraction failed (%s); heuristics stand", exc)

    note = message.strip()
    if note and note not in merged.notes:
        merged.notes.append(note[:300])
    return merged


def assess(
    scheme_name: str, chunks: list[RetrievedChunk], profile: UserProfile
) -> EligibilityAssessment:
    """Judge the profile against one scheme's retrieved eligibility text."""
    criteria_text = "\n\n".join(
        f"[{rc.chunk.section_title}] {rc.chunk.text}" for rc in chunks
    )[:4000]

    if not criteria_text.strip():
        return EligibilityAssessment(
            scheme_name=scheme_name,
            overall="insufficient_info",
            next_question=None,
        )

    # Replies to the agent's own questions live in notes as "Q: ... A: ...".
    # They carry facts the slot schema cannot hold -- "Is the institution
    # AICTE approved? -> yes" is not a profile field, but it settles a
    # criterion -- so they go to the model verbatim.
    #
    # Rendered one per line as question -> ANSWER rather than passed as the
    # raw note string. The pairing is the whole value here: a model given
    # "1)yes,2)yes,3)no" against six criteria cannot reconstruct which answer
    # belonged to which rule, and the observed result was an empty re-ask
    # followed by "not enough information" about a scheme it had just been
    # told six facts about.
    answered = [
        n[3:].split(" A: ", 1) for n in profile.notes
        if n.startswith("Q: ") and " A: " in n
    ]
    answers_block = ""
    if answered:
        lines = "\n".join(
            f'- "{q.strip()}" -> {a.strip().upper()}' for q, a in answered[-20:]
        )
        answers_block = (
            "\nTHE USER HAS ALREADY ANSWERED THESE -- treat each as an "
            "established fact and do not ask again:\n" + lines + "\n"
        )

    try:
        raw = get_llm().complete(
            ASSESS_SYSTEM,
            ASSESS_USER.format(
                scheme_name=scheme_name,
                criteria_text=criteria_text,
                profile=profile.describe(),
                answers=answers_block,
            ),
            temperature=0.0,
            # Sized for the OUTPUT: eight criteria, each with a restatement, a
            # verdict and a reason, plus a question per unresolved one, runs to
            # well over a thousand tokens of JSON. This was 700, which fit the
            # first turn and truncated on the second -- where the model also
            # has to reconcile the user's answers against its own criteria.
            # Deliberation is paid for separately by
            # llm_thinking_headroom_tokens, so this is output budget only.
            max_tokens=1800,
            name="eligibility.assess",
            json_mode=True,
        )
    except Exception as exc:
        # failed=True, NOT a bare insufficient_info: the two are different
        # facts and must not read the same to the user. "Not enough
        # information to decide" after they answered every question is a lie
        # about THEM; the truth is that our call failed.
        logger.warning("Eligibility assessment call failed: %s", exc, exc_info=True)
        return EligibilityAssessment(
            scheme_name=scheme_name, overall="insufficient_info", failed=True
        )

    data = _extract_json(raw)
    if not isinstance(data, dict):
        # Log the actual output. Without this the only evidence of the most
        # likely cause -- a truncated response, because thinking ate the token
        # budget -- is discarded, which turns a diagnosis into a guess.
        logger.warning(
            "Assessment returned unparseable JSON (%d chars). This is usually "
            "truncation: raise LLM_THINKING_HEADROOM_TOKENS. Raw output:\n%s",
            len(raw or ""), (raw or "")[:600],
        )
        return EligibilityAssessment(
            scheme_name=scheme_name, overall="insufficient_info", failed=True
        )

    verdicts: list[CriterionVerdict] = []
    for item in data.get("criteria") or []:
        if not isinstance(item, dict) or not item.get("criterion"):
            continue
        verdict = str(item.get("verdict", "unknown")).strip().lower()
        if verdict not in {"met", "not_met", "unknown"}:
            verdict = "unknown"
        slot = item.get("needs_slot")
        verdicts.append(
            CriterionVerdict(
                criterion=str(item["criterion"])[:300],
                verdict=verdict,
                reason=str(item.get("reason") or "")[:300],
                needs_slot=str(slot) if slot else None,
            )
        )

    # Derive `overall` from the verdicts rather than trusting the model's
    # summary: a 4B frequently says "likely_eligible" while listing an unmet
    # criterion, and the rule is simple enough to enforce ourselves.
    if any(v.verdict == "not_met" for v in verdicts):
        overall = "likely_ineligible"
    elif verdicts and all(v.verdict == "met" for v in verdicts):
        overall = "likely_eligible"
    elif verdicts:
        overall = "insufficient_info"
    else:
        overall = str(data.get("overall") or "insufficient_info")

    known = set(profile.known_slots())
    missing = [
        v.needs_slot for v in verdicts
        if v.verdict == "unknown" and v.needs_slot and v.needs_slot not in known
    ]
    missing = list(dict.fromkeys(missing))

    raw_questions = data.get("questions")
    if isinstance(raw_questions, str):
        raw_questions = [raw_questions]
    questions = [
        str(q).strip() for q in (raw_questions or []) if str(q).strip()
    ][:6]

    # Never repeat something already answered: a 4B that cannot map an answer
    # onto its criterion will otherwise ask the same thing forever.
    already = {
        n.split(" A: ")[0][3:].strip().lower()
        for n in profile.notes if n.startswith("Q: ") and " A: " in n
    }
    questions = [q for q in questions if q.strip().lower() not in already]

    if overall != "insufficient_info":
        questions = []  # the verdict is settled; asking more is noise

    # The model sometimes marks criteria unknown yet returns no questions.
    # Fall back to asking about the slots we derived ourselves.
    if overall == "insufficient_info" and not questions and missing:
        pretty = {
            "occupation": "What is your occupation?",
            "category": "Which social category do you belong to (SC/ST/OBC/General)?",
            "gender": "What is your gender?",
            "education_level": "What is your highest level of education?",
            "age": "How old are you?",
            "annual_income": "What is your annual family income, in rupees?",
            "landholding_ha": "How much land do you hold, in acres or hectares?",
            "state": "Which state or union territory do you live in?",
            "special_statuses": "Do any of these apply: below poverty line, "
                                "disability, widow, ex-serviceman family?",
        }
        questions = [pretty.get(slot, f"What is your {slot}?") for slot in missing[:6]]

    return EligibilityAssessment(
        scheme_name=scheme_name,
        overall=overall,
        criteria=verdicts,
        missing_slots=missing,
        questions=questions,
    )


def _basis(profile: UserProfile) -> str:
    """What the verdict rests on: the profile AND the user's own answers.

    Showing only the profile printed "Based on: (nothing known yet)" under a
    verdict reached entirely from an answered question -- which reads as a
    bug, because the system plainly did know something.
    """
    parts = []
    described = profile.describe()
    if described != "(nothing known yet)":
        parts.append(described)
    answers = [n[3:] for n in profile.notes if n.startswith("Q: ")]
    if answers:
        parts.append("your answers: " + "; ".join(answers[-4:]))
    return " | ".join(parts) if parts else "(nothing known yet)"


def render_questions(assessment: EligibilityAssessment, profile: UserProfile) -> str:
    """Show the criteria being checked, then ask everything outstanding at once."""
    lines = [f"Checking your eligibility for {assessment.scheme_name}.", ""]

    if assessment.criteria:
        symbol = {"met": "[yes]", "not_met": "[no] ", "unknown": "[?]  "}
        lines.append("Criteria in the scheme document:")
        for v in assessment.criteria:
            lines.append(f"  {symbol.get(v.verdict, '[?]  ')} {v.criterion}")
        lines.append("")

    if len(assessment.questions) == 1:
        lines.append(assessment.questions[0] + "  (yes / no)")
    else:
        lines.append("To finish the check, please answer yes or no to each:")
        for i, q in enumerate(assessment.questions, 1):
            lines.append(f"  {i}. {q}")
        lines.append("")
        # The web UI never shows this line -- it reads assessment.questions
        # and walks them one at a time with buttons. This is the terminal
        # path, where a numbered reply is all the user can give.
        lines.append('Reply like "1 yes, 2 no, 3 yes".')

    return "\n".join(lines)


def render(assessment: EligibilityAssessment, profile: UserProfile) -> str:
    """Plain-text verdict."""
    # A failed call is reported as a failure, not as a verdict. Telling someone
    # who just answered five questions that there is "not enough information"
    # blames them for our error and invites them to answer again, which will
    # fail the same way.
    if assessment.failed:
        answered = len([n for n in profile.notes if n.startswith("Q: ")])
        lines = [
            f"I could not complete the eligibility check for "
            f"{assessment.scheme_name}.",
            "",
            "This is a failure on my side, not missing information from you"
            + (f" — your {answered} answer(s) were recorded." if answered else "."),
            "",
            "The server log says which call failed. The usual cause is the "
            "model running out of token budget mid-answer; raise "
            "LLM_THINKING_HEADROOM_TOKENS in .env if the log shows truncated "
            "or unparseable output.",
        ]
        return "\n".join(lines)

    headline = {
        "likely_eligible": "You appear to be ELIGIBLE",
        "likely_ineligible": "You appear to be NOT eligible",
        "insufficient_info": "Not enough information to decide",
    }.get(assessment.overall, assessment.overall)

    lines = [f"{headline} for {assessment.scheme_name}.", ""]
    symbol = {"met": "[yes]", "not_met": "[no] ", "unknown": "[?]  "}
    for v in assessment.criteria:
        lines.append(f"  {symbol.get(v.verdict, '[?]  ')} {v.criterion}")
        if v.reason:
            lines.append(f"         {v.reason}")
    if assessment.criteria:
        lines.append("")

    unknown = [v for v in assessment.criteria if v.verdict == "unknown"]
    if unknown and assessment.overall == "insufficient_info":
        answered = len([n for n in profile.notes if n.startswith("Q: ")])
        if answered:
            # Say plainly that the answers arrived and the model still did not
            # apply them. Otherwise this reads as "you did not tell me enough"
            # to someone who answered everything they were asked.
            lines.append(
                f"You answered {answered} question(s), and they were recorded, "
                f"but {len(unknown)} criterion(s) above could still not be "
                f"resolved against them. That is a limitation of the "
                f"assessment, not something missing from your answers."
            )
        else:
            lines.append(
                f"{len(unknown)} criterion(s) could not be checked from what I "
                "know. The verdict above is therefore incomplete."
            )
        lines.append("")

    lines.append(f"Based on: {_basis(profile)}")
    if assessment.overall != "insufficient_info":
        lines.append("")
        lines.append(
            "This is an assessment against the published criteria, not a "
            "decision. The implementing authority decides."
        )
    return "\n".join(lines)
