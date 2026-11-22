"""Controlled vocabularies for eligibility facets.

WHY CLOSED VOCABULARIES

To answer "I'm a farmer below poverty line, what can I get?" we must FILTER
725 schemes by occupation, age, income and category. A filter only works if
both sides use identical strings. If the extractor is free to write "farmer",
"farmers", "agriculturist" and "cultivator" for the same concept, the payload
index is useless and every query silently misses most matches.

So the extraction prompt is given these lists and must map into them. Anything
it cannot map becomes OTHER rather than a new string, which keeps the index
dense and makes gaps visible instead of invisible.

WHY SENTINELS INSTEAD OF NULLS

Numeric eligibility bounds are often absent ("no age limit"). Qdrant range
filters cannot express "this field is null OR within range" in one clause, so
an unconstrained bound is stored as a sentinel at the edge of the domain:

    age_min=0, age_max=200, income_max=1e12, land_max_ha=1e6

A user's age then matches with two plain range conditions, and schemes with no
age rule match everyone automatically. No null handling, no `should` clauses,
no post-filtering.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------- sentinels
AGE_MIN_UNSET = 0
AGE_MAX_UNSET = 200
INCOME_MAX_UNSET = 1_000_000_000_000.0  # 1e12 INR/year
LAND_MAX_UNSET = 1_000_000.0  # hectares

# "any" is stored alongside real values when a scheme places no restriction on
# that axis, so MatchAny([user_value, "any"]) matches unrestricted schemes too.
ANY = "any"


class Occupation(str, Enum):
    FARMER = "farmer"
    AGRI_LABOURER = "agricultural_labourer"
    FISHERMAN = "fisherman"
    LIVESTOCK_FARMER = "livestock_farmer"
    ARTISAN = "artisan"
    WEAVER = "weaver"
    HANDICRAFT_WORKER = "handicraft_worker"
    STREET_VENDOR = "street_vendor"
    MANUAL_SCAVENGER = "manual_scavenger"
    SANITATION_WORKER = "sanitation_worker"
    CONSTRUCTION_WORKER = "construction_worker"
    PLANTATION_WORKER = "plantation_worker"
    MINE_WORKER = "mine_worker"
    STUDENT = "student"
    RESEARCHER = "researcher"
    TEACHER = "teacher"
    SCIENTIST = "scientist"
    DOCTOR = "doctor"
    ENGINEER = "engineer"
    SOFTWARE_PROFESSIONAL = "software_professional"
    ENTREPRENEUR = "entrepreneur"
    MSME_OWNER = "msme_owner"
    EXPORTER = "exporter"
    TRADER = "trader"
    SHOPKEEPER = "shopkeeper"
    SELF_EMPLOYED = "self_employed"
    SALARIED = "salaried"
    GOVT_EMPLOYEE = "government_employee"
    ARMED_FORCES = "armed_forces"
    POLICE = "police"
    EX_SERVICEMAN = "ex_serviceman"
    ARTIST = "artist"
    SPORTSPERSON = "sportsperson"
    JOURNALIST = "journalist"
    LAWYER = "lawyer"
    HOMEMAKER = "homemaker"
    UNEMPLOYED = "unemployed"
    INSTITUTION = "institution"  # scheme's beneficiary is an org, not a person
    OTHER = "other"


class Category(str, Enum):
    SC = "sc"
    ST = "st"
    OBC = "obc"
    EBC = "ebc"
    GENERAL = "general"
    MINORITY = "minority"
    DNT = "denotified_tribe"
    PVTG = "particularly_vulnerable_tribal_group"
    OTHER = "other"


class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    TRANSGENDER = "transgender"


class EducationLevel(str, Enum):
    NONE = "none"
    PRIMARY = "primary"
    CLASS_8 = "class_8"
    CLASS_10 = "class_10"
    CLASS_12 = "class_12"
    ITI_DIPLOMA = "iti_diploma"
    GRADUATE = "graduate"
    POSTGRADUATE = "postgraduate"
    MPHIL = "mphil"
    PHD = "phd"
    POSTDOC = "postdoc"


class SpecialStatus(str, Enum):
    """Flags that gate a lot of this corpus and are asked about directly."""

    BPL = "below_poverty_line"
    DISABLED = "person_with_disability"
    WIDOW = "widow"
    SENIOR_CITIZEN = "senior_citizen"
    PREGNANT_LACTATING = "pregnant_or_lactating"
    ORPHAN = "orphan"
    SINGLE_GIRL_CHILD = "single_girl_child"
    TRANSGENDER = "transgender"
    EX_SERVICEMAN_FAMILY = "ex_serviceman_family"
    MIGRANT_WORKER = "migrant_worker"
    HOMELESS = "homeless"
    SMALL_MARGINAL_FARMER = "small_or_marginal_farmer"
    FIRST_GENERATION_LEARNER = "first_generation_learner"
    RURAL = "rural_resident"
    URBAN = "urban_resident"
    NORTH_EAST = "north_east_resident"


def values(enum_cls) -> list[str]:
    return [m.value for m in enum_cls]


def vocabulary_block() -> str:
    """Rendered into the extraction prompt, so the model sees the exact
    strings it is allowed to emit."""
    return (
        f"occupations: {', '.join(values(Occupation))}\n\n"
        f"categories: {', '.join(values(Category))}\n\n"
        f"genders: {', '.join(values(Gender))}\n\n"
        f"education_levels: {', '.join(values(EducationLevel))}\n\n"
        f"special_statuses: {', '.join(values(SpecialStatus))}"
    )


# Synonyms a 4B actually emits instead of the vocabulary value. Without these
# "agriculturist" and "cultivator" are dropped, the axis falls back to ANY, and
# a farmers-only scheme becomes visible to everyone. Fail-open is the safe
# direction, but each alias recovers real precision for one cheap line.
_ALIASES: dict[str, str] = {
    # occupations
    "agriculturist": "farmer", "cultivator": "farmer", "agriculture": "farmer",
    "kisan": "farmer", "grower": "farmer", "agri_labourer": "agricultural_labourer",
    "labourer": "agricultural_labourer", "fisher": "fisherman",
    "fisherfolk": "fisherman", "craftsman": "artisan", "craftsperson": "artisan",
    "handloom_weaver": "weaver", "vendor": "street_vendor",
    "hawker": "street_vendor", "safai_karamchari": "sanitation_worker",
    "scavenger": "manual_scavenger", "scholar": "researcher",
    "research_scholar": "researcher", "faculty": "teacher",
    "professor": "teacher", "it_professional": "software_professional",
    "software_engineer": "software_professional", "developer": "software_professional",
    "startup": "entrepreneur", "founder": "entrepreneur",
    "business_owner": "entrepreneur", "msme": "msme_owner",
    "soldier": "armed_forces", "defence_personnel": "armed_forces",
    "veteran": "ex_serviceman", "esm": "ex_serviceman",
    "housewife": "homemaker", "jobless": "unemployed",
    "organisation": "institution", "organization": "institution",
    "ngo": "institution", "college": "institution", "university": "institution",
    "company": "institution", "state_government": "institution",
    # categories
    "scheduled_caste": "sc", "scheduled_castes": "sc",
    "scheduled_tribe": "st", "scheduled_tribes": "st",
    "other_backward_class": "obc", "other_backward_classes": "obc",
    "backward": "obc", "backward_class": "obc",
    "economically_backward_class": "ebc", "economically_weaker_section": "ebc",
    "ews": "ebc", "minorities": "minority", "muslim": "minority",
    "denotified": "denotified_tribe", "dnt": "denotified_tribe",
    "nomadic_tribe": "denotified_tribe", "unreserved": "general",
    # genders
    "women": "female", "woman": "female", "girl": "female", "girls": "female",
    "men": "male", "man": "male", "boy": "male", "boys": "male",
    "trans": "transgender",
    # special statuses
    "bpl": "below_poverty_line", "poor": "below_poverty_line",
    "poverty": "below_poverty_line", "disabled": "person_with_disability",
    "handicapped": "person_with_disability", "pwd": "person_with_disability",
    "divyangjan": "person_with_disability",
    "differently_abled": "person_with_disability",
    "specially_abled": "person_with_disability",
    "widows": "widow", "elderly": "senior_citizen", "aged": "senior_citizen",
    "old_age": "senior_citizen", "pregnant": "pregnant_or_lactating",
    "lactating": "pregnant_or_lactating", "small_farmer": "small_or_marginal_farmer",
    "marginal_farmer": "small_or_marginal_farmer", "rural": "rural_resident",
    "urban": "urban_resident", "north_eastern": "north_east_resident",
    "ner": "north_east_resident",
    # education
    "matric": "class_10", "matriculation": "class_10",
    "post_matric": "class_12", "higher_secondary": "class_12",
    "intermediate": "class_12", "diploma": "iti_diploma", "iti": "iti_diploma",
    "bachelor": "graduate", "ug": "graduate", "undergraduate": "graduate",
    "master": "postgraduate", "pg": "postgraduate", "doctorate": "phd",
    "doctoral": "phd", "post_doctoral": "postdoc",
}


def coerce(raw, enum_cls, *, default=None) -> list[str]:
    """Map model output into the vocabulary, dropping anything unmappable.

    Tolerant of the shapes a 4B actually emits: a bare string, a list, mixed
    case, spaces instead of underscores, plurals.
    """
    if raw is None:
        return list(default or [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return list(default or [])

    allowed = {m.value for m in enum_cls}
    # alias map: "farmers" -> "farmer", "Farmer" -> "farmer", "sc/st" handled
    # by the caller splitting on separators first
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip().lower().replace(" ", "_").replace("-", "_")
        key = _ALIASES.get(key, key)
        if key in allowed:
            candidate = key
        elif _ALIASES.get(key.rstrip("s"), key.rstrip("s")) in allowed:
            candidate = _ALIASES.get(key.rstrip("s"), key.rstrip("s"))
        elif key.rstrip("s") in allowed:
            candidate = key.rstrip("s")
        elif f"{key}s" in allowed:
            candidate = f"{key}s"
        else:
            continue
        if candidate not in out:
            out.append(candidate)
    return out or list(default or [])
