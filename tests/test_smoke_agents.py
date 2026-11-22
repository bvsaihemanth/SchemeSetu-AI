"""Drives scripts/smoke_agents.py against the fake stack.

The smoke script is the thing you run when something looks wrong with the
real provider, so it has to be trustworthy itself: a smoke test that reports
PASS on a broken pipeline, or crashes on a healthy one, is worse than none.

Three runs here:
  1. a healthy pipeline           -> exit 0, every agent PASS
  2. an interview that re-asks    -> exit 1, eligibility.answer FAIL
  3. a model that invents a [9]   -> caught as an invented citation

Run 2 is the important one. It simulates exactly the behaviour that made the
local model unusable -- answering every question and being asked again -- and
proves the smoke test would have caught it rather than reporting success.

    python -m tests.test_smoke_agents
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

from tests.test_agents import CORPUS, build_env  # noqa: E402


class SmokeLLM:
    """Answer-aware stand-in.

    The one behaviour that matters: once the assessment prompt contains the
    user's answers, this resolves the criteria instead of re-asking. A fake
    that ignored the answers would make run 1 fail for the wrong reason and
    would not distinguish a working pipeline from the broken one.
    """

    def __init__(self, *, reask: bool = False, invent_citation: bool = False):
        self.reask = reask
        self.invent_citation = invent_citation
        self.calls: list[str] = []

    def complete(self, system, user, temperature=None, max_tokens=None, json_mode=False, **_):
        if "Classify the user's question" in system:
            self.calls.append("router")
            q = user.lower()
            intent = (
                "documents" if "document" in q
                else "comparison" if "compare" in q
                else "application" if "how do i apply" in q
                else "eligibility" if "eligible" in q
                else "discovery" if ("what schemes" in q or "can i get" in q)
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
            return '{"occupation": "farmer", "state": "Telangana"}'

        if "You assess whether a person qualifies" in system:
            self.calls.append("assess")
            answered = "ALREADY ANSWERED" in user
            if answered and not self.reask:
                # Resolved: the paired answers settled both criteria.
                return (
                    '{"criteria": ['
                    '{"criterion": "Applicant must be a cultivator",'
                    ' "verdict": "met", "reason": "answered yes", "needs_slot": null},'
                    '{"criterion": "Land must be in the applicant\'s name",'
                    ' "verdict": "met", "reason": "answered yes", "needs_slot": null}],'
                    '"overall": "likely_eligible", "questions": []}'
                )
            # Unresolved -- and with reask=True it stays unresolved even after
            # the answers arrive, which is the regression being simulated.
            return (
                '{"criteria": ['
                '{"criterion": "Applicant must be a cultivator",'
                ' "verdict": "unknown", "reason": "", "needs_slot": "occupation"},'
                '{"criterion": "Land must be in the applicant\'s name",'
                ' "verdict": "unknown", "reason": "", "needs_slot": "landholding_ha"}],'
                '"overall": "insufficient_info", "questions": ['
                '"Are you a cultivator of the land?",'
                ' "Is the land held in your own name?"]}'
            )

        if "You write a short passage" in system:
            self.calls.append("hyde")
            return (
                "Eligible beneficiaries include small and marginal farmers "
                "holding cultivable land who receive financial assistance and "
                "income support subject to conditions notified by the "
                "implementing department in rural areas."
            )

        if "You compare Indian government schemes" in system:
            self.calls.append("compare")
            first: dict[str, int] = {}
            for m in re.finditer(r"\[(\d+)\] \(([^|]+)\|", user):
                first.setdefault(m.group(2).strip(), int(m.group(1)))
            return " ".join(f"{n} provides cover [{i}]." for n, i in first.items())

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
        if self.invent_citation:
            # [9] does not exist in the context: the Evidence agent must strip
            # it, and if it ever leaks, citations_resolve must catch it.
            return "Beneficiaries receive income support [1]. An invented claim [9]."
        return "Beneficiaries receive income support [1]."

    def health(self):
        return True

    def available_models(self):
        from schemes_rag.config import get_settings

        return [get_settings().llm_model]


def run(llm, *, argv: list[str] | None = None) -> tuple[int, list]:
    """Build the fake stack, then run the smoke script's main()."""
    build_env(llm)

    import smoke_agents

    # The script reads these from the live settings; point them at the fake.
    from schemes_rag.config import get_settings

    s = get_settings()
    s.llm_api_key = "test-key-looks-real"
    s.llm_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    s.llm_model = "gemini-3.5-flash-lite"

    import schemes_rag.graph as graph_mod

    smoke_agents.run_query = graph_mod.run_query
    smoke_agents._results.clear()

    old = sys.argv
    sys.argv = ["smoke_agents.py", *(argv or [])]
    try:
        code = smoke_agents.main()
    finally:
        sys.argv = old
    return code, list(smoke_agents._results)


def statuses(results) -> dict[str, str]:
    return {agent: status for status, agent, _, _ in results}


def main() -> int:
    if sys.flags.hash_randomization and os.environ.get("_REEXEC") != "1":
        os.environ["_REEXEC"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "tests.test_smoke_agents"])

    if not os.path.isdir(CORPUS):
        print(f"SKIP: corpus not available at {CORPUS}")
        return 0

    failures: list[str] = []

    # ---------------- 1. healthy pipeline ----------------
    print("\n" + "#" * 76)
    print("# RUN 1: healthy pipeline -> expect exit 0")
    print("#" * 76)
    code, results = run(SmokeLLM())
    got = statuses(results)
    print(f"\n>>> exit={code}  {got}")

    if code != 0:
        bad = [f"{a}: {d}" for st, a, d, _ in results if st == "FAIL"]
        failures.append(f"healthy run exited {code}: {bad}")
    for agent in ("eligibility.ask", "eligibility.answer", "evidence", "refusal"):
        if got.get(agent) != "PASS":
            failures.append(f"healthy run: {agent} was {got.get(agent)}, expected PASS")
    if not failures:
        print("  OK  every agent passed on a healthy pipeline")

    # ---------------- 2. the interview regression ----------------
    print("\n" + "#" * 76)
    print("# RUN 2: assessor ignores the answers -> expect eligibility.answer FAIL")
    print("#" * 76)
    code, results = run(SmokeLLM(reask=True))
    got = statuses(results)
    print(f"\n>>> exit={code}  {got}")

    if code == 0:
        failures.append("re-asking pipeline exited 0 -- the smoke test is blind "
                        "to the exact regression it exists to catch")
    if got.get("eligibility.answer") != "FAIL":
        failures.append(
            f"eligibility.answer was {got.get('eligibility.answer')}, expected FAIL"
        )
    else:
        detail = next(d for st, a, d, _ in results if a == "eligibility.answer")
        print(f"  OK  caught it: {detail}")
        assert "REPEATS" in detail or "re-asked" in detail, detail
    # The other agents must still report honestly rather than cascading.
    if got.get("documents") != "PASS":
        failures.append(f"documents was {got.get('documents')} -- failure cascaded")

    # ---------------- 3. invented citation ----------------
    print("\n" + "#" * 76)
    print("# RUN 3: model invents [9] -> Evidence must strip it")
    print("#" * 76)
    code, results = run(SmokeLLM(invent_citation=True))
    got = statuses(results)
    print(f"\n>>> exit={code}  {got}")

    # Evidence strips unresolvable markers upstream, so the right outcome is
    # PASS: the guard worked. What must NOT happen is a leaked [9].
    if got.get("evidence") != "PASS":
        detail = next(d for st, a, d, _ in results if a == "evidence")
        failures.append(f"invented marker reached the user: {detail}")
    else:
        print("  OK  invented [9] stripped before it reached the answer")

    print("\n" + "=" * 76)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL SMOKE-TEST BEHAVIOURS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
