# Rubrics for evaluating editorial qualities of documentation.
#
# This module defines both rule-based and learned scoring functions for
# the editorial environment. The rule-based portion checks for the
# presence of prerequisite declarations, pinned version examples
# and error handling sections. The learned portion leverages a judge
# model to score actionability and clarity on a continuous scale based
# on a fixed set of criteria.

from __future__ import annotations

import re
from verifiers import Rubric, JudgeRubric


def has_prereqs(text: str) -> float:
    return 1.0 if re.search(r"\b(requirements|prerequisites|deps|dependencies)\b", text.lower()) else 0.0


def has_pinned_versions(text: str) -> float:
    return 1.0 if re.search(r"(pip install [^\n]+==\d|conda env create|pyproject\.toml)", text.lower()) else 0.0


def has_error_playbook(text: str) -> float:
    return 1.0 if re.search(r"(troubleshooting|common errors|faq)", text.lower()) else 0.0


def rule_editorial(prompt, completion, answer, state, info, **_) -> float:
    t = info.get("candidate_doc", "")
    return 0.5 * has_prereqs(t) + 0.3 * has_pinned_versions(t) + 0.2 * has_error_playbook(t)


def judge_editorial() -> JudgeRubric:
    return JudgeRubric(
        judge_model="gpt-4o-mini",
        judge_prompt=(
            "Rate ACTIONABILITY 0.0-1.0.\n"
            "Criteria: (1) explicit preconditions; (2) step ordering; "
            "(3) copy/paste-ready commands with language tags; (4) failure modes.\n"
            "Document:\n{response}\n\nScore only; do not quote."
        ),
        judge_sampling_args={"temperature": 0.1},
    )
