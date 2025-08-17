# Deterministic checkers for task-based evaluation.
#
# Each checker in this module validates whether a response from the
# language model satisfies a particular type of task specification.
# Currently supported types are:
# - span: direct span extraction; expected answer must appear verbatim
#   (case-insensitive, whitespace-normalised).
# - procedure: sequence of required command substrings that must appear
#   in order within the response.
# - rationale: qualitative explanation requiring certain discourse markers
#   and minimum length.

from __future__ import annotations

import re
from typing import Dict, Any, List


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def check_span(response: str, task: Dict[str, Any]) -> bool:
    expected = _normalise(task.get("expected_span", ""))
    return bool(expected and expected in _normalise(response))


def check_procedure(response: str, task: Dict[str, Any]) -> bool:
    steps: List[Dict[str, str]] = task.get("policy_steps", [])
    resp_lower = response.lower()
    position = 0
    for step in steps:
        must = step.get("must", "").lower()
        if not must:
            continue
        idx = resp_lower.find(must, position)
        if idx < 0:
            return False
        position = idx + len(must)
    return True


def check_rationale(response: str, task: Dict[str, Any]) -> bool:
    min_words = int(task.get("min_words", 40))
    lower_resp = response.lower()
    has_indicator = any(
        kw in lower_resp for kw in ["because", "therefore", "so that", "given"]
    )
    return has_indicator and len(response.split()) >= min_words


def run_checker(response: str, task: Dict[str, Any]) -> bool:
    t = task.get("type")
    if t == "span":
        return check_span(response, task)
    if t == "procedure":
        return check_procedure(response, task)
    if t == "rationale":
        return check_rationale(response, task)
    return False
