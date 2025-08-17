# Checkers for vf_docpair_task environment
import re
from typing import Dict, Any, List


def norm(s: str) -> str:
    """Normalize whitespace and lowercase the string."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def check_span_qa(response: str, task: Dict[str, Any]) -> bool:
    """Check if the expected span is present in the response (normalized)."""
    return norm(task.get("expected_span", "")) in norm(response)


def check_procedure(response: str, task: Dict[str, Any]) -> bool:
    """Check if the response follows the required ordered steps."""
    steps: List[Dict[str, str]] = task.get("policy_steps", [])
    pos = 0
    r = response.lower()
    for step in steps:
        pat = step.get("must", "").lower()
        i = r.find(pat, pos)
        if i < 0:
            return False
        pos = i + len(pat)
    return True


def check_rationale(response: str, task: Dict[str, Any]) -> bool:
    """Check if the response provides rationale with key words and minimum length."""
    rationale_keywords = ["because", "therefore", "so that", "given"]
    has_reason = any(w in response.lower() for w in rationale_keywords)
    min_words = task.get("min_words", 40)
    return has_reason and len(response.split()) >= min_words


def run_checker(response: str, task: Dict[str, Any]) -> bool:
    """Dispatch to the appropriate checker based on task type."""
    t = task.get("type")
    if t == "span":
        return check_span_qa(response, task)
    if t == "procedure":
        return check_procedure(response, task)
    if t == "rationale":
        return check_rationale(response, task)
    return False
