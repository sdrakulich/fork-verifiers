"""Structural coverage evaluation environment for verifiers.

This environment scores the structural similarity between two
documentation texts.  It computes recall-style metrics on the set
of section headings as well as a simple depth balance heuristic.  The
resulting reward is a weighted combination of these submetrics.  All
processing is done with pure Python and no language model calls are
issued.

Each dataset row for this environment is expected to contain a
``question`` string (ignored), and an ``info`` dictionary with the
following keys:

``gold_doc`` (str)
    The full text of the reference document.

``candidate_doc`` (str)
    The full text of the document under test.

Other fields such as ``project_id`` or checksums are passed through
without interpretation.  The computed metrics are stored in the
environment state under the ``metrics`` key for downstream analysis.
"""

from __future__ import annotations

from typing import Dict

from verifiers import ThinkParser, Rubric
from verifiers.envs import SingleTurnEnv

from .outline import extract_outline, canonicalize_outline
from .metrics import depth_balance


def _compute_structure_metrics(gold_text: str, cand_text: str) -> Dict[str, float]:
    """Compute structural similarity metrics between two documents.

    The function extracts Markdown headings from both the gold and
    candidate documents, normalises them, and then computes
    intersection sizes to derive recall-like measures.  It also
    evaluates the balance of heading depths to penalise overly flat or
    overly deep structures.

    Parameters
    ----------
    gold_text:
        The full text of the reference (gold) document.
    cand_text:
        The full text of the candidate document.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the following keys:
        ``outline_recall``: proportion of gold headings present in the candidate.
        ``critical_recall``: recall over a subset of high‑value section names.
        ``depth_balance``: similarity in depth distribution of headings.
    """
    g_outline = canonicalize_outline(extract_outline(gold_text))
    c_outline = canonicalize_outline(extract_outline(cand_text))
    gold_set = set(g_outline)
    cand_set = set(c_outline)
    # simple intersection recall
    outline_recall = len(gold_set & cand_set) / max(1, len(gold_set))
    # define a handful of critical sections that are especially important
    critical_keywords = {
        "install",
        "installation",
        "quickstart",
        "usage",
        "config",
        "configuration",
        "cli",
        "environment",
        "troubleshooting",
        "faq",
        "reference",
    }
    gold_crit = [s for s in g_outline if any(k in s for k in critical_keywords)]
    cand_crit = [s for s in c_outline if any(k in s for k in critical_keywords)]
    crit_recall = (
        len(set(gold_crit) & set(cand_crit)) / max(1, len(gold_crit))
        if gold_crit
        else 1.0
    )
    balance = depth_balance(g_outline, c_outline)
    return {
        "outline_recall": outline_recall,
        "critical_recall": crit_recall,
        "depth_balance": balance,
    }


def load_environment(**kwargs) -> SingleTurnEnv:
    """Instantiate the structural coverage environment.

    The environment uses a trivial parser (ThinkParser) and a single
    reward function that sums weighted metrics.  All other options
    passed via ``kwargs`` are forwarded to the ``SingleTurnEnv``
    constructor.
    """
    parser = ThinkParser()

    # reward function: compute metrics and return weighted sum
    def structure_reward(prompt, completion, answer, state, info, **_):
        metrics = _compute_structure_metrics(info["gold_doc"], info["candidate_doc"])
        # store raw metrics for auditing
        state.setdefault("metrics", {}).update(metrics)
        # weight metrics: emphasise outline recall, then critical recall,
        # then depth balance
        return (
            0.6 * metrics["outline_recall"]
            + 0.3 * metrics["critical_recall"]
            + 0.1 * metrics["depth_balance"]
        )

    rubric = Rubric(funcs=[structure_reward], weights=[1.0])
    env = SingleTurnEnv(
        dataset=kwargs.get("dataset"),
        parser=parser,
        rubric=rubric,
        message_type=kwargs.get("message_type", "chat"),
    )
    return env
