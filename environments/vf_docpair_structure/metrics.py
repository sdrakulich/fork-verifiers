"""Helper metrics for structural comparison.

This module provides simple functions to quantify similarities
between two outlines.  For now, only a depth balance heuristic is
implemented.  Additional functions may be added here to support
future scoring dimensions.
"""

from __future__ import annotations

from typing import List


def depth_balance(gold: List[str], candidate: List[str]) -> float:
    """Compute a heuristic score for depth distribution similarity.

    The distribution of heading levels (1 through 6) in both the
    gold and candidate outlines is computed and normalised.  The
    absolute differences between corresponding bins are summed and
    scaled to yield a score in [0, 1], where 1 indicates identical
    distributions and 0 indicates maximal imbalance.

    Parameters
    ----------
    gold : List[str]
        Canonicalised outline of the reference document.
    candidate : List[str]
        Canonicalised outline of the candidate document.

    Returns
    -------
    float
        A value in the range [0, 1] representing similarity of
        heading depth distributions.
    """
    def hist(out: List[str]):
        counts = {i: 0 for i in range(1, 7)}
        for entry in out:
            try:
                lvl = int(entry.split(":", 1)[0])
            except (ValueError, IndexError):
                continue
            if 1 <= lvl <= 6:
                counts[lvl] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    g_hist = hist(gold)
    c_hist = hist(candidate)
    # compute total variation distance and convert to similarity
    diff = sum(abs(g_hist[i] - c_hist[i]) for i in range(1, 7))
    return max(0.0, 1.0 - diff / 2.0)