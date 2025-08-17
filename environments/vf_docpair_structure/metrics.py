# Implements metrics for vf_docpair_structure environment
from typing import List


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))


def depth_balance(gold: List[str], cand: List[str]) -> float:
    """Compute balance of heading depths between gold and candidate outlines.
    Returns a value in [0, 1], where 1 means perfect depth distribution match.
    """
    def depth_hist(xs):
        # Build a histogram of depths from 1 to 6
        h = {i: 0 for i in range(1, 7)}
        for x in xs:
            lvl = int(x.split(":")[0])
            h[lvl] += 1
        tot = sum(h.values()) or 1
        return {k: v / tot for k, v in h.items()}

    gh = depth_hist(gold)
    ch = depth_hist(cand)
    diff = sum(abs(gh[i] - ch[i]) for i in range(1, 7))
    return max(0.0, 1.0 - diff / 2.0)
