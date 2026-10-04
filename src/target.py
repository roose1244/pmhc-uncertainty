"""Half-life transform.

20% of Stability.txt is recorded as 0.00 h (below the assay floor).
log10(0) is undefined, so we shift by the table's 0.1 h resolution.
"""

from __future__ import annotations

import math

from src.hla import normalise_hla

PSEUDOCOUNT = 0.1


def to_log(half_life: float) -> float:
    return math.log10(half_life + PSEUDOCOUNT)


def from_log(log_half_life: float) -> float:
    return max(10**log_half_life - PSEUDOCOUNT, 0.0)


def make_id(hla: str, peptide: str) -> str:
    return f"{normalise_hla(hla)}|{peptide.strip().upper()}"
