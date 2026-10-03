"""Canonical HLA names.

Developer 2's plan: use HLA-A*02:01 everywhere, via normalise_hla().
The raw Stability.txt already uses that form. The three engineered
mutants keep their (C67S) suffix because they are not IMGT alleles.
"""

from __future__ import annotations

import re

_ENGINEERED = re.compile(r"\(([^)]+)\)$")
_CANON = re.compile(
    r"^(?:HLA[-_])?([ABC])\*?(\d{2}):?(\d{2})$",
    re.IGNORECASE,
)


def normalise_hla(raw: str) -> str:
    """Return HLA-A*02:01 or HLA-B*14:02(C67S). Raise on unknown input."""
    text = raw.strip().replace(" ", "")
    suffix = ""
    engineered = _ENGINEERED.search(text)
    if engineered:
        suffix = f"({engineered.group(1)})"
        text = text[: engineered.start()]

    match = _CANON.match(text)
    if match is None:
        raise ValueError(f"Unrecognised HLA allele: {raw!r}")

    locus, family, protein = match.group(1).upper(), match.group(2), match.group(3)
    return f"HLA-{locus}*{family}:{protein}{suffix}"
