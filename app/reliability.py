"""Reliability signals for a single peptide-HLA query.

This is deliberately separate from the model call. Everything here is computed
from the training data we already have, so the familiarity gauges are real
from day one even while predict() is still returning placeholders -- and they
stay real if the endpoint goes down.

WHY THE BADGE IS NOT DRIVEN BY MODEL AGREEMENT. The obvious design is to read
the ensemble's y_std and colour the badge by it. Measured on held-out alleles
(scripts/analyse_novelty_uncertainty.py), that does not work:

    novelty vs mean |error| = +0.546
    novelty vs mean y_std   = -0.055

Ensemble disagreement does not rise when the allele is unfamiliar, because M1
one-hot encodes the allele name and every unseen allele is the same zero
vector, so all five members see identical input. A badge built on y_std would
look principled and carry almost no information about allele novelty. Novelty
is therefore the primary signal and model agreement is shown as a secondary,
explicitly weak one.

Thresholds are set from the measured data, not from intuition, and each one
cites what it is based on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

CACHE = Path(__file__).parent / "cache"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Training rows per allele range from 7 (B*13:02) to 1070 (A*02:01). Conformal
# coverage on the thinnest alleles was the least reliable in the leave-one-out
# test, so density is the primary HLA signal for alleles the model has seen.
HLA_RICH = 300
HLA_THIN = 100

# Peptide novelty is Hamming distance over the 9-mer. Conditional coverage on
# the peptide split fell from 90.5% in the lowest novelty tercile to 86.6% in
# the highest, where the mean distance was ~6 of 9 positions.
PEP_CLOSE = 2
PEP_FAR = 5


@dataclass
class Signal:
    name: str
    level: str  # "green" | "amber" | "red"
    value: str
    detail: str


@dataclass
class Reliability:
    badge: str
    signals: list[Signal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def known_peptides() -> tuple[str, ...]:
    return tuple(json.loads((CACHE / "known_peptides.json").read_text()))


@lru_cache(maxsize=1)
def alleles() -> dict[str, dict]:
    rows = {}
    lines = (CACHE / "alleles.csv").read_text().splitlines()
    header = lines[0].split(",")
    for line in lines[1:]:
        parts = line.split(",")
        r = dict(zip(header, parts))
        r["n_rows"] = int(r["n_rows"])
        rows[r["hla"]] = r
    return rows


def validate_peptide(peptide: str) -> str | None:
    """Return an error message, or None when the peptide is usable."""
    p = peptide.strip().upper()
    if not p:
        return "Enter a peptide sequence."
    if not 8 <= len(p) <= 11:
        return f"Peptide must be 8-11 amino acids ({len(p)} given)."
    bad = sorted(set(p) - VALID_AA)
    if bad:
        return f"Not valid amino acid letters: {', '.join(bad)}"
    if len(p) != 9:
        # Every measurement in the training table is a 9-mer, so anything else
        # is an extrapolation even when the model will accept it.
        return None
    return None


def peptide_distance(peptide: str) -> tuple[int, str]:
    """Minimum Hamming distance to any training peptide, and that peptide.

    Only defined between equal lengths; for a non-9-mer there is no training
    peptide to compare against position by position, so this reports -1 and
    the caller treats it as maximally novel.
    """
    p = peptide.strip().upper()
    best, who = 99, ""
    for other in known_peptides():
        if len(other) != len(p):
            continue
        d = sum(1 for a, b in zip(p, other) if a != b)
        if d < best:
            best, who = d, other
            if d == 0:
                break
    return (best, who) if who else (-1, "")


def assess(peptide: str, hla: str, model_std: float | None = None) -> Reliability:
    signals: list[Signal] = []
    reasons: list[str] = []

    # --- HLA familiarity ------------------------------------------------
    meta = alleles().get(hla)
    if meta is None:
        signals.append(
            Signal("HLA familiarity", "red", "not in training",
                   "This allele was never seen during training.")
        )
        reasons.append(
            "The model has no measurements for this allele. Its predictions "
            "for a wholly unseen allele are not better than peptide-only "
            "guesses, and the ensemble does not widen to reflect that."
        )
    else:
        n = meta["n_rows"]
        level = "green" if n >= HLA_RICH else "amber" if n >= HLA_THIN else "red"
        signals.append(
            Signal("HLA familiarity", level, f"{n} training examples",
                   f"Supertype {meta['supertype']} "
                   f"({meta['supertype_confidence']}).")
        )
        if level != "green":
            reasons.append(
                f"This allele has only {n} measurements in training, so the "
                f"model has seen comparatively little of how it behaves."
            )
        if meta["supertype_confidence"] == "low":
            reasons.append(
                "Its supertype label was inferred from pocket similarity "
                "rather than published, so comparisons to related alleles "
                "are less certain."
            )

    # --- peptide familiarity --------------------------------------------
    dist, nearest = peptide_distance(peptide)
    if dist < 0:
        signals.append(
            Signal("Peptide familiarity", "red", "no comparable peptide",
                   "Every training peptide is a 9-mer.")
        )
        reasons.append(
            "All training data is 9-mers, so a peptide of another length is "
            "an extrapolation beyond anything the model was fitted on."
        )
    else:
        level = "green" if dist <= PEP_CLOSE else "amber" if dist < PEP_FAR else "red"
        word = "exact match in training" if dist == 0 else f"{dist} of {len(peptide)} differ"
        signals.append(
            Signal("Peptide familiarity", level, word,
                   f"Closest training peptide: {nearest}" if nearest else "")
        )
        if dist >= PEP_FAR:
            reasons.append(
                f"The closest peptide in training differs at {dist} of "
                f"{len(peptide)} positions. Interval coverage measured on "
                f"peptides this distant fell to about 87%, below the stated 90%."
            )

    # --- model agreement, secondary and explicitly weak ------------------
    if model_std is not None:
        signals.append(
            Signal("Model agreement", "grey", f"std {model_std:.3f}",
                   "Weak signal: ensemble spread tracks error poorly "
                   "(within-allele Spearman +0.05) and does not respond to "
                   "an unfamiliar allele at all.")
        )

    ranked = {"green": 0, "amber": 1, "red": 2}
    scored = [s for s in signals if s.level in ranked]
    badge = max(scored, key=lambda s: ranked[s.level]).level if scored else "amber"
    if badge == "green" and not reasons:
        reasons.append(
            "This allele is well represented in training and the peptide is "
            "close to sequences the model has seen."
        )
    return Reliability(badge=badge, signals=signals, reasons=reasons)
