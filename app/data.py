"""Single source of truth for every number the interface shows.

Every figure in the UI is read from a file under results/tables/ that an
experiment actually wrote. Nothing is typed in by hand, so a number cannot
drift from the result that produced it, and a missing file makes a panel
disappear rather than show something invented.

Each loader returns None when its file is absent. Callers render an explicit
"not computed" state instead of a placeholder value.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

TABLES = Path("results/tables")

SPLITS = ["random", "peptide", "cluster", "hla"]

# What each split actually tests. Used to label charts so a reader knows why
# four numbers exist rather than one.
SPLIT_MEANING = {
    "random": ("RANDOM", "Seen peptide, seen allele",
               "General performance when nothing is new."),
    "peptide": ("PEPTIDE", "Unseen peptide",
                "Can it handle a peptide it has never measured?"),
    "cluster": ("CLUSTER", "Unseen peptide family",
                "Can it handle a whole family of related peptides it has not seen?"),
    "hla": ("HLA", "Unseen allele",
            "Can it handle an HLA allele it has never measured? The hardest test."),
}


def _read(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def comparison() -> pd.DataFrame | None:
    """Dev 1's headline table: every model and NetMHCstabpan, per split."""
    return _read("comparison.csv")


@lru_cache(maxsize=1)
def seed_variance() -> pd.DataFrame | None:
    """Per-model mean and spread over repeated training runs.

    The only table with error bars. It exists because a single run on the HLA
    split spans 0.007 to 0.113, so single-run differences there are not
    interpretable and the interface should not draw them as if they were.
    """
    return _read("seed_variance.csv")


@lru_cache(maxsize=1)
def head_to_head() -> pd.DataFrame | None:
    """This system against NetMHCstabpan on identical rows, with paired CIs."""
    return _read("head_to_head.csv")


@lru_cache(maxsize=1)
def nms_baseline() -> pd.DataFrame | None:
    return _read("nms_baseline.csv")


@lru_cache(maxsize=1)
def nms_in_sample() -> pd.DataFrame | None:
    """The seven alleles outside NetMHCstabpan's own training set."""
    return _read("nms_in_sample.csv")


@lru_cache(maxsize=1)
def selective() -> pd.DataFrame | None:
    return _read("selective_prediction.csv")


@lru_cache(maxsize=1)
def per_allele() -> pd.DataFrame | None:
    return _read("per_allele_uncertainty.csv")


@lru_cache(maxsize=1)
def uncertainty_tables() -> dict[str, pd.DataFrame | None]:
    return {"m1ens": _read("m1ens.csv"), "m2nens": _read("m2nens.csv")}


@lru_cache(maxsize=1)
def in_sample_fact() -> tuple[int, int, float] | None:
    """How much of the evaluation set is inside NetMHCstabpan's training data.

    Derived rather than hardcoded: nms_in_sample.csv lists the alleles found
    OUTSIDE its training.pseudo together with their row counts, so the
    in-sample remainder follows from the dataset size.
    """
    out = nms_in_sample()
    if out is None or out.empty:
        return None
    total = 28166  # master.parquet row count, locked in the project README
    outside = int(out.n.sum())
    inside = total - outside
    return inside, total, 100.0 * inside / total


def model_label(name: str) -> str:
    return {
        "m1": "M1 · baseline",
        "m1pep": "M1 · peptide only",
        "m1ens": "M1 ensemble",
        "m2": "M2 · sequence",
        "m2n": "M2n · sequence, scaled",
        "m2pep": "M2 · peptide only",
        "m2nens": "M2n ensemble",
        "m3": "M3 · cross-attention",
        "netmhcstabpan": "NetMHCstabpan",
    }.get(name, name)


def is_reference(name: str) -> bool:
    return name == "netmhcstabpan"
