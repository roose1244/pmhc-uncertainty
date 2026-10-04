"""Leak-proof train / val / test / calib assignments."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path("data/splits")
SEED = 42
FOLDS = ("train", "val", "test", "calib")
# test 10%, val 10%, then 10% of the remainder as calib → 8% / 72%
TARGETS = {"test": 0.10, "val": 0.10, "calib": 0.08, "train": 0.72}

HLA_TEST = "HLA-B*15:02"
HLA_VAL = "HLA-B*27:02"
HLA_TRAIN_SIBLINGS = ("HLA-B*15:01", "HLA-B*27:05")
RANDOM_LEAKAGE_MIN = 0.90


def _assign_groups(group_sizes: dict[str, int], rng: np.random.Generator) -> dict[str, str]:
    """Greedy assignment of groups onto target row fractions."""
    names = list(group_sizes)
    rng.shuffle(names)
    assigned: dict[str, str] = {}
    used = {fold: 0 for fold in FOLDS}
    total = sum(group_sizes.values())
    caps = {fold: TARGETS[fold] * total for fold in FOLDS}
    for name in names:
        size = group_sizes[name]
        # fill the most under-full fold, preferring the locked order for ties
        fold = min(
            FOLDS,
            key=lambda f: ((used[f] + size) / caps[f], FOLDS.index(f)),
        )
        assigned[name] = fold
        used[fold] += size
    return assigned


def _hamming3_components(peptides: list[str]) -> dict[str, str]:
    """Union-find: peptides within Hamming distance 3 share a component id."""
    parent = {p: p for p in peptides}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for peptide in peptides:
        for dropped in combinations(range(9), 3):
            key = (dropped, "".join(peptide[i] for i in range(9) if i not in dropped))
            buckets[key].append(peptide)
    for group in buckets.values():
        first = group[0]
        for other in group[1:]:
            union(first, other)
    return {peptide: find(peptide) for peptide in peptides}


def random_split(master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sizes = {row_id: 1 for row_id in master["id"]}
    assigned = _assign_groups(sizes, rng)
    return pd.DataFrame({"id": master["id"], "fold": master["id"].map(assigned)})


def peptide_split(master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sizes = master.groupby("peptide").size().to_dict()
    assigned = _assign_groups(sizes, rng)
    return pd.DataFrame({"id": master["id"], "fold": master["peptide"].map(assigned)})


def cluster_split(master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    peptides = sorted(master["peptide"].unique())
    component = _hamming3_components(peptides)
    master = master.copy()
    master["component"] = master["peptide"].map(component)
    sizes = master.groupby("component").size().to_dict()
    assigned = _assign_groups(sizes, rng)
    return pd.DataFrame({"id": master["id"], "fold": master["component"].map(assigned)})


def hla_split(master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    assigned_allele: dict[str, str] = {}
    remaining = []
    for allele, count in master.groupby("hla").size().items():
        if allele == HLA_TEST:
            assigned_allele[allele] = "test"
        elif allele == HLA_VAL:
            assigned_allele[allele] = "val"
        elif allele in HLA_TRAIN_SIBLINGS:
            assigned_allele[allele] = "train"
        else:
            remaining.append((allele, int(count)))

    leftover_rows = sum(count for _, count in remaining)
    calib_target = TARGETS["calib"] / (TARGETS["calib"] + TARGETS["train"]) * leftover_rows
    order = remaining[:]
    rng.shuffle(order)
    calib_rows = 0
    for allele, count in order:
        if calib_rows < calib_target and abs((calib_rows + count) - calib_target) <= abs(
            calib_rows - calib_target
        ):
            assigned_allele[allele] = "calib"
            calib_rows += count
        else:
            assigned_allele[allele] = "train"
    return pd.DataFrame({"id": master["id"], "fold": master["hla"].map(assigned_allele)})


def peptide_leakage(master: pd.DataFrame, split: pd.DataFrame) -> float:
    merged = master.merge(split, on="id", how="inner")
    train_peptides = set(merged.loc[merged["fold"] == "train", "peptide"])
    test = merged.loc[merged["fold"] == "test"]
    if test.empty:
        return 0.0
    leaked = test["peptide"].isin(train_peptides).mean()
    return float(leaked)


def assert_split(master: pd.DataFrame, split: pd.DataFrame, name: str) -> dict:
    if set(split.columns) != {"id", "fold"}:
        raise AssertionError(f"{name}: columns must be id, fold")
    if split["id"].duplicated().any():
        raise AssertionError(f"{name}: duplicate ids")
    if set(split["id"]) != set(master["id"]):
        raise AssertionError(f"{name}: ids do not cover the master table")
    if set(split["fold"]) - set(FOLDS):
        raise AssertionError(f"{name}: unknown fold labels")

    merged = master.merge(split, on="id", how="inner")
    by_fold = {fold: merged.loc[merged["fold"] == fold] for fold in FOLDS}
    for left, right in combinations(FOLDS, 2):
        if set(by_fold[left]["id"]) & set(by_fold[right]["id"]):
            raise AssertionError(f"{name}: {left} and {right} share ids")

    if name in {"peptide", "cluster"}:
        for left, right in combinations(FOLDS, 2):
            leak = set(by_fold[left]["peptide"]) & set(by_fold[right]["peptide"])
            if leak:
                raise AssertionError(f"{name}: peptide leak {left}/{right}: {next(iter(leak))}")

    if name == "cluster":
        peptides = sorted(master["peptide"].unique())
        component = _hamming3_components(peptides)
        merged = merged.copy()
        merged["component"] = merged["peptide"].map(component)
        for left, right in combinations(FOLDS, 2):
            leak = set(merged.loc[merged["fold"] == left, "component"]) & set(
                merged.loc[merged["fold"] == right, "component"]
            )
            if leak:
                raise AssertionError(f"{name}: cluster leak {left}/{right}")

    if name == "hla":
        test_alleles = set(by_fold["test"]["hla"])
        val_alleles = set(by_fold["val"]["hla"])
        train_alleles = set(by_fold["train"]["hla"])
        if test_alleles != {HLA_TEST}:
            raise AssertionError(f"hla: test alleles are {test_alleles}")
        if val_alleles != {HLA_VAL}:
            raise AssertionError(f"hla: val alleles are {val_alleles}")
        if HLA_TEST in train_alleles or HLA_VAL in train_alleles:
            raise AssertionError("hla: held-out allele leaked into train")
        for sibling in HLA_TRAIN_SIBLINGS:
            if sibling not in train_alleles:
                raise AssertionError(f"hla: sibling {sibling} is not in train")

    leakage = peptide_leakage(master, split)
    if name == "random" and leakage < RANDOM_LEAKAGE_MIN:
        raise AssertionError(
            f"random split peptide leakage is {leakage:.3f}; expected ~0.94. "
            "This is not the random split."
        )
    counts = split["fold"].value_counts().to_dict()
    return {"name": name, "counts": counts, "peptide_leakage": leakage}


def build_all(master: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    builders = {
        "random": random_split,
        "peptide": peptide_split,
        "cluster": cluster_split,
        "hla": hla_split,
    }
    out = {}
    for name, builder in builders.items():
        rng = np.random.default_rng(seed)
        split = builder(master, rng)
        assert_split(master, split, name)
        out[name] = split
    return out


def write_splits(splits: dict[str, pd.DataFrame], directory: Path = SPLITS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
