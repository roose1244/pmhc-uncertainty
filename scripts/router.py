"""Route each query to the model that handles its regime.

M1 and M2n fail in opposite places. One-hot encoding the allele name is a
perfect lookup for an allele the model has measured and a zero vector for one
it has not: M1 scores 0.765 on the random split and -0.113 on a held-out
allele. The ESM representation generalises instead of memorising, so M2n is
worse on familiar alleles and the only one of the two with any signal on
unfamiliar ones (+0.090, non-overlapping seed ranges).

Rather than pick one, route on whether the queried allele was in training:

    allele seen in training  -> M1    (memorisation is legitimate here)
    allele not seen          -> M2n   (the only model with signal)

The routing feature is known at prediction time and needs no labels, so this
is not leakage: a deployed tool always knows which alleles its own training
set contained.

WHY THIS NEEDS A PURPOSE-BUILT EVALUATION. No existing split can test a
router, because each one is homogeneous: every test allele in the random,
peptide and cluster splits is seen, and every test allele in the HLA split is
unseen. A router evaluated on any single one of them never actually switches.

So this simulates deployment. Starting from the HLA split's train fold (63
alleles), a random 10% of those rows is withheld as a familiar-allele test
set, both models train on the remaining 90%, and they are evaluated on that
withheld set together with the HLA split's own test allele. The result is a
test set containing both regimes, which is what a real user population looks
like.

Run:  python scripts/router.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.train_m1 as m1mod  # noqa: E402
import src.train_m2_scaled as m2mod  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/router.csv")

SEEDS = [42, 0, 1, 2]
HOLDOUT_FRAC = 0.10
SEED_SPLIT = 7


def build_evaluation(master: pd.DataFrame) -> pd.DataFrame:
    """A split whose test fold contains both familiar and novel alleles."""
    hla_split = pd.read_parquet(SPLIT_DIR / "hla.parquet")
    d = master[["id", "hla"]].merge(hla_split, on="id")

    rng = np.random.default_rng(SEED_SPLIT)
    train_rows = d[d.fold == "train"].copy()
    held = rng.random(len(train_rows)) < HOLDOUT_FRAC

    fold = pd.Series("train", index=d.index)
    fold[d.fold == "val"] = "val"
    # The HLA split's own test allele stays the novel half of the test fold.
    fold[d.fold == "test"] = "test"
    fold[d.fold == "calib"] = "drop"
    idx = train_rows.index[held]
    fold[idx] = "test"

    out = pd.DataFrame({"id": d.id, "fold": fold.to_numpy(),
                        "hla": d.hla.to_numpy()})
    # Regime label, known at prediction time.
    seen = set(out[out.fold == "train"].hla)
    out["regime"] = np.where(out.hla.isin(seen), "familiar", "novel")
    return out[out.fold != "drop"]


def spear(x, y) -> float:
    if len(x) < 10 or len(np.unique(x)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def main() -> int:
    master = pd.read_parquet(MASTER)
    ev = build_evaluation(master)
    split = ev[["id", "fold"]]

    test = ev[ev.fold == "test"]
    print("deployment-style evaluation")
    print(f"   train rows   : {int((ev.fold == 'train').sum()):6d}")
    print(f"   test rows    : {len(test):6d}  "
          f"({int((test.regime == 'familiar').sum())} familiar, "
          f"{int((test.regime == 'novel').sum())} novel)")
    print(f"   test alleles : {test.hla.nunique()} "
          f"({test[test.regime == 'novel'].hla.nunique()} never seen)\n")

    pep_store = m2mod.load_store("esm_peptide")
    hla_store = m2mod.load_store("esm_hla")

    rows = []
    for seed in SEEDS:
        p1, _, _ = m1mod.train_one(master, split, peptide_only=False, seed=seed)
        m2mod.SEED = seed
        p2, _ = m2mod.train_one(master, split, pep_store, hla_store,
                                peptide_only=False, scale=True)

        j = (test.merge(p1[["id", "y_true", "y_mean"]].rename(
                 columns={"y_mean": "m1"}), on="id")
                 .merge(p2[["id", "y_mean"]].rename(columns={"y_mean": "m2n"}), on="id"))
        # The router: regime decides which column is used for that row.
        j["routed"] = np.where(j.regime == "familiar", j.m1, j.m2n)

        for scope in ["familiar", "novel", "all"]:
            s = j if scope == "all" else j[j.regime == scope]
            rows.append({
                "seed": seed, "scope": scope, "n": len(s),
                "m1": spear(s.y_true, s.m1),
                "m2n": spear(s.y_true, s.m2n),
                "routed": spear(s.y_true, s.routed),
            })
        print(f"   seed {seed}: done", flush=True)

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("\nSpearman by regime, mean over seeds (sd in brackets)")
    print(f"   {'scope':10s} {'n':>6s} {'M1':>16s} {'M2n':>16s} {'routed':>16s}")
    for scope in ["familiar", "novel", "all"]:
        s = t[t.scope == scope]
        cells = []
        for c in ["m1", "m2n", "routed"]:
            cells.append(f"{s[c].mean():+8.3f} ({s[c].std(ddof=1):.3f})")
        print(f"   {scope:10s} {int(s.n.iloc[0]):6d} " + " ".join(cells))

    a = t[t.scope == "all"]
    print(f"\n   routed beats M1 overall by "
          f"{a.routed.mean() - a.m1.mean():+.3f}, "
          f"M2n by {a.routed.mean() - a.m2n.mean():+.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
