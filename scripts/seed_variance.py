"""How much of each model's score is the model, and how much is the seed?

Every number in the ladder so far comes from a single training run at seed 42.
That is fine where the test fold is thousands of rows, and misleading on the
HLA split, whose val and test folds are a single allele each (359 and 349
rows) and whose training stops after a handful of epochs. A first pass found
the same unscaled M2 scoring anywhere from 0.007 to 0.113 across six seeds --
a spread wider than every model difference anyone has reported on that split.

This runs each model over several seeds and reports mean, standard deviation
and range, so a difference between two models can be judged against the noise
of either. Two models whose intervals overlap are not distinguishable by this
evidence, whatever their single-seed numbers said.

The random split is included as a control. If its spread is small, the problem
is specific to the HLA split's fold sizes rather than to the training recipe,
and only HLA-split claims need restating.

Run:  python scripts/seed_variance.py [n_seeds]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.train_m1 as m1mod  # noqa: E402
import src.train_m2_scaled as m2mod  # noqa: E402
import src.train_m3 as m3mod  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/seed_variance.csv")

SPLITS = ["hla", "random"]
CONTROL_SEEDS = 3  # random is slow and only needs to answer "is it stable?"
DEFAULT_SEEDS = 5


def run_model(name: str, master, split, split_name: str, seed: int, stores) -> float:
    """One training run of one model at one seed, returning test Spearman."""
    pep_mean, hla_mean, pep_res, pep_lu, hla_res, hla_lu = stores

    if name in ("m1", "m1pep"):
        # train_one takes `seed` as a default argument, bound at definition
        # time, so patching the module global has no effect. Pass it.
        _, metrics, _ = m1mod.train_one(
            master, split, peptide_only=(name == "m1pep"), seed=seed
        )
    elif name in ("m2", "m2n"):
        m2mod.SEED = seed
        _, metrics = m2mod.train_one(
            master, split, pep_mean, hla_mean,
            peptide_only=False, scale=(name == "m2n"),
        )
    elif name == "m3":
        m3mod.SEED = seed
        _, metrics, _ = m3mod.train_one(
            master, split, pep_res, pep_lu, hla_res, hla_lu, split_name
        )
    else:
        raise ValueError(name)
    return float(metrics["spearman"])


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEEDS
    seeds = [42, 0, 1, 2, 3, 4, 5, 6][:n_seeds]

    master = pd.read_parquet(MASTER)
    stores = (
        m2mod.load_store("esm_peptide"), m2mod.load_store("esm_hla"),
        *m3mod.load_residue_store("esm_peptide"),
        *m3mod.load_residue_store("esm_hla"),
    )
    # load_residue_store returns (array, lookup); unpack into the order run_model wants
    pep_res, pep_lu, hla_res, hla_lu = stores[2], stores[3], stores[4], stores[5]
    stores = (stores[0], stores[1], pep_res, pep_lu, hla_res, hla_lu)

    rows = []
    for split_name in SPLITS:
        split = pd.read_parquet(SPLIT_DIR / f"{split_name}.parquet")
        for model in ["m1", "m1pep", "m2", "m2n", "m3"]:
            scores = []
            use = seeds if split_name == 'hla' else seeds[:CONTROL_SEEDS]
            for seed in use:
                try:
                    scores.append(run_model(model, master, split, split_name, seed, stores))
                except Exception as exc:
                    print(f"   {model} {split_name} seed {seed} failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            if not scores:
                continue
            a = np.array(scores)
            rows.append({
                "split": split_name, "model": model, "n_seeds": len(a),
                "mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                "spread": round(float(a.max() - a.min()), 4),
            })
            print(f"   {split_name:7s} {model:6s} mean={a.mean():+.3f} "
                  f"sd={a.std(ddof=1):.3f} range=[{a.min():+.3f}, {a.max():+.3f}]",
                  flush=True)

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("\n" + "=" * 68)
    for split_name in SPLITS:
        s = t[t.split == split_name]
        if s.empty:
            continue
        print(f"\n{split_name} split, {len(seeds)} seeds")
        print(f"   {'model':7s} {'mean':>8s} {'sd':>7s} {'range':>20s}")
        for _, r in s.sort_values("mean", ascending=False).iterrows():
            print(f"   {r.model:7s} {r['mean']:+8.3f} {r.sd:7.3f} "
                  f"  [{r['min']:+.3f}, {r['max']:+.3f}]")
        # Can the best be told apart from the rest?
        best = s.loc[s["mean"].idxmax()]
        others = s[s.model != best.model]
        overlap = others[others["max"] >= best["min"]].model.tolist()
        print(f"   best mean: {best.model}. Overlapping ranges: "
              f"{overlap if overlap else 'none - the ordering is resolvable'}")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
