"""Can a hybrid beat M1 everywhere, including where M1 currently wins?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_hybrid import RECIPES, load_mean, train_one  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/hybrid.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
SEEDS = [42, 0, 1, 2, 3]
# M1 at seed 42 for the stable splits; seed-averaged for hla, where one run
# cannot resolve differences below about 0.1.
M1 = {"random": 0.765, "peptide": 0.642, "cluster": 0.629, "hla": -0.113}


def main() -> int:
    master = pd.read_parquet(MASTER)
    pep_store = load_mean("esm_peptide")
    hla_store = load_mean("esm_hla")

    rows = []
    for split_name in SPLITS:
        split = pd.read_parquet(SPLIT_DIR / f"{split_name}.parquet")
        for name in RECIPES:
            if split_name == "hla":
                scores = []
                for seed in SEEDS:
                    _, m = train_one(master, split, name, pep_store, hla_store, seed)
                    scores.append(m["spearman"])
                    rows.append({"split": split_name, "model": name, "seed": seed, **m})
                a = np.array(scores)
                print(f"  {split_name:8s} {name:20s} mean={a.mean():+.3f} "
                      f"sd={a.std(ddof=1):.3f} range=[{a.min():+.3f}, {a.max():+.3f}]",
                      flush=True)
            else:
                _, m = train_one(master, split, name, pep_store, hla_store)
                rows.append({"split": split_name, "model": name, "seed": 42, **m})
                print(f"  {split_name:8s} {name:20s} spearman={m['spearman']:+.3f}  "
                      f"MAE={m['mae_hours']:.2f}h  dims={m['n_in']}  "
                      f"epoch={m['best_epoch']}", flush=True)

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("\nagainst M1")
    print(f"   {'split':9s} " + " ".join(f"{n[:18]:>19s}" for n in RECIPES) + f" {'M1':>8s}")
    for sp in SPLITS:
        cells = []
        for n in RECIPES:
            g = t[(t.split == sp) & (t.model == n)].spearman
            v = float(g.mean())
            mark = "*" if v > M1[sp] else " "
            cells.append(f"{v:+18.3f}{mark}")
        print(f"   {sp:9s} " + " ".join(cells) + f" {M1[sp]:+8.3f}")
    print("\n   * = beats M1 on that split")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
