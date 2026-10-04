"""Compare HLA representations: whole groove against the 34 contact residues.

Runs mean182 / mean34 / attn182 / attn34 on every split, then repeats the two
attention variants over several seeds on the HLA split, where a single run
cannot resolve differences smaller than about 0.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_contact import contact_positions, load_residue, train_one  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/contact_residues.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
REPS = ["mean182", "mean34", "attn182", "attn34"]
SEEDS = [42, 0, 1, 2, 3]
M1 = {"random": 0.765, "peptide": 0.642, "cluster": 0.629, "hla": -0.113}


def main() -> int:
    master = pd.read_parquet(MASTER)
    pep_arr, pep_lu = load_residue("esm_peptide")
    hla_arr, hla_lu = load_residue("esm_hla")
    print(f"contact positions: {len(contact_positions())} of {hla_arr.shape[1]}\n")

    rows = []
    for split_name in SPLITS:
        split = pd.read_parquet(SPLIT_DIR / f"{split_name}.parquet")
        for rep in REPS:
            _, m = train_one(master, split, rep, pep_arr, pep_lu, hla_arr, hla_lu)
            rows.append({"split": split_name, "rep": rep, "seed": 42, **m})
            print(f"  {split_name:8s} {rep:8s} spearman={m['spearman']:+.3f}  "
                  f"MAE={m['mae_hours']:.2f}h  epoch={m['best_epoch']}", flush=True)

    # The HLA split needs seeds before any of its differences mean anything.
    print("\nHLA split, attention variants over seeds:")
    split = pd.read_parquet(SPLIT_DIR / "hla.parquet")
    for rep in ["attn182", "attn34"]:
        scores = []
        for seed in SEEDS:
            _, m = train_one(master, split, rep, pep_arr, pep_lu,
                             hla_arr, hla_lu, seed=seed)
            scores.append(m["spearman"])
            rows.append({"split": "hla", "rep": rep, "seed": seed, **m})
        a = np.array(scores)
        print(f"  {rep:8s} mean={a.mean():+.3f} sd={a.std(ddof=1):.3f} "
              f"range=[{a.min():+.3f}, {a.max():+.3f}]", flush=True)

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("\ndoes restricting to contact residues help? (seed 42)")
    print(f"   {'split':9s} {'mean182':>9s} {'mean34':>9s} {'d':>7s} "
          f"{'attn182':>9s} {'attn34':>9s} {'d':>7s} {'M1':>8s}")
    s42 = t[t.seed == 42]
    for sp in SPLITS:
        g = s42[s42.split == sp].set_index("rep").spearman
        if len(g) < 4:
            continue
        print(f"   {sp:9s} {g['mean182']:9.3f} {g['mean34']:9.3f} "
              f"{g['mean34'] - g['mean182']:+7.3f} {g['attn182']:9.3f} "
              f"{g['attn34']:9.3f} {g['attn34'] - g['attn182']:+7.3f} {M1[sp]:8.3f}")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
