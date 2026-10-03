"""Train M3 on all four splits and compare it to M1 and M2.

Unlike M1 and M2, this also saves the trained weights. Nothing on the Volume
or in the repo currently persists a model, so Phase G would otherwise have to
retrain from scratch to deploy an endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_m3 import run_all  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
WEIGHTS = Path("models")
SPLITS = ["random", "peptide", "cluster", "hla"]


def main() -> int:
    master = pd.read_parquet(MASTER)
    splits = {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet") for s in SPLITS}
    table = run_all(master, splits, save_weights=WEIGHTS)

    print("\nladder comparison (test-fold Spearman)")
    prior = {
        "m1": {"random": 0.765, "peptide": 0.642, "cluster": 0.629, "hla": -0.210},
        "m2": {"random": 0.634, "peptide": 0.543, "cluster": 0.529, "hla": 0.037},
    }
    print(f"   {'split':9s} {'M1':>7s} {'M2':>7s} {'M3':>7s} {'M3-best':>9s}")
    for s in SPLITS:
        row = table.loc[table.split == s]
        if row.empty:
            continue
        m3 = float(row.spearman.iloc[0])
        best_prior = max(prior["m1"][s], prior["m2"][s])
        print(f"   {s:9s} {prior['m1'][s]:7.3f} {prior['m2'][s]:7.3f} "
              f"{m3:7.3f} {m3 - best_prior:+9.3f}")

    print(f"\nweights saved to {WEIGHTS}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
