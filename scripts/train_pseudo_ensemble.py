"""Five-member ensemble of the 34-residue pocket model, then split conformal.

The quantile is fit on the calibration fold only. On the HLA split that fold
is other held-out alleles, so the B*15:02 interval was not tuned on B*15:02.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH
from src.splits import SPLITS_DIR
from src.train_pseudo import run_ensemble


def _allele_coverage(name: str) -> None:
    if name != "hla":
        return
    preds = pd.read_parquet(Path("results/predictions") / f"mpocketens_{name}.parquet")
    master = pd.read_parquet(MASTER_PATH)[["id", "hla"]]
    frame = preds.merge(master, on="id")
    print("\nHLA-split coverage by allele (quantile fit on calib, not per allele)")
    print(f"{'fold':6} {'allele':22} {'n':>5} {'cover':>7} {'spearman':>9} {'err-unc':>8}")
    for fold in ("calib", "test"):
        part = frame.loc[frame["fold"] == fold]
        for hla, group in part.groupby("hla"):
            y = group["y_true"].to_numpy()
            covered = float(np.mean((y >= group["lo"]) & (y <= group["hi"])))
            rho = float("nan")
            err = float("nan")
            if len(group) >= 20 and group["y_true"].nunique() >= 3:
                rho = float(spearmanr(group["y_true"], group["y_mean"]).statistic)
                err = float(
                    spearmanr(group["y_std"], np.abs(group["y_true"] - group["y_mean"])).statistic
                )
            print(f"{fold:6} {hla:22} {len(group):5d} {covered:7.3f} {rho:9.3f} {err:8.3f}")


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    splits = {
        name: pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
        for name in ("random", "peptide", "cluster", "hla")
    }
    table = run_ensemble(master, splits)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _allele_coverage("hla")


if __name__ == "__main__":
    main()
