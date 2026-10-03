"""NetMHCstabpan baseline metrics per split and fold.

This is the incumbent number every model result is compared against, so it is
reported in the same shape as Dev 1's M1 table: Spearman with a bootstrap 95%
CI over 1,000 resamples, plus MAE in hours.

Two caveats travel with every number here and are printed with the table:

1. These rows ARE NetMHCstabpan's own training data. The scores are circular
   and will look unrealistically strong, especially on the random split. They
   are a reference point, not a fair baseline.
2. 1,135 rows (three engineered C67S variants) have no NetMHCstabpan score at
   all, because the tool has no pseudo-sequence for them. Coverage is reported
   per fold rather than silently dropped, since a baseline evaluated on a
   different row set from the model is not a comparison.

Run:  python scripts/nms_baseline_metrics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MASTER = Path("data/processed/master.parquet")
PREDS = Path("data/external/netmhcstabpan_preds.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/nms_baseline.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
FOLDS = ["train", "val", "test", "calib"]
SEED = 42
N_BOOT = 1000


def spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Spearman with a percentile bootstrap CI, matching Dev 1's M1 table."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        # A resample can be constant (tiny folds), which makes Spearman
        # undefined; carry it as NaN rather than letting it read as 0.
        boots[i] = (
            spearmanr(xs, ys).statistic
            if len(np.unique(xs)) > 1 and len(np.unique(ys)) > 1
            else np.nan
        )
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


def main() -> int:
    master = pd.read_parquet(MASTER)
    preds = pd.read_parquet(PREDS)
    data = master.merge(preds, on="id", how="left")

    rows = []
    for split in SPLITS:
        sp = pd.read_parquet(SPLIT_DIR / f"{split}.parquet")
        d = data.merge(sp, on="id")
        for fold in FOLDS:
            sub = d[d.fold == fold]
            scored = sub.dropna(subset=["nms_half_life"])
            if scored.empty:
                continue
            rho, lo, hi = spearman_ci(
                scored.half_life.to_numpy(float),
                scored.nms_half_life.to_numpy(float),
            )
            mae = float(np.abs(scored.half_life - scored.nms_half_life).mean())
            rows.append(
                {
                    "split": split,
                    "fold": fold,
                    "n_rows": len(sub),
                    "n_scored": len(scored),
                    "coverage": round(len(scored) / len(sub), 4),
                    "spearman": round(rho, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "mae_hours": round(mae, 3),
                }
            )

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    print("NetMHCstabpan vs measured half-life (IN-SAMPLE: these rows are its")
    print("own training data; treat as a reference point, not a fair baseline)\n")
    for split in SPLITS:
        t = table[table.split == split]
        if t.empty:
            continue
        print(f"{split}:")
        for _, r in t.iterrows():
            print(
                f"   {r.fold:6s} n={int(r.n_scored):6d}/{int(r.n_rows):<6d} "
                f"cov={r.coverage:5.1%}  "
                f"spearman={r.spearman:6.3f} [{r.ci_lo:6.3f}, {r.ci_hi:6.3f}]  "
                f"MAE={r.mae_hours:6.2f} h"
            )
        print()

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
