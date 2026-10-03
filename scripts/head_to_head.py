"""Our model against NetMHCstabpan on exactly the same rows, and are they complementary?

Phase E. Two questions the project has to answer about the incumbent:

1. On identical test rows, who wins? Comparing published headline numbers is
   not enough, because NetMHCstabpan cannot score the three engineered C67S
   variants at all. Dropping those rows from one side and not the other would
   flatter whichever model kept them, so both are restricted to the rows where
   both produce a number.

2. Is the incumbent complementary, or redundant? If the two disagree in
   useful places, a combination beats either. If the ensemble is simply worse
   everywhere, the honest conclusion is that our contribution is the
   uncertainty, not the point prediction.

The circularity caveat stands throughout and is not a detail: 99.5% of these
rows are in NetMHCstabpan's own training set (scripts/nms_in_sample_check.py),
so it is being tested on data it was fitted on and our model is not. This is
not a fair fight and the table should never be shown without saying so.

Run:  python scripts/head_to_head.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MASTER = Path("data/processed/master.parquet")
NMS = Path("data/external/netmhcstabpan_preds.parquet")
SPLIT_DIR = Path("data/splits")
PRED_DIR = Path("results/predictions")
OUT = Path("results/tables/head_to_head.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
MODEL = "m1ens"
N_BOOT = 1000
SEED = 42


def boot_spearman_diff(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float, float]:
    """Bootstrap CI for rho(y,a) - rho(y,b) on paired rows.

    Paired because both models see identical rows, so the resample must take
    the same indices for both; otherwise the CI is far too wide.
    """
    rng = np.random.default_rng(SEED)
    out = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            out[i] = np.nan
            continue
        out[i] = spearmanr(ys, a[idx]).statistic - spearmanr(ys, b[idx]).statistic
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def main() -> int:
    master = pd.read_parquet(MASTER)
    nms = pd.read_parquet(NMS)

    rows = []
    for split in SPLITS:
        sp = pd.read_parquet(SPLIT_DIR / f"{split}.parquet")
        pred_path = PRED_DIR / f"{MODEL}_{split}.parquet"
        if not pred_path.exists():
            continue
        pred = pd.read_parquet(pred_path)

        d = (
            master[["id", "half_life", "log_half_life"]]
            .merge(nms, on="id", how="left")
            .merge(sp, on="id")
            .merge(pred[["id", "y_mean"]], on="id")
        )
        test = d[d.fold == "test"]
        both = test.dropna(subset=["nms_half_life", "y_mean"])
        if len(both) < 30:
            continue

        y = both.log_half_life.to_numpy(float)
        ours = both.y_mean.to_numpy(float)
        # NetMHCstabpan reports hours; put it on the same monotone scale so
        # Spearman is comparing like with like (rank-invariant either way,
        # but it keeps the MAE below honest).
        theirs_log = np.log10(both.nms_half_life.to_numpy(float) + 0.1)

        r_ours = float(spearmanr(y, ours).statistic)
        r_theirs = float(spearmanr(y, theirs_log).statistic)
        lo, hi = boot_spearman_diff(y, ours, theirs_log)

        # Complementarity: does averaging the two beat the better one alone?
        # Averaged as ranks, since the two are on different calibrations.
        blend = (
            pd.Series(ours).rank().to_numpy() + pd.Series(theirs_log).rank().to_numpy()
        )
        r_blend = float(spearmanr(y, blend).statistic)

        rows.append(
            {
                "split": split,
                "n_test": len(test),
                "n_scored_by_both": len(both),
                "dropped": len(test) - len(both),
                "spearman_ours": round(r_ours, 4),
                "spearman_nms": round(r_theirs, 4),
                "diff": round(r_ours - r_theirs, 4),
                "diff_ci_lo": round(lo, 4),
                "diff_ci_hi": round(hi, 4),
                "spearman_blend": round(r_blend, 4),
                "blend_gain": round(r_blend - max(r_ours, r_theirs), 4),
                "corr_between_models": round(
                    float(spearmanr(ours, theirs_log).statistic), 4
                ),
            }
        )

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("HEAD TO HEAD on identical test rows (m1ens vs NetMHCstabpan)")
    print("CAVEAT: 99.5% of these rows are NetMHCstabpan's own training data.\n")
    print(f"{'split':9s} {'n both':>7s} {'drop':>5s} {'ours':>7s} {'NMS':>7s} "
          f"{'diff':>7s} {'95% CI':>18s} {'blend':>7s} {'gain':>7s} {'r(a,b)':>7s}")
    for _, r in t.iterrows():
        print(
            f"{r.split:9s} {int(r.n_scored_by_both):7d} {int(r.dropped):5d} "
            f"{r.spearman_ours:7.3f} {r.spearman_nms:7.3f} {r['diff']:7.3f} "
            f"  [{r.diff_ci_lo:6.3f}, {r.diff_ci_hi:6.3f}] "
            f"{r.spearman_blend:7.3f} {r.blend_gain:7.3f} {r.corr_between_models:7.3f}"
        )

    print("\ndiff = ours - NMS; a CI excluding 0 is a real difference on these rows.")
    print("gain = blend minus the better single model; >0 means complementary.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
