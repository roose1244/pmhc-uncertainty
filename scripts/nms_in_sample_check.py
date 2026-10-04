"""Which of our alleles was NetMHCstabpan actually trained on, and does it matter?

The README dismisses the NetMHCstabpan baseline wholesale as circular, because
master.parquet is its published training table. The tool's own data directory
ships `training.pseudo`, which names the 68 alleles it was fitted on, so the
circularity can be quantified instead of assumed -- and the handful of alleles
outside that list are the only rows where the comparison is honest.

The obvious comparison is confounded. Every allele inside its training set has
at least 220 measurements here; every allele outside it has at most 32. The
ranges do not overlap at all, so a raw "in 0.824 vs out 0.558" gap cannot be
read as evidence of anything -- a Spearman over 16 points is attenuated and
noisy whatever the model.

So the test is a subsampling control: draw an in-training allele, cut it down
to exactly the size of the out-of-training allele being judged, and recompute
Spearman. Two thousand draws give a null distribution of "what a same-sized
in-training allele looks like". Where the real allele falls in that null is a
one-sided p-value, and the seven are combined because each alone is
underpowered.

Run:  python scripts/nms_in_sample_check.py [path/to/training.pseudo]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, combine_pvalues, spearmanr

MASTER = Path("data/processed/master.parquet")
PREDS = Path("data/external/netmhcstabpan_preds.parquet")
DEFAULT_TRAINING = Path.home() / "tools/netMHCstabpan-1.0/data/training.pseudo"
OUT = Path("results/tables/nms_in_sample.csv")

N_DRAWS = 2000
SEED = 42


def dtu_name(hla: str) -> str:
    """HLA-B*14:01(C67S) -> HLA-B14:01, which is how the tool names alleles."""
    return re.sub(r"\(.*\)$", "", hla).replace("*", "")


def main() -> int:
    training_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRAINING
    if not training_path.exists():
        print(f"training.pseudo not found at {training_path}")
        print("It ships in the NetMHCstabpan data directory; pass the path as an arg.")
        return 1

    trained = {
        line.split()[0] for line in training_path.read_text().splitlines() if line.split()
    }

    master = pd.read_parquet(MASTER)
    preds = pd.read_parquet(PREDS)
    d = master.merge(preds, on="id").dropna(subset=["nms_half_life"])
    d["in_tr"] = d.hla.map(lambda h: dtu_name(h) in trained)

    n_in = int(d[d.in_tr].hla.nunique())
    n_out = int(d[~d.in_tr].hla.nunique())
    print(f"NetMHCstabpan training alleles listed: {len(trained)}")
    print(f"our alleles in its training set : {n_in} ({int(d.in_tr.sum())} rows)")
    print(f"our alleles outside it          : {n_out} ({int((~d.in_tr).sum())} rows)")
    print(f"=> {100 * d.in_tr.mean():.1f}% of scored rows are in-sample for the tool\n")

    per = (
        d.groupby("hla")
        .apply(
            lambda x: pd.Series(
                {
                    "n": len(x),
                    "rho": spearmanr(x.half_life, x.nms_half_life).statistic,
                    "in_tr": bool(x.in_tr.iloc[0]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    lo_in, hi_out = per[per.in_tr].n.min(), per[~per.in_tr].n.max()
    print(f"CONFOUND: smallest in-training allele n={int(lo_in)}, "
          f"largest out-of-training n={int(hi_out)}")
    print("  the ranges do not overlap, so raw group means are uninterpretable\n")

    rng = np.random.default_rng(SEED)
    pools = [g for _, g in d[d.in_tr].groupby("hla")]
    rows = []
    print("subsampling control, 2000 draws per allele:")
    print(f"  {'allele':14s} {'n':>4s} {'rho':>7s} {'null med':>9s} "
          f"{'null 5-95%':>18s} {'pctile':>8s}")
    for _, r in per[~per.in_tr].sort_values("rho").iterrows():
        n = int(r.n)
        draws = []
        for _ in range(N_DRAWS):
            g = pools[rng.integers(len(pools))]
            s = g.sample(n, random_state=int(rng.integers(1 << 31)))
            if s.half_life.nunique() > 1 and s.nms_half_life.nunique() > 1:
                draws.append(spearmanr(s.half_life, s.nms_half_life).statistic)
        draws = np.asarray(draws)
        pct = float((draws < r.rho).mean())
        lo, hi = np.percentile(draws, [5, 95])
        rows.append({"hla": r.hla, "n": n, "rho": round(float(r.rho), 4),
                     "null_median": round(float(np.median(draws)), 4),
                     "percentile": round(pct, 4)})
        print(f"  {r.hla:14s} {n:4d} {r.rho:7.3f} {np.median(draws):9.3f} "
              f"  [{lo:6.3f}, {hi:6.3f}] {100 * pct:7.1f}%")

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    p = res.percentile.to_numpy(float)
    # Clip away exact zeros so Fisher's -2*sum(log p) stays finite when an
    # allele falls below every one of the 2000 draws.
    p = np.clip(p, 1.0 / N_DRAWS, 1.0)
    print("\ncombined over the 7 alleles (percentiles are uniform under the null):")
    print(f"  mean percentile      = {p.mean():.3f}  (0.50 expected)")
    print(f"  Fisher combined p    = {combine_pvalues(p, method='fisher').pvalue:.5f}")
    print(f"  Stouffer combined p  = {combine_pvalues(p, method='stouffer').pvalue:.5f}")
    k = int((p < 0.5).sum())
    print(f"  sign test            = {k}/{len(p)}, "
          f"p = {binomtest(k, len(p), 0.5, alternative='greater').pvalue:.4f}")
    bonf = 0.05 / len(p)
    print(f"  survives Bonferroni ({bonf:.4f}): "
          f"{[r.hla for _, r in res.iterrows() if r.percentile < bonf] or 'none'}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
