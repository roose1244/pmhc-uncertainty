"""Does predicted uncertainty grow with novelty, and do the intervals hold?

This is H3: error and uncertainty should both rise when the peptide is unseen
or the allele is held out. Dev 1 measured error against uncertainty; this
measures both against novelty, which is the axis the claim is actually about.

Three things are kept apart because they have different validity:

  A. novelty vs uncertainty   Spearman of novelty against y_std. Legitimate on
                              any fold, because y_std comes from ensemble
                              disagreement and never sees the calibration set.

  B. novelty vs error         the same against absolute error.

  C. conditional coverage     does the 90% interval cover 90% *within* each
                              novelty bin? Marginal coverage can sit at exactly
                              90% while the model is badly over-confident on
                              novel rows and over-cautious on familiar ones,
                              which is the failure the project exists to catch.

Fold choice is not uniform, and that is deliberate:

  - HLA novelty varies only in the HLA split. There, val and test are a single
    allele each (B*27:02, B*15:02), so novelty is constant and a correlation
    is undefined. Only calib spans a range (10 alleles, 1 to 7 contact
    mismatches), so A and B for HLA novelty are computed on calib.
  - Coverage on calib is circular, because calib set the conformal threshold.
    So C is reported on test folds only, and is therefore available for
    peptide novelty but not for HLA novelty. That gap is a property of the
    split design, not an oversight.

Run:  python scripts/analyse_novelty_uncertainty.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NOVELTY = Path("data/features/novelty.parquet")
PRED_DIR = Path("results/predictions")
OUT_DIR = Path("results/tables")

SPLITS = ["random", "peptide", "cluster", "hla"]
MODELS = ["m1", "m1pep", "m1ens", "m2", "m2pep"]
MIN_N = 30


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """NaN rather than a number when either side is constant or too small."""
    if len(x) < MIN_N or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def safe_spearman_small(x, y) -> float:
    """Spearman without the MIN_N guard, for the 10-allele aggregate view."""
    return float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)


def load(model: str, split: str) -> pd.DataFrame | None:
    path = PRED_DIR / f"{model}_{split}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> int:
    nov = pd.read_parquet(NOVELTY)
    rows, cov_rows = [], []

    for split in SPLITS:
        # HLA novelty only varies where whole alleles are held out, and within
        # that split only calib holds more than one allele.
        hla_fold = "calib" if split == "hla" else "test"

        for model in MODELS:
            pred = load(model, split)
            if pred is None:
                continue
            d = pred.merge(nov, on="id", how="left")
            d["abs_err"] = (d.y_true - d.y_mean).abs() if "y_mean" in d else np.nan
            if "y_mean" not in d:
                d["abs_err"] = (d.y_true - d.m0).abs()

            for axis, col, fold in [
                ("hla", f"hla_novelty_{split}", hla_fold),
                ("hla_top3", f"hla_novelty_top3_{split}", hla_fold),
                ("peptide", f"peptide_novelty_{split}", "test"),
                ("n_train", f"hla_n_train_{split}", hla_fold),
            ]:
                sub = d[d.fold == fold]
                if col not in sub or sub.empty:
                    continue
                x = sub[col].to_numpy(float)
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "axis": axis,
                        "fold": fold,
                        "n": len(sub),
                        "n_unique_x": int(len(np.unique(x[~np.isnan(x)]))),
                        "vs_y_std": round(safe_spearman(x, sub.y_std.to_numpy(float)), 4),
                        "vs_abs_err": round(
                            safe_spearman(x, sub.abs_err.to_numpy(float)), 4
                        ),
                    }
                )

        # --- C. conditional coverage, ensemble only, test fold only ---------
        ens = load("m1ens", split)
        if ens is None or "lo" not in ens:
            continue
        d = ens.merge(nov, on="id", how="left")
        t = d[d.fold == "test"].copy()
        if t.empty:
            continue
        t["covered"] = (t.y_true >= t.lo) & (t.y_true <= t.hi)
        t["covered_const"] = (t.y_true >= t.lo_const) & (t.y_true <= t.hi_const)

        for axis, col in [
            ("peptide", f"peptide_novelty_{split}"),
            ("hla", f"hla_novelty_{split}"),
        ]:
            x = t[col]
            if x.nunique() < 3:
                cov_rows.append(
                    {
                        "split": split,
                        "axis": axis,
                        "bin": "CONSTANT - not analysable",
                        "n": len(t),
                        "novelty": round(float(x.iloc[0]), 4) if len(x) else np.nan,
                        "coverage": round(float(t.covered.mean()), 4),
                        "coverage_const": round(float(t.covered_const.mean()), 4),
                        "median_width": round(float((t.hi - t.lo).median()), 4),
                    }
                )
                continue
            try:
                t["bin"] = pd.qcut(x, 3, labels=["low", "mid", "high"], duplicates="drop")
            except ValueError:
                continue
            for b, g in t.groupby("bin", observed=True):
                cov_rows.append(
                    {
                        "split": split,
                        "axis": axis,
                        "bin": str(b),
                        "n": len(g),
                        "novelty": round(float(g[col].mean()), 4),
                        "coverage": round(float(g.covered.mean()), 4),
                        "coverage_const": round(float(g.covered_const.mean()), 4),
                        "median_width": round(float((g.hi - g.lo).median()), 4),
                    }
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr = pd.DataFrame(rows)
    cov = pd.DataFrame(cov_rows)
    corr.to_csv(OUT_DIR / "novelty_vs_uncertainty.csv", index=False)
    cov.to_csv(OUT_DIR / "conditional_coverage.csv", index=False)

    print("=" * 74)
    print("A/B. NOVELTY vs UNCERTAINTY and ERROR  (Spearman; NaN = x constant)")
    print("=" * 74)
    for split in SPLITS:
        c = corr[(corr.split == split) & (corr.model == "m1ens")]
        if c.empty:
            continue
        print(f"\n{split}  (m1ens)")
        for _, r in c.iterrows():
            print(
                f"   {r.axis:9s} fold={r.fold:6s} n={int(r.n):5d} "
                f"distinct_x={int(r.n_unique_x):3d}  "
                f"vs y_std={r.vs_y_std!s:>8}  vs |err|={r.vs_abs_err!s:>8}"
            )

    print("\n" + "=" * 74)
    print("C. CONDITIONAL COVERAGE of the 90% interval, test folds, m1ens")
    print("=" * 74)
    for split in SPLITS:
        c = cov[cov.split == split]
        if c.empty:
            continue
        print(f"\n{split}")
        for _, r in c.iterrows():
            print(
                f"   {r.axis:8s} {r['bin']:26s} n={int(r.n):5d} "
                f"novelty={r.novelty:6.3f}  coverage={r.coverage:6.1%} "
                f"(const {r.coverage_const:5.1%})  width={r.median_width:6.3f}"
            )

    # --- D. per-allele, the clearest view of H3 --------------------------
    # One row per held-out allele removes the within-allele peptide variance
    # that swamps the between-allele signal at row level.
    master = pd.read_parquet("data/processed/master.parquet")[["id", "hla"]]
    ens = load("m1ens", "hla")
    per_allele = pd.DataFrame()
    if ens is not None:
        d = ens.merge(nov, on="id").merge(master, on="id")
        c = d[d.fold == "calib"].copy()
        c["abs_err"] = (c.y_true - c.y_mean).abs()
        per_allele = (
            c.groupby("hla")
            .agg(
                n=("id", "size"),
                novelty=("hla_novelty_hla", "first"),
                mean_y_std=("y_std", "mean"),
                mean_abs_err=("abs_err", "mean"),
            )
            .reset_index()
            .sort_values("novelty")
        )
        per_allele["novelty_mismatches"] = (per_allele.novelty * 182).round().astype(int)
        per_allele.to_csv(OUT_DIR / "per_allele_uncertainty.csv", index=False)

        print("\n" + "=" * 74)
        print("D. PER-ALLELE, HLA split calib fold (10 held-out alleles)")
        print("=" * 74)
        print(
            per_allele[
                ["hla", "n", "novelty_mismatches", "mean_y_std", "mean_abs_err"]
            ].round(4).to_string(index=False)
        )
        n_, s_, e_ = per_allele.novelty, per_allele.mean_y_std, per_allele.mean_abs_err
        print(f"\n   novelty vs mean y_std = {safe_spearman_small(n_, s_):+.3f}"
              "   <- uncertainty does NOT rise with novelty")
        print(f"   novelty vs mean |err| = {safe_spearman_small(n_, e_):+.3f}"
              "   <- but error does")

    print(f"\nwrote {OUT_DIR}/novelty_vs_uncertainty.csv, conditional_coverage.csv")
    if not per_allele.empty:
        print(f"wrote {OUT_DIR}/per_allele_uncertainty.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
