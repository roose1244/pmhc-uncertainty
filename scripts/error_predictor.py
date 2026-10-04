"""Predict the error instead of reading it off ensemble spread.

The ensemble's y_std is a poor uncertainty signal on its own: within an allele
it correlates with absolute error at only +0.050, and across held-out alleles
it does not respond to novelty at all (-0.055) because M1 receives the same
zero vector for every unseen allele.

Novelty has the opposite shape. Across held-out alleles it tracks error at
+0.546, but it is constant within an allele, so it cannot say which peptide on
that allele is harder.

Neither is sufficient alone and they fail in different directions, so this
trains a small gradient-boosted model to predict absolute error from both,
which is Dev 1's "error predictor" item. Its output is a per-row uncertainty
that can vary within an allele (via y_std and peptide novelty) and across
alleles (via HLA novelty and training density).

HONEST EVALUATION. The predictor is fitted on the calib fold only -- never on
train, which the stability model saw, and never on test, which it is judged
on. calib is reserved for exactly this kind of post-hoc fitting. The question
asked is whether predicted error ranks true error better than y_std does, on
test rows, which is the same err-unc Spearman Dev 1 reports for the ensemble.

A win here is not a better stability prediction. It is a better answer to
"should you trust this one", which is the project's actual claim.

Run:  python scripts/error_predictor.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MASTER = Path("data/processed/master.parquet")
NOVELTY = Path("data/features/novelty.parquet")
PRED_DIR = Path("results/predictions")
OUT = Path("results/tables/error_predictor.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
MODEL = "m1ens"
SEED = 42


def features_for(split: str) -> list[str]:
    """Per-split novelty columns, plus the ensemble's own spread."""
    return [
        "y_std",
        f"hla_novelty_{split}",
        f"hla_novelty_top3_{split}",
        f"hla_novelty_pseudo_{split}",
        f"hla_n_train_{split}",
        f"peptide_novelty_{split}",
    ]


def fit_predictor(train: pd.DataFrame, cols: list[str]):
    """Small LightGBM. Deliberately tiny: calib is ~2,300 rows.

    Falls back to a linear model if LightGBM cannot load, which on Apple
    Silicon means the OpenMP runtime is missing rather than the package.
    """
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
        )
        model.fit(train[cols], train.abs_err)
        return model, "lightgbm"
    except Exception as exc:  # pragma: no cover - environment dependent
        from sklearn.ensemble import HistGradientBoostingRegressor

        print(f"   (lightgbm unavailable: {type(exc).__name__}; using HistGradientBoosting)")
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            random_state=SEED,
        )
        model.fit(train[cols], train.abs_err)
        return model, "hist_gradient_boosting"


def main() -> int:
    master = pd.read_parquet(MASTER)[["id", "hla"]]
    nov = pd.read_parquet(NOVELTY)

    rows, importances, saved = [], [], []
    for split in SPLITS:
        path = PRED_DIR / f"{MODEL}_{split}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path).merge(nov, on="id").merge(master, on="id")
        d["abs_err"] = (d.y_true - d.y_mean).abs()

        cols = [c for c in features_for(split) if c in d.columns]
        calib = d[d.fold == "calib"].dropna(subset=cols + ["abs_err"])
        test = d[d.fold == "test"].dropna(subset=cols + ["abs_err"])
        if len(calib) < 100 or len(test) < 30:
            continue

        model, kind = fit_predictor(calib, cols)
        pred_err = model.predict(test[cols])

        # The incumbent signal, and the new one, judged the same way.
        r_std = float(spearmanr(test.y_std, test.abs_err).statistic)
        r_hat = float(spearmanr(pred_err, test.abs_err).statistic)

        # Also report novelty alone, so any gain is attributable rather than
        # just "the ensemble of features did something".
        nov_col = f"hla_novelty_{split}"
        pep_col = f"peptide_novelty_{split}"
        r_nov = (
            float(spearmanr(test[nov_col], test.abs_err).statistic)
            if test[nov_col].nunique() > 1
            else float("nan")
        )
        r_pep = (
            float(spearmanr(test[pep_col], test.abs_err).statistic)
            if pep_col in test and test[pep_col].nunique() > 1
            else float("nan")
        )

        rows.append(
            {
                "split": split,
                "fitted_on": len(calib),
                "tested_on": len(test),
                "model": kind,
                "err_unc_y_std": round(r_std, 4),
                "err_unc_hla_novelty": round(r_nov, 4) if r_nov == r_nov else "",
                "err_unc_peptide_novelty": round(r_pep, 4) if r_pep == r_pep else "",
                "err_unc_predicted": round(r_hat, 4),
                "gain_over_y_std": round(r_hat - r_std, 4),
            }
        )
        saved.append(pd.DataFrame({"id": test["id"].to_numpy(), "split": split, "pred_err": pred_err}))

        if hasattr(model, "feature_importances_"):
            imp = dict(zip(cols, model.feature_importances_))
            total = sum(imp.values()) or 1
            importances.append(
                {"split": split, **{k: round(v / total, 3) for k, v in imp.items()}}
            )

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("ERROR PREDICTOR vs ENSEMBLE SPREAD")
    print("Spearman of each signal against true absolute error, test folds.")
    print("Fitted on calib only; test never seen.\n")
    print(f"   {'split':9s} {'calib':>6s} {'test':>6s} {'y_std':>8s} "
          f"{'hla_nov':>8s} {'pep_nov':>8s} {'predicted':>10s} {'gain':>8s}")
    for _, r in t.iterrows():
        def cell(value: object) -> str:
            return f"{'const':>8}" if value == "" else f"{float(value):8.3f}"

        print(f"   {r.split:9s} {int(r.fitted_on):6d} {int(r.tested_on):6d} "
              f"{r.err_unc_y_std:8.3f} {cell(r.err_unc_hla_novelty)} "
              f"{cell(r.err_unc_peptide_novelty)} {r.err_unc_predicted:10.3f} "
              f"{r.gain_over_y_std:+8.3f}")

    if importances:
        print("\nfeature importance (share of total split gain)")
        imp = pd.DataFrame(importances).set_index("split").fillna(0)
        print(imp.to_string())

    if saved:
        pred_path = PRED_DIR / "errpred_m1ens.parquet"
        pd.concat(saved, ignore_index=True).to_parquet(pred_path, index=False)
        print(f"wrote {pred_path}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
