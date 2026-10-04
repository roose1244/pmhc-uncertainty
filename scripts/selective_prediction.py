"""Can PepShield pick out which individual predictions are wrong?

"This allele is unseen, so be careful" is a lookup, not a finding: a tool
always knows its own training set, and the warning is available without any
model. The question that is not tautological is row-level -- among predictions
the model is equally entitled to make, can it rank the wrong ones to the top?

Selective prediction is the instrument. Rank the test rows by a confidence
signal, discard the least confident, and measure error on what is kept. A
signal that carries information makes the retained error fall faster than
discarding at random. Two reference curves bound what is achievable:

    random   discard at random; no information
    oracle   discard by TRUE absolute error; perfect information, unreachable

The headline is the fraction of the oracle's gain a signal captures, which is
comparable across splits with different base error rates. Four signals:

    y_std       ensemble disagreement, the incumbent
    pred_err    the fitted error predictor (y_std + novelty features)
    novelty     HLA novelty alone, the tautological warning
    nms_rank    NetMHCstabpan's own %rank, a naive confidence proxy

The four splits are four KINDS of novelty, which is the more interesting
comparison than any single number: random is in-distribution, peptide holds
out unseen sequences, cluster holds out whole peptide families, and hla holds
out entire alleles. A signal that works in-distribution and fails under shift
is a signal that cannot be trusted where it matters.

Run:  python scripts/selective_prediction.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MASTER = Path("data/processed/master.parquet")
NOVELTY = Path("data/features/novelty.parquet")
NMS = Path("data/external/netmhcstabpan_preds.parquet")
PRED_DIR = Path("results/predictions")
OUT = Path("results/tables/selective_prediction.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
KIND = {"random": "none (in-distribution)", "peptide": "unseen peptide",
        "cluster": "unseen peptide family", "hla": "unseen allele"}
COVERAGES = [1.0, 0.8, 0.5, 0.2]
SEED = 42


def to_hours(x):
    return np.maximum(10.0**np.asarray(x, float) - 0.1, 0.0)


def mae_at(err_hours: np.ndarray, score: np.ndarray, cov: float) -> float:
    """MAE over the `cov` fraction with the LOWEST score (most confident)."""
    n = max(1, int(round(cov * len(err_hours))))
    keep = np.argsort(score, kind="stable")[:n]
    return float(err_hours[keep].mean())


def curve(err_hours: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {f"mae_{int(c*100)}": mae_at(err_hours, score, c) for c in COVERAGES}


def aurc(err_hours: np.ndarray, score: np.ndarray) -> float:
    """Area under the risk-coverage curve; lower is better."""
    order = np.argsort(score, kind="stable")
    risks = np.cumsum(err_hours[order]) / np.arange(1, len(order) + 1)
    return float(risks.mean())


def main() -> int:
    master = pd.read_parquet(MASTER)[["id", "hla"]]
    nov = pd.read_parquet(NOVELTY)
    nms = pd.read_parquet(NMS)
    rng = np.random.default_rng(SEED)

    rows = []
    for split in SPLITS:
        path = PRED_DIR / f"m1ens_{split}.parquet"
        if not path.exists():
            continue
        d = (pd.read_parquet(path).merge(nov, on="id")
             .merge(master, on="id").merge(nms[["id", "nms_rank"]], on="id", how="left"))
        d["abs_err"] = (d.y_true - d.y_mean).abs()
        d["err_h"] = np.abs(to_hours(d.y_true) - to_hours(d.y_mean))

        cols = [c for c in ["y_std", f"hla_novelty_{split}",
                            f"hla_novelty_top3_{split}", f"hla_novelty_pseudo_{split}",
                            f"hla_n_train_{split}", f"peptide_novelty_{split}"]
                if c in d.columns]
        calib = d[d.fold == "calib"].dropna(subset=cols + ["abs_err"])
        test = d[d.fold == "test"].dropna(subset=cols + ["abs_err"]).copy()
        if len(calib) < 100 or len(test) < 50:
            continue

        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=15, learning_rate=0.05,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbose=-1).fit(calib[cols], calib.abs_err)

        err = test.err_h.to_numpy()
        signals = {
            "y_std": test.y_std.to_numpy(float),
            "pred_err": model.predict(test[cols]),
            "novelty": test[f"hla_novelty_{split}"].to_numpy(float),
            "nms_rank": test.nms_rank.to_numpy(float),
            "random": rng.random(len(test)),
            "oracle": err,
        }

        base = {k: v for k, v in signals.items()}
        ref_rand = mae_at(err, base["random"], 0.5)
        ref_orac = mae_at(err, base["oracle"], 0.5)

        for name, score in signals.items():
            if np.all(np.isnan(score)):
                continue
            s = np.where(np.isnan(score), np.nanmax(score), score)
            c = curve(err, s)
            gain = ref_rand - c["mae_50"]
            best = ref_rand - ref_orac
            rows.append({
                "split": split, "kind": KIND[split], "signal": name,
                "n_test": len(test), **{k: round(v, 3) for k, v in c.items()},
                "aurc": round(aurc(err, s), 3),
                "pct_of_oracle": round(100 * gain / best, 1) if best > 0 else np.nan,
            })

    t = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print("SELECTIVE PREDICTION: can the signal rank wrong predictions to the top?")
    print("MAE in hours over the most-confident fraction. "
          "'% of oracle' is the share of achievable gain at 50% coverage.\n")
    for split in SPLITS:
        s = t[t.split == split]
        if s.empty:
            continue
        print(f"{split}  ({KIND[split]}, n={int(s.n_test.iloc[0])})")
        print(f"   {'signal':10s} {'keep100':>8s} {'keep80':>8s} {'keep50':>8s} "
              f"{'keep20':>8s} {'AURC':>7s} {'% of oracle':>12s}")
        for _, r in s.iterrows():
            print(f"   {r.signal:10s} {r.mae_100:8.3f} {r.mae_80:8.3f} "
                  f"{r.mae_50:8.3f} {r.mae_20:8.3f} {r.aurc:7.3f} "
                  f"{r.pct_of_oracle:11.1f}%")
        print()

    print("summary: % of oracle gain captured, by kind of novelty")
    piv = t[~t.signal.isin(["oracle", "random"])].pivot(
        index="signal", columns="split", values="pct_of_oracle")
    print(piv.reindex(columns=SPLITS).to_string())
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
