"""Can novelty-scaled conformal intervals fix conditional coverage?

The ensemble's y_std does not rise with HLA novelty (see
analyse_novelty_uncertainty.py), so intervals scaled by y_std are the same
width on a familiar allele and a distant one. Novelty, unlike y_std, does
track error (+0.546 across held-out alleles). This asks whether substituting
novelty for y_std as the conformal scaling term restores conditional coverage.

Split conformal: with a per-row scale s_i, calibrate
q = Quantile_{1-alpha}(|y - y_hat| / s_i) on held-out data, then the interval
is y_hat +/- q * s_i. Marginal coverage is guaranteed for any s_i; the choice
of s_i only decides *where* the coverage comes from. A good s_i buys
conditional coverage -- roughly 90% within every subgroup, not just overall.

Four scalings are compared:

    const     s = 1                      Dev 1's q_const
    y_std     s = ensemble std           Dev 1's q_norm, the incumbent
    novelty   s = E[|err|] from novelty  fitted, no y_std at all
    combined  s = E[|err|] from both

EVALUATION DESIGN. The honest test of "does this generalise to an allele we
have never seen" is leave-one-allele-out: fit the scale model and the
conformal quantile on nine held-out alleles, then evaluate coverage on the
tenth, and rotate. Calibrating on calib and evaluating on the HLA split's
test fold would not work, because that fold is the single allele B*15:02 and
its novelty is constant -- there is nothing to be conditional across.

The metric is mean |coverage - 0.90| across alleles. Marginal coverage is
near 90% for every scaling by construction, so it cannot separate them; the
spread across alleles is what does.

Run:  python scripts/novelty_conformal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

NOVELTY = Path("data/features/novelty.parquet")
MASTER = Path("data/processed/master.parquet")
PRED = Path("results/predictions/m1ens_hla.parquet")
OUT = Path("results/tables/novelty_conformal.csv")

ALPHA = 0.10
FLOOR = 0.05  # keeps a scale strictly positive; matches Dev 1's sigma_floor


def fit_scale(train: pd.DataFrame, cols: list[str]) -> LinearRegression:
    """Predict |error| from the given columns. The fitted value is the scale."""
    model = LinearRegression()
    model.fit(train[cols].to_numpy(float), train.abs_err.to_numpy(float))
    return model


def scale_for(
    model: LinearRegression | None, frame: pd.DataFrame, cols: list[str]
) -> np.ndarray:
    if model is None:
        return np.ones(len(frame))
    s = model.predict(frame[cols].to_numpy(float))
    return np.maximum(s, FLOOR)


def main() -> int:
    nov = pd.read_parquet(NOVELTY)
    master = pd.read_parquet(MASTER)[["id", "hla"]]
    ens = pd.read_parquet(PRED).merge(nov, on="id").merge(master, on="id")

    d = ens[ens.fold == "calib"].copy()
    d["abs_err"] = (d.y_true - d.y_mean).abs()
    d["nov"] = d.hla_novelty_hla
    d["nov3"] = d.hla_novelty_top3_hla

    alleles = sorted(d.hla.unique())
    print(f"leave-one-allele-out over {len(alleles)} held-out alleles, "
          f"{len(d)} rows, target coverage {1 - ALPHA:.0%}\n")

    schemes: dict[str, list[str] | None] = {
        "const": None,
        "y_std": ["y_std"],
        "novelty": ["nov", "nov3"],
        "combined": ["nov", "nov3", "y_std"],
    }

    records = []
    for name, cols in schemes.items():
        for held in alleles:
            train = d[d.hla != held]
            test = d[d.hla == held]
            if len(test) < 5:
                continue

            if cols is None:
                s_tr, s_te = np.ones(len(train)), np.ones(len(test))
            elif cols == ["y_std"]:
                # Use the ensemble std directly rather than regressing on it,
                # so this reproduces Dev 1's q_norm instead of a variant.
                s_tr = np.maximum(train.y_std.to_numpy(float), FLOOR)
                s_te = np.maximum(test.y_std.to_numpy(float), FLOOR)
            else:
                m = fit_scale(train, cols)
                s_tr, s_te = scale_for(m, train, cols), scale_for(m, test, cols)

            scores = train.abs_err.to_numpy(float) / s_tr
            # Finite-sample correction: the conformal quantile is taken at
            # ceil((n+1)(1-alpha))/n, not simply 1-alpha.
            n = len(scores)
            k = min(int(np.ceil((n + 1) * (1 - ALPHA))), n)
            q = float(np.sort(scores)[k - 1])

            half = q * s_te
            covered = (test.abs_err.to_numpy(float) <= half).mean()
            records.append(
                {
                    "scheme": name,
                    "hla": held,
                    "n": len(test),
                    "novelty_mism": int(round(test.nov.iloc[0] * 182)),
                    "coverage": float(covered),
                    "median_half_width": float(np.median(half)),
                }
            )

    res = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("per-allele coverage by scheme (target 0.90)")
    piv = res.pivot(index="hla", columns="scheme", values="coverage")
    order = (
        res[res.scheme == "const"].set_index("hla").novelty_mism.sort_values().index
    )
    piv = piv.loc[order, ["const", "y_std", "novelty", "combined"]]
    piv.insert(0, "nov", res[res.scheme == "const"].set_index("hla").novelty_mism)
    print(piv.round(3).to_string())

    print("\nsummary")
    print(f"   {'scheme':10s} {'marginal':>9s} {'mean|cov-.90|':>14s} "
          f"{'worst allele':>13s} {'med width':>10s}")
    summary = []
    for name in schemes:
        r = res[res.scheme == name]
        # Row-weighted marginal coverage, so big alleles are not outvoted by
        # the 7-row ones.
        marg = float((r.coverage * r.n).sum() / r.n.sum())
        gap = float((r.coverage - 0.90).abs().mean())
        worst = float(r.coverage.min())
        width = float(r.median_half_width.median())
        summary.append(
            {"scheme": name, "marginal": marg, "mean_abs_gap": gap,
             "worst_allele": worst, "median_half_width": width}
        )
        print(f"   {name:10s} {marg:9.3f} {gap:14.3f} {worst:13.3f} {width:10.3f}")

    best = min(summary, key=lambda r: r["mean_abs_gap"])
    print(f"\nbest conditional coverage: {best['scheme']} "
          f"(mean gap {best['mean_abs_gap']:.3f})")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
