"""Failure case gallery: where the model is confidently wrong, and where it isn't.

Phase E, and the slide the plan expects to carry the talk. Four quadrants over
the test rows of a split, cut at the medians of predicted uncertainty and
actual error:

    confidently wrong   low y_std, high error   the dangerous cases
    correctly unsure    high y_std, high error  the system working
    confidently right   low y_std, low error    the happy path
    needlessly unsure   high y_std, low error   wasted caution

The reason this is worth a slide rather than a table: each confidently-wrong
row is a prediction a user would have believed. The question the gallery has
to answer is whether anything we ship would have warned them.

So each case also carries what the app's novelty-driven badge would have said.
That is the real test of the badge: not whether it correlates with something
in aggregate, but whether it fires on the specific rows where the model's own
confidence failed. If it does not, the gallery says so.

Run:  python scripts/failure_gallery.py [split]   (default: hla)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.reliability import assess  # noqa: E402

MASTER = Path("data/processed/master.parquet")
NOVELTY = Path("data/features/novelty.parquet")
PRED_DIR = Path("results/predictions")
OUT = Path("results/tables/failure_gallery.csv")

N_PER_QUADRANT = 6


def to_hours(x: float) -> float:
    return max(10.0**x - 0.1, 0.0)


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "hla"
    pred_path = PRED_DIR / f"m1ens_{split}.parquet"
    if not pred_path.exists():
        print(f"missing {pred_path}")
        return 1

    master = pd.read_parquet(MASTER)[["id", "peptide", "hla", "half_life"]]
    nov = pd.read_parquet(NOVELTY)
    d = pd.read_parquet(pred_path).merge(master, on="id").merge(nov, on="id")

    t = d[d.fold == "test"].copy()
    t["abs_err"] = (t.y_true - t.y_mean).abs()
    t["pred_h"] = t.y_mean.map(to_hours)
    t["true_h"] = t.half_life

    med_std, med_err = t.y_std.median(), t.abs_err.median()
    t["unsure"] = t.y_std > med_std
    t["wrong"] = t.abs_err > med_err

    quadrants = {
        "confidently_wrong": (~t.unsure) & t.wrong,
        "correctly_unsure": t.unsure & t.wrong,
        "confidently_right": (~t.unsure) & (~t.wrong),
        "needlessly_unsure": t.unsure & (~t.wrong),
    }

    rows = []
    for name, mask in quadrants.items():
        sub = t[mask]
        if sub.empty:
            continue
        # Worst (or best) examples first, so the slide shows real extremes
        # rather than borderline rows that happen to fall in the quadrant.
        ordering = sub.abs_err.rank(ascending="right" not in name)
        pick = sub.assign(_o=ordering).nlargest(N_PER_QUADRANT, "_o")
        for _, r in pick.iterrows():
            verdict = assess(r.peptide, r.hla, model_std=float(r.y_std))
            rows.append(
                {
                    "quadrant": name,
                    "peptide": r.peptide,
                    "hla": r.hla,
                    "true_hours": round(float(r.true_h), 2),
                    "pred_hours": round(float(r.pred_h), 2),
                    "abs_err_log": round(float(r.abs_err), 3),
                    "y_std": round(float(r.y_std), 4),
                    "badge": verdict.badge,
                    "covered_by_90pct": bool(r.lo <= r.y_true <= r.hi),
                }
            )

    g = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT, index=False)

    print(f"FAILURE GALLERY - split '{split}', test fold, n={len(t)}")
    print(f"cut at medians: y_std={med_std:.4f}, |err|={med_err:.4f}\n")
    for name in quadrants:
        sub = g[g.quadrant == name]
        if sub.empty:
            continue
        n_all = int(quadrants[name].sum())
        print(f"--- {name.replace('_', ' ').upper()}  ({n_all} rows in quadrant)")
        print(f"    {'peptide':11s} {'hla':16s} {'true':>7s} {'pred':>7s} "
              f"{'|err|':>6s} {'y_std':>6s} {'badge':>6s} {'in 90%':>7s}")
        for _, r in sub.iterrows():
            print(f"    {r.peptide:11s} {r.hla:16s} {r.true_hours:7.2f} "
                  f"{r.pred_hours:7.2f} {r.abs_err_log:6.3f} {r.y_std:6.3f} "
                  f"{r.badge:>6s} {str(r.covered_by_90pct):>7s}")
        print()

    cw = g[g.quadrant == "confidently_wrong"]
    if not cw.empty:
        caught = (cw.badge != "green").sum()
        print(f"Of the {len(cw)} worst confidently-wrong cases shown, the "
              f"novelty badge warns on {caught}.")
        print(f"Interval covered the truth in "
              f"{int(cw.covered_by_90pct.sum())}/{len(cw)}.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
