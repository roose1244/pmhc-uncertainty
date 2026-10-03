"""Four result figures and the comparison table. Test folds are read, not refit."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH
from src.evaluate import hours_from_log
from src.splits import SPLITS_DIR

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")
PRED_DIR = Path("results/predictions")
NAVY = "#1B3A4B"
TEAL = "#1F7A8C"
GREY = "#8E99A4"
FLOOR_H = 0.1
SPLITS = ("random", "peptide", "cluster", "hla")
SPLIT_LABELS = {
    "random": "Random",
    "peptide": "Unseen peptide",
    "cluster": "Unseen cluster",
    "hla": "B*15:02",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _test(model: str, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(PRED_DIR / f"{model}_{split}.parquet")
    return frame.loc[frame["fold"] == "test"].copy()


def _hours(log_values: np.ndarray) -> np.ndarray:
    return np.maximum(hours_from_log(np.asarray(log_values, dtype=np.float64)), FLOOR_H)


def _spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(spearmanr(y, yhat).statistic)


def comparison_table() -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH)
    nms = pd.read_parquet("data/external/netmhcstabpan_preds.parquet")
    rows = []
    notes = {
        "m1": "BLOSUM50 + HLA one-hot. On the HLA split the one-hot is all zeros.",
        "m1pep": "BLOSUM50 only. Trained without an HLA input.",
        "m2": "Frozen ESM-2 mean-pools. Retrained after the three groove corrections.",
        "m2pep": "ESM-2 peptide mean-pool only.",
        "m1ens": "Five bootstrapped M1 members. y_std is their disagreement.",
        "netmhcstabpan": "In-sample. NetMHCstabpan was trained on this file. Contaminated ceiling, not a fair baseline.",
    }
    for model in ("m1", "m1pep", "m2", "m2pep", "m1ens"):
        for split in SPLITS:
            test = _test(model, split)
            row = {
                "model": model,
                "split": split,
                "n": len(test),
                "spearman": _spearman(test["y_true"], test["y_mean"]),
                "mae_hours": float(
                    np.mean(np.abs(hours_from_log(test["y_mean"]) - hours_from_log(test["y_true"])))
                ),
                "err_unc_spearman": np.nan,
                "coverage_90": np.nan,
                "note": notes[model],
            }
            if model == "m1ens":
                err = np.abs(test["y_true"] - test["y_mean"])
                row["err_unc_spearman"] = _spearman(test["y_std"], err)
                covered = (test["y_true"] >= test["lo"]) & (test["y_true"] <= test["hi"])
                row["coverage_90"] = float(covered.mean())
            rows.append(row)
    for split in SPLITS:
        assignment = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
        joined = master.merge(assignment, on="id").merge(nms, on="id", how="left")
        test = joined.loc[joined["fold"] == "test"].dropna(subset=["nms_half_life"])
        rows.append(
            {
                "model": "netmhcstabpan",
                "split": split,
                "n": len(test),
                "spearman": _spearman(test["half_life"], test["nms_half_life"]),
                "mae_hours": float(np.mean(np.abs(test["nms_half_life"] - test["half_life"]))),
                "err_unc_spearman": np.nan,
                "coverage_90": np.nan,
                "note": notes["netmhcstabpan"],
            }
        )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "comparison.csv", index=False)
    return table


def fig_pred_vs_measured() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    panels = (
        (_test("m1ens", "peptide"), "Unseen peptides"),
        (_test("m1ens", "hla"), "Held-out HLA-B*15:02"),
    )
    for ax, (test, title) in zip(axes, panels):
        x = _hours(test["y_true"])
        y = _hours(test["y_mean"])
        ax.scatter(x, y, s=12, c=TEAL, alpha=0.35, linewidths=0, rasterized=True)
        limits = (0.08, 400)
        ax.plot(limits, limits, color=NAVY, lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        rho = _spearman(test["y_true"], test["y_mean"])
        ax.set_title(f"{title}\nSpearman {rho:.2f}")
        ax.set_xlabel("Measured half-life (hours)")
    axes[0].set_ylabel("Predicted half-life (hours)")
    fig.suptitle("M1 ensemble", color=NAVY, fontsize=14, y=1.02)
    fig.text(
        0.0,
        -0.02,
        "Measured zeros are drawn at 0.1 h, the resolution of the assay. Identity line in navy.",
        color=GREY,
        fontsize=9,
    )
    _save(fig, "fig1_pred_vs_measured")


def fig_uncertainty() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    cmap = LinearSegmentedColormap.from_list("teal", ["#F3F6F7", TEAL, NAVY])
    panels = (
        (_test("m1ens", "peptide"), "Unseen peptides"),
        (_test("m1ens", "hla"), "Held-out HLA-B*15:02"),
    )
    for ax, (test, title) in zip(axes, panels):
        x = test["y_std"].to_numpy(dtype=np.float64)
        y = np.abs(test["y_true"].to_numpy() - test["y_mean"].to_numpy())
        if len(test) > 800:
            ax.hexbin(x, y, gridsize=28, cmap=cmap, mincnt=1, linewidths=0)
        else:
            ax.scatter(x, y, s=16, c=TEAL, alpha=0.45, linewidths=0)
        order = np.argsort(x)
        bins = np.array_split(order, 6)
        bx = [x[b].mean() for b in bins if len(b)]
        by = [y[b].mean() for b in bins if len(b)]
        ax.plot(bx, by, color=NAVY, lw=1.8, marker="o", ms=4)
        rho = _spearman(x, y)
        ax.set_title(f"{title}\nError–uncertainty Spearman {rho:.2f}")
        ax.set_xlabel("Ensemble std, log10(half-life + 0.1)")
    axes[0].set_ylabel("Absolute error, log10(half-life + 0.1)")
    fig.suptitle("Does disagreement track error?", color=NAVY, fontsize=14, y=1.02)
    _save(fig, "fig2_uncertainty_vs_error")


def fig_shift_bars(table: pd.DataFrame) -> None:
    ens = table.loc[table["model"] == "m1ens"].set_index("split").loc[list(SPLITS)]
    x = np.arange(len(SPLITS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - width / 2, ens["spearman"], width, color=TEAL, label="Prediction Spearman")
    ax.bar(
        x + width / 2,
        ens["err_unc_spearman"],
        width,
        color=NAVY,
        label="Error–uncertainty Spearman",
    )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-0.35, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("M1 ensemble under distribution shift")
    _save(fig, "fig3_shift_bars")


def _prefix_mae(y_true: np.ndarray, y_pred: np.ndarray, order: np.ndarray, fraction: float) -> float:
    keep = max(int(round(len(order) * fraction)), 1)
    take = order[:keep]
    return float(np.mean(np.abs(hours_from_log(y_pred[take]) - hours_from_log(y_true[take]))))


def fig_selective() -> None:
    test = _test("m1ens", "peptide")
    y = test["y_true"].to_numpy(dtype=np.float64)
    yhat = test["y_mean"].to_numpy(dtype=np.float64)
    err = np.abs(y - yhat)
    fractions = np.linspace(1.0, 0.2, 17)
    certain = np.argsort(test["y_std"].to_numpy(), kind="mergesort")
    oracle = np.argsort(err, kind="mergesort")
    model_mae = [_prefix_mae(y, yhat, certain, f) for f in fractions]
    oracle_mae = [_prefix_mae(y, yhat, oracle, f) for f in fractions]
    rng = np.random.default_rng(0)
    random_curves = []
    for _ in range(40):
        order = rng.permutation(len(y))
        random_curves.append([_prefix_mae(y, yhat, order, f) for f in fractions])
    random_mae = np.mean(random_curves, axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(fractions * 100, model_mae, color=TEAL, lw=2.2, label="Ranked by ensemble std")
    ax.plot(fractions * 100, random_mae, color=GREY, lw=2.0, label="Random retention")
    ax.plot(fractions * 100, oracle_mae, color=NAVY, lw=1.6, ls="--", label="Oracle, ranked by true error")
    ax.set_xlim(100, 20)
    ax.set_xlabel("Percent of predictions retained")
    ax.set_ylabel("MAE (hours)")
    ax.set_title("Unseen peptides: keep the more certain half")
    ax.legend(frameon=False)
    _save(fig, "fig4_selective_peptide")


def main() -> None:
    _style()
    table = comparison_table()
    fig_pred_vs_measured()
    fig_uncertainty()
    fig_shift_bars(table)
    fig_selective()
    show = table.loc[
        table["split"].isin(["peptide", "hla"])
        & table["model"].isin(["m1", "m1pep", "m2", "m1ens", "netmhcstabpan"])
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {FIG_DIR} and {TABLE_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
