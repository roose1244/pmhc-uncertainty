"""Altair charts for the comparison views.

Every chart takes a dataframe that came from results/tables/ and adds nothing
to it. Where a table carries uncertainty — seed spread, bootstrap CIs — the
chart draws it, because the project's own finding is that single-run
differences on the HLA split are not interpretable, and a bare bar would
contradict that.

Colour is consistent across every chart: teal for this system, violet for
NetMHCstabpan, and a muted grey for ablations that exist only as controls.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from app.theme import ACCENT, FAINT, INK, MUTED, NEUTRAL, REFERENCE

AXIS = alt.Axis(labelColor=MUTED, titleColor=MUTED, tickColor=FAINT,
                domainColor=FAINT, labelFontSize=11, titleFontSize=11,
                grid=False)
GRID = alt.Axis(labelColor=MUTED, titleColor=MUTED, tickColor=FAINT,
                domainColor=FAINT, labelFontSize=11, titleFontSize=11,
                grid=True, gridColor="rgba(15,23,42,0.06)")

SPLIT_ORDER = ["random", "peptide", "cluster", "hla"]


def _base(df: pd.DataFrame, height: int) -> alt.Chart:
    return alt.Chart(df).properties(height=height).configure_view(
        strokeWidth=0).configure_axis(labelFont="system-ui", titleFont="system-ui")


def spearman_by_split(df: pd.DataFrame, models: list[str], height: int = 230):
    """Grouped bars: rank correlation per split, this system vs the reference."""
    d = df[df.model.isin(models)].copy()
    d["colour"] = d.model.map(
        lambda m: REFERENCE if m == "netmhcstabpan" else (
            ACCENT if m in ("m1ens", "m2nens", "m2n") else NEUTRAL))

    chart = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("split:N", sort=SPLIT_ORDER, title=None, axis=AXIS),
            xOffset=alt.XOffset("model:N", sort=models),
            y=alt.Y("spearman:Q", title="Spearman (rank correlation)", axis=GRID,
                    scale=alt.Scale(domain=[-0.3, 1.0])),
            color=alt.Color("model:N", sort=models,
                            scale=alt.Scale(
                                domain=models,
                                range=[REFERENCE if m == "netmhcstabpan"
                                       else (ACCENT if m in ("m1ens", "m2nens", "m2n")
                                             else NEUTRAL) for m in models]),
                            legend=alt.Legend(orient="top", title=None,
                                              labelColor=MUTED, labelFontSize=11)),
            tooltip=[alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("split:N", title="Split"),
                     alt.Tooltip("spearman:Q", title="Spearman", format=".3f"),
                     alt.Tooltip("n:Q", title="Test rows")],
        )
        .properties(height=height)
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=FAINT, strokeDash=[3, 3]).encode(y="y:Q")
    return (chart + zero).configure_view(strokeWidth=0)


def seed_spread(df: pd.DataFrame, split: str, height: int = 190):
    """Mean with min-max whiskers over repeated runs, for one split.

    Drawn as a range rather than a bar because on the HLA split the spread is
    wider than the differences between models, and a bar chart there would
    invite a conclusion the data cannot support.
    """
    d = df[df.split == split].copy().sort_values("mean")
    d["label"] = d.model

    rule = alt.Chart(d).mark_rule(color=FAINT, size=2).encode(
        y=alt.Y("label:N", sort=list(d.label), title=None, axis=AXIS),
        x=alt.X("min:Q", title="Spearman across repeated runs", axis=GRID),
        x2="max:Q",
    )
    point = alt.Chart(d).mark_point(filled=True, size=110, color=ACCENT).encode(
        y=alt.Y("label:N", sort=list(d.label), title=None, axis=AXIS),
        x=alt.X("mean:Q", axis=GRID),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                 alt.Tooltip("sd:Q", title="SD", format=".3f"),
                 alt.Tooltip("min:Q", title="Lowest run", format=".3f"),
                 alt.Tooltip("max:Q", title="Highest run", format=".3f"),
                 alt.Tooltip("n_seeds:Q", title="Runs")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=FAINT, strokeDash=[3, 3]).encode(x="x:Q")
    return (rule + zero + point).properties(height=height).configure_view(strokeWidth=0)


def mae_by_split(df: pd.DataFrame, models: list[str], height: int = 230):
    d = df[df.model.isin(models)].copy()
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("split:N", sort=SPLIT_ORDER, title=None, axis=AXIS),
            xOffset=alt.XOffset("model:N", sort=models),
            y=alt.Y("mae_hours:Q", title="Mean absolute error (hours)", axis=GRID),
            color=alt.Color("model:N", sort=models,
                            scale=alt.Scale(
                                domain=models,
                                range=[REFERENCE if m == "netmhcstabpan" else ACCENT
                                       for m in models]),
                            legend=alt.Legend(orient="top", title=None,
                                              labelColor=MUTED, labelFontSize=11)),
            tooltip=[alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("split:N", title="Split"),
                     alt.Tooltip("mae_hours:Q", title="MAE (h)", format=".2f")],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )


def paired_difference(df: pd.DataFrame, height: int = 200):
    """Our Spearman minus NetMHCstabpan's, with the paired bootstrap CI."""
    d = df.copy()
    rule = alt.Chart(d).mark_rule(color=FAINT, size=2).encode(
        y=alt.Y("split:N", sort=SPLIT_ORDER, title=None, axis=AXIS),
        x=alt.X("diff_ci_lo:Q", title="Difference in Spearman (ours − NetMHCstabpan)",
                axis=GRID),
        x2="diff_ci_hi:Q",
    )
    point = alt.Chart(d).mark_point(filled=True, size=110, color=ACCENT).encode(
        y=alt.Y("split:N", sort=SPLIT_ORDER, title=None, axis=AXIS),
        x=alt.X("diff:Q", axis=GRID),
        tooltip=[alt.Tooltip("split:N", title="Split"),
                 alt.Tooltip("spearman_ours:Q", title="Ours", format=".3f"),
                 alt.Tooltip("spearman_nms:Q", title="NetMHCstabpan", format=".3f"),
                 alt.Tooltip("diff:Q", title="Difference", format=".3f"),
                 alt.Tooltip("n_scored_by_both:Q", title="Rows scored by both")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=INK, strokeDash=[4, 4], size=1).encode(x="x:Q")
    return (rule + zero + point).properties(height=height).configure_view(strokeWidth=0)


def novelty_vs_error(df: pd.DataFrame, height: int = 240):
    """Per held-out allele: error against novelty, with uncertainty alongside.

    The project's central result. Error climbs with novelty while the model's
    own uncertainty does not, so both are plotted on the same x-axis.
    """
    d = df.copy()
    d["novelty_res"] = d.novelty_mismatches

    err = alt.Chart(d).mark_point(filled=True, size=95, color=ACCENT).encode(
        x=alt.X("novelty_res:Q", title="Residues differing from nearest trained allele",
                axis=GRID),
        y=alt.Y("mean_abs_err:Q", title="Mean absolute error (log units)", axis=GRID),
        tooltip=[alt.Tooltip("hla:N", title="Allele"),
                 alt.Tooltip("n:Q", title="Rows"),
                 alt.Tooltip("novelty_mismatches:Q", title="Residues differing"),
                 alt.Tooltip("mean_abs_err:Q", title="Mean error", format=".3f"),
                 alt.Tooltip("mean_y_std:Q", title="Model uncertainty", format=".3f")],
    )
    trend = err.transform_regression("novelty_res", "mean_abs_err").mark_line(
        color=ACCENT, strokeDash=[4, 3], opacity=0.6)
    return (err + trend).properties(height=height).configure_view(strokeWidth=0)


def novelty_vs_uncertainty(df: pd.DataFrame, height: int = 240):
    d = df.copy()
    d["novelty_res"] = d.novelty_mismatches
    unc = alt.Chart(d).mark_point(filled=True, size=95, color=REFERENCE).encode(
        x=alt.X("novelty_res:Q", title="Residues differing from nearest trained allele",
                axis=GRID),
        y=alt.Y("mean_y_std:Q", title="Model's own uncertainty", axis=GRID),
        tooltip=[alt.Tooltip("hla:N", title="Allele"),
                 alt.Tooltip("novelty_mismatches:Q", title="Residues differing"),
                 alt.Tooltip("mean_y_std:Q", title="Model uncertainty", format=".3f")],
    )
    trend = unc.transform_regression("novelty_res", "mean_y_std").mark_line(
        color=REFERENCE, strokeDash=[4, 3], opacity=0.6)
    return (unc + trend).properties(height=height).configure_view(strokeWidth=0)


def selective_curve(df: pd.DataFrame, split: str, height: int = 240):
    """Error against how much of the test set you keep, per confidence signal."""
    d = df[df.split == split].copy()
    rows = []
    for _, r in d.iterrows():
        for cov, col in [(100, "mae_100"), (80, "mae_80"), (50, "mae_50"), (20, "mae_20")]:
            rows.append({"signal": r.signal, "coverage": cov, "mae": r[col]})
    long = pd.DataFrame(rows)
    keep = ["pred_err", "y_std", "novelty", "random", "oracle"]
    long = long[long.signal.isin(keep)]

    palette = {"pred_err": ACCENT, "y_std": REFERENCE, "novelty": "#f59e0b",
               "random": NEUTRAL, "oracle": INK}
    return (
        alt.Chart(long)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("coverage:Q", title="% of predictions kept (most confident first)",
                    scale=alt.Scale(reverse=True), axis=GRID),
            y=alt.Y("mae:Q", title="Mean absolute error (hours)", axis=GRID),
            color=alt.Color("signal:N",
                            scale=alt.Scale(domain=list(palette), range=list(palette.values())),
                            legend=alt.Legend(orient="top", title=None,
                                              labelColor=MUTED, labelFontSize=11)),
            tooltip=[alt.Tooltip("signal:N", title="Signal"),
                     alt.Tooltip("coverage:Q", title="% kept"),
                     alt.Tooltip("mae:Q", title="MAE (h)", format=".2f")],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )
