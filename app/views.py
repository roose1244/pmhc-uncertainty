"""PepShield screens: Predict, Findings, About.

Each page has one job. Secondary detail sits behind expanders. Nothing here
computes a scientific result; figures are read through app.data from files an
experiment wrote, and a missing file produces an explicit empty state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import charts, data
from app.components import gauge, interval_bar, routing_sentence, stat_block
from app.theme import INK, MUTED, OK, RISK, SUPPORT, page_head, section, term


def _caption(text: str) -> None:
    st.markdown(
        f"<p style='font-size:0.88rem;color:{MUTED};margin:0.2rem 0 1rem 0;"
        f"line-height:1.45'>{text}</p>",
        unsafe_allow_html=True,
    )


def _missing(what: str) -> None:
    st.markdown(
        f"<p style='color:{MUTED}'>{what} has not been computed in this checkout.</p>",
        unsafe_allow_html=True,
    )


def predict_view(pred, rel, hla: str, peptide: str) -> None:
    colour, level, gloss = SUPPORT[rel.badge]
    lo_h, hi_h = pred.interval_hours

    if lo_h < 0.1:
        span = "lower end below the 0.1 h assay floor"
    else:
        span = f"about {hi_h / lo_h:.0f}× wide"

    st.markdown(
        stat_block(
            "Predicted half-life",
            f"{pred.half_life_hours:.1f} h",
            f"Likely range {lo_h:.1f}–{hi_h:.1f} h ({span}). "
            f"Training coverage: {level.lower()}.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(interval_bar(pred.half_life_hours, lo_h, hi_h, colour),
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.88rem;color:{MUTED};margin-top:-0.6rem'>"
        f"{term('half-life', 'Half-life')} is how long the complex holds together. "
        f"The band is the {term('90% interval')} — wider means less precise. "
        f"{gloss}</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Training coverage for this query"):
        gcols = st.columns(len(rel.signals), gap="large")
        for col, s in zip(gcols, rel.signals):
            with col:
                st.markdown(
                    gauge(s.name, s.value, s.level, s.fraction, s.detail),
                    unsafe_allow_html=True,
                )
        for reason in rel.reasons:
            st.markdown(f"<p style='font-size:0.88rem;color:{MUTED}'>{reason}</p>",
                        unsafe_allow_html=True)
        st.caption(
            "Coverage is about evidence, not correctness: a well-covered query "
            "can still have a wide range."
        )

    which = pred.source.split(":", 1)[1] if pred.source.startswith("local:") else None
    with st.expander("Which model answered"):
        st.markdown(
            f"<p>{routing_sentence(which, rel.badge != 'red')}</p>",
            unsafe_allow_html=True,
        )
        if which == "m1":
            st.markdown(
                f"<p style='color:{MUTED}'>{term('M1')} treats the allele as a "
                f"name tag. That works when the allele is in training and is "
                f"empty when it is not.</p>",
                unsafe_allow_html=True,
            )
        elif which == "m2n":
            st.markdown(
                f"<p style='color:{MUTED}'>{term('M2n')} reads the allele's "
                f"protein sequence, so it can still score an allele never "
                f"measured.</p>",
                unsafe_allow_html=True,
            )
        if pred.source == "placeholder":
            st.markdown(
                "<div class='note'>No trained weights found, so the half-life "
                "figures are stand-ins. Training coverage above is still "
                "computed from real training data.</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Binding groove for this allele"):
        biology_view(hla, compact=True)


def compare_view() -> None:
    st.markdown(
        page_head(
            "What the experiments show",
            "The question is whether predicted uncertainty can flag unreliable "
            f"answers when the {term('peptide')} or {term('HLA')} is new — not "
            "whether an unseen allele is risky, which the tool already knows.",
        ),
        unsafe_allow_html=True,
    )

    topic = st.radio(
        "Choose a finding",
        [
            "Why a number is not enough",
            "Compared with NetMHCstabpan",
            "New alleles",
            "Does uncertainty flag mistakes?",
            "HLA-C, with no measurements",
        ],
        index=1,
    )

    if topic == "Why a number is not enough":
        problem_view()
    elif topic == "Compared with NetMHCstabpan":
        _performance_view()
    elif topic == "New alleles":
        _new_allele_view()
    elif topic == "Does uncertainty flag mistakes?":
        _reliability_view()
    else:
        transfer_view()


def _performance_view() -> None:
    st.markdown(
        section(
            "How does this system compare with the published tool?",
            f"{term('NetMHCstabpan')} is the reference. Ranking ({term('Spearman')}) "
            f"is the main metric; error in hours ({term('MAE')}) is secondary.",
        ),
        unsafe_allow_html=True,
    )
    fact = data.in_sample_fact()
    if fact:
        inside, total, pct = fact
        st.markdown(
            f"<div class='note'><b>{pct:.1f}%</b> of this dataset "
            f"({inside:,} of {total:,} rows) sits inside NetMHCstabpan's own "
            f"training data. Read the bars as a comparison against a reference, "
            f"not as a clean independent test. The unseen-allele split is the "
            f"more informative one.</div>",
            unsafe_allow_html=True,
        )

    metric = st.radio("Metric", ["Ranking", "Error in hours"], horizontal=True)
    comp = data.comparison()
    if comp is None:
        _missing("The comparison table")
        return
    if metric == "Ranking":
        st.altair_chart(
            charts.spearman_by_split(comp, ["m1ens", "m2", "netmhcstabpan"]),
            use_container_width=True,
        )
        _caption(
            "Higher is better. Sage is this system; plum is NetMHCstabpan. "
            "NetMHCstabpan scores highest on every split — it was trained on "
            "this data, so those bars are a contaminated ceiling. The rightmost "
            "group (unseen allele) is the informative comparison."
        )
    else:
        st.altair_chart(
            charts.mae_by_split(comp, ["m1ens", "netmhcstabpan"]),
            use_container_width=True,
        )
        _caption(
            "Lower is better. Error looks smallest on the unseen-allele split "
            "because that allele's complexes are short-lived, not because the "
            "ranking is right. Ranking is the honest metric there."
        )


def _new_allele_view() -> None:
    st.markdown(
        section(
            "What happens when the allele has never been seen?",
            "Repeated training runs, because a single run on this split is too "
            "noisy to interpret.",
        ),
        unsafe_allow_html=True,
    )
    sv = data.seed_variance()
    if sv is None:
        _missing("The repeated-run table")
        return
    st.altair_chart(charts.seed_spread(sv, "hla"), use_container_width=True)
    _caption(
        "Dot is the mean over repeated runs; the line spans the lowest and "
        "highest run. The sequence model (m2n) is the only one above zero on "
        "an unseen allele. Bars are wide because this split holds out a "
        "single allele — differences smaller than the bars are not real."
    )
    with st.expander("Control: the ordinary split, where nothing is held out"):
        st.altair_chart(charts.seed_spread(sv, "random"), use_container_width=True)
        _caption(
            "On the ordinary split the runs agree closely, so the wide bars "
            "above are a property of holding out one allele, not of the "
            "training recipe."
        )


def _reliability_view() -> None:
    st.markdown(
        section(
            "Does the model know when it is likely to be wrong?",
            "Each point is one held-out allele.",
        ),
        unsafe_allow_html=True,
    )
    pa = data.per_allele()
    if pa is None:
        _missing("The per-allele table")
    else:
        which = st.radio(
            "Plot",
            ["Error vs how new the allele is", "The model's own uncertainty"],
            horizontal=True,
        )
        if which.startswith("Error"):
            st.altair_chart(charts.novelty_vs_error(pa), use_container_width=True)
            _caption(
                "Error rises with novelty. That is expected: more distant "
                "alleles are harder."
            )
        else:
            st.altair_chart(charts.novelty_vs_uncertainty(pa), use_container_width=True)
            _caption(
                "The model's own uncertainty does not rise with novelty — it "
                "is flat to slightly falling. Ensemble disagreement does not "
                "detect an unfamiliar allele. That negative result is the "
                "project's central finding."
            )

    sel = data.selective()
    if sel is None:
        return
    with st.expander("Can it rank which individual predictions are wrong?"):
        which_split = st.radio("Split", data.SPLITS, index=2, horizontal=True)
        st.altair_chart(charts.selective_curve(sel, which_split),
                        use_container_width=True)
        _caption(
            "Discard the least confident predictions and measure error on what "
            "is kept. Lower is better. 'oracle' is the unreachable best case; "
            "'random' is no information."
        )
        row = sel[(sel.split == which_split) & (sel.signal == "pred_err")]
        if not row.empty:
            _caption(
                f"On this split the fitted error signal captures "
                f"{row.pct_of_oracle.iloc[0]:.0f}% of the achievable gain. "
                f"Novelty on its own performs worse than discarding at random, "
                f"so “unseen, therefore warn” is not usable per prediction."
            )


def biology_view(hla: str, compact: bool = False) -> None:
    from app.structure import groove_for, nearest_training_allele, viewer_html

    if not compact:
        st.markdown(
            section("Why does the HLA sequence matter?"),
            unsafe_allow_html=True,
        )

    groove = groove_for(hla)
    nn_name, novel_pos = nearest_training_allele(hla, groove)
    try:
        contacts = [p + 1 for p in json.loads(
            Path("data/external/contact_positions.json").read_text())]
    except Exception:
        contacts = []

    html = viewer_html(hla, contacts, novel_pos, height=360)
    if html is None:
        _missing("The AlphaFold structure")
    else:
        import streamlit.components.v1 as components

        components.html(html, height=375)
        legend = "Amber = peptide-contacting residues"
        if novel_pos:
            legend += (
                f". Red = {len(novel_pos)} position(s) where {hla} differs "
                f"from {nn_name}, the closest allele in training."
            )
        st.caption(legend)

    st.markdown(
        f"<p style='font-size:0.92rem;color:{MUTED};line-height:1.55'>"
        f"{term('HLA')} molecules form the groove that holds a {term('peptide')}. "
        f"Alleles differ at residues lining that groove. The sequence model uses "
        f"those differences; the name-tag model cannot. The structure is context, "
        f"not the predictor — no docking was performed, AlphaFold supplies one "
        f"representative fold per locus, and the peptide itself is not modelled."
        f"</p>",
        unsafe_allow_html=True,
    )


def about_view() -> None:
    from app.landing import pipeline_diagram
    from app.theme import GLOSSARY

    st.markdown(
        page_head(
            "How this works",
            "A short account of the data, the models, and the question the "
            "project is actually asking.",
        ),
        unsafe_allow_html=True,
    )

    stages = [
        ("Data", "28,166 measured peptide–HLA half-lives across 75 alleles."),
        ("Splits", "Four held-out designs, each testing a different kind of novelty."),
        ("Baseline (M1)", "BLOSUM peptide features plus the allele as a name tag."),
        ("Sequence model (M2n)", "ESM-2 embeddings of the peptide and the HLA groove."),
        ("Ensemble", "Five runs per model; their disagreement is one uncertainty estimate."),
        ("Coverage", "How far this query sits from anything in training."),
        ("Support", "Coverage combined into a plain-language training-coverage call."),
    ]
    for i, (name, desc) in enumerate(stages, start=1):
        st.markdown(
            f"<p style='margin:0.35rem 0'><span class='num' style='color:{MUTED}'>"
            f"{i:02d}</span> &nbsp;<b style='color:{INK}'>{name}.</b> "
            f"<span style='color:{MUTED}'>{desc}</span></p>",
            unsafe_allow_html=True,
        )

    st.markdown(section("What is actually built"), unsafe_allow_html=True)
    st.markdown(pipeline_diagram(), unsafe_allow_html=True)

    st.markdown(
        section("What this project is asking"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='line-height:1.55'>The question is not whether an unseen "
        f"allele is risky — a tool always knows which alleles it trained on. "
        f"The question is whether predicted {term('uncertainty')} and "
        f"{term('training coverage')} can identify which individual predictions "
        f"are unreliable under {term('distribution shift')}.</p>"
        f"<p style='line-height:1.55;color:{MUTED}'>Measured here, the answer "
        f"differs by axis. Under peptide novelty the fitted reliability signal "
        f"does rank mistakes usefully. Under allele novelty it does not, and "
        f"the model's own ensemble uncertainty is flat to inversely related to "
        f"how novel an allele is. That negative result is reported rather than "
        f"smoothed over.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Glossary"):
        gl = pd.DataFrame(
            {"Term": list(GLOSSARY), "Meaning": list(GLOSSARY.values())}
        )
        st.dataframe(gl, hide_index=True, use_container_width=True)


def problem_view() -> None:
    st.markdown(
        section(
            "A prediction alone does not say whether to believe it.",
            "Two queries can produce similar half-lives and deserve very "
            "different trust.",
        ),
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    examples = [
        (c1, "HLA-A*02:01 · VTTEVAFGL", "2.4 h",
         "1,023 measurements on this allele; this exact peptide was measured."),
        (c2, "HLA-C*07:02 · VTTEVAFGL", "0.6 h",
         "No measurement exists for this allele, or for any HLA-C allele, "
         "in training."),
    ]
    for col, who, pred, note in examples:
        with col:
            st.markdown(
                f"<div class='num' style='font-size:0.88rem;color:{MUTED}'>{who}</div>"
                f"<div class='num' style='font-size:1.8rem;font-weight:600;"
                f"color:{INK};margin:0.2rem 0 0.45rem 0'>{pred}</div>"
                f"<p style='font-size:0.9rem;color:{MUTED};line-height:1.45;"
                f"margin:0'>{note}</p>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<p style='margin-top:1.2rem;line-height:1.55'>These are the model's "
        "real outputs for those two queries. Nothing in the prediction itself "
        "distinguishes them. The rest of this page asks whether anything the "
        "model computes can.</p>",
        unsafe_allow_html=True,
    )


def transfer_view() -> None:
    st.markdown(
        section(
            "What happens on a locus with no training data at all?",
            f"{term('HLA-C')} has no stability measurement in this dataset, "
            f"and none in {term('NetMHCstabpan')}'s either. Both systems will "
            f"still answer.",
        ),
        unsafe_allow_html=True,
    )

    probe = Path("results/tables/hla_c_probe.csv")
    if not probe.exists():
        _missing("The HLA-C probe")
        return
    p = pd.read_csv(probe)
    g = (
        p.groupby("tier")
        .agg(
            alleles=("hla", "nunique"),
            n=("peptide", "size"),
            novelty=("novelty", "mean"),
            m1=("m1_std", "mean"),
            m2n=("m2n_std", "mean"),
        )
        .reset_index()
        .sort_values("tier")
    )

    st.caption(f"{int(g.n.sum())} predictions, grouped by how far the allele is from training.")
    display = g.rename(columns={
        "tier": "Distance from training",
        "alleles": "Alleles",
        "novelty": "Mean residue differences",
        "m1": "M1 disagreement",
        "m2n": "M2n disagreement",
    }).drop(columns=["n"])
    st.dataframe(display, hide_index=True, use_container_width=True)

    t1, t3 = g.iloc[0], g.iloc[-1]
    d1 = 100 * (t3.m1 / t1.m1 - 1)
    d2 = 100 * (t3.m2n / t1.m2n - 1)
    st.markdown(
        f"<p style='line-height:1.55'>From the closest tier to HLA-C, novelty "
        f"rises from {t1.novelty:.1f} to {t3.novelty:.1f} residues. "
        f"<b>M1</b> (name tag) disagreement changes by "
        f"<span style='color:{RISK}'>{d1:+.0f}%</span>. "
        f"<b>M2n</b> (sequence) disagreement changes by "
        f"<span style='color:{OK}'>{d2:+.0f}%</span>.</p>",
        unsafe_allow_html=True,
    )

    c = p[p.tier.str.startswith("3")]
    spread_m1 = c.groupby("peptide").m1_mean.std().mean()
    spread_m2 = c.groupby("peptide").m2n_mean.std().mean()
    st.markdown(
        f"<p style='line-height:1.55'>Holding the peptide fixed across seven "
        f"HLA-C alleles, M1's prediction varies by "
        f"<span class='num'><b>{spread_m1:.2e}</b></span> and M2n's by "
        f"<span class='num'><b>{spread_m2:.3f}</b></span>. M1 returns the "
        f"same number for every HLA-C allele: an unseen name is a vector of "
        f"zeros, so the answer is a peptide-only guess wearing an allele's name.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='caveat'>No HLA-C stability measurements exist here, so "
        "error on HLA-C cannot be computed. Everything above is behaviour — "
        "what the models output and how much they disagree — not accuracy.</div>",
        unsafe_allow_html=True,
    )
