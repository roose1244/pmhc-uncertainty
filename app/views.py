"""The four screens: Predict, Compare, Biology, About.

Each section states the question it answers before showing anything, so a
reader never meets a chart without knowing what it is for. Terms a non
specialist would not know carry a hover definition from theme.GLOSSARY rather
than a paragraph of explanation.

No view computes a scientific result. Everything is read through app.data from
files an experiment wrote, and a missing file produces an explicit "not
computed" state rather than a plausible-looking number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import charts, data
from app.components import gauge, interval_bar, routing_diagram, stat_block
from app.theme import (ACCENT, FAINT, INK, MONO, MUTED, REFERENCE, SUPPORT,
                       section, term)


def _caption(text: str) -> None:
    st.markdown(f"<div style='font-size:0.8rem;color:{MUTED};margin-top:-0.3rem'>"
                f"{text}</div>", unsafe_allow_html=True)


def _takeaway(text: str) -> None:
    """One sentence under every chart saying what to take from it."""
    st.markdown(f"<div style='font-size:0.82rem;color:{INK};background:#f8fafc;"
                f"border-left:3px solid {ACCENT};padding:0.5rem 0.75rem;"
                f"border-radius:0 6px 6px 0;margin-top:0.4rem'>{text}</div>",
                unsafe_allow_html=True)


def _missing(what: str) -> None:
    st.markdown(f"<div style='font-size:0.82rem;color:{MUTED};font-style:italic'>"
                f"{what} has not been computed in this checkout.</div>",
                unsafe_allow_html=True)


# ------------------------------------------------------------------ PREDICT --
def predict_view(pred, rel, hla: str, peptide: str) -> None:
    colour, level, gloss = SUPPORT[rel.badge]
    lo_h, hi_h = pred.interval_hours

    st.markdown(section("PREDICTION", "What does the model predict?"),
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        if lo_h < 0.1:
            span = "lower bound below the 0.1 h assay floor"
        else:
            span = f"{hi_h / lo_h:.0f}× wide"
        st.markdown(
            stat_block("Predicted half-life", f"{pred.half_life_hours:.1f} h",
                       f"90% interval {lo_h:.1f} – {hi_h:.1f} h &nbsp;·&nbsp; {span}"),
            unsafe_allow_html=True)
        st.markdown(interval_bar(pred.half_life_hours, lo_h, hi_h, colour),
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.78rem;color:{MUTED};margin-top:-0.8rem'>"
            f"{term('half-life', 'Half-life')} is how long the complex holds "
            f"together. The band is the {term('90% interval')} — wider means "
            f"less precise.</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            f"<div style='border:1px solid rgba(15,23,42,0.08);border-radius:10px;"
            f"padding:0.85rem 1rem'>"
            f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.1em;"
            f"color:{MUTED}'>PREDICTION SUPPORT</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{colour};"
            f"line-height:1.2;margin-top:2px'>{level}</div>"
            f"<div style='font-size:0.78rem;color:{MUTED};margin-top:4px'>{gloss}</div>"
            f"</div>", unsafe_allow_html=True)

    # ---- support -----------------------------------------------------------
    st.markdown(section("SUPPORT",
                        "How well is this query represented by training data?",
                        "Support is about evidence, not correctness: a well-covered "
                        "query can still carry a wide interval."),
                unsafe_allow_html=True)

    gcols = st.columns(len(rel.signals), gap="medium")
    for col, s in zip(gcols, rel.signals):
        with col:
            st.markdown(gauge(s.name, s.value, s.level, s.fraction, s.detail),
                        unsafe_allow_html=True)
    for reason in rel.reasons:
        st.markdown(f"<div style='font-size:0.82rem;color:{MUTED}'>• {reason}</div>",
                    unsafe_allow_html=True)

    # ---- routing -----------------------------------------------------------
    st.markdown(section("ROUTING", "Why did the system choose this model?"),
                unsafe_allow_html=True)
    which = pred.source.split(":", 1)[1] if pred.source.startswith("local:") else None
    st.markdown(routing_diagram(which, rel.badge != "red"), unsafe_allow_html=True)
    if which:
        note = ({"m1": f"{term('M1')} encodes the allele as an identity label. That "
                       "works well when the allele is in training and carries no "
                       "information when it is not.",
                 "m2n": f"{term('M2n')} reads the allele's protein sequence, so it can "
                        "still say something about an allele never measured."}
                .get(which, ""))
        st.markdown(f"<div style='font-size:0.8rem;color:{MUTED}'>{note}</div>",
                    unsafe_allow_html=True)
    elif pred.source == "placeholder":
        st.markdown("<div class='note'>No trained weights found, so the stability "
                    "figures are placeholders. The support panel above is still "
                    "computed from real training data.</div>",
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ COMPARE --
def compare_view() -> None:
    st.markdown(section("THE QUESTION", "Can a model recognise when it is outside "
                        "its training experience?"), unsafe_allow_html=True)

    cols = st.columns(4, gap="small")
    for col, sp in zip(cols, data.SPLITS):
        name, what, why = data.SPLIT_MEANING[sp]
        hard = sp == "hla"
        with col:
            st.markdown(
                f"<div style='border:1px solid {'rgba(190,18,60,0.35)' if hard else 'rgba(15,23,42,0.08)'};"
                f"border-radius:8px;padding:0.6rem 0.7rem;height:100%;"
                f"background:{'#fff1f2' if hard else '#fff'}'>"
                f"<div style='font-size:0.66rem;font-weight:700;letter-spacing:0.08em;"
                f"color:{'#be123c' if hard else ACCENT}'>{name}</div>"
                f"<div style='font-size:0.8rem;font-weight:600;color:{INK};"
                f"margin-top:2px'>{what}</div>"
                f"<div style='font-size:0.72rem;color:{MUTED};margin-top:3px'>{why}</div>"
                f"</div>", unsafe_allow_html=True)

    # ---- benchmark caveat, stated before any comparison --------------------
    fact = data.in_sample_fact()
    if fact:
        inside, total, pct = fact
        st.markdown(
            f"<div class='note' style='margin-top:1rem'><b>Benchmark note.</b> "
            f"{term('NetMHCstabpan')} is the established reference tool, but this "
            f"dataset overlaps heavily with its own training data: "
            f"<b>{pct:.1f}%</b> of rows ({inside:,} of {total:,}) sit inside it. "
            f"Read what follows as a comparison against a reference model, not as "
            f"a clean independent benchmark. The unseen-allele split is the more "
            f"informative test, and only 7 alleles fall outside its training set."
            f"</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Performance", "Error", "Unseen alleles", "Reliability"])
    comp = data.comparison()

    with tabs[0]:
        st.markdown(section("COMPARISON",
                            "How does this system compare with the baseline?"),
                    unsafe_allow_html=True)
        if comp is None:
            _missing("The comparison table")
        else:
            models = ["m1ens", "m2", "netmhcstabpan"]
            st.altair_chart(charts.spearman_by_split(comp, models),
                            use_container_width=True)
            _caption(f"{term('Spearman')} — higher is better. Teal is this system, "
                     f"violet is the reference tool.")
            _takeaway(
                "NetMHCstabpan scores highest on every split — but it was trained "
                "on this data, so those bars are a contaminated ceiling rather "
                "than a target. The informative comparison is the rightmost group.")

    with tabs[1]:
        st.markdown(section("ERROR", "How large are the mistakes, in hours?"),
                    unsafe_allow_html=True)
        if comp is None:
            _missing("The comparison table")
        else:
            st.altair_chart(
                charts.mae_by_split(comp, ["m1ens", "netmhcstabpan"]),
                use_container_width=True)
            _caption(f"{term('MAE')} — lower is better.")
            _takeaway(
                "Error looks smallest on the unseen-allele split for both models, "
                "which is misleading: that allele's complexes are short-lived, so "
                "the numbers are small whether or not the ranking is right. "
                "Ranking is the honest metric there.")

    with tabs[2]:
        st.markdown(section("DISTRIBUTION SHIFT",
                            "What happens when the allele has never been seen?",
                            "The hardest test, and the one this project is about."),
                    unsafe_allow_html=True)
        sv = data.seed_variance()
        if sv is None:
            _missing("The repeated-run table")
        else:
            st.altair_chart(charts.seed_spread(sv, "hla"), use_container_width=True)
            _caption("Dot is the mean over repeated training runs; the line spans "
                     "the lowest and highest run.")
            _takeaway(
                "The sequence model (m2n) is the only one above zero on an unseen "
                "allele; the identity-based models sit below it. The bars are wide "
                "because this split holds out a single allele — differences "
                "smaller than the bars are not real.")

            st.markdown(section("CONTROL", "Is that spread specific to this split?"),
                        unsafe_allow_html=True)
            st.altair_chart(charts.seed_spread(sv, "random"),
                            use_container_width=True)
            _takeaway("On the ordinary split the runs agree closely and the ranking "
                      "is unambiguous, so the wide bars above are a property of "
                      "holding out one allele, not of the training recipe.")

    with tabs[3]:
        st.markdown(section("RELIABILITY",
                            "Does the model know when it is likely to be wrong?"),
                    unsafe_allow_html=True)
        pa = data.per_allele()
        if pa is None:
            _missing("The per-allele table")
        else:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.altair_chart(charts.novelty_vs_error(pa), use_container_width=True)
                _caption("Each point is one held-out allele.")
            with c2:
                st.altair_chart(charts.novelty_vs_uncertainty(pa),
                                use_container_width=True)
                _caption("Same alleles, the model's own uncertainty.")
            _takeaway(
                "Error rises with novelty. The model's own uncertainty does not — "
                "it is flat to slightly falling. This is the project's central "
                "finding, and it is a negative one: ensemble disagreement does not "
                "detect an unfamiliar allele.")

        sel = data.selective()
        if sel is not None:
            st.markdown(section("SELECTIVE PREDICTION",
                                "Can it rank which individual predictions are wrong?"),
                        unsafe_allow_html=True)
            which = st.radio("Split", data.SPLITS, index=2, horizontal=True,
                             label_visibility="collapsed")
            st.altair_chart(charts.selective_curve(sel, which),
                            use_container_width=True)
            _caption("Discard the least confident predictions and measure error on "
                     "what is kept. Lower is better; 'oracle' is the unreachable "
                     "best case and 'random' is no information.")
            row = sel[(sel.split == which) & (sel.signal == "pred_err")]
            if not row.empty:
                _takeaway(
                    f"On this split the fitted error signal captures "
                    f"{row.pct_of_oracle.iloc[0]:.0f}% of the achievable gain. "
                    f"Novelty on its own performs worse than discarding at random, "
                    f"so 'unseen, therefore warn' is not usable per prediction.")


# ------------------------------------------------------------------ BIOLOGY --
def biology_view(hla: str) -> None:
    from app.structure import groove_for, locus_of, nearest_training_allele, viewer_html

    st.markdown(section("BIOLOGICAL CONTEXT", "Why does the HLA sequence matter?"),
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    groove = groove_for(hla)
    nn_name, novel_pos = nearest_training_allele(hla, groove)
    try:
        contacts = [p + 1 for p in json.loads(
            Path("data/external/contact_positions.json").read_text())]
    except Exception:
        contacts = []

    with left:
        html = viewer_html(hla, contacts, novel_pos, height=380)
        if html is None:
            _missing("The AlphaFold structure")
        else:
            import streamlit.components.v1 as components
            components.html(html, height=395)
            legend = (f"<span style='color:#b45309;font-weight:600'>amber</span> "
                      f"= the 34 peptide-contacting residues")
            if novel_pos:
                legend += (f" &nbsp;·&nbsp; <span style='color:#be123c;"
                           f"font-weight:600'>red</span> = {len(novel_pos)} "
                           f"position(s) where <span class='mono'>{hla}</span> "
                           f"differs from <span class='mono'>{nn_name}</span>, "
                           f"its closest allele in training")
            st.markdown(f"<div style='font-size:0.78rem;color:{MUTED}'>{legend}</div>",
                        unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                 color:{ACCENT}'>WHY THIS MATTERS</div>
<div style='font-size:0.85rem;color:#334155;line-height:1.55;margin-top:0.5rem'>
<p style='margin:0 0 0.6rem 0'><b>1.</b> {term('HLA')} molecules form the groove
that holds a {term('peptide')} and presents it to T cells.</p>
<p style='margin:0 0 0.6rem 0'><b>2.</b> Alleles differ at specific residues lining
that groove, which changes which peptides bind and how long they stay.</p>
<p style='margin:0 0 0.6rem 0'><b>3.</b> The sequence model uses those differences;
the identity-based model cannot, because a label carries no sequence.</p>
<p style='margin:0 0 0.6rem 0'><b>4.</b> Red marks show where this allele departs
from the closest one the model has measured — the pockets it has no evidence about.</p>
<p style='margin:0'><b>5.</b> The structure is context, not the predictor.</p>
</div>""", unsafe_allow_html=True)

        st.markdown(
            f"<div class='note' style='margin-top:0.9rem'><b>What this is not.</b> "
            f"No docking or structure-based prediction was performed. AlphaFold "
            f"provides one representative structure per HLA locus, not one per "
            f"allele, and the peptide is not modelled here — its footprint is shown "
            f"as the contact residues instead.</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------- ABOUT --
def about_view() -> None:
    st.markdown(section("METHODOLOGY", "How does this work?"), unsafe_allow_html=True)

    stages = [
        ("Data", "28,166 measured peptide–HLA half-lives across 75 alleles."),
        ("Splits", "Four held-out designs, each testing a different kind of novelty."),
        ("M1 baseline", "BLOSUM peptide features plus the allele as an identity label."),
        ("M2n sequence", "ESM-2 embeddings of the peptide and the HLA groove."),
        ("Ensemble", "Five runs per model; their disagreement is one uncertainty estimate."),
        ("Coverage", "How far this query sits from anything in training."),
        ("Support verdict", "Coverage and uncertainty combined into a reliability call."),
    ]
    for i, (name, desc) in enumerate(stages):
        st.markdown(
            f"<div style='display:flex;gap:0.8rem;align-items:flex-start;"
            f"padding:0.45rem 0'>"
            f"<div style='font-family:{MONO};font-size:0.7rem;color:{FAINT};"
            f"min-width:1.4rem;padding-top:2px'>{i+1:02d}</div>"
            f"<div><div style='font-weight:600;font-size:0.88rem;color:{INK}'>{name}</div>"
            f"<div style='font-size:0.8rem;color:{MUTED}'>{desc}</div></div></div>",
            unsafe_allow_html=True)

    st.markdown(section("THE CONTRIBUTION", "What is this project actually asking?"),
                unsafe_allow_html=True)
    st.markdown(
        f"""<div style='font-size:0.9rem;color:#334155;line-height:1.6'>
The question is not whether an unseen allele is risky — a tool always knows which
alleles it trained on, so that warning is free. The question is whether
<b>predicted uncertainty and training coverage can identify which individual
predictions are unreliable</b> under {term('distribution shift')}.<br><br>
Measured here, the answer differs by axis. Under peptide novelty the fitted
reliability signal does rank mistakes usefully. Under allele novelty it does not,
and the model's own ensemble uncertainty is flat to inversely related to how
novel an allele is. That negative result is reported rather than smoothed over,
because an uncertainty estimate that fails silently is worse than none.
</div>""", unsafe_allow_html=True)

    st.markdown(section("GLOSSARY", "What do the terms mean?"), unsafe_allow_html=True)
    from app.theme import GLOSSARY
    gl = pd.DataFrame({"Term": list(GLOSSARY), "Meaning": list(GLOSSARY.values())})
    st.dataframe(gl, hide_index=True, use_container_width=True)
