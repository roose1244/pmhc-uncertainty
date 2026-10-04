"""PepShield: predicted stability with an honest reliability verdict.

Run:  streamlit run app/streamlit_app.py

The layout answers two questions side by side rather than stacked, because
they are different facts and the old stacked version let each be read as a
verdict on the other:

    how long?        the estimate and its 90% interval
    should I trust?  the badge, from how well training data covers this query

A model can have plenty of relevant data and still be imprecise, and it can be
falsely precise on an allele it has never measured. Showing "Reliable" above a
fifty-fold interval invites the obvious objection, so the two now sit in
adjacent columns with their own headings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.client import predict  # noqa: E402
from app.components import gauge, interval_bar, stat_block, verdict_card  # noqa: E402
from app.reliability import alleles, assess, validate_peptide  # noqa: E402
from app.structure import (  # noqa: E402
    groove_for, locus_of, nearest_training_allele, viewer_html,
)

BADGE = {
    "green": ("#2da44e", "Well supported",
              "Training data covers this peptide and this allele."),
    "amber": ("#bf8700", "Thin support",
              "Something about this query is sparsely represented."),
    "red": ("#cf222e", "Outside training data",
            "Do not rely on this number without experimental confirmation."),
    "grey": ("#8c959f", "", ""),
}

st.set_page_config(page_title="PepShield", page_icon="🛡", layout="centered")
st.markdown("<style>div.block-container{padding-top:2.2rem}</style>",
            unsafe_allow_html=True)

st.title("PepShield")
st.caption("Peptide–MHC class I stability, with an explicit account of how far "
           "the query sits from the training data.")

# ---------------------------------------------------------------- inputs ----
allele_meta = alleles()
names = sorted(allele_meta)

col1, col2 = st.columns([3, 2])
with col1:
    peptide = st.text_input(
        "Peptide", value="VTTEVAFGL",
        help="8–11 amino acids. Every training measurement is a 9-mer.",
    ).strip().upper()
with col2:
    default = names.index("HLA-A*02:01") if "HLA-A*02:01" in names else 0
    picked = st.selectbox("HLA allele", names + ["Other (type below)…"], index=default)

# The dropdown holds only the 75 alleles with training data, so without this
# the panel could never show a genuinely unseen allele, which is the one case
# the project exists to warn about.
if picked.startswith("Other"):
    hla = st.text_input(
        "Allele not in the training set", value="HLA-C*07:02",
        help="Any IMGT name. The groove is looked up in IPD-IMGT/HLA and "
             "embedded on demand, so all 46,406 alleles are reachable.",
    ).strip()
    if not hla:
        st.warning("Enter an allele name.")
        st.stop()
else:
    hla = picked

if (error := validate_peptide(peptide)):
    st.warning(error)
    st.stop()

pred = predict(peptide, hla)
rel = assess(peptide, hla, model_std=pred.std)
colour, verdict, gloss = BADGE[rel.badge]
lo_h, hi_h = pred.interval_hours

# ----------------------------------------------------------------- result ---
left, right = st.columns([3, 2], gap="large")

with left:
    # A ratio is only meaningful when the lower bound is resolvable. Below
    # 0.1 h the assay floor dominates and "138x wide" would be an artefact of
    # clamping a zero, not a property of the interval.
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

with right:
    st.markdown(verdict_card(colour, verdict, gloss), unsafe_allow_html=True)
    if pred.source.startswith("local:"):
        which = pred.source.split(":", 1)[1]
        why = ("allele is in training, so the one-hot model answers"
               if which == "m1" else
               "allele is not in training, so the sequence model answers")
        st.markdown(
            f"<div style='font-size:0.72rem;color:#8c959f;margin-top:0.5rem'>"
            f"Routed to <b>{which}</b> — {why}.</div>", unsafe_allow_html=True)
    elif pred.source == "placeholder":
        st.markdown(
            "<div style='font-size:0.72rem;color:#bf8700;margin-top:0.5rem'>"
            "Placeholder numbers — no trained weights found. The panel below "
            "is still computed from real training data.</div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- gauges ----
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
st.markdown("##### How well is this query covered?")

gcols = st.columns(len(rel.signals), gap="medium")
for col, s in zip(gcols, rel.signals):
    with col:
        st.markdown(gauge(s.name, s.value, s.level, s.fraction, s.detail),
                    unsafe_allow_html=True)

for reason in rel.reasons:
    st.markdown(f"<div style='font-size:0.84rem;color:#8c959f;margin-top:0.2rem'>"
                f"• {reason}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ notes ---
# ------------------------------------------------------------- structure ---
with st.expander("Where this allele sits on the binding groove", expanded=False):
    import json as _json

    import streamlit.components.v1 as _components

    try:
        contacts = [p + 1 for p in _json.loads(
            Path("data/external/contact_positions.json").read_text())]
    except Exception:
        contacts = []

    # Mark where this allele differs from its nearest TRAINING allele. That is
    # the project's own novelty measure put on the structure: not "this allele
    # is unusual" in the abstract, but which pockets differ from anything the
    # model has measured.
    nn_name, novel_pos = nearest_training_allele(hla, groove_for(hla))

    html = viewer_html(hla, contacts, novel_pos)
    if html is None:
        st.caption("Structure unavailable — AlphaFold could not be reached.")
    else:
        _components.html(html, height=360)
        legend = ("<span style='color:#bf8700'>amber</span> = the 34 "
                  "peptide-contacting residues")
        if novel_pos:
            legend += (f" &nbsp;·&nbsp; <span style='color:#cf222e'>red</span> = "
                       f"{len(novel_pos)} position(s) where {hla} differs from "
                       f"{nn_name}, its closest allele in training")
        st.markdown(f"<div style='font-size:0.78rem;color:#8c959f'>{legend}</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.72rem;color:#8c959f;margin-top:0.5rem'>"
            f"Backbone is AlphaFold's model of the HLA-{locus_of(hla)} heavy "
            f"chain. AlphaFold has one structure per locus, not per allele, so "
            f"the fold is the locus consensus — alleles differ at side chains, "
            f"not fold. The peptide is not modelled: a 9-mer has no AlphaFold "
            f"entry and the complex would need co-folding, so its footprint is "
            f"shown as the contact residues instead.</div>",
            unsafe_allow_html=True)

with st.expander("What the verdict is based on, and what it is not"):
    st.markdown(
        """
**The badge is driven by novelty, not by the model's own confidence.** That is
a measured choice. Across held-out alleles, error rises with novelty
(Spearman **+0.546**) while ensemble spread does not (**−0.055**). M1 one-hot
encodes the allele *name*, so every unseen allele is the same zero vector and
the ensemble members see identical input — they cannot disagree more simply
because an allele is unfamiliar.

**Giving the ensemble sequence features makes this worse, not better.** The
ESM-based ensemble's spread is *inversely* related to novelty (**−0.46**): it
grows more confident the further an allele sits from training, because
out-of-distribution inputs pull every member toward the same default answer.
Model agreement is shown above for completeness and never drives the badge.

**The two columns answer different questions.** Support says whether training
data covers this query. The interval says how precise the estimate is. A
well-supported query can still carry a wide interval, and that is not a
contradiction.

**Known limits.** 90% coverage is an average. Per allele it ranged 65%–100% in
leave-one-allele-out testing, so it is not a promise for any single allele.
Uncertainty tracks peptide novelty as a graded signal (it rises monotonically
with the number of mutations) but under-reports by roughly 2×: at eight
mutations the prediction moves about twice as far as the stated uncertainty
admits.
        """
    )
