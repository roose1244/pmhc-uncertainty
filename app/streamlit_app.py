"""pMHC Guardian: predicted stability with an honest reliability verdict.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.client import predict  # noqa: E402
from app.reliability import alleles, assess, validate_peptide  # noqa: E402

BADGE = {
    "green": ("#1a7f37", "Reliable", "The model has relevant training data for this query."),
    "amber": ("#9a6700", "Treat with caution", "Something about this query is thin."),
    "red": ("#b3261e", "Do not rely on this", "This query is outside what the model has seen."),
    "grey": ("#57606a", "", ""),
}

st.set_page_config(page_title="pMHC Guardian", page_icon="🛡", layout="centered")

st.title("pMHC Guardian")
st.caption(
    "Predicted peptide–MHC class I complex stability, with a verdict on "
    "whether that prediction should be trusted."
)

allele_meta = alleles()
names = sorted(allele_meta)
OTHER = "Other (type below)…"

st.session_state.setdefault("peptide", "FVRQCFNPM")
st.session_state.setdefault("allele_pick", "HLA-A*02:01")
st.session_state.setdefault("custom_hla", "HLA-C*07:02")


def _show_seen() -> None:
    st.session_state.peptide = "FVRQCFNPM"
    st.session_state.allele_pick = "HLA-A*02:01"


def _show_unseen_peptide() -> None:
    # Held out of the peptide-split training fold. Three substitutions from
    # the nearest training peptide, so the badge is caution, not failure.
    st.session_state.peptide = "GLYGNGILV"
    st.session_state.allele_pick = "HLA-A*02:01"


def _show_unseen() -> None:
    st.session_state.peptide = "FVRQCFNPM"
    st.session_state.allele_pick = OTHER
    st.session_state.custom_hla = "HLA-C*07:02"


b1, b2, b3 = st.columns(3)
b1.button("Seen pair", on_click=_show_seen)
b2.button("Unseen peptide", on_click=_show_unseen_peptide)
b3.button("Unseen HLA", on_click=_show_unseen)

col1, col2 = st.columns([3, 2])
with col1:
    peptide = st.text_input(
        "Peptide",
        key="peptide",
        help="9 amino acids. Every training measurement is a 9-mer.",
    ).strip().upper()
with col2:
    picked = st.selectbox("HLA allele", names + [OTHER], key="allele_pick")

# The dropdown holds only the 75 alleles with training data, so without this
# the reliability panel can never show a genuinely unseen allele.
if picked.startswith("Other"):
    hla = st.text_input(
        "Allele not in the training set",
        key="custom_hla",
        help="Any name outside the 75 stability-table alleles. "
             "HLA-C*07:02 was never measured. B*15:02 is in the table, "
             "so this frozen model has seen it.",
    ).strip()
    if not hla:
        st.warning("Enter an allele name.")
        st.stop()
else:
    hla = picked

error = validate_peptide(peptide)
if error:
    st.warning(error)
    st.stop()

pred = predict(peptide, hla)
if pred.error:
    st.error(pred.error)
    st.stop()
rel = assess(peptide, hla, model_std=pred.std)

if pred.source.startswith("local:"):
    which = pred.source.split(":", 1)[1]
    st.caption(
        f"Served by the frozen **{which}** model running locally. "
        + ("This allele was in training, so the one-hot model answers."
           if which == "m1" else
           "This allele was not in training, so the sequence-based model "
           "answers — the one-hot model would see an all-zero allele vector.")
    )
elif pred.source == "endpoint":
    st.caption("Served by the frozen M1 ensemble. " + (pred.verdict or ""))
elif pred.source != "endpoint":
    st.info(
        {
            "placeholder": "**Placeholder numbers.** No trained weights were "
            "found and no endpoint is configured, so the stability figures "
            "below are stand-ins. The reliability panel is computed from real "
            "training data and is already meaningful.",
            "cache": "Served from the offline cache rather than the live model.",
        }[pred.source]
    )

colour, verdict, gloss = BADGE[rel.badge]
st.markdown(
    f"<div style='background:{colour};color:#fff;padding:0.75rem 1rem;"
    f"border-radius:8px;font-weight:600;font-size:1.1rem'>{verdict}</div>",
    unsafe_allow_html=True,
)
st.caption(gloss)

lo_h, hi_h = pred.interval_hours
st.metric("Predicted half-life", f"{pred.half_life_hours:.1f} h")
if pred.hla_in_training is False:
    st.write(f"**Uncalibrated range:** {lo_h:.1f} – {hi_h:.1f} hours")
    st.caption(
        "Not a 90% interval. This allele was outside training, and the "
        "ensemble does not widen its spread to show that."
    )
else:
    st.write(f"**90% interval:** {lo_h:.1f} – {hi_h:.1f} hours")
    st.caption(
        "The interval is asymmetric in hours because the model works in log space. "
        "That is expected, not a display bug."
    )
st.progress(min(pred.half_life_hours / 24.0, 1.0))

st.subheader("Why")
for s in rel.signals:
    c, _, _ = BADGE[s.level]
    st.markdown(
        f"<span style='color:{c};font-weight:600'>●</span> **{s.name}** — "
        f"{s.value}<br><span style='color:#57606a;font-size:0.9em'>{s.detail}</span>",
        unsafe_allow_html=True,
    )

for reason in rel.reasons:
    st.write(f"- {reason}")

with st.expander("What this verdict is based on"):
    st.markdown(
        """
The verdict is driven by **novelty**, not by how much the ensemble members
disagree with each other.

That is a deliberate choice backed by measurement. Across held-out alleles,
error rises with novelty (Spearman **+0.546**) while ensemble spread does not
(**−0.055**). The model one-hot encodes the allele *name*, so every unseen
allele becomes the same zero vector and all ensemble members see identical
input — they cannot disagree *more* just because an allele is unfamiliar.

A badge driven by ensemble spread would therefore look principled and tell you
almost nothing about allele novelty. Model agreement is still shown, marked as
the weak signal it is.

**Known limits.** Interval coverage is about 90% overall but is not uniform:
per-allele it ranged 65%–100% in leave-one-allele-out testing. The 90% figure
is an average, not a promise for any single allele.
        """
    )
