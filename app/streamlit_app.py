"""pMHC Guardian: predicted stability with an honest reliability verdict.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from app.client import predict  # noqa: E402
from app.reliability import alleles, assess, validate_peptide  # noqa: E402
from src.target import make_id  # noqa: E402


@st.cache_data
def _table() -> pd.DataFrame:
    master = pd.read_parquet("data/processed/master.parquet")
    nms = pd.read_parquet("data/external/netmhcstabpan_preds.parquet")
    split = pd.read_parquet("data/splits/peptide.parquet")
    return master.merge(nms, on="id", how="left").merge(split[["id", "fold"]], on="id")


def _reference(peptide: str, hla: str) -> dict | None:
    try:
        row_id = make_id(hla, peptide)
    except ValueError:
        return None
    hit = _table().loc[_table()["id"] == row_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "half_life": float(row.half_life),
        "nms_half_life": float(row.nms_half_life) if pd.notna(row.nms_half_life) else float("nan"),
        "fold": str(row.fold),
    }

BADGE = {
    "green": ("#1a7f37", "Reliable", "The model has relevant training data for this query."),
    "amber": ("#9a6700", "Treat with caution", "Something about this query is thin."),
    "red": ("#b3261e", "Do not rely on this", "This query is outside what the model has seen."),
    "grey": ("#57606a", "", ""),
}

st.set_page_config(page_title="pMHC Guardian", page_icon="🛡", layout="centered")

st.title("pMHC Guardian")
st.caption(
    "Peptide–HLA stability for immunotherapy design. NetMHCstabpan was trained "
    "on this public table. This screen says whether the number should be trusted."
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

if pred.source == "local":
    st.caption("Served by the frozen ensemble on this machine. " + (pred.verdict or ""))
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
    st.write("**No calibrated interval.**")
    st.caption(
        "This allele was outside training. The ensemble stays narrow there, "
        "so a numeric range would look precise without being a 90% interval. "
        "Scaling the width by HLA distance was tested on ten held-out alleles "
        "and did not improve coverage."
    )
else:
    st.write(f"**90% interval:** {lo_h:.1f} – {hi_h:.1f} hours")
    st.caption(
        "The interval is asymmetric in hours because the model works in log space. "
        "That is expected, not a display bug."
    )
st.progress(min(pred.half_life_hours / 24.0, 1.0))

reference = _reference(peptide, hla)
if reference is not None and reference["nms_half_life"] == reference["nms_half_life"]:
    st.write(f"**NetMHCstabpan:** {reference['nms_half_life']:.2f} h")
    st.caption(
        "Trained on this same public table, including rows we held out. "
        "Not an independent comparison."
    )
if reference is not None and reference["fold"] == "test":
    reveal = f"reveal:{hla}|{peptide}"
    if st.button("Show held-out measurement"):
        st.session_state[reveal] = True
    if st.session_state.get(reveal):
        measured = reference["half_life"]
        lo_raw, hi_raw = pred.interval_hours
        inside = lo_raw <= measured <= hi_raw
        st.write(f"**Measured half-life:** {measured:.1f} h")
        st.caption(
            "Held out of this model's training fold. "
            + ("Inside the 90% interval." if inside else "Outside the 90% interval.")
        )

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
