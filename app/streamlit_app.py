"""PepShield — peptide–MHC class I stability prediction with training-data awareness.

Run:  streamlit run app/streamlit_app.py

Four screens, in the order a reader needs them: what is predicted, how it
compares with the established tool, why the biology matters, and how the whole
thing works. The query composer stays above the tabs so a prediction is always
in context.

No scientific logic lives here. Models, routing, uncertainty and novelty are
unchanged; this module only arranges what they produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.client import predict  # noqa: E402
from app.reliability import alleles, assess, validate_peptide  # noqa: E402
from app.theme import CSS, INK, MUTED, term  # noqa: E402
from app.views import about_view, biology_view, compare_view, predict_view  # noqa: E402

st.set_page_config(page_title="PepShield", page_icon="🛡",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ header --
st.markdown(
    f"""<div style='display:flex;align-items:baseline;gap:0.75rem;
         border-bottom:1px solid rgba(15,23,42,0.08);padding-bottom:0.7rem'>
      <div style='font-size:1.6rem;font-weight:700;color:{INK};
                  letter-spacing:-0.02em'>PepShield</div>
      <div style='font-size:0.85rem;color:{MUTED}'>
        Peptide–MHC class I stability prediction, with training-data awareness
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown(
    f"<div style='font-size:0.84rem;color:{MUTED};margin:0.6rem 0 0.2rem 0'>"
    f"Predicts how long a {term('peptide')} stays bound to an {term('HLA')} "
    f"molecule — and, separately, how much evidence the model actually has for "
    f"that particular query.</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------- composer --
allele_meta = alleles()
names = sorted(allele_meta)

c1, c2, c3 = st.columns([3, 3, 4], gap="medium")
with c1:
    peptide = st.text_input(
        "PEPTIDE", value="VTTEVAFGL",
        help="8–11 amino acids. Every training measurement is a 9-mer.",
    ).strip().upper()
with c2:
    default = names.index("HLA-A*02:01") if "HLA-A*02:01" in names else 0
    picked = st.selectbox("HLA CLASS I", names + ["Other (type below)…"], index=default,
                          help="The 75 alleles with training data, or any other "
                               "IMGT allele via 'Other'.")
with c3:
    # The dropdown holds only trained alleles, so without this the one case the
    # project exists to warn about would be unreachable from the interface.
    if picked.startswith("Other"):
        hla = st.text_input(
            "ALLELE OUTSIDE TRAINING", value="HLA-C*07:02",
            help="Any IMGT name. The groove is looked up in IPD-IMGT/HLA and "
                 "embedded on demand, so all 46,406 alleles are reachable.",
        ).strip()
    else:
        hla = picked
        st.markdown(
            f"<div style='padding-top:1.9rem;font-size:0.78rem;color:{MUTED}'>"
            f"Choose <b>Other</b> to score an allele the model has never seen.</div>",
            unsafe_allow_html=True)

if not hla:
    st.warning("Enter an allele name.")
    st.stop()
if (error := validate_peptide(peptide)):
    st.warning(error)
    st.stop()

pred = predict(peptide, hla)
rel = assess(peptide, hla, model_std=pred.std)

# -------------------------------------------------------------------- tabs --
tab_predict, tab_compare, tab_biology, tab_about = st.tabs(
    ["Predict", "Compare", "Biology", "About"])

with tab_predict:
    predict_view(pred, rel, hla, peptide)

with tab_compare:
    compare_view()

with tab_biology:
    biology_view(hla)

with tab_about:
    about_view()

st.markdown(
    f"<div style='margin-top:2rem;padding-top:0.8rem;"
    f"border-top:1px solid rgba(15,23,42,0.08);font-size:0.76rem;color:{MUTED}'>"
    f"PepShield does not only predict stability. It asks whether we can "
    f"recognise when a prediction falls outside the model's reliable experience "
    f"— and reports honestly where that works and where it does not."
    f"</div>", unsafe_allow_html=True)
