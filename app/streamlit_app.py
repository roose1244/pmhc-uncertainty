"""PepShield — peptide–HLA stability with training-data awareness.

Run:  streamlit run app/streamlit_app.py

Three pages. Predict is the tool. Findings is the evidence. About is the
method. Query inputs live only on Predict; every other page has one job.

No scientific logic lives here. Models, routing, uncertainty and novelty are
untouched; this module arranges what they produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.client import predict  # noqa: E402
from app.reliability import alleles, assess, validate_peptide  # noqa: E402
from app.theme import CSS, MUTED, page_head, term  # noqa: E402
from app.views import about_view, compare_view, predict_view  # noqa: E402

st.set_page_config(
    page_title="PepShield",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

PAGES = ["Predict", "Findings", "About"]

top = st.columns([2, 5])
with top[0]:
    st.markdown(
        f"<div style='font-size:1.05rem;font-weight:600;color:#1c1917;"
        f"padding-top:0.35rem'>PepShield</div>",
        unsafe_allow_html=True,
    )
with top[1]:
    nav = st.radio("Page", PAGES, index=0, horizontal=True,
                   label_visibility="collapsed")

if nav == "Predict":
    st.markdown(
        page_head(
            "How long will this complex last?",
            f"Enter a {term('peptide')} and an {term('HLA')} allele. The model "
            f"estimates {term('half-life')} and how much training data supports "
            f"that estimate.",
        ),
        unsafe_allow_html=True,
    )

    allele_meta = alleles()
    names = sorted(allele_meta)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        peptide = st.text_input(
            "Peptide",
            value="VTTEVAFGL",
            help="8–11 amino acids. All training data is 9-mers.",
        ).strip().upper()
    with c2:
        default = names.index("HLA-A*02:01") if "HLA-A*02:01" in names else 0
        picked = st.selectbox(
            "HLA class I allele",
            names + ["Other (type below)…"],
            index=default,
            help="The 75 alleles with training data, or any IMGT allele via Other.",
        )

    if picked.startswith("Other"):
        hla = st.text_input(
            "Allele not in training",
            value="HLA-C*07:02",
            help="Any IMGT name. The groove is looked up in IPD-IMGT/HLA and "
                 "embedded on demand.",
        ).strip()
    else:
        hla = picked

    if not hla:
        st.warning("Enter an allele name.")
        st.stop()
    if (error := validate_peptide(peptide)):
        st.warning(error)
        st.stop()

    pred = predict(peptide, hla)
    rel = assess(peptide, hla, model_std=pred.std)
    predict_view(pred, rel, hla, peptide)

elif nav == "Findings":
    compare_view()

else:
    about_view()

st.markdown(
    f"<p style='margin-top:2.8rem;padding-top:0.8rem;"
    f"border-top:1px solid rgba(28,25,23,0.10);font-size:0.8rem;color:{MUTED}'>"
    f"Every figure is read from a results file an experiment wrote. "
    f"Where something has not been measured, this interface says so.</p>",
    unsafe_allow_html=True,
)
