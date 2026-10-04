"""PepShield: a research-style view of the frozen stability ensemble.

Run:  streamlit run app/streamlit_app.py

Every number on the results screens is read from results/tables. A live card
calls the frozen ensemble. A placeholder source is labelled as such.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.client import predict  # noqa: E402
from app.reliability import assess, validate_peptide  # noqa: E402
from app.structure import describe, viewer_html  # noqa: E402
from src.target import make_id  # noqa: E402

TABLES = ROOT / "results" / "tables"

EXAMPLES = (
    ("Seen pair", "FVRQCFNPM", "HLA-A*02:01"),
    ("Unseen peptide", "GLYGNGILV", "HLA-A*02:01"),
    ("Unseen HLA", "FVRQCFNPM", "HLA-C*07:02"),
)

PAGES = (
    "01  OVERVIEW",
    "02  THE PROBLEM",
    "03  HOW IT WORKS",
    "04  UNCERTAINTY",
    "05  MODEL COMPARISON",
)


def _css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
}
.stApp { background: #f3eee6; color: #1c1410; }
/* Streamlit's header is fixed and opaque. Without the clearance below it the
   section tabs sit underneath it and look missing. */
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 58rem; padding-top: 4.2rem; }
.marquee { overflow: hidden; white-space: nowrap; border-top: 1px solid #ddd4c8; border-bottom: 1px solid #ddd4c8; margin: 0 0 1rem; }
.marquee span { display: inline-block; padding-right: 2.5rem; letter-spacing: 0.18em; font-size: 0.68rem; color: #6a5e54; animation: ticker 28s linear infinite; }
@keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.sub { color: #6a5e54; font-size: 0.68rem; letter-spacing: 0.14em; text-align: center; margin: 0.2rem 0 1.4rem; }
div.st-key-title_enter { text-align: center; }
div.st-key-title_enter button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #1c1410 !important; font-size: 3.4rem !important; letter-spacing: 0.18em !important;
  padding: 0.2rem 0 !important; border-radius: 0 !important;
}
div.st-key-title_enter button:hover { opacity: 0.65; text-decoration: underline; text-underline-offset: 0.12em; }
div.st-key-title_enter button p {
  display: inline-block; overflow: hidden; white-space: nowrap;
  width: 0; margin: 0 auto; animation: typing 1.35s steps(9, end) forwards;
}
@keyframes typing { to { width: 11.2em; } }
.block-container { animation: rise 0.4s ease; }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.pipe { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; margin: 0.8rem 0 1.2rem; }
.pipe b { font-weight: 500; border: 1px solid #1c1410; padding: 0.28rem 0.45rem; font-size: 0.72rem; letter-spacing: 0.06em; }
.pipe i { font-style: normal; color: #6a5e54; }
.member { display: grid; grid-template-columns: 4.2rem 1fr 4.4rem; gap: 0.55rem; align-items: center; margin: 0.28rem 0; font-size: 0.82rem; }
.mtrack { height: 7px; background: #ddd4c8; border-radius: 99px; }
.mfill { height: 7px; background: #3d5346; border-radius: 99px; }
div.st-key-press_enter { text-align: center; }
div.st-key-press_enter button {
  background: transparent; color: #1c1410; border: 1px solid #1c1410;
  border-radius: 0; letter-spacing: 0.14em;
}
div.st-key-press_enter button:hover { background: #1c1410; color: #f3eee6; }
div.st-key-go_back button {
  background: transparent; color: #1c1410; border: 1px solid #1c1410;
  border-radius: 0; letter-spacing: 0.12em;
}
div.st-key-go_back button:hover { background: #1c1410; color: #f3eee6; }
div[data-testid="stProgress"] > div > div > div { background: #3d5346; }
.ibar {
  position: relative; height: 8px; margin: 0.45rem 0 0.35rem;
  border-radius: 999px; background: #d9d0c4; overflow: hidden;
}
.ibar .fill {
  position: absolute; top: 0; bottom: 0; background: #3d5346; border-radius: 999px;
}
h1, h2, h3 { letter-spacing: -0.03em; font-weight: 500; }
p, li { color: #1c1410; }
.kicker {
  color: #6a5e54; font-size: 0.72rem; letter-spacing: 0.16em;
  text-transform: uppercase; margin-bottom: 0.6rem;
}
.rule { border-top: 1px solid #ddd4c8; margin: 1.4rem 0; }
.card {
  border: 1px solid #1c1410; padding: 1rem 1.1rem; margin: 0.8rem 0 1rem;
  background: #f7f3ec;
}
.hours { font-size: 1.7rem; margin: 0.15rem 0 0.4rem; }
.muted { color: #6a5e54; font-size: 0.92rem; }
.warn { color: #8d2f2f; }
.olive { color: #3d5346; }
.illu {
  border: 1px dashed #b7ab9e; padding: 0.8rem 0.9rem; margin-top: 0.4rem;
}
div[data-testid="stRadio"] label { font-size: 0.78rem !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def _kicker(text: str) -> None:
    st.markdown(f"<div class='kicker'>{text}</div>", unsafe_allow_html=True)


def _note(text: str) -> None:
    st.markdown(f"<p class='muted'>{text}</p>", unsafe_allow_html=True)


@st.cache_data
def _table() -> pd.DataFrame:
    master = pd.read_parquet(ROOT / "data/processed/master.parquet")
    nms = pd.read_parquet(ROOT / "data/external/netmhcstabpan_preds.parquet")
    split = pd.read_parquet(ROOT / "data/splits/peptide.parquet")
    return master.merge(nms, on="id", how="left").merge(split[["id", "fold"]], on="id")


@st.cache_data
def _metrics() -> dict[str, pd.DataFrame]:
    frames = {}
    for name in ("m1ens", "comparison", "mpocket", "mhetero", "error_predictor"):
        path = TABLES / f"{name}.csv"
        if path.exists():
            frames[name] = pd.read_csv(path)
    return frames


def _row(frame: pd.DataFrame, split: str) -> pd.Series | None:
    hit = frame.loc[frame["split"] == split]
    if hit.empty:
        return None
    return hit.iloc[0]


def _reference(peptide: str, hla: str) -> dict | None:
    try:
        row_id = make_id(hla, peptide)
    except ValueError:
        return None
    hit = _table().loc[_table()["id"] == row_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    nms = row.nms_half_life
    return {
        "half_life": float(row.half_life),
        "nms_half_life": None if nms != nms else float(nms),
        "fold": str(row.fold),
    }


def _use_example(peptide: str, hla: str) -> None:
    st.session_state.peptide = peptide
    st.session_state.hla_text = hla
    st.session_state.show_pred = True
    st.session_state.reveal = False


def _enter() -> None:
    st.session_state.inside = True


def landing() -> None:
    """Full-viewport title. Motion follows the GSAP showcase: staggered type, then a quiet enter."""
    letters = "".join(f"<span class='ch'>{ch}</span>" for ch in "PEPSHIELD")
    components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
<style>
  html, body {{ margin: 0; background: #f3eee6; color: #1c1410; font-family: "IBM Plex Mono", monospace; }}
  .stage {{ min-height: 88vh; display: flex; flex-direction: column; justify-content: flex-end; padding: 6vh 4vw 8vh; box-sizing: border-box; }}
  .rail {{ overflow: hidden; white-space: nowrap; font-size: 11px; letter-spacing: 0.22em; color: #6a5e54; margin-bottom: auto; }}
  .rail span {{ display: inline-block; padding-right: 3rem; }}
  h1 {{ font-weight: 500; font-size: clamp(52px, 11vw, 112px); letter-spacing: 0.02em; line-height: 0.88; margin: 0; }}
  h1 .ch {{ display: inline-block; }}
  .sub {{ margin-top: 1.2rem; font-size: 12px; letter-spacing: 0.16em; color: #6a5e54; }}
</style>
</head>
<body>
  <div class="stage">
    <div class="rail" id="rail">
      <span>PEPTIDE → HLA → PREDICTION → UNCERTAINTY</span>
      <span>ENSEMBLE · INTERVAL · STRUCTURE</span>
      <span>PEPTIDE → HLA → PREDICTION → UNCERTAINTY</span>
      <span>ENSEMBLE · INTERVAL · STRUCTURE</span>
    </div>
    <h1 id="title">{letters}</h1>
    <div class="sub" id="sub">UNCERTAINTY-AWARE PEPTIDE–HLA STABILITY PREDICTION</div>
  </div>
  <script>
    gsap.set(".ch", {{ y: 110, opacity: 0 }});
    gsap.to(".ch", {{ y: 0, opacity: 1, duration: 0.9, stagger: 0.055, ease: "power4.out", delay: 0.12 }});
    gsap.from("#sub", {{ y: 18, opacity: 0, duration: 0.7, delay: 0.85, ease: "power3.out" }});
    gsap.to("#rail", {{ x: -420, duration: 22, repeat: -1, ease: "none" }});
  </script>
</body>
</html>
""",
        height=560,
    )
    # The button has to be a real Streamlit widget. A button inside the component
    # iframe cannot navigate the page when the iframe is cross-origin, which is
    # what happens once the app is served from Modal rather than localhost.
    st.button("PRESS TO ENTER", key="press_enter", on_click=_enter, use_container_width=True)


def _interval_bar(lo: float, hi: float) -> None:
    """Thin bar under the interval. The track is log hours, so the fill is uneven."""
    if not (hi > lo >= 0):
        return

    def lx(hours: float) -> float:
        return math.log10(max(hours, 0.05))

    left, right = lx(0.1), lx(max(hi, 24.0))
    span = right - left or 1.0

    def pct(hours: float) -> float:
        return min(100.0, max(0.0, (lx(hours) - left) / span * 100.0))

    start, end = pct(lo), pct(hi)
    st.markdown(
        f"""
<div class="ibar" title="90% interval on a log-hour scale">
  <div class="fill" style="left:{start:.2f}%;width:{max(end - start, 1.2):.2f}%"></div>
</div>
""",
        unsafe_allow_html=True,
    )
    _note("The interval is asymmetric in hours because the model works in log space. That is expected, not a display bug.")


def _structure_panel(hla: str, peptide: str, groove_only: bool, show_contacts: bool, height: int) -> None:
    _kicker("HLA structure")
    info = describe(hla)
    if not info.available:
        st.markdown("**3D structure unavailable.**")
        _note("No AlphaFold structure for this locus.")
        return
    html = viewer_html(hla, groove_only=groove_only, show_contacts=show_contacts, height=height)
    if html is None:
        st.markdown("**3D structure unavailable.**")
        return
    components.html(html, height=height + 8)
    st.markdown(f"**{peptide}** · **{hla}**")
    plddt = f"{info.plddt_groove:.0f}" if info.plddt_groove is not None else "—"
    _note(
        f"AlphaFold prediction of a reference HLA-{info.locus} protein ({info.entry_id}). "
        f"Not this allele, and the peptide is not in the file. Groove pLDDT {plddt}."
    )


def prediction_card() -> None:
    c1, c2, c3 = st.columns(3)
    for col, (name, peptide, hla) in zip((c1, c2, c3), EXAMPLES):
        col.button(name, on_click=_use_example, args=(peptide, hla), use_container_width=True)

    peptide = st.text_input("Peptide", key="peptide").strip().upper()
    hla = st.text_input("HLA allele", key="hla_text").strip()
    if st.button("Run prediction"):
        st.session_state.show_pred = True
        st.session_state.reveal = False

    if not st.session_state.get("show_pred"):
        _note("Choose a measured example, or type a 9-mer and an allele. Nothing on this card is filled in until the ensemble runs.")
        return
    if not peptide or not hla:
        return
    error = validate_peptide(peptide)
    if error:
        st.warning(error)
        return

    with st.spinner("Asking the frozen ensemble"):
        pred = predict(peptide, hla)
    if pred.error:
        st.error(pred.error)
        return
    if pred.source == "placeholder":
        st.markdown(
            "<p class='warn'>Placeholder numbers. No weights and no endpoint answered, so this card is not a result.</p>",
            unsafe_allow_html=True,
        )
        return

    rel = assess(peptide, hla, model_std=pred.std)
    support = {
        "green": "Well supported by training data",
        "amber": "Limited support",
        "red": "Outside the measured alleles",
    }[rel.badge]

    ref = _reference(peptide, hla)

    groove_only = st.toggle("Show binding groove", key="groove_view")
    show_contacts = st.toggle("Highlight peptide-contact residues", key="show_contacts")
    larger = st.toggle("Larger view", key="structure_large")
    pred_col, structure_col = st.columns(2)
    with pred_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kicker'>{peptide} / {hla}</div>", unsafe_allow_html=True)
        st.markdown("**Predicted stability**")
        st.progress(min(pred.half_life_hours / 24.0, 1.0))
        st.markdown(f"<div class='hours'>{pred.half_life_hours:.1f} h</div>", unsafe_allow_html=True)
        st.markdown(f"**Ensemble disagreement:** {pred.std:.3f}")
        if pred.hla_in_training is False:
            st.markdown("<p class='warn'><strong>No calibrated interval.</strong></p>", unsafe_allow_html=True)
        else:
            lo_h, hi_h = pred.interval_hours
            st.markdown(f"**90% interval:** {lo_h:.1f} – {hi_h:.1f} hours")
            _interval_bar(lo_h, hi_h)
        st.markdown(f"**Support:** {support}")
        st.markdown("</div>", unsafe_allow_html=True)
    with structure_col:
        _structure_panel(hla, peptide, groove_only, show_contacts, 520 if larger else 360)

    _members(pred)

    if ref and ref["nms_half_life"] is not None:
        st.markdown(f"**NetMHCstabpan:** {ref['nms_half_life']:.2f} h")
        _note("Trained on this same table. Not an independent comparison.")
    if ref and ref["fold"] == "test":
        if st.button("Show held-out measurement"):
            st.session_state.reveal = True
            st.rerun()
        if st.session_state.get("reveal"):
            measured = ref["half_life"]
            lo_h, hi_h = pred.interval_hours
            where = "Inside the 90% interval." if lo_h <= measured <= hi_h else "Outside the 90% interval."
            st.markdown(f"**Measured half-life:** {measured:.1f} h")
            _note(where)
    elif ref is None:
        _note("No measured half-life for this pair.")


def _explain(what: str, why: str) -> None:
    with st.expander("What am I looking at?"):
        st.write(what)
        st.write(why)


def _members(pred) -> None:
    hours = tuple(getattr(pred, "member_hours", ()) or ())
    if len(hours) < 2:
        return
    scale = max(hours) or 1.0
    rows = []
    for index, value in enumerate(hours, start=1):
        width = 100.0 * value / scale
        rows.append(
            f"<div class='member'><span>Copy {index}</span>"
            f"<div class='mtrack'><div class='mfill' style='width:{width:.1f}%'></div></div>"
            f"<span>{value:.2f} h</span></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    _explain(
        "Each row is one of the five copies. The length is that copy’s predicted half-life. These are the model’s own outputs for this query, not an illustration.",
        "If the rows end in nearly the same place, the copies agree. Agreement is model disagreement, which is separate from the calibrated interval.",
    )


def screen_overview() -> None:
    _kicker("PepShield")
    st.header("Can the model know when it is wrong?")
    st.write(
        "PepShield predicts peptide–HLA stability and reports how far five copies of the model disagree."
    )
    st.markdown(
        """
<div class="pipe">
  <b>PEPTIDE + HLA</b><i>→</i>
  <b>PREDICTOR</b><i>→</i>
  <b>FIVE COPIES</b><i>→</i>
  <b>PREDICTION + DISAGREEMENT</b><i>→</i>
  <b>STRUCTURE</b>
</div>
""",
        unsafe_allow_html=True,
    )
    prediction_card()


def screen_problem() -> None:
    _kicker("02  The problem")
    st.header("A model can agree with itself and still be wrong")
    st.write(
        "A stability tool usually returns one number. That number does not say whether the peptide was in training, "
        "whether this HLA allele was measured, whether the five copies agree, or whether a claimed interval was calibrated."
    )
    st.markdown(
        """
<div class="card">
<div class="kicker">A more informative prediction</div>
<p>Predicted half-life</p>
<p>+ ensemble disagreement</p>
<p>+ 90% interval, only if the allele was in the training table</p>
</div>
""",
        unsafe_allow_html=True,
    )
    _note(
        "What you are looking at: the three fields the served model can actually return. "
        "Disagreement is the spread of five copies. The interval is that spread multiplied by a factor fit on held-out rows. "
        "They answer different questions."
    )
    st.write(
        "On peptides the model has not seen, higher disagreement goes with larger error, weakly. "
        "On a held-out HLA allele, the copies do not spread out as the error grows. "
        "That is why an allele outside the table gets a prediction and no interval."
    )


def screen_how() -> None:
    _kicker("03  How it works")
    st.header("From sequence to reliability")
    steps = [
        ("Peptide", "Nine amino acids. Every training measurement is a 9-mer."),
        ("HLA", "The allele the peptide is bound to. The served model sees the allele’s name. A second model sees the 34 residues that touch the peptide."),
        ("Representation", "The served path encodes each peptide letter with BLOSUM50, a fixed table of which amino-acid swaps are common. ESM-2, a frozen protein language model, was the other representation: it turns a sequence into a vector. Averaging that vector over the whole binding groove washed out the few substitutions that change stability."),
        ("Stability model", "A small network trained to match log10(half-life + 0.1). The 0.1 is there because about 20% of the published measurements are written as 0.00 hours, below the assay floor."),
        ("Ensemble", "Five copies, different random seeds, each trained on a resample of the training rows. They never see the calibration rows or the test rows."),
        ("Prediction", "The average of the five copies, converted back to hours."),
        ("Uncertainty", "The standard deviation of the five copies. Call it ensemble predictive uncertainty. If they disagree, the model has less agreement about the answer. This is not a measurement of biological uncertainty."),
    ]
    for title, body in steps:
        st.markdown(f"<div class='kicker'>{title}</div>", unsafe_allow_html=True)
        st.write(body)
    _note(
        "The 90% interval is a separate step. On a calibration fold, we measure how many disagreement-units the real error usually spans, and multiply. "
        "That makes average coverage about 90% where the allele was seen. It does not make the disagreement itself a perfect ranking of errors."
    )


def _figure(stem: str, title: str, what: str, why: str) -> None:
    path = ROOT / "results" / "figures" / f"{stem}.svg"
    st.subheader(title)
    if not path.exists():
        st.markdown("**Analysis not available.**")
        return
    st.image(str(path))
    _explain(what, why)


def screen_uncertainty() -> None:
    _kicker("04  Uncertainty")
    st.header("Prediction is not the whole answer")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='kicker'>Illustration · copies agree</div>", unsafe_allow_html=True)
        st.write("0.69 · 0.70 · 0.71 · 0.70 · 0.69")
    with right:
        st.markdown("<div class='kicker'>Illustration · copies disagree</div>", unsafe_allow_html=True)
        st.write("0.48 · 0.61 · 0.72 · 0.55 · 0.79")
    _note("Those two rows are a sketch of the idea. They are not outputs from this model.")
    _figure(
        "fig2_uncertainty_vs_error",
        "Disagreement against absolute error",
        "Each point is a test prediction. Horizontal is how far the five copies disagreed. Vertical is how far the prediction missed the measured half-life.",
        "If disagreement were a strong warning, the points would rise together. On unseen peptides the rise is real and modest.",
    )
    st.header("Does disagreement point at the larger errors?")
    frames = _metrics()
    ens = frames.get("m1ens")
    if ens is None:
        st.warning("results/tables/m1ens.csv is missing. This screen has nothing to show.")
        return
    peptide = _row(ens, "peptide")
    held = _row(ens, "hla")
    st.write(
        "The test below uses the peptide split: no peptide in the test set was in training, and the alleles were seen. "
        "Disagreement is correlated with absolute error at "
        f"**{peptide.err_unc_spearman_test:.2f}**. "
        "A claimed 90% interval covers "
        f"**{peptide.coverage_90:.0%}** of those peptides."
    )
    kept = pd.DataFrame(
        {
            "Predictions kept": ["100%", "80% most agreed", "50% most agreed", "Random 50%"],
            "Mean absolute error (hours)": [
                round(float(peptide.mae_100), 2),
                round(float(peptide.mae_80), 2),
                round(float(peptide.mae_50), 2),
                round(float(peptide.random_mae_50), 2),
            ],
        }
    )
    st.dataframe(kept, hide_index=True, use_container_width=True)
    _note(
        "What you are looking at: error in hours as we keep only the predictions whose five copies agree most. "
        "On unseen peptides, the most-agreed half is more accurate than a random half. The signal is real and modest."
    )
    if held is not None:
        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.subheader("The same signal on a held-out allele")
        st.write(
            f"On HLA-B*15:02, which was removed from training, disagreement and error correlate at "
            f"**{held.err_unc_spearman_test:.2f}**. "
            f"Coverage is still **{held.coverage_90:.0%}**, because the intervals are wide "
            f"(median about {held.median_width_hours:.0f} hours), not because the copies noticed the new allele."
        )
        _note("Keeping the most-agreed half does not reduce the error on that allele. Disagreement is not a warning for an unseen HLA.")


def screen_comparison() -> None:
    _kicker("05  Model comparison")
    st.header("What changed the ranking, and what carries a warning")
    _note(
        "Spearman rank correlation should be high. Mean absolute error in hours should be low. "
        "Pearson correlation was not in the locked tables, so it is not shown. "
        "ProtT5 and a cross-attention model are not in these results. "
        "NetMHCstabpan was trained on this file. It is a comparator, not the laboratory answer."
    )
    frames = _metrics()
    comp = frames.get("comparison")
    pocket = frames.get("mpocket")
    if comp is None:
        st.warning("results/tables/comparison.csv is missing.")
        return
    rows = []
    labels = {
        "m1": "Allele name",
        "m2": "ESM-2 average",
        "m1ens": "Ensemble of the allele-name model",
        "netmhcstabpan": "NetMHCstabpan (in-sample)",
    }
    for split, split_name in (("peptide", "Unseen peptides"), ("hla", "Held-out B*15:02")):
        part = comp.loc[comp["split"] == split]
        for _, row in part.iterrows():
            if row.model not in labels:
                continue
            rows.append(
                {
                    "Model": labels[row.model],
                    "Split": split_name,
                    "Rank correlation": round(float(row.spearman), 2),
                    "Disagreement vs error": "—"
                    if pd.isna(row.err_unc_spearman)
                    else f"{float(row.err_unc_spearman):.2f}",
                }
            )
    if pocket is not None:
        for split, split_name in (("peptide", "Unseen peptides"), ("hla", "Held-out B*15:02")):
            row = _row(pocket, split)
            if row is None:
                continue
            rows.append(
                {
                    "Model": "34 contact residues",
                    "Split": split_name,
                    "Rank correlation": round(float(row.spearman), 2),
                    "Disagreement vs error": "",
                }
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    _note(
        "What you are looking at: Spearman rank correlation with measured half-life. "
        "Empty cells mean that model does not produce an ensemble disagreement. "
        "The contact-residue model is the one that ranks a new allele above chance. "
        "The ensemble is the one the live card serves, because it is the model with a calibrated interval."
    )


def main() -> None:
    st.set_page_config(page_title="PepShield", page_icon="◻", layout="centered")
    _css()
    st.session_state.setdefault("inside", False)
    st.session_state.setdefault("show_pred", True)
    st.session_state.setdefault("reveal", False)
    st.session_state.setdefault("peptide", "GLYGNGILV")
    st.session_state.setdefault("hla_text", "HLA-A*02:01")
    if st.query_params.get("enter") == "1":
        _enter()
    if not st.session_state.inside:
        landing()
        return

    page = st.radio(
        "Section",
        PAGES,
        horizontal=True,
        label_visibility="collapsed",
        key="section",
    )
    if page != PAGES[0]:
        st.button("GO BACK", key="go_back", on_click=lambda: st.session_state.update(section=PAGES[0]))
    st.markdown(
        "<div class='marquee'><span>PEPTIDE → HLA → PREDICTION → UNCERTAINTY · "
        "ENSEMBLE · INTERVAL · STRUCTURE · "
        "PEPTIDE → HLA → PREDICTION → UNCERTAINTY · "
        "ENSEMBLE · INTERVAL · STRUCTURE · </span></div>",
        unsafe_allow_html=True,
    )
    {
        PAGES[0]: screen_overview,
        PAGES[1]: screen_problem,
        PAGES[2]: screen_how,
        PAGES[3]: screen_uncertainty,
        PAGES[4]: screen_comparison,
    }[page]()


if __name__ == "__main__":
    main()
else:
    main()
