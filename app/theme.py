"""Design tokens and global CSS for PepShield.

One accent colour, used only for the product's own identity and for chart
marks that represent this system. Green, amber and red are reserved for
support states and never used decoratively, so a colour in the interface
always means the same thing.

Streamlit's defaults are deliberately overridden rather than themed: the stock
chrome reads as an internal dashboard, and the point of this screen is that a
reader who has never heard of pMHC can follow it.
"""

from __future__ import annotations

# Core palette
INK = "#0f172a"          # headings
BODY = "#334155"         # body text
MUTED = "#64748b"        # captions
FAINT = "#94a3b8"        # axis labels, tick text
HAIRLINE = "rgba(15,23,42,0.08)"
SURFACE = "#ffffff"
CANVAS = "#f8fafc"

ACCENT = "#0e7490"       # teal: this system
ACCENT_SOFT = "#ecfeff"
REFERENCE = "#7c3aed"    # violet: NetMHCstabpan, the reference model

OK = "#15803d"
WARN = "#b45309"
RISK = "#be123c"
NEUTRAL = "#94a3b8"

SUPPORT = {
    "green": (OK, "HIGH", "Training data covers this peptide and this allele."),
    "amber": (WARN, "MODERATE", "Part of this query is sparsely represented."),
    "red": (RISK, "LIMITED", "This query sits outside the training data."),
    "grey": (NEUTRAL, "—", ""),
}

MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', monospace")

CSS = f"""
<style>
  .block-container {{ padding-top: 3.2rem; max-width: 1080px; }}
  #MainMenu, footer {{ visibility: hidden; }}

  h1, h2, h3, h4, h5 {{ color: {INK}; letter-spacing: -0.015em; }}

  /* Section headers: a label and the question that section answers. */
  .sec {{ margin: 1.5rem 0 0.7rem 0; }}
  .sec-label {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {ACCENT};
  }}
  .sec-q {{ font-size: 1.02rem; font-weight: 600; color: {INK}; margin-top: 1px; }}
  .sec-sub {{ font-size: 0.82rem; color: {MUTED}; margin-top: 2px; }}

  .mono {{ font-family: {MONO}; }}

  /* Inline glossary: dotted underline, native tooltip on hover. */
  .term {{
    border-bottom: 1px dotted {FAINT}; cursor: help;
  }}

  .card {{
    background: {SURFACE}; border: 1px solid {HAIRLINE};
    border-radius: 10px; padding: 0.95rem 1.05rem;
  }}

  .note {{
    border-left: 3px solid {WARN}; background: #fffbeb;
    padding: 0.7rem 0.9rem; border-radius: 0 8px 8px 0;
    font-size: 0.84rem; color: #78350f;
  }}

  .pill {{
    display:inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
  }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {HAIRLINE}; }}
  .stTabs [data-baseweb="tab"] {{
    height: 40px; padding: 0 18px; font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase; color: {MUTED};
  }}
  .stTabs [aria-selected="true"] {{ color: {ACCENT}; }}

  div[data-testid="stTextInput"] input,
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    font-family: {MONO}; font-size: 0.9rem;
  }}
</style>
"""


def section(label: str, question: str, sub: str = "") -> str:
    """Every section says which question it answers."""
    s = f"<div class='sec-sub'>{sub}</div>" if sub else ""
    return (f"<div class='sec'><div class='sec-label'>{label}</div>"
            f"<div class='sec-q'>{question}</div>{s}</div>")


GLOSSARY = {
    "pMHC": "A peptide bound to an MHC molecule — the complex a T cell sees.",
    "HLA": "Human Leukocyte Antigen: the human MHC protein that holds a peptide "
           "fragment and presents it to T cells.",
    "peptide": "A short protein fragment. Every measurement here is 9 amino acids long.",
    "half-life": "How long the peptide–HLA complex holds together before half of it "
                 "has come apart. Longer usually means better presented.",
    "90% interval": "A range the true value is expected to fall in 9 times out of 10. "
                    "Wider means less precise.",
    "ensemble": "Several copies of the same model trained with different random "
                "starts. How much they disagree is one way to estimate uncertainty.",
    "M1": "The baseline model. Encodes the allele as an identity label, which works "
          "well for alleles it has measured and carries no information for new ones.",
    "M2n": "The sequence model. Reads the allele's protein sequence, so it can say "
           "something about alleles it has never measured.",
    "novelty": "How far a query sits from anything in the training data, measured as "
               "sequence difference.",
    "distribution shift": "When new inputs differ systematically from the examples a "
                          "model learned from.",
    "Spearman": "Rank correlation: does the predicted ordering match the measured "
                "ordering? 1 is perfect, 0 is no relationship.",
    "MAE": "Mean absolute error — the average size of the mistake, in hours.",
    "NetMHCstabpan": "The established published tool for predicting peptide–HLA "
                     "stability. The reference this project is measured against.",
    "training coverage": "How much relevant data the model saw for this kind of query.",
}


def term(word: str, shown: str | None = None) -> str:
    """Inline term with a hover definition."""
    d = GLOSSARY.get(word, "")
    return f"<span class='term' title='{d}'>{shown or word}</span>"
