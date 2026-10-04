"""Design tokens and global CSS for PepShield.

One accent, reserved for this system and for the primary action. Green, amber
and red are used only for training-coverage states. Longer text is a system
sans; monospace is limited to sequences, alleles and numbers.
"""

from __future__ import annotations

INK = "#1c1917"
BODY = "#44403c"
MUTED = "#78716c"
FAINT = "#a8a29e"
HAIRLINE = "rgba(28, 25, 23, 0.10)"
SURFACE = "#fffcf8"
CANVAS = "#f4f1eb"

ACCENT = "#3d6b5a"
ACCENT_SOFT = "#e7f0eb"
REFERENCE = "#6b5678"

OK = "#3d6b3a"
WARN = "#9a6b2f"
RISK = "#8f3d35"
NEUTRAL = "#a8a29e"

SUPPORT = {
    "green": (OK, "Well covered",
              "Training data covers this peptide and this allele."),
    "amber": (WARN, "Some gaps",
              "Part of this query is sparsely represented in training."),
    "red": (RISK, "Little training data",
              "This query sits outside the training data."),
    "grey": (NEUTRAL, "—", ""),
}

SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', monospace")

CSS = f"""
<style>
  .stApp {{ background: {CANVAS}; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 820px; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  html, body, [class*="css"] {{ color: {BODY}; font-family: {SANS}; }}
  h1, h2, h3, h4, h5 {{
    color: {INK}; letter-spacing: -0.02em; font-weight: 600;
    font-family: {SANS};
  }}

  .page-kicker {{
    font-size: 0.8rem; font-weight: 500; color: {ACCENT}; margin-bottom: 0.35rem;
  }}
  .page-title {{
    font-size: 1.85rem; font-weight: 600; color: {INK};
    letter-spacing: -0.03em; line-height: 1.2; margin: 0;
  }}
  .page-lede {{
    font-size: 1.02rem; line-height: 1.55; color: {MUTED};
    max-width: 40rem; margin: 0.55rem 0 1.6rem 0;
  }}

  .sec {{ margin: 1.8rem 0 0.6rem 0; }}
  .sec-q {{ font-size: 1.05rem; font-weight: 600; color: {INK}; }}
  .sec-sub {{ font-size: 0.92rem; color: {MUTED}; margin-top: 0.25rem; line-height: 1.5; }}

  .mono {{ font-family: {MONO}; }}
  .num {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
  .term {{ border-bottom: 1px dotted {FAINT}; cursor: help; }}

  .note {{
    border-left: 2px solid {WARN}; padding: 0.65rem 0 0.65rem 0.85rem;
    font-size: 0.9rem; color: {BODY}; line-height: 1.5; margin: 0.8rem 0;
  }}
  .caveat {{
    border-left: 2px solid {MUTED}; padding: 0.65rem 0 0.65rem 0.85rem;
    font-size: 0.9rem; color: {MUTED}; line-height: 1.5; margin: 1rem 0;
  }}

  /* Navigation: plain text, one selected state. */
  div[role="radiogroup"] {{ gap: 0.2rem !important; flex-wrap: wrap; }}
  div[role="radiogroup"] label {{
    font-family: {SANS} !important; font-size: 0.92rem !important;
    padding: 0.25rem 0.85rem 0.25rem 0 !important; color: {MUTED} !important;
  }}
  div[role="radiogroup"] label:has(input:checked) {{
    color: {INK} !important; font-weight: 600;
  }}

  div[data-testid="stTextInput"] label,
  div[data-testid="stSelectbox"] label {{
    font-size: 0.8rem !important; font-weight: 500; color: {MUTED} !important;
    text-transform: none !important; letter-spacing: 0 !important;
  }}
  div[data-testid="stTextInput"] input,
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    font-family: {MONO}; font-size: 0.92rem; background: {SURFACE};
    border-radius: 4px;
  }}
  div[data-testid="stTextInput"] input:focus,
  div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {{
    outline: 2px solid {ACCENT}; outline-offset: 2px;
  }}

  details, [data-testid="stExpander"] {{
    border: none !important; box-shadow: none !important;
  }}

  @media (max-width: 640px) {{
    .block-container {{ padding-top: 1.2rem; }}
    .page-title {{ font-size: 1.45rem; }}
  }}
</style>
"""


def page_head(title: str, lede: str, kicker: str = "") -> str:
    kick = f"<div class='page-kicker'>{kicker}</div>" if kicker else ""
    return (f"{kick}<h1 class='page-title'>{title}</h1>"
            f"<p class='page-lede'>{lede}</p>")


def section(question: str, sub: str = "") -> str:
    s = f"<div class='sec-sub'>{sub}</div>" if sub else ""
    return f"<div class='sec'><div class='sec-q'>{question}</div>{s}</div>"


GLOSSARY = {
    "pMHC": "A peptide bound to an MHC molecule — the complex a T cell sees.",
    "HLA": "Human Leukocyte Antigen: the human MHC protein that holds a peptide "
           "fragment and presents it to T cells.",
    "HLA-C": "The third class I locus. Neither this model nor NetMHCstabpan has "
             "any stability measurement for it, which makes it a transfer target.",
    "peptide": "A short protein fragment. Every measurement here is 9 amino acids long.",
    "half-life": "How long the peptide–HLA complex holds together before half of it "
                 "has come apart. Longer usually means better presented.",
    "transfer": "Applying a model to inputs of a kind it never trained on.",
    "90% interval": "A range the true value is expected to fall in 9 times out of 10. "
                    "Wider means less precise.",
    "ensemble": "Several copies of the same model trained with different random "
                "starts. How much they disagree is one way to estimate uncertainty.",
    "uncertainty": "Here, how much five independently trained models disagree. "
                   "High means they disagree about this prediction.",
    "foundation model": "A large model pretrained on millions of protein sequences, "
                        "reused as a feature extractor rather than retrained.",
    "M1": "The baseline model. Encodes the allele as a name tag, which works well "
          "for alleles it has measured and carries no information for new ones.",
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
    d = GLOSSARY.get(word, "")
    return f"<span class='term' title='{d}'>{shown or word}</span>"
