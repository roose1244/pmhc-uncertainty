"""Honest pipeline sketch for the About page.

The project plan imagines three foundation models feeding one predictor; two
ESM-2 scales have been run and ProtT5 has not, so ProtT5 is labelled as not
run rather than drawn as if it were working.
"""

from __future__ import annotations

from app.theme import FAINT, INK, MUTED


def pipeline_diagram() -> str:
    def row(items: list[tuple[str, str, bool]]) -> str:
        cells = []
        for label, sub, on in items:
            color = INK if on else FAINT
            hint = "" if on else " · not run"
            cells.append(
                f"<div style='flex:1;min-width:6rem'>"
                f"<div style='font-size:0.88rem;color:{color}'>{label}{hint}</div>"
                f"<div style='font-size:0.78rem;color:{MUTED}'>{sub}</div></div>"
            )
        return (f"<div style='display:flex;flex-wrap:wrap;gap:1rem;"
                f"margin:0.35rem 0 0.9rem 0'>{''.join(cells)}</div>")

    return f"""
<div style='font-size:0.9rem;color:{MUTED};line-height:1.5'>
  <div style='color:{INK};margin-bottom:0.2rem'>Peptide + HLA groove</div>
  {row([("ESM-2 35M", "run", True),
        ("ESM-2 150M", "run", True),
        ("ProtT5", "planned", False)])}
  <div style='color:{INK};margin-bottom:0.2rem'>Stability predictor (MLP, 5-member ensemble)</div>
  {row([("Half-life", "hours", True),
        ("Uncertainty", "how much the members disagree", True)])}
  <div style='color:{INK}'>HLA-C transfer — a locus with no training measurements</div>
</div>
"""
