"""Visual components for the PepShield UI.

Two things the previous layout got wrong, both worth stating because the fix
is the design:

A progress bar showed the point estimate as a fraction of 24 hours. It carried
no information about uncertainty, and sitting under a confidence badge it read
as though it did. A bar that looks like a confidence meter and is not one is
worse than no bar.

The badge and the interval answered different questions while appearing to
answer the same one. "Reliable" above an interval of 0.3 to 16.4 hours invites
the obvious objection that a 50-fold range is not reliable. Data support and
precision are separate facts: the model can have plenty of relevant training
data and still be imprecise, and it can be falsely precise on an allele it has
never seen. The layout now shows them side by side rather than stacked, so
neither is read as a verdict on the other.

Everything here renders as inline HTML. Streamlit's widgets cannot express a
log-scale interval, and half-life spans roughly 0.01 to 250 hours, so a linear
bar would compress every short-lived complex into the first pixel.
"""

from __future__ import annotations

import math

# Half-life axis, log10. The data runs 0 to 256 hours with a median near 1.
AXIS_MIN, AXIS_MAX = -1.0, 2.0  # 0.1 h to 100 h
TICKS = [(0.1, "0.1h"), (1, "1h"), (10, "10h"), (100, "100h")]

LEVEL_COLOUR = {
    "green": "#2da44e",
    "amber": "#bf8700",
    "red": "#cf222e",
    "grey": "#8c959f",
}


def _pos(hours: float) -> float:
    """Position on the log axis as a percentage, clamped to the visible range."""
    h = max(hours, 10**AXIS_MIN)
    frac = (math.log10(h) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)
    return max(0.0, min(1.0, frac)) * 100


def interval_bar(point: float, lo: float, hi: float, colour: str = "#2da44e") -> str:
    """The 90% interval on a log half-life axis, with the estimate marked.

    The width of the band is the honest headline: a point estimate with a
    50-fold interval is a different claim from the same estimate with a 2-fold
    one, and only the band shows that.
    """
    left, right = _pos(lo), _pos(hi)
    width = max(right - left, 0.8)
    mark = _pos(point)

    ticks = "".join(
        f"<div style='position:absolute;left:{_pos(v)}%;top:0;bottom:0;"
        f"width:1px;background:rgba(128,128,128,0.25)'></div>"
        f"<div style='position:absolute;left:{_pos(v)}%;top:34px;"
        f"transform:translateX(-50%);font-size:0.7rem;color:#8c959f'>{lab}</div>"
        for v, lab in TICKS
    )

    return f"""
<div style='margin:0.5rem 0 1.9rem 0'>
  <div style='position:relative;height:30px'>
    <div style='position:absolute;top:13px;left:0;right:0;height:4px;
                background:rgba(128,128,128,0.18);border-radius:2px'></div>
    {ticks}
    <div style='position:absolute;top:9px;left:{left}%;width:{width}%;height:12px;
                background:{colour};opacity:0.35;border-radius:6px'></div>
    <div style='position:absolute;top:5px;left:{mark}%;width:3px;height:20px;
                background:{colour};transform:translateX(-50%);border-radius:2px'></div>
  </div>
</div>"""


def gauge(label: str, value: str, level: str, fraction: float, detail: str = "") -> str:
    """One familiarity gauge: a labelled bar whose fill is the strength."""
    colour = LEVEL_COLOUR.get(level, LEVEL_COLOUR["grey"])
    pct = max(3.0, min(100.0, fraction * 100))
    # Label above value rather than beside it: in three narrow columns a
    # flex row wraps both halves and the gauges stop lining up.
    sub = (f"<div style='font-size:0.72rem;color:#8c959f;margin-top:3px;"
           f"line-height:1.35'>{detail}</div>" if detail else "")
    return f"""
<div style='margin-bottom:0.9rem'>
  <div style='font-weight:600;font-size:0.82rem;white-space:nowrap;
              overflow:hidden;text-overflow:ellipsis'>{label}</div>
  <div style='font-size:0.82rem;color:{colour};font-weight:600;
              margin-top:1px'>{value}</div>
  <div style='height:6px;background:rgba(128,128,128,0.18);border-radius:3px;
              margin-top:6px;overflow:hidden'>
    <div style='height:100%;width:{pct}%;background:{colour};border-radius:3px'></div>
  </div>
  {sub}
</div>"""


def verdict_card(colour: str, verdict: str, gloss: str) -> str:
    return f"""
<div style='background:{colour};color:#fff;padding:0.7rem 1rem;border-radius:10px'>
  <div style='font-weight:700;font-size:1.05rem;line-height:1.2'>{verdict}</div>
  <div style='font-size:0.78rem;opacity:0.92;margin-top:3px'>{gloss}</div>
</div>"""


def stat_block(label: str, value: str, sub: str) -> str:
    return f"""
<div>
  <div style='font-size:0.75rem;color:#8c959f;text-transform:uppercase;
              letter-spacing:0.04em'>{label}</div>
  <div style='font-size:2.1rem;font-weight:700;line-height:1.1;
              margin-top:2px'>{value}</div>
  <div style='font-size:0.78rem;color:#8c959f;margin-top:2px'>{sub}</div>
</div>"""


def routing_diagram(selected: str | None, allele_known: bool) -> str:
    """Compact pathway showing which model answered and why.

    Routing is a decision the system makes from a fact it can check at query
    time -- whether the allele was in training -- so the diagram shows that
    fact as the branch point rather than presenting the model choice as
    arbitrary.
    """
    on, off = "#0e7490", "#cbd5e1"
    m1_c = on if selected == "m1" else off
    m2_c = on if selected == "m2n" else off
    branch = ("allele IS in training" if allele_known or selected == "m1"
              else "allele is NOT in training")

    def node(label, sub, colour, strong):
        weight = "700" if strong else "500"
        return (f"<div style='flex:1;text-align:center'>"
                f"<div style='border:1.5px solid {colour};border-radius:8px;"
                f"padding:0.4rem 0.3rem;background:{'#ecfeff' if strong else '#fff'}'>"
                f"<div style='font-size:0.78rem;font-weight:{weight};color:"
                f"{'#0f172a' if strong else '#94a3b8'}'>{label}</div>"
                f"<div style='font-size:0.64rem;color:#94a3b8'>{sub}</div>"
                f"</div></div>")

    arrow = ("<div style='align-self:center;color:#cbd5e1;font-size:0.9rem;"
             "padding:0 0.35rem'>&rarr;</div>")

    return f"""
<div style='display:flex;align-items:stretch;margin:0.3rem 0 0.2rem 0'>
  {node("Query", "peptide + allele", "#cbd5e1", False)}
  {arrow}
  {node("Coverage check", branch, "#0e7490", True)}
  {arrow}
  <div style='flex:1.2;display:flex;flex-direction:column;gap:3px'>
    {node("M1", "allele identity", m1_c, selected == "m1")}
    {node("M2n", "allele sequence", m2_c, selected == "m2n")}
  </div>
  {arrow}
  {node("Prediction", "+ support verdict", "#0e7490", True)}
</div>"""
