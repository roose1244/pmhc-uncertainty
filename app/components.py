"""Small visual pieces for the Predict screen.

The interval is drawn on a log axis because half-life spans roughly 0.01 to
250 hours; a linear bar would crush every short-lived complex into one pixel.
Gauges show training coverage, not correctness.
"""

from __future__ import annotations

import math

from app.theme import ACCENT, FAINT, INK, MUTED, OK, RISK, WARN

AXIS_MIN, AXIS_MAX = -1.0, 2.0  # 0.1 h to 100 h
TICKS = [(0.1, "0.1 h"), (1, "1 h"), (10, "10 h"), (100, "100 h")]

LEVEL_COLOUR = {
    "green": OK,
    "amber": WARN,
    "red": RISK,
    "grey": FAINT,
}


def _pos(hours: float) -> float:
    h = max(hours, 10**AXIS_MIN)
    frac = (math.log10(h) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)
    return max(0.0, min(1.0, frac)) * 100


def interval_bar(point: float, lo: float, hi: float, colour: str = ACCENT) -> str:
    left, right = _pos(lo), _pos(hi)
    width = max(right - left, 0.8)
    mark = _pos(point)
    ticks = "".join(
        f"<div style='position:absolute;left:{_pos(v)}%;top:0;bottom:0;"
        f"width:1px;background:rgba(28,25,23,0.12)'></div>"
        f"<div style='position:absolute;left:{_pos(v)}%;top:28px;"
        f"transform:translateX(-50%);font-size:0.72rem;color:{MUTED}'>{lab}</div>"
        for v, lab in TICKS
    )
    return f"""
<div style='margin:0.4rem 0 1.6rem 0' aria-hidden='true'>
  <div style='position:relative;height:26px'>
    <div style='position:absolute;top:12px;left:0;right:0;height:2px;
                background:rgba(28,25,23,0.12)'></div>
    {ticks}
    <div style='position:absolute;top:7px;left:{left}%;width:{width}%;height:12px;
                background:{colour};opacity:0.28;border-radius:2px'></div>
    <div style='position:absolute;top:3px;left:{mark}%;width:2px;height:20px;
                background:{colour};transform:translateX(-50%)'></div>
  </div>
</div>"""


def gauge(label: str, value: str, level: str, fraction: float, detail: str = "") -> str:
    colour = LEVEL_COLOUR.get(level, FAINT)
    pct = max(3.0, min(100.0, fraction * 100))
    sub = (f"<div style='font-size:0.78rem;color:{MUTED};margin-top:4px;"
           f"line-height:1.4'>{detail}</div>" if detail else "")
    return f"""
<div style='margin-bottom:1rem'>
  <div style='font-size:0.88rem;color:{INK}'>{label}</div>
  <div style='font-size:0.82rem;color:{colour};margin-top:2px'>{value}</div>
  <div style='height:4px;background:rgba(28,25,23,0.10);margin-top:8px'>
    <div style='height:100%;width:{pct}%;background:{colour}'></div>
  </div>
  {sub}
</div>"""


def stat_block(label: str, value: str, sub: str) -> str:
    return f"""
<div>
  <div style='font-size:0.8rem;color:{MUTED}'>{label}</div>
  <div class='num' style='font-size:2.4rem;font-weight:600;line-height:1.1;
              color:{INK};margin-top:4px'>{value}</div>
  <div style='font-size:0.88rem;color:{MUTED};margin-top:6px'>{sub}</div>
</div>"""


def routing_sentence(selected: str | None, allele_known: bool) -> str:
    """One sentence instead of a pathway diagram."""
    if selected == "m1":
        why = ("This allele was in training, so the model that treats the allele "
               "as a name tag answered.")
    elif selected == "m2n":
        why = ("This allele was not in training, so the model that reads the "
               "allele's protein sequence answered.")
    else:
        why = "The system chose a model from whether this allele was in training."
    if not allele_known and selected == "m1":
        why = "The name-tag model answered even though this allele was not in training."
    return why
