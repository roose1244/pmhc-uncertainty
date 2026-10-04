"""Public demo page. Same frozen ensemble as the POST endpoint.

The page is server-rendered. A missing HLA-C measurement stays missing.
An allele outside training does not get a numeric 90% interval.

    modal deploy modal_app/serve.py
"""

from __future__ import annotations

import html
import os
from functools import lru_cache

import modal

from modal_app.common import VOL, app, image

api_image = image.pip_install("fastapi[standard]").add_local_python_source("src", "modal_app")

EXAMPLES = (
    ("Seen pair", "FVRQCFNPM", "HLA-A*02:01"),
    ("Unseen peptide", "GLYGNGILV", "HLA-A*02:01"),
    ("Unseen HLA", "FVRQCFNPM", "HLA-C*07:02"),
)


def _page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pMHC Guardian</title>
<style>
  :root {{
    --paper: #f4efe6;
    --ink: #1c1410;
    --muted: #5c5148;
    --line: #d9cfc3;
    --olive: #3f5344;
    --amber: #8a5a12;
    --red: #8d2f2f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--paper); color: var(--ink);
    font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
    line-height: 1.45;
  }}
  main {{ max-width: 42rem; margin: 0 auto; padding: 3rem 1.25rem 4rem; }}
  h1 {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 0.4rem; }}
  p {{ margin: 0.4rem 0; }}
  .muted {{ color: var(--muted); font-size: 0.92rem; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.5rem 0; }}
  a.btn, button {{
    background: transparent; color: var(--ink); border: 1px solid var(--ink);
    padding: 0.45rem 0.7rem; text-decoration: none; font: inherit; cursor: pointer;
  }}
  a.btn:hover, button:hover {{ background: var(--ink); color: var(--paper); }}
  form {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 1.5rem; }}
  input {{
    font: inherit; background: transparent; color: var(--ink);
    border: 1px solid var(--line); padding: 0.45rem 0.6rem;
  }}
  input[name=peptide] {{ width: 12rem; }}
  input[name=hla] {{ width: 11rem; }}
  section {{ border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 0.5rem; }}
  .hours {{ font-size: 1.8rem; margin: 0.2rem 0 0.6rem; }}
  .warn {{ color: var(--red); }}
  .note {{ color: var(--amber); }}
</style>
</head>
<body>
<main>
<h1>pMHC Guardian</h1>
<p class="muted">Peptide–HLA stability. The number below is the frozen ensemble. An interval is shown only when the allele was in training.</p>
<nav>
{''.join(f'<a class="btn" href="/?peptide={p}&amp;hla={html.escape(h)}">{html.escape(name)}</a>' for name, p, h in EXAMPLES)}
</nav>
<form method="get" action="/">
  <input name="peptide" placeholder="9-mer peptide" value="" required>
  <input name="hla" placeholder="HLA allele" value="" required>
  <button type="submit">Predict</button>
</form>
{body}
</main>
</body>
</html>"""


@lru_cache(maxsize=1)
def _table():
    import pandas as pd

    master = pd.read_parquet("/vol/data/processed/master.parquet")
    nms = pd.read_parquet("/vol/data/external/netmhcstabpan_preds.parquet")
    split = pd.read_parquet("/vol/data/splits/peptide.parquet")
    return master.merge(nms, on="id", how="left").merge(split[["id", "fold"]], on="id")


def _lookup(peptide: str, hla: str) -> dict | None:
    from src.hla import normalise_hla
    from src.target import make_id

    try:
        row_id = make_id(normalise_hla(hla), peptide)
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


def _render(peptide: str, hla: str, reveal: bool) -> str:
    from src.serve import predict

    peptide = peptide.strip().upper()
    hla = hla.strip()
    try:
        result = predict(peptide, hla)
    except ValueError as exc:
        return f"<section><p class='warn'>{html.escape(str(exc))}</p></section>"

    shown_hla = html.escape(result["hla"])
    shown_pep = html.escape(result["peptide"])
    hours = result["half_life_hours"]
    parts = [
        "<section>",
        f"<p class='muted'>{shown_pep} / {shown_hla}</p>",
        f"<p class='hours'>{hours:.1f} h</p>",
        f"<p>{html.escape(result['verdict'])}</p>",
    ]
    if result["allele_known"]:
        parts.append(
            f"<p><strong>90% interval:</strong> {result['lo_hours']:.1f} – {result['hi_hours']:.1f} hours</p>"
        )
        parts.append(
            "<p class='muted'>Asymmetric in hours because the model works in log space.</p>"
        )
    else:
        parts.append("<p class='warn'><strong>No calibrated interval.</strong></p>")
        parts.append(
            "<p class='muted'>This allele was outside training. Ensemble agreement is not a 90% interval for a new HLA locus.</p>"
        )

    ref = _lookup(result["peptide"], result["hla"])
    if ref and ref["nms_half_life"] is not None:
        parts.append(f"<p><strong>NetMHCstabpan:</strong> {ref['nms_half_life']:.2f} h</p>")
        parts.append(
            "<p class='muted'>Trained on this same public table, including rows we held out. Not an independent comparison.</p>"
        )
    if ref and ref["fold"] == "test":
        if reveal:
            measured = ref["half_life"]
            inside = result["lo_hours"] <= measured <= result["hi_hours"]
            where = "Inside the 90% interval." if inside else "Outside the 90% interval."
            parts.append(f"<p><strong>Measured half-life:</strong> {measured:.1f} h</p>")
            parts.append(
                f"<p class='muted'>Held out of this model's training fold. {where}</p>"
            )
        else:
            q = f"/?peptide={result['peptide']}&hla={html.escape(result['hla'])}&reveal=1"
            parts.append(f"<p><a class='btn' href='{q}'>Show held-out measurement</a></p>")
    elif ref is None:
        parts.append("<p class='muted'>No experimental half-life for this pair in the stability table.</p>")
    parts.append("</section>")
    return "\n".join(parts)


@app.function(image=api_image, volumes=VOL, timeout=300, scaledown_window=3600)
@modal.asgi_app(label="demo")
def demo():
    os.environ["PMHC_MODELS"] = "/vol/models"
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from src.serve import _bundle

    _bundle()
    api = FastAPI()

    @api.get("/", response_class=HTMLResponse)
    def home(peptide: str = "", hla: str = "", reveal: int = 0) -> str:
        body = _render(peptide, hla, bool(reveal)) if peptide and hla else ""
        return _page(body)

    return api
