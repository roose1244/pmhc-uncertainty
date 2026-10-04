"""AlphaFold view of the HLA protein. There is no peptide in these files.

AlphaFold DB has one predicted structure per UniProt entry, and UniProt has
one entry per class I locus: P04439 (HLA-A), P01889 (HLA-B), P10321 (HLA-C).
That model is not allele-specific, it is not an experimental crystal, and it
does not contain the peptide or β2-microglobulin.

The mature groove starts at AlphaFold residue 25 (a 24-residue signal peptide
precedes GSHSM). The 34 peptide-contact positions are the NetMHCpan set; they
match pseudo_best for every allele in the stability table.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external" / "structures"
UNIPROT = {"A": "P04439", "B": "P01889", "C": "P10321"}
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
SIGNAL = 24
GROOVE = 182
# 1-based positions in the mature α1–α2 domain. Verified against pseudo_best.
CONTACTS = (
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84,
    95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171,
)


@dataclass(frozen=True)
class StructureInfo:
    available: bool
    locus: str
    accession: str
    entry_id: str
    description: str
    plddt_global: float | None
    plddt_groove: float | None
    complex_available: bool = False


def locus_of(hla: str) -> str:
    core = hla.replace("HLA-", "").strip()
    return core[:1].upper() if core else ""


def af_residue(mature_position: int) -> int:
    """Mature groove position 1 is AlphaFold residue 25."""
    return mature_position + SIGNAL


def _ca_records(pdb: str) -> list[tuple[int, float]]:
    rows = []
    for line in pdb.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            rows.append((int(line[22:26]), float(line[60:66])))
    return rows


def groove_plddt(pdb: str) -> float | None:
    start, end = af_residue(1), af_residue(GROOVE)
    vals = [b for resi, b in _ca_records(pdb) if start <= resi <= end]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


@lru_cache(maxsize=8)
def load_locus(locus: str) -> tuple[str, dict] | None:
    """Return (pdb text, metadata) for one locus, fetching once into the cache."""
    acc = UNIPROT.get(locus)
    if acc is None:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    pdb_path = CACHE / f"AF-{acc}.pdb"
    meta_path = CACHE / f"AF-{acc}.json"
    if pdb_path.exists() and meta_path.exists():
        return pdb_path.read_text(), json.loads(meta_path.read_text())

    try:
        import certifi
        import requests

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        meta_resp = requests.get(AFDB.format(acc=acc), timeout=30)
        meta_resp.raise_for_status()
        item = meta_resp.json()[0]
        pdb_resp = requests.get(item["pdbUrl"], timeout=60)
        pdb_resp.raise_for_status()
    except Exception:
        return None

    meta = {
        "entryId": item.get("entryId", ""),
        "uniprotAccession": acc,
        "uniprotDescription": item.get("uniprotDescription", ""),
        "globalMetricValue": item.get("globalMetricValue"),
        "isComplex": bool(item.get("isComplex")),
        "pdbUrl": item.get("pdbUrl", ""),
    }
    pdb_path.write_text(pdb_resp.text)
    meta_path.write_text(json.dumps(meta))
    return pdb_resp.text, meta


def describe(hla: str) -> StructureInfo:
    locus = locus_of(hla)
    loaded = load_locus(locus)
    if loaded is None:
        return StructureInfo(False, locus, UNIPROT.get(locus, ""), "", "", None, None)
    pdb, meta = loaded
    global_plddt = meta.get("globalMetricValue")
    return StructureInfo(
        available=True,
        locus=locus,
        accession=meta.get("uniprotAccession", ""),
        entry_id=meta.get("entryId", ""),
        description=meta.get("uniprotDescription", ""),
        plddt_global=float(global_plddt) if global_plddt is not None else None,
        plddt_groove=groove_plddt(pdb),
        complex_available=bool(meta.get("isComplex")),
    )


def viewer_html(
    hla: str,
    *,
    groove_only: bool = False,
    show_contacts: bool = False,
    height: int = 380,
) -> str | None:
    """Interactive cartoon. Contacts are HLA residues, not a peptide chain."""
    loaded = load_locus(locus_of(hla))
    if loaded is None:
        return None
    pdb, _meta = loaded

    import py3Dmol

    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb, "pdb")
    groove = {"resi": f"{af_residue(1)}-{af_residue(GROOVE)}"}
    if groove_only:
        view.setStyle({"cartoon": {"color": "#e6dfd4", "opacity": 0.12}})
        view.setStyle(groove, {"cartoon": {"color": "#3d5346"}})
        view.zoomTo(groove)
    else:
        view.setStyle({"cartoon": {"color": "#cfc6b8"}})
        view.setStyle(groove, {"cartoon": {"color": "#3d5346"}})
        view.zoomTo()
    if show_contacts:
        view.addStyle(
            {"resi": [af_residue(p) for p in CONTACTS]},
            {"stick": {"color": "#6e6248", "radius": 0.22}},
        )
    view.setBackgroundColor(0xF3EEE6)
    return view._make_html()
