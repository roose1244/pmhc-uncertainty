"""The HLA groove in 3D, with the contact residues and this allele's novelty marked.

WHAT THIS IS AND IS NOT. AlphaFold DB holds one predicted structure per
UniProt entry, and UniProt has one entry per class I locus -- P04439 for the A
chain, P01889 for B, P10321 for C. There is no AlphaFold model of A*02:01 as
distinct from A*02:12: alleles differ by a handful of side chains, not by
fold, so the backbone shown is the locus consensus and is honest only at that
level. The viewer says so rather than implying an allele-specific model.

There is also no AlphaFold peptide. A 9-mer is not a UniProt entry, and the
peptide-MHC complex would have to be co-folded with Boltz, Chai-1 or
AlphaFold-Multimer, one GPU job per pair. The plan caps that at a few hundred
structures and gates it on everything else being finished, so the peptide is
represented by marking the groove positions it contacts rather than by a
modelled chain.

WHAT IT ADDS. The 34 peptide-contacting positions were recovered from
NetMHCstabpan's pseudo-sequence and verified against the published set, so
they can be drawn on the structure. More usefully, the positions where the
queried allele differs from its nearest training allele can be drawn too --
which turns "this allele is 11 of 182 residues from anything we trained on"
from a number into a place on the binding groove. That is the project's own
novelty measure made spatial, and it is not something AlphaFold provides.

Structures are cached on disk after first fetch.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

CACHE = Path("data/external/structures")
UNIPROT = {"A": "P04439", "B": "P01889", "C": "P10321"}
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

# The mature chain starts after a 24-residue signal peptide, so groove
# position i maps to residue i + 24 in the AlphaFold numbering.
SIGNAL_OFFSET = 24


def locus_of(hla: str) -> str:
    """HLA-B*15:02 -> B."""
    core = hla.replace("HLA-", "")
    return core[0].upper() if core else "A"


@lru_cache(maxsize=8)
def fetch_structure(locus: str) -> str | None:
    """AlphaFold PDB text for one class I locus, cached on disk."""
    acc = UNIPROT.get(locus)
    if acc is None:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"AF-{acc}.pdb"
    if path.exists():
        return path.read_text()

    try:
        import certifi
        import requests

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        meta = requests.get(AFDB.format(acc=acc), timeout=20)
        meta.raise_for_status()
        url = json.loads(meta.text)[0]["pdbUrl"]
        pdb = requests.get(url, timeout=30)
        pdb.raise_for_status()
        path.write_text(pdb.text)
        return pdb.text
    except Exception:
        return None


def viewer_html(
    hla: str,
    contact_positions: list[int],
    novel_positions: list[int] | None = None,
    height: int = 340,
) -> str | None:
    """Cartoon of the groove: contacts in amber, novel positions in red.

    Positions are 1-based into the mature 182-mer and shifted by the signal
    peptide to match AlphaFold residue numbers.
    """
    pdb = fetch_structure(locus_of(hla))
    if pdb is None:
        return None

    import py3Dmol

    novel = set(novel_positions or [])
    contacts = [p for p in contact_positions if p not in novel]

    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb, "pdb")
    # Grey the whole chain, then show only the alpha1-alpha2 groove in colour:
    # the alpha3 domain and signal peptide are not where peptides bind and
    # dominate the view if left prominent.
    view.setStyle({"cartoon": {"color": "#d0d7de", "opacity": 0.45}})
    view.setStyle(
        {"resi": f"{1 + SIGNAL_OFFSET}-{182 + SIGNAL_OFFSET}"},
        {"cartoon": {"color": "#8c959f"}},
    )
    if contacts:
        view.addStyle(
            {"resi": [p + SIGNAL_OFFSET for p in contacts]},
            {"stick": {"color": "#bf8700", "radius": 0.18}},
        )
    if novel:
        view.addStyle(
            {"resi": [p + SIGNAL_OFFSET for p in sorted(novel)]},
            {"stick": {"color": "#cf222e", "radius": 0.30}},
        )
        view.addStyle(
            {"resi": [p + SIGNAL_OFFSET for p in sorted(novel)]},
            {"sphere": {"color": "#cf222e", "radius": 0.9, "opacity": 0.55}},
        )
    view.zoomTo({"resi": f"{1 + SIGNAL_OFFSET}-{182 + SIGNAL_OFFSET}"})
    view.setBackgroundColor(0xFFFFFF, 0.0)  # transparent, so it sits on the page background
    return view._make_html()


def differing_positions(groove_a: str, groove_b: str) -> list[int]:
    """1-based positions where two equal-length grooves differ."""
    if not groove_a or not groove_b or len(groove_a) != len(groove_b):
        return []
    return [i + 1 for i, (x, y) in enumerate(zip(groove_a, groove_b)) if x != y]


@lru_cache(maxsize=1)
def _groove_table() -> dict[str, str]:
    import pandas as pd

    df = pd.read_csv("data/external/hla_sequences.csv")
    return dict(zip(df.hla, df.groove_seq))


@lru_cache(maxsize=1)
def _known_alleles() -> tuple[str, ...]:
    """Alleles the deployed model actually trained on."""
    try:
        return tuple(json.loads(Path("models/manifest.json").read_text())["known_alleles"])
    except Exception:
        return tuple(_groove_table())


def nearest_training_allele(hla: str, groove: str | None = None) -> tuple[str, list[int]]:
    """Closest allele the model has measured, and where this one differs.

    Computed here rather than read from novelty.parquet, whose wide form keeps
    the distances but not the neighbour's name. For an allele already in
    training the nearest OTHER allele is used, so the panel still shows which
    pockets separate it from its closest sibling instead of comparing it with
    itself.
    """
    table = _groove_table()
    groove = groove or table.get(hla)
    if not groove:
        return "", []

    best, best_d = "", 10**6
    for other in _known_alleles():
        if other == hla:
            continue
        seq = table.get(other)
        if not seq or len(seq) != len(groove):
            continue
        d = sum(1 for a, b in zip(groove, seq) if a != b)
        if d < best_d:
            best, best_d = other, d
    return best, differing_positions(groove, table.get(best, ""))


@lru_cache(maxsize=128)
def groove_for(hla: str) -> str | None:
    """Groove sequence for any allele: the training table, else IMGT.

    A novel allele is exactly the case this panel exists for, and it is by
    definition absent from hla_sequences.csv, so fall through to the same IMGT
    rebuild the predictor uses rather than showing nothing.
    """
    table = _groove_table()
    if hla in table:
        return table[hla]
    try:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.verify_hla_seq import read_fasta, rebuild

        fasta = Path("data/raw/hla_prot.fasta")
        if not fasta.exists():
            return None
        groove, _, _ = rebuild(read_fasta(fasta), hla)
        return groove
    except Exception:
        return None
