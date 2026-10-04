"""Mature HLA α1–α2 sequences from IPD-IMGT/HLA protein FASTA.

Numbering: skip the 24-residue signal peptide, take mature residues 1–182.
That is the peptide-binding groove (α1 + α2). Developer 2 should verify.
Engineered (C67S) alleles are the parent sequence with mature Cys67 → Ser.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

from src.hla import normalise_hla

FASTA_URLS = {
    "A": "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/A_prot.fasta",
    "B": "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/B_prot.fasta",
}
SIGNAL = 24
GROOVE = 182
HLA_SEQ_PATH = Path("data/external/hla_seq.parquet")
HEADER = re.compile(
    r"(?:HLA-)?([ABC])\*(\d{2}):(\d{2})",
    re.IGNORECASE,
)


def _two_field(name: str) -> str:
    return normalise_hla(name)


def _parent_allele(name: str) -> str:
    return name.split("(")[0]


def _fasta_score(seq: str) -> tuple[int, int]:
    """Prefer canonical preprotein, then mature groove, then usable fragments."""
    if seq.startswith(("MAV", "MLV", "MRV")) and len(seq) >= SIGNAL + GROOVE:
        return (3, len(seq))
    if seq.startswith("GSHSM") and len(seq) >= GROOVE:
        return (2, len(seq))
    if seq.startswith("SHSMR") and len(seq) >= GROOVE - 1:
        return (1, len(seq))
    return (0, len(seq))


def parse_imgt_fasta(text: str) -> dict[str, str]:
    """Map HLA-A*02:01 → longest protein sequence for that 2-field name."""
    best: dict[str, str] = {}
    current_name = None
    chunks: list[str] = []

    def flush() -> None:
        if current_name is None:
            return
        seq = "".join(chunks)
        if len(seq) < GROOVE - 2:
            return
        prev = best.get(current_name)
        if prev is None or _fasta_score(seq) > _fasta_score(prev):
            best[current_name] = seq

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            chunks = []
            match = HEADER.search(line)
            current_name = (
                f"HLA-{match.group(1).upper()}*{match.group(2)}:{match.group(3)}"
                if match
                else None
            )
        else:
            chunks.append(line.strip())
    flush()
    return best


def mature_groove(full: str) -> str:
    """Locate the invariant GSHSM motif and take the next 182 residues."""
    start = full.find("GSHSM")
    if start < 0 and full.startswith("SHSMR"):
        full = "G" + full
        start = 0
    if start < 0:
        raise ValueError(f"no GSHSM motif in {full[:20]!r} len={len(full)}")
    groove = full[start : start + GROOVE]
    if len(groove) != GROOVE:
        raise ValueError(f"groove truncated at {len(groove)} after GSHSM")
    return groove


def apply_engineered(name: str, groove: str) -> str:
    if name.endswith("(C67S)"):
        residues = list(groove)
        residues[66] = "S"
        return "".join(residues)
    return groove


def fetch_locus(locus: str) -> str:
    import shutil
    import subprocess

    dest = Path("data/raw") / f"IMGT_{locus}_prot.fasta"
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-fsSL", "-H", "User-Agent: pMHC-Guardian/0.1", "-o", str(dest), FASTA_URLS[locus]],
            check=True,
        )
        return dest.read_text()
    request = urllib.request.Request(
        FASTA_URLS[locus],
        headers={"User-Agent": "pMHC-Guardian/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    dest.write_text(text)
    return text


def build_hla_table(alleles: list[str], fasta_by_locus: dict[str, str] | None = None) -> pd.DataFrame:
    if fasta_by_locus is None:
        fasta_by_locus = {locus: fetch_locus(locus) for locus in FASTA_URLS}
    lookup: dict[str, str] = {}
    for text in fasta_by_locus.values():
        lookup.update(parse_imgt_fasta(text))

    rows = []
    missing = []
    for allele in sorted(set(alleles)):
        parent = _parent_allele(allele)
        full = lookup.get(parent)
        if full is None:
            missing.append(allele)
            continue
        try:
            groove = apply_engineered(allele, mature_groove(full))
        except ValueError as exc:
            raise ValueError(f"{allele} parent={parent} len={len(full)} start={full[:20]!r}: {exc}") from exc
        rows.append(
            {
                "hla": allele,
                "hla_seq": groove,
                "hla_seq_source": "IPD-IMGT/HLA Latest protein FASTA, mature 1-182",
                "parent": parent,
                "engineered": allele != parent,
            }
        )
    if missing:
        raise ValueError(f"No IMGT sequence for: {missing}")
    return pd.DataFrame(rows)


def attach_to_master(master: pd.DataFrame, hla_table: pd.DataFrame) -> pd.DataFrame:
    out = master.drop(columns=["hla_seq"]).merge(
        hla_table[["hla", "hla_seq"]], on="hla", how="left"
    )
    if out["hla_seq"].isna().any():
        absent = sorted(out.loc[out["hla_seq"].isna(), "hla"].unique())
        raise ValueError(f"master rows missing hla_seq: {absent}")
    return out
