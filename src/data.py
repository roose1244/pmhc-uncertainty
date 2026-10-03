"""Build the master table from Rasmussen 2016 Stability.txt."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.hla import normalise_hla
from src.target import make_id, to_log

RAW_PATH = Path("data/raw/Stability.txt")
MASTER_PATH = Path("data/processed/master.parquet")
SOURCE = "Rasmussen2016"
ASSAY = "SPA_half_life_37C"
EXPECTED_ROWS = 28166


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python scripts/download_stability.py"
        )
    rows: list[dict] = []
    with path.open() as handle:
        header = handle.readline().split()
        if header[:3] != ["HLA", "Pep", "Thalf"]:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for line_no, line in enumerate(handle, start=2):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no} is not HLA / peptide / Thalf")
            hla_raw, peptide, half_life_raw = parts[0], parts[1], parts[2]
            peptide = peptide.strip().upper()
            if len(peptide) != 9 or not peptide.isalpha():
                raise ValueError(f"{path}:{line_no} is not a 9-mer peptide: {peptide}")
            half_life = float(half_life_raw)
            hla = normalise_hla(hla_raw)
            rows.append(
                {
                    "id": make_id(hla, peptide),
                    "peptide": peptide,
                    "hla": hla,
                    "hla_seq": None,
                    "half_life": half_life,
                    "log_half_life": to_log(half_life),
                    "source": SOURCE,
                    "assay": ASSAY,
                    "censored": half_life == 0.0,
                    "allele_class": "engineered" if "(" in hla else "natural",
                }
            )
    columns = [
        "id",
        "peptide",
        "hla",
        "hla_seq",
        "log_half_life",
        "half_life",
        "source",
        "assay",
        "censored",
        "allele_class",
    ]
    frame = pd.DataFrame(rows)[columns]
    if frame["id"].duplicated().any():
        dupes = frame.loc[frame["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate ids: {dupes[:5]}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(frame)}")
    return frame


def write_master(frame: pd.DataFrame, path: Path = MASTER_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
