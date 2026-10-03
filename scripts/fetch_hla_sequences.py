"""Attach IMGT mature α1–α2 sequences to master.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH, write_master
from src.hla_seq import HLA_SEQ_PATH, attach_to_master, build_hla_table


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    table = build_hla_table(master["hla"].tolist())
    HLA_SEQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(HLA_SEQ_PATH, index=False)
    updated = attach_to_master(master, table)
    write_master(updated)
    print(f"alleles {len(table)}  seq_len {updated['hla_seq'].str.len().unique().tolist()}")
    print(f"wrote {HLA_SEQ_PATH} and updated {MASTER_PATH}")
    engineered = table.loc[table["engineered"]]
    for _, row in engineered.iterrows():
        print(f"  {row['hla']} pos67={row['hla_seq'][66]} (parent {row['parent']})")


if __name__ == "__main__":
    main()
