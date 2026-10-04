"""Train the 34-residue HLA pocket model on the four locked splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH
from src.splits import SPLITS_DIR
from src.train_pseudo import run_all


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    splits = {
        name: pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
        for name in ("random", "peptide", "cluster", "hla")
    }
    table = run_all(master, splits)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
