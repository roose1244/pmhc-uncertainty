"""Train the five-member M1 ensemble and score its uncertainty once."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH
from src.ensemble import PRED_DIR, TABLE_DIR, ensemble_split
from src.splits import SPLITS_DIR


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    rows = []
    for name in ("random", "peptide", "cluster", "hla"):
        split = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
        scored, report = ensemble_split(master, split, name)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(PRED_DIR / f"m1ens_{name}.parquet", index=False)
        rows.append(report)
        print(
            f"{name:8} spearman={report['spearman']:.3f}  "
            f"err-unc={report['err_unc_spearman_test']:.3f}  "
            f"cover={report['coverage_90']:.3f}  "
            f"MAE100={report['mae_100']:.2f}  MAE50={report['mae_50']:.2f}"
        )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "m1ens.csv", index=False)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
