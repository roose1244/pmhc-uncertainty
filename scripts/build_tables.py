"""Build master.parquet and the four split files. Fail if a split leaks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw, write_master
from src.splits import assert_split, build_all, write_splits


def main() -> None:
    master = load_raw()
    write_master(master)
    print(f"master  rows={len(master)}  alleles={master['hla'].nunique()}  "
          f"peptides={master['peptide'].nunique()}  "
          f"censored={int(master['censored'].sum())}")

    splits = build_all(master)
    write_splits(splits)
    for name, frame in splits.items():
        summary = assert_split(master, frame, name)
        counts = summary["counts"]
        print(
            f"{name:8}  train={counts.get('train', 0):5}  "
            f"val={counts.get('val', 0):5}  "
            f"test={counts.get('test', 0):5}  "
            f"calib={counts.get('calib', 0):5}  "
            f"peptide_in_train={summary['peptide_leakage']:.3f}"
        )


if __name__ == "__main__":
    main()
