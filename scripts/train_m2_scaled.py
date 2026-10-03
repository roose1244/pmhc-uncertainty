"""Does standardising the ESM features fix M2?

Trains M2 and its peptide-only twin twice per split, unscaled and scaled, so
the comparison is a controlled A/B rather than a comparison against Dev 1's
logged numbers from a different run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_m2_scaled import run_all  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
SPLITS = ["random", "peptide", "cluster", "hla"]

M1 = {"random": 0.765, "peptide": 0.642, "cluster": 0.629, "hla": -0.210}


def main() -> int:
    master = pd.read_parquet(MASTER)
    splits = {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet") for s in SPLITS}
    t = run_all(master, splits)

    print("\neffect of standardising ESM features (test-fold Spearman)")
    print(f"   {'split':9s} {'M2 raw':>8s} {'M2 scaled':>10s} {'delta':>8s} "
          f"{'M1':>8s} {'beats M1':>9s}")
    for s in SPLITS:
        raw = t[(t.split == s) & (t.model == "m2")].spearman
        sca = t[(t.split == s) & (t.model == "m2n")].spearman
        if raw.empty or sca.empty:
            continue
        r, c = float(raw.iloc[0]), float(sca.iloc[0])
        print(f"   {s:9s} {r:8.3f} {c:10.3f} {c - r:+8.3f} {M1[s]:8.3f} "
              f"{'YES' if c > M1[s] else 'no':>9s}")

    print("\npeptide-only twin")
    print(f"   {'split':9s} {'raw':>8s} {'scaled':>10s} {'delta':>8s}")
    for s in SPLITS:
        raw = t[(t.split == s) & (t.model == "m2pep")].spearman
        sca = t[(t.split == s) & (t.model == "m2pepn")].spearman
        if raw.empty or sca.empty:
            continue
        r, c = float(raw.iloc[0]), float(sca.iloc[0])
        print(f"   {s:9s} {r:8.3f} {c:10.3f} {c - r:+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
