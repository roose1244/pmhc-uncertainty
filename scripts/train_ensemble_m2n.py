"""Build the M2n ensemble on every split and compare its uncertainty to m1ens.

This is the direct test of the project's hypothesis on a model whose members
can disagree about an unseen allele.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ensemble_m2n import PRED_DIR, TABLE_DIR, ensemble_split  # noqa: E402
from src.train_m2_scaled import load_store  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
SPLITS = ["random", "peptide", "cluster", "hla"]


def main() -> int:
    master = pd.read_parquet(MASTER)
    pep_store = load_store("esm_peptide")
    hla_store = load_store("esm_hla")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for split_name in SPLITS:
        split = pd.read_parquet(SPLIT_DIR / f"{split_name}.parquet")
        scored, report = ensemble_split(master, split, split_name,
                                        pep_store, hla_store)
        scored.to_parquet(PRED_DIR / f"m2nens_{split_name}.parquet", index=False)
        reports.append(report)
        print(f"{split_name:8s} spearman={report['spearman']:+.3f}  "
              f"err_unc={report['err_unc_spearman_test']:+.3f}  "
              f"coverage={report['coverage_90']:.3f}", flush=True)

    t = pd.DataFrame(reports)
    t.to_csv(TABLE_DIR / "m2nens.csv", index=False)

    print("\nuncertainty quality: m2nens against m1ens")
    old = pd.read_csv(TABLE_DIR / "m1ens.csv").set_index("split")
    print(f"   {'split':9s} {'err-unc m1ens':>14s} {'err-unc m2nens':>15s} "
          f"{'delta':>8s} {'cov m1ens':>10s} {'cov m2nens':>11s}")
    for _, r in t.iterrows():
        o = old.loc[r.split]
        print(f"   {r.split:9s} {o.err_unc_spearman_test:14.3f} "
              f"{r['err_unc_spearman_test']:15.3f} "
              f"{r['err_unc_spearman_test'] - o.err_unc_spearman_test:+8.3f} "
              f"{o.coverage_90:10.3f} {r['coverage_90']:11.3f}")

    print(f"\nwrote {PRED_DIR}/m2nens_*.parquet and {TABLE_DIR}/m2nens.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
