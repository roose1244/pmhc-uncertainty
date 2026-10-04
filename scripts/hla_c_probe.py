"""What do the models do on HLA-C, a locus nobody trained on?

There is no HLA-C stability data: the training table is 13,335 HLA-A rows and
14,831 HLA-B rows and nothing else, so HLA-C cannot be trained on and no error
can be measured against it. What can be measured is BEHAVIOUR, and that needs
no labels.

HLA-C is the strongest out-of-distribution case available, for both models at
once. Our 75 alleles are A and B. NetMHCstabpan's own training.pseudo lists 32
A and 36 B alleles and no C at all, while its MHC_pseudo.dat will happily
score 617 C alleles. So both systems will answer confidently about a locus
neither has ever seen a measurement from.

The experiment holds a fixed peptide set and walks it up three tiers of
novelty:

    tier 1  a familiar allele, in training
    tier 2  a held-out A/B allele, same loci as training
    tier 3  HLA-C alleles, a locus never trained on

If uncertainty is doing its job, spread should widen from tier 1 to tier 3.
The project's existing measurements predict it will not -- ensemble spread was
inversely related to allele novelty (-0.46) -- and this is the cleanest test
of that, because tier 3 is not a slightly unusual allele but an entire locus
outside the training distribution.

Degeneracy is checked too. M1 one-hot encodes the allele name, so every allele
outside training is the same zero vector and M1 must return an identical
prediction for a given peptide across every tier-3 allele. If that is what
happens, the model is not reasoning about HLA-C at all, and the number it
returns is a peptide-only guess wearing an allele's name.

Run:  python scripts/hla_c_probe.py [n_peptides]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/hla_c_probe.csv")

N_MEMBERS = 5
SEED = 42
EPS = 1e-6

TIERS = {
    "1 · in training": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*03:01"],
    "2 · held-out A/B": ["HLA-B*15:02", "HLA-B*27:02"],
    "3 · HLA-C (locus never trained)": [
        "HLA-C*01:02", "HLA-C*03:04", "HLA-C*04:01",
        "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02",
    ],
}


def main() -> int:
    n_pep = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    rng = np.random.default_rng(SEED)

    from app.structure import groove_for, nearest_training_allele
    from src.ensemble_m2n import _fit_member, _predict
    from src.features import encode_hla, encode_peptides, fit_hla_encoder
    from src.train_m2_scaled import load_store

    master = pd.read_parquet(MASTER)
    split = pd.read_parquet(SPLIT_DIR / "random.parquet")
    d = master.merge(split, on="id")
    train, val = d[d.fold == "train"], d[d.fold == "val"]

    peptides = (train.peptide.drop_duplicates()
                .sample(n_pep, random_state=SEED).tolist())

    queries = []
    for tier, alleles in TIERS.items():
        for a in alleles:
            g = groove_for(a)
            if not g:
                print(f"   no groove for {a}, skipping")
                continue
            nn, diffs = nearest_training_allele(a, g)
            for p in peptides:
                queries.append({"tier": tier, "hla": a, "peptide": p,
                                "groove": g, "nearest_train": nn,
                                "novelty": len(diffs)})
    q = pd.DataFrame(queries)
    print(f"{len(q)} queries · {q.hla.nunique()} alleles · {n_pep} peptides\n")

    # ---- M1 ensemble: BLOSUM peptide + one-hot allele --------------------
    encoder = fit_hla_encoder(train.hla.tolist())

    def m1_x(pep, hla):
        return np.concatenate([encode_peptides(list(pep)),
                               encode_hla(encoder, list(hla))], axis=1).astype(np.float32)

    x_tr = m1_x(train.peptide, train.hla)
    y_tr = train.log_half_life.to_numpy(np.float32)
    x_va = m1_x(val.peptide, val.hla)
    y_va = val.log_half_life.to_numpy(np.float32)
    x_q = m1_x(q.peptide, q.hla)

    print("training M1 ensemble")
    preds = []
    for s in range(N_MEMBERS):
        preds.append(_predict(_fit_member(x_tr, y_tr, x_va, y_va, s), x_q))
        print(f"   member {s}", flush=True)
    stack = np.stack(preds, 1)
    q["m1_mean"], q["m1_std"] = stack.mean(1), stack.std(1)

    # ---- M2n ensemble: standardised ESM of peptide and groove ------------
    pep_store = load_store("esm_peptide")
    hla_store = load_store("esm_hla")

    import os

    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from src.embed import embed_sequences, load_esm

    missing = [a for a in q.hla.unique() if a not in hla_store]
    if missing:
        model, alphabet, device = load_esm()
        grooves = [q[q.hla == a].groove.iloc[0] for a in missing]
        mean, _ = embed_sequences(grooves, model, alphabet, device, batch_size=8)
        hla_store.update(dict(zip(missing, mean)))
        print(f"embedded {len(missing)} unseen alleles on demand")

    def m2_raw(pep, hla):
        return np.concatenate([np.stack([pep_store[p] for p in pep], 0),
                               np.stack([hla_store[h] for h in hla], 0)],
                              axis=1).astype(np.float32)

    raw_tr = m2_raw(train.peptide, train.hla)
    raw_va = m2_raw(val.peptide, val.hla)
    raw_q = m2_raw(q.peptide, q.hla)

    print("training M2n ensemble")
    preds = []
    for s in range(N_MEMBERS):
        r = np.random.default_rng(s)
        take = r.choice(len(raw_tr), size=len(raw_tr), replace=True)
        mu = raw_tr[take].mean(0, keepdims=True)
        sd = np.maximum(raw_tr[take].std(0, keepdims=True), EPS)
        sc = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
        preds.append(_predict(_fit_member(sc(raw_tr), y_tr, sc(raw_va), y_va, s),
                              sc(raw_q)))
        print(f"   member {s}", flush=True)
    stack = np.stack(preds, 1)
    q["m2n_mean"], q["m2n_std"] = stack.mean(1), stack.std(1)

    q.drop(columns=["groove"]).to_csv(OUT, index=False)

    # ---- what happened ---------------------------------------------------
    print("\n" + "=" * 74)
    print("BEHAVIOUR BY NOVELTY TIER")
    print("=" * 74)
    g = q.groupby("tier").agg(
        alleles=("hla", "nunique"), n=("peptide", "size"),
        novelty=("novelty", "mean"),
        m1_pred=("m1_mean", "mean"), m1_std=("m1_std", "mean"),
        m2n_pred=("m2n_mean", "mean"), m2n_std=("m2n_std", "mean"))
    print(f"\n   {'tier':34s} {'alleles':>7s} {'novelty':>8s} "
          f"{'M1 y_std':>9s} {'M2n y_std':>10s}")
    for t, r in g.iterrows():
        print(f"   {t:34s} {int(r.alleles):7d} {r.novelty:8.1f} "
              f"{r.m1_std:9.4f} {r.m2n_std:10.4f}")

    # Degeneracy: does M1 give the same answer for every tier-3 allele?
    c = q[q.tier.str.startswith("3")]
    spread = c.groupby("peptide").m1_mean.std()
    print(f"\n   M1 prediction spread ACROSS HLA-C alleles, per peptide:")
    print(f"      mean sd = {spread.mean():.2e}   max sd = {spread.max():.2e}")
    if spread.max() < 1e-6:
        print("      -> identical for every HLA-C allele. M1 is not using the "
              "allele at all; the number is a peptide-only guess.")
    spread2 = c.groupby("peptide").m2n_mean.std()
    print(f"   M2n prediction spread across HLA-C alleles, per peptide:")
    print(f"      mean sd = {spread2.mean():.4f}   max sd = {spread2.max():.4f}")

    t1 = g.loc[[i for i in g.index if i.startswith("1")][0]]
    t3 = g.loc[[i for i in g.index if i.startswith("3")][0]]
    print(f"\n   uncertainty change, tier 1 -> tier 3:")
    print(f"      M1  {t1.m1_std:.4f} -> {t3.m1_std:.4f}  "
          f"({100*(t3.m1_std/t1.m1_std-1):+.0f}%)")
    print(f"      M2n {t1.m2n_std:.4f} -> {t3.m2n_std:.4f}  "
          f"({100*(t3.m2n_std/t1.m2n_std-1):+.0f}%)")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
