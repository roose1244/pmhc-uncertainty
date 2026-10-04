"""Does predicted uncertainty rise with a controlled dose of novelty?

The splits test novelty as a category -- a peptide is held out or it is not.
That invites the objection that "unseen, so warn" is a lookup rather than a
finding. A dose-response asks the sharper question: if novelty is applied in
measured amounts, does the uncertainty signal track the amount?

Take peptides the model was trained on, mutate k positions for k = 0..8, and
measure what the ensemble says. No ground-truth half-life exists for a
synthetic mutant, so error cannot be measured here -- but error is not the
question. The question is whether the signal the product relies on responds to
novelty it can be given in known quantities. A signal that only distinguishes
"seen" from "unseen" is a lookup; one that rises with k is a measurement.

Two ensembles are compared because they fail differently. M1 one-hot encodes
the allele but reads the peptide through BLOSUM, so peptide novelty is visible
to it. M2n reads both through ESM. Both are retrained here rather than loaded,
since only their predictions were ever saved.

Mutations are drawn uniformly over the 19 alternative amino acids at k
positions chosen without replacement. The realised novelty is measured rather
than assumed: mutating k positions does not guarantee distance k to the
NEAREST training peptide, because a mutant can land closer to a different one.

Run:  python scripts/dose_response.py [n_peptides] [n_replicates]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ensemble_m2n import _fit_member, _predict  # noqa: E402
from src.features import encode_hla, encode_peptides, fit_hla_encoder  # noqa: E402
from src.train_m2_scaled import load_store  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
OUT = Path("results/tables/dose_response.csv")

AMINO = list("ACDEFGHIKLMNPQRSTVWY")
DOSES = list(range(0, 9))
N_MEMBERS = 5
EPS = 1e-6
SEED = 42
DEFAULT_PEPTIDES = 60
DEFAULT_REPLICATES = 3


def mutate(peptide: str, k: int, rng: np.random.Generator) -> str:
    if k == 0:
        return peptide
    pos = rng.choice(len(peptide), size=min(k, len(peptide)), replace=False)
    out = list(peptide)
    for p in pos:
        out[p] = rng.choice([a for a in AMINO if a != out[p]])
    return "".join(out)


def min_hamming(query: str, pool: np.ndarray) -> int:
    q = np.frombuffer(query.encode(), dtype=np.uint8)
    return int((pool != q).sum(axis=1).min())


def main() -> int:
    n_pep = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PEPTIDES
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REPLICATES
    rng = np.random.default_rng(SEED)

    master = pd.read_parquet(MASTER)
    split = pd.read_parquet(SPLIT_DIR / "random.parquet")
    d = master.merge(split, on="id")
    train = d[d.fold == "train"]
    val = d[d.fold == "val"]

    train_peps = np.array([list(p.encode()) for p in sorted(train.peptide.unique())],
                          dtype=np.uint8)
    print(f"training peptides: {len(train_peps)}")

    # One well-represented allele, held fixed, so only peptide novelty varies.
    allele = train.hla.value_counts().index[0]
    seeds_pep = (train[train.hla == allele].peptide.drop_duplicates()
                 .sample(n_pep, random_state=SEED).tolist())
    print(f"allele held fixed at {allele}, {len(seeds_pep)} seed peptides\n")

    # --- build the query set --------------------------------------------
    rows = []
    for pep in seeds_pep:
        for k in DOSES:
            for r in range(1 if k == 0 else n_rep):
                rows.append({"seed_peptide": pep, "dose": k,
                             "peptide": mutate(pep, k, rng)})
    q = pd.DataFrame(rows).drop_duplicates(subset=["peptide", "dose"])
    q["hla"] = allele
    q["realised_novelty"] = [min_hamming(p, train_peps) for p in q.peptide]
    print(f"queries: {len(q)}  (doses {DOSES[0]}-{DOSES[-1]})")

    # --- train both ensembles -------------------------------------------
    pep_store = load_store("esm_peptide")
    hla_store = load_store("esm_hla")

    encoder = fit_hla_encoder(train.hla.tolist())

    def m1_x(frame):
        return np.concatenate(
            [encode_peptides(frame.peptide.tolist()),
             encode_hla(encoder, frame.hla.tolist())], axis=1).astype(np.float32)

    print("\ntraining M1 ensemble...")
    x_tr, y_tr = m1_x(train), train.log_half_life.to_numpy(np.float32)
    x_va, y_va = m1_x(val), val.log_half_life.to_numpy(np.float32)
    x_q = m1_x(q)
    m1_preds = []
    for s in range(N_MEMBERS):
        mdl = _fit_member(x_tr, y_tr, x_va, y_va, s)
        m1_preds.append(_predict(mdl, x_q))
        print(f"   member {s} done", flush=True)
    m1_stack = np.stack(m1_preds, 1)
    q["m1_mean"], q["m1_std"] = m1_stack.mean(1), m1_stack.std(1)

    # M2n needs ESM vectors for mutants, which are not precomputed.
    print("\nembedding mutant peptides with ESM-2...")
    import os

    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from src.embed import embed_sequences, load_esm

    model, alphabet, device = load_esm()
    uniq = sorted(set(q.peptide) - set(pep_store))
    if uniq:
        mean, _ = embed_sequences(uniq, model, alphabet, device, batch_size=64)
        pep_store.update(dict(zip(uniq, mean)))
    print(f"   embedded {len(uniq)} new peptides")

    def m2_raw(frame):
        return np.concatenate(
            [np.stack([pep_store[p] for p in frame.peptide], 0),
             np.stack([hla_store[h] for h in frame.hla], 0)], axis=1).astype(np.float32)

    print("\ntraining M2n ensemble...")
    raw_tr, raw_va, raw_q = m2_raw(train), m2_raw(val), m2_raw(q)
    m2_preds = []
    for s in range(N_MEMBERS):
        r = np.random.default_rng(s)
        take = r.choice(len(raw_tr), size=len(raw_tr), replace=True)
        mu = raw_tr[take].mean(0, keepdims=True)
        sd = np.maximum(raw_tr[take].std(0, keepdims=True), EPS)
        sc = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
        mdl = _fit_member(sc(raw_tr), y_tr, sc(raw_va), y_va, s)
        m2_preds.append(_predict(mdl, sc(raw_q)))
        print(f"   member {s} done", flush=True)
    m2_stack = np.stack(m2_preds, 1)
    q["m2n_mean"], q["m2n_std"] = m2_stack.mean(1), m2_stack.std(1)

    # Drift from the unmutated prediction: confirms the input really changed.
    base = q[q.dose == 0].set_index("seed_peptide")
    for tag in ["m1", "m2n"]:
        q[f"{tag}_drift"] = (
            q[f"{tag}_mean"].to_numpy()
            - base[f"{tag}_mean"].reindex(q.seed_peptide).to_numpy()
        )
        q[f"{tag}_drift"] = q[f"{tag}_drift"].abs()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(OUT, index=False)

    from scipy.stats import spearmanr

    print("\n" + "=" * 72)
    print("DOSE RESPONSE: uncertainty against a controlled amount of novelty")
    print("=" * 72)
    g = q.groupby("dose").agg(
        n=("peptide", "size"),
        novelty=("realised_novelty", "mean"),
        m1_std=("m1_std", "mean"), m1_drift=("m1_drift", "mean"),
        m2n_std=("m2n_std", "mean"), m2n_drift=("m2n_drift", "mean"),
    )
    print(f"\n   {'dose':>5s} {'n':>5s} {'novelty':>8s} {'m1 y_std':>9s} "
          f"{'m1 drift':>9s} {'m2n y_std':>10s} {'m2n drift':>10s}")
    for k, r in g.iterrows():
        print(f"   {k:5d} {int(r.n):5d} {r.novelty:8.2f} {r.m1_std:9.4f} "
              f"{r.m1_drift:9.4f} {r.m2n_std:10.4f} {r.m2n_drift:10.4f}")

    print("\n   Spearman against realised novelty (per query, not per dose):")
    for tag in ["m1_std", "m2n_std", "m1_drift", "m2n_drift"]:
        print(f"      {tag:10s} {spearmanr(q.realised_novelty, q[tag]).statistic:+.3f}")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
