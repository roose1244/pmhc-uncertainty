"""Build data/processed/novelty.parquet: how new is each row, per split?

The uncertainty story needs a defensible x-axis. "Error grows with novelty"
only means something if novelty is measured against what the model actually
trained on, which differs per split: B*15:02 is wholly unseen in the HLA split
and completely ordinary in the random split. So novelty is computed per
(split, row), not once per allele.

Three HLA measures, because the two models can use different information:

  hla_in_train      binary. This is the *only* HLA signal M1 receives, because
                    src/features.py one-hot encodes the allele name with
                    handle_unknown="ignore" -- an unseen allele is a zero
                    vector, so M1 cannot interpolate between alleles at all.
  hla_groove_dist   continuous, over the mature 182-mer. What a sequence-based
                    model (M2) could in principle exploit.
  hla_pseudo_dist   continuous, over the 34 NetMHC contact residues. Closer to
                    the biology: two alleles can be near-identical overall and
                    still differ where the peptide actually binds.

Peptide novelty is plain Hamming distance, which is exact here because every
peptide in the dataset is a 9-mer -- no alignment or gap penalty needed.

Distances are 1 - fractional identity to the *nearest training allele*, so 0
means "this exact thing was in train" and larger means further out.

Run:  python scripts/build_novelty.py            # IMGT-corrected sequences
      python scripts/build_novelty.py --seq dev1 # master.parquet as-is
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.verify_hla_seq import read_fasta, rebuild  # noqa: E402

MASTER = Path("data/processed/master.parquet")
FASTA = Path("data/raw/hla_prot.fasta")
PSEUDO = Path("data/external/hla_pseudo.csv")
SPLIT_DIR = Path("data/splits")
OUT = Path("data/processed/novelty.parquet")

SPLITS = ["random", "hla", "peptide", "cluster"]


def groove_sequences(master: pd.DataFrame, which: str) -> tuple[dict[str, str], str]:
    """Per-allele 182-mer, either Dev 1's column or an independent IMGT rebuild.

    Recorded in the output as seq_version, because novelty computed from a
    wrong sequence is wrong in a way that is invisible downstream: A*24:19
    looks like a near-copy of A*24:02 under Dev 1's column and is not one.
    """
    dev1 = master.groupby("hla").hla_seq.first().to_dict()
    if which == "dev1":
        return dev1, "dev1_master"

    seqs = read_fasta(FASTA)
    out, fell_back = {}, []
    for hla in dev1:
        mine, _, _ = rebuild(seqs, hla)
        if mine is None:
            out[hla] = dev1[hla]
            fell_back.append(hla)
        else:
            out[hla] = mine
    if fell_back:
        print(f"  ! no IMGT rebuild for {fell_back}, kept master's sequence")
    changed = [h for h in out if out[h] != dev1[h]]
    print(f"  IMGT rebuild differs from master for {len(changed)} alleles: {changed}")
    return out, "imgt_corrected"


def pseudo_sequences() -> dict[str, str]:
    ps = pd.read_csv(PSEUDO)
    return dict(zip(ps.hla, ps.pseudo_best))


def nearest(
    query: dict[str, str], pool: list[str], table: dict[str, str]
) -> tuple[dict[str, float], dict[str, str]]:
    """For each allele, 1 - identity to the closest allele in `pool`.

    Compared position-wise on equal-length strings, so an allele whose
    sequence is absent from `table` yields NaN rather than a silent 0.
    """
    dist: dict[str, float] = {}
    who: dict[str, str] = {}
    pool_seqs = [(p, table[p]) for p in pool if table.get(p)]
    for hla in query:
        mine = table.get(hla)
        if not mine or not pool_seqs:
            dist[hla], who[hla] = np.nan, ""
            continue
        best_d, best_p = 1.0, ""
        for p, other in pool_seqs:
            if len(other) != len(mine):
                continue
            mism = sum(1 for a, b in zip(mine, other) if a != b)
            d = mism / len(mine)
            if d < best_d:
                best_d, best_p = d, p
                if d == 0.0:
                    break
        dist[hla], who[hla] = best_d, best_p
    return dist, who


def peptide_novelty(
    queries: np.ndarray, train_peps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum Hamming distance from each query 9-mer to any training 9-mer.

    Exact rather than approximate: all peptides are 9-mers, so this is a
    straight position-wise comparison. Chunked over queries to keep the
    broadcast array small.
    """
    qa = np.array([list(p.encode()) for p in queries], dtype=np.uint8)
    ta = np.array([list(p.encode()) for p in train_peps], dtype=np.uint8)

    best = np.full(len(qa), 9, dtype=np.int16)
    arg = np.zeros(len(qa), dtype=np.int64)
    step = max(1, 4_000_000 // max(1, len(ta)))
    for i in range(0, len(qa), step):
        block = qa[i : i + step]
        # (block, train, 9) -> mismatch count per pair
        mism = (block[:, None, :] != ta[None, :, :]).sum(axis=2).astype(np.int16)
        best[i : i + len(block)] = mism.min(axis=1)
        arg[i : i + len(block)] = mism.argmin(axis=1)
    return best, train_peps[arg]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", choices=["imgt", "dev1"], default="imgt")
    args = ap.parse_args()

    master = pd.read_parquet(MASTER)
    print(f"master: {len(master)} rows, {master.hla.nunique()} alleles")

    print(f"\nHLA sequences ({args.seq}):")
    groove, seq_version = groove_sequences(master, args.seq)
    pseudo = pseudo_sequences()

    frames = []
    for split in SPLITS:
        sp = pd.read_parquet(SPLIT_DIR / f"{split}.parquet")
        d = master[["id", "hla", "peptide"]].merge(sp, on="id", how="inner")
        train = d[d.fold == "train"]
        train_alleles = sorted(train.hla.unique())
        train_peps = np.array(sorted(train.peptide.unique()))

        alleles = {h: groove.get(h, "") for h in d.hla.unique()}
        g_dist, g_nn = nearest(alleles, train_alleles, groove)
        p_dist, p_nn = nearest(alleles, train_alleles, pseudo)

        uniq = np.array(sorted(d.peptide.unique()))
        ham, ham_nn = peptide_novelty(uniq, train_peps)
        ham_map = dict(zip(uniq, ham))
        hamnn_map = dict(zip(uniq, ham_nn))

        in_train_hla = set(train_alleles)
        in_train_pep = set(train_peps.tolist())

        out = pd.DataFrame(
            {
                "id": d.id.to_numpy(),
                "split": split,
                "fold": d.fold.to_numpy(),
                "hla": d.hla.to_numpy(),
                "hla_in_train": d.hla.isin(in_train_hla).to_numpy(),
                "hla_groove_dist": d.hla.map(g_dist).to_numpy(),
                "hla_groove_nn": d.hla.map(g_nn).to_numpy(),
                "hla_pseudo_dist": d.hla.map(p_dist).to_numpy(),
                "hla_pseudo_nn": d.hla.map(p_nn).to_numpy(),
                "pep_in_train": d.peptide.isin(in_train_pep).to_numpy(),
                "pep_min_hamming": d.peptide.map(ham_map).to_numpy(),
                "pep_nn": d.peptide.map(hamnn_map).to_numpy(),
                "seq_version": seq_version,
            }
        )
        frames.append(out)

        held = out[out.fold != "train"]
        print(
            f"  {split:8s} train alleles={len(train_alleles):3d} "
            f"peps={len(train_peps):5d} | held-out rows={len(held):5d} "
            f"unseen HLA={100 * (~held.hla_in_train).mean():5.1f}% "
            f"unseen peptide={100 * (~held.pep_in_train).mean():5.1f}%"
        )

    novelty = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    novelty.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  rows={len(novelty)}  seq_version={seq_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
