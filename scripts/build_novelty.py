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
OUT = Path("data/features/novelty.parquet")  # the file contract's path
LONG_OUT = Path("data/processed/novelty_long.parquet")  # diagnostic, Dev 2 only

SPLITS = ["random", "hla", "peptide", "cluster"]


def groove_sequences(master: pd.DataFrame, which: str) -> tuple[dict[str, str], str]:
    """Per-allele 182-mer, either Dev 1's column or an independent IMGT rebuild.

    Recorded in the output as seq_version, because novelty computed from a
    wrong sequence is wrong in a way that is invisible downstream: A*24:19
    looks like a near-copy of A*24:02 under Dev 1's column and is not one.
    """
    dev1 = master.groupby("hla").hla_seq.first().to_dict()
    if which == "dev1":
        return dev1, "master_corrected_imgt_5b915f27"

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
) -> tuple[dict[str, float], dict[str, str], dict[str, float]]:
    """For each allele: distance to the closest allele in `pool`, and to the top 3.

    Compared position-wise on equal-length strings, so an allele whose
    sequence is absent from `table` yields NaN rather than a silent 0.
    """
    dist: dict[str, float] = {}
    who: dict[str, str] = {}
    top3: dict[str, float] = {}
    pool_seqs = [(p, table[p]) for p in pool if table.get(p)]
    for hla in query:
        mine = table.get(hla)
        if not mine or not pool_seqs:
            dist[hla], who[hla], top3[hla] = np.nan, "", np.nan
            continue
        scored = []
        for p, other in pool_seqs:
            if len(other) != len(mine):
                continue
            mism = sum(1 for a, b in zip(mine, other) if a != b)
            scored.append((mism / len(mine), p))
        if not scored:
            dist[hla], who[hla], top3[hla] = np.nan, "", np.nan
            continue
        scored.sort()
        dist[hla], who[hla] = scored[0]
        # Mean over the three closest training alleles. A single nearest
        # neighbour can be misleadingly reassuring when it is the only close
        # relative in train; the top-3 mean reflects how well the allele is
        # covered rather than whether one sibling happens to be present.
        top3[hla] = float(np.mean([s for s, _ in scored[:3]]))
    return dist, who, top3


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


def to_contract_shape(long: pd.DataFrame, seq_version: str) -> pd.DataFrame:
    """Pivot to the file contract: one row per id, novelty as named columns.

    The contract in the project plan is `id, hla_novelty, peptide_novelty`,
    with per-split variants (hla_novelty_hla, hla_novelty_random, ...). But
    novelty is only defined relative to a training fold, so the per-split
    columns are the real quantity and the bare ones are aliases.

    The bare columns alias the split that actually holds that dimension out:
    hla_novelty      <- the HLA split, where whole alleles are unseen
    peptide_novelty  <- the peptide split, where whole peptides are unseen

    Anywhere else the bare columns would be mostly zero and would read as
    "nothing is novel" rather than "this split does not test that axis".
    """
    wide = long[["id"]].drop_duplicates().set_index("id")

    for split in SPLITS:
        s = long[long.split == split].set_index("id")
        # hla_novelty is 1 - identity to the nearest training allele over the
        # mature 182-mer; the _pseudo_ variant is the same over the 34
        # peptide-contacting residues, which is the more biological distance.
        wide[f"hla_novelty_{split}"] = s.hla_groove_dist
        wide[f"hla_novelty_top3_{split}"] = s.hla_groove_top3
        wide[f"hla_n_train_{split}"] = s.hla_n_train
        wide[f"hla_novelty_pseudo_{split}"] = s.hla_pseudo_dist
        wide[f"hla_seen_{split}"] = s.hla_in_train
        # Hamming over 9 positions, scaled to [0, 1] so both novelty axes are
        # comparable fractional-mismatch quantities.
        wide[f"peptide_novelty_{split}"] = s.pep_min_hamming / 9.0
        wide[f"peptide_seen_{split}"] = s.pep_in_train

    wide["hla_novelty"] = wide["hla_novelty_hla"]
    wide["peptide_novelty"] = wide["peptide_novelty_peptide"]
    wide["seq_version"] = seq_version

    return wide.reset_index()


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
        g_dist, g_nn, g_top3 = nearest(alleles, train_alleles, groove)
        p_dist, p_nn, _ = nearest(alleles, train_alleles, pseudo)

        # Training examples per allele: the plan flags this as a strong, cheap
        # uncertainty predictor independent of sequence distance -- an allele
        # can be close to train and still barely measured.
        density = train.hla.value_counts().to_dict()

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
                "hla_groove_top3": d.hla.map(g_top3).to_numpy(),
                "hla_n_train": d.hla.map(density).fillna(0).astype(int).to_numpy(),
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

    long = pd.concat(frames, ignore_index=True)
    LONG_OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(LONG_OUT, index=False)
    print(f"\nwrote {LONG_OUT}  rows={len(long)}  (diagnostic long form)")

    wide = to_contract_shape(long, seq_version)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(wide)}  cols={len(wide.columns)}  "
          f"seq_version={seq_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
