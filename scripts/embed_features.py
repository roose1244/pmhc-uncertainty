"""Embed unique peptides and unique HLA grooves with frozen ESM-2 35M."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH
from src.embed import load_esm, embed_sequences, write_embedding_store


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    if master["hla_seq"].isna().any():
        raise SystemExit("hla_seq is empty. Run scripts/fetch_hla_sequences.py first.")

    peptides = sorted(master["peptide"].unique())
    hla_map = (
        master[["hla", "hla_seq"]]
        .drop_duplicates()
        .sort_values("hla")
    )
    hla_names = hla_map["hla"].tolist()
    hla_seqs = hla_map["hla_seq"].tolist()

    model, alphabet, device = load_esm()
    print(f"ESM-2 35M on {device}  peptides={len(peptides)}  alleles={len(hla_seqs)}")

    pep_mean, pep_res = embed_sequences(peptides, model, alphabet, device)
    write_embedding_store(peptides, pep_mean, pep_res, "esm_peptide")
    print(f"peptide mean {pep_mean.shape}  residue {pep_res.shape}")

    hla_mean, hla_res = embed_sequences(hla_seqs, model, alphabet, device, batch_size=8)
    write_embedding_store(hla_names, hla_mean, hla_res, "esm_hla")
    print(f"hla mean {hla_mean.shape}  residue {hla_res.shape}")


if __name__ == "__main__":
    main()
