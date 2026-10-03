"""M1 features: BLOSUM50 peptide + train-only HLA one-hot."""

from __future__ import annotations

import numpy as np
from Bio.Align import substitution_matrices
from sklearn.preprocessing import OneHotEncoder

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


def _blosum50_table() -> dict[str, np.ndarray]:
    matrix = substitution_matrices.load("BLOSUM50")
    alphabet = list(matrix.alphabet)
    index = {aa: alphabet.index(aa) for aa in AMINO_ACIDS}
    table = {}
    for aa in AMINO_ACIDS:
        table[aa] = np.array(
            [float(matrix[index[aa], index[other]]) for other in AMINO_ACIDS],
            dtype=np.float32,
        )
    return table


_BLOSUM = _blosum50_table()


def blosum_peptide(peptide: str) -> np.ndarray:
    if len(peptide) != 9:
        raise ValueError(f"M1 expects a 9-mer, got {peptide!r}")
    return np.concatenate([_BLOSUM[residue] for residue in peptide], dtype=np.float32)


def encode_peptides(peptides: list[str] | np.ndarray) -> np.ndarray:
    return np.stack([blosum_peptide(str(p)) for p in peptides], axis=0)


def fit_hla_encoder(train_alleles: list[str] | np.ndarray) -> OneHotEncoder:
    """Unknown alleles become a zero vector. Fit on train only."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(np.asarray(train_alleles, dtype=object).reshape(-1, 1))
    return encoder


def encode_hla(encoder: OneHotEncoder, alleles: list[str] | np.ndarray) -> np.ndarray:
    return encoder.transform(np.asarray(alleles, dtype=object).reshape(-1, 1)).astype(
        np.float32
    )
