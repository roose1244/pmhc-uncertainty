"""Frozen ESM-2 embeddings. Embed each unique string once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "esm2_t12_35M_UR50D"
LAYER = 12
FEATURES_DIR = Path("data/features")


def load_esm(device: torch.device | None = None):
    import esm

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.eval().to(device)
    return model, alphabet, device


@torch.no_grad()
def embed_sequences(
    sequences: list[str],
    model,
    alphabet,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_pool, per_residue) for a list of amino-acid strings.

    BOS and EOS are dropped. per_residue is ragged in length, so it is only
    returned when every sequence has the same length; otherwise the second
    array is empty.
    """
    converter = alphabet.get_batch_converter()
    means = []
    residues = []
    same_len = len({len(s) for s in sequences}) == 1
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        labels = [(str(i), seq) for i, seq in enumerate(chunk)]
        _, _, tokens = converter(labels)
        tokens = tokens.to(device)
        reps = model(tokens, repr_layers=[LAYER])["representations"][LAYER]
        for i, seq in enumerate(chunk):
            # tokens: BOS, residues..., EOS (and possible padding)
            res = reps[i, 1 : 1 + len(seq)].detach().cpu().numpy().astype(np.float32)
            means.append(res.mean(axis=0))
            if same_len:
                residues.append(res)
    mean = np.stack(means, axis=0)
    residue = np.stack(residues, axis=0) if same_len else np.empty((0,))
    return mean, residue


def write_embedding_store(
    strings: list[str],
    mean: np.ndarray,
    residue: np.ndarray | None,
    stem: str,
    directory: Path = FEATURES_DIR,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{stem}.npy", mean)
    index = directory / f"{stem}_index.csv"
    with index.open("w") as handle:
        handle.write("string,row\n")
        for row, text in enumerate(strings):
            handle.write(f"{text},{row}\n")
    if residue is not None and residue.size:
        np.save(directory / f"{stem}_residue.npy", residue)
