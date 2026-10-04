"""Use the better encoder for each half, instead of forcing one everywhere.

The peptide-only ablations say BLOSUM beats frozen ESM-2 35M on every split
(0.411 vs 0.363 on random, 0.327 vs 0.206 on peptide, 0.298 vs 0.184 on
cluster). A 9-mer gives a language model almost no context to work with, while
BLOSUM encodes exactly the right prior -- which amino acids substitute for
which -- without needing any.

The HLA side is the opposite. One-hot encoding the allele name wins where every
allele is seen, because it is a perfect lookup, and collapses to a zero vector
where one is not: on the HLA split M1 scores -0.113 against M2's +0.090 over
five seeds, with non-overlapping ranges.

So M1 and M2 each pair the better encoder for one half with the worse for the
other. Three combinations are trained here:

    h_blosum_esm       BLOSUM peptide + ESM HLA
    h_blosum_onehot_esm BLOSUM peptide + one-hot HLA + ESM HLA
    h_esm_onehot       ESM peptide + one-hot HLA + ESM HLA

The second strictly nests M1 -- every feature M1 receives, plus the ESM HLA
block -- so it should not lose to M1 on any split. If it does, the cause is
optimisation rather than information, which is a different and more tractable
problem.

Only the ESM block is standardised. BLOSUM is already well scaled (sd 2.6) and
one-hot columns are indicator variables whose meaning standardisation would
destroy, inflating rare alleles. Mixing a scaled block with unscaled ones is
deliberate, not an oversight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import regression_metrics
from src.features import encode_hla, encode_peptides, fit_hla_encoder
from src.model import StabilityMLP

FEATURES_DIR = Path("data/features")
PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")

SEED = 42
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12
EPS = 1e-6

RECIPES = {
    "h_blosum_esm": dict(peptide="blosum", onehot=False, esm_hla=True),
    "h_blosum_onehot_esm": dict(peptide="blosum", onehot=True, esm_hla=True),
    "h_esm_onehot": dict(peptide="esm", onehot=True, esm_hla=True),
}


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_mean(stem: str) -> dict[str, np.ndarray]:
    arr = np.load(FEATURES_DIR / f"{stem}.npy")
    idx = pd.read_csv(FEATURES_DIR / f"{stem}_index.csv")
    return {str(r.string): arr[int(r.row)] for r in idx.itertuples()}


def build(frame, recipe, encoder, pep_store, hla_store, scaler):
    """Assemble the feature matrix; `scaler` standardises the ESM block only."""
    blocks = []
    if recipe["peptide"] == "blosum":
        blocks.append(encode_peptides(frame["peptide"].tolist()))
    else:
        blocks.append(np.stack([pep_store[p] for p in frame["peptide"]], 0))
    if recipe["onehot"]:
        blocks.append(encode_hla(encoder, frame["hla"].tolist()))
    if recipe["esm_hla"]:
        esm = np.stack([hla_store[h] for h in frame["hla"]], 0)
        if scaler is not None:
            mu, sd = scaler
            esm = (esm - mu) / sd
        blocks.append(esm.astype(np.float32))
    x = np.concatenate(blocks, axis=1).astype(np.float32)
    return x, frame["log_half_life"].to_numpy(np.float32)


def train_one(master, split, name, pep_store, hla_store, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    recipe = RECIPES[name]

    merged = master.merge(split, on="id")
    train = merged[merged.fold == "train"]
    encoder = fit_hla_encoder(train["hla"].tolist())

    # Scaler fitted on the train fold's ESM HLA rows only.
    esm_tr = np.stack([hla_store[h] for h in train["hla"]], 0)
    scaler = (esm_tr.mean(0, keepdims=True),
              np.maximum(esm_tr.std(0, keepdims=True), EPS))

    x_tr, y_tr = build(train, recipe, encoder, pep_store, hla_store, scaler)
    x_va, y_va = build(merged[merged.fold == "val"], recipe, encoder,
                       pep_store, hla_store, scaler)

    model = StabilityMLP(n_in=x_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH, shuffle=True,
    )
    va_x = torch.from_numpy(x_va).to(device)
    va_y = torch.from_numpy(y_va).to(device)

    best_state, best_val, bad, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(va_x), va_y))
        if val < best_val - 1e-4:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    x_all, y_all = build(merged, recipe, encoder, pep_store, hla_store, scaler)
    with torch.no_grad():
        pred = model(torch.from_numpy(x_all).to(device)).cpu().numpy()

    out = pd.DataFrame({"id": merged.id.to_numpy(), "y_true": y_all,
                        "y_mean": pred, "y_std": np.nan,
                        "fold": merged.fold.to_numpy()})
    test = out[out.fold == "test"]
    m = regression_metrics(test.y_true.to_numpy(), test.y_mean.to_numpy())
    m["best_epoch"] = best_epoch
    m["val_mse"] = best_val
    m["n_in"] = int(x_tr.shape[1])
    return out, m
