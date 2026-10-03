"""M2 with standardised ESM features. Same model, same protocol, better conditioning.

M2 loses to M1 on every split, and the inputs explain why. ESM-2 mean-pool
embeddings carry per-dimension offsets as large as 6.16 while the per-dimension
spread is 0.046 to 0.308 -- some dimensions are a near-constant 6 carrying a
signal of sd 0.07, a 90:1 offset-to-signal ratio, and the spreads themselves
differ 6.7x across dimensions.

A small MLP with weight decay handles that badly on two counts: the first
layer spends capacity representing constant offsets, and the weights needed to
exploit a 0.05-sd direction are large enough for weight decay to suppress.
BLOSUM does not have this problem (sd 2.6, roughly centred), which is why M1
needs no scaling and M2 does.

The fix is per-dimension standardisation, fitted on the TRAIN fold only so no
information from val, test or calib reaches the scaler. Everything else --
the MLP, Adam, MSE on log half-life, early stopping on val, seed 42 -- is
Dev 1's protocol unchanged, so any difference is attributable to the scaling
alone.

Dev 1's src/train_m2.py is left untouched; this is a parallel module.

Run:  python scripts/train_m2_scaled.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import regression_metrics
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


def load_store(stem: str) -> dict[str, np.ndarray]:
    mean = np.load(FEATURES_DIR / f"{stem}.npy")
    index = pd.read_csv(FEATURES_DIR / f"{stem}_index.csv")
    return {str(r.string): mean[int(r.row)] for r in index.itertuples()}


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _raw_xy(frame, pep_store, hla_store, peptide_only: bool):
    pep = np.stack([pep_store[p] for p in frame["peptide"]], axis=0)
    x = pep if peptide_only else np.concatenate(
        [pep, np.stack([hla_store[h] for h in frame["hla"]], axis=0)], axis=1
    )
    y = frame["log_half_life"].to_numpy(dtype=np.float32)
    return x.astype(np.float32), y


def train_one(master, split, pep_store, hla_store, peptide_only: bool, scale: bool):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()

    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]

    x_train, y_train = _raw_xy(train, pep_store, hla_store, peptide_only)
    x_val, y_val = _raw_xy(val, pep_store, hla_store, peptide_only)

    if scale:
        # Fitted on train only. A dimension that is constant in train gets a
        # unit divisor rather than exploding, via EPS.
        mu = x_train.mean(axis=0, keepdims=True)
        sd = x_train.std(axis=0, keepdims=True)
        sd = np.maximum(sd, EPS)
        apply = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
    else:
        apply = lambda a: a  # noqa: E731

    x_train, x_val = apply(x_train), apply(x_val)

    model = StabilityMLP(n_in=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH, shuffle=True,
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_state, best_val, bad, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y))
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    x_all, y_all = _raw_xy(merged, pep_store, hla_store, peptide_only)
    with torch.no_grad():
        pred = model(torch.from_numpy(apply(x_all)).to(device)).cpu().numpy()

    out = pd.DataFrame({
        "id": merged["id"].to_numpy(), "y_true": y_all, "y_mean": pred,
        "y_std": np.nan, "fold": merged["fold"].to_numpy(),
    })
    test = out.loc[out["fold"] == "test"]
    metrics = regression_metrics(test["y_true"].to_numpy(), test["y_mean"].to_numpy())
    metrics["best_epoch"] = best_epoch
    metrics["val_mse"] = best_val
    return out, metrics


def run_all(master, splits, write_predictions: bool = True):
    pep_store = load_store("esm_peptide")
    hla_store = load_store("esm_hla")
    rows = []
    for split_name, split in splits.items():
        for scale in (False, True):
            for peptide_only, base in ((False, "m2"), (True, "m2pep")):
                name = f"{base}n" if scale else base
                preds, metrics = train_one(
                    master, split, pep_store, hla_store, peptide_only, scale
                )
                if write_predictions and scale:
                    PRED_DIR.mkdir(parents=True, exist_ok=True)
                    preds.to_parquet(
                        PRED_DIR / f"{name}_{split_name}.parquet", index=False
                    )
                rows.append({"model": name, "split": split_name,
                             "scaled": scale, **metrics})
                print(f"  {name:7s} {split_name:8s} spearman={metrics['spearman']:+.3f}  "
                      f"MAE={metrics['mae_hours']:.2f}h  epoch={metrics['best_epoch']}",
                      flush=True)
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "m2_scaled.csv", index=False)
    return table
