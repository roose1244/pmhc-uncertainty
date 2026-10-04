"""M2: frozen ESM-2 mean-pools. Same MLP and protocol as M1."""

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


def load_store(stem: str) -> dict[str, np.ndarray]:
    mean = np.load(FEATURES_DIR / f"{stem}.npy")
    index = pd.read_csv(FEATURES_DIR / f"{stem}_index.csv")
    return {str(row.string): mean[int(row.row)] for row in index.itertuples()}


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _xy(frame, pep_store, hla_store, peptide_only: bool):
    pep = np.stack([pep_store[p] for p in frame["peptide"]], axis=0)
    if peptide_only:
        x = pep
    else:
        hla = np.stack([hla_store[h] for h in frame["hla"]], axis=0)
        x = np.concatenate([pep, hla], axis=1)
    y = frame["log_half_life"].to_numpy(dtype=np.float32)
    return x.astype(np.float32), y


def train_one(master, split, pep_store, hla_store, peptide_only: bool):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]
    x_train, y_train = _xy(train, pep_store, hla_store, peptide_only)
    x_val, y_val = _xy(val, pep_store, hla_store, peptide_only)
    model = StabilityMLP(n_in=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH,
        shuffle=True,
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
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    x_all, y_all = _xy(merged, pep_store, hla_store, peptide_only)
    with torch.no_grad():
        pred = model(torch.from_numpy(x_all).to(device)).cpu().numpy()
    out = pd.DataFrame(
        {
            "id": merged["id"].to_numpy(),
            "y_true": y_all,
            "y_mean": pred,
            "y_std": np.nan,
            "fold": merged["fold"].to_numpy(),
        }
    )
    test = out.loc[out["fold"] == "test"]
    metrics = regression_metrics(test["y_true"].to_numpy(), test["y_mean"].to_numpy())
    metrics["best_epoch"] = best_epoch
    metrics["val_mse"] = best_val
    return out, metrics, x_train.shape[1]


def run_all(master, splits):
    pep_store = load_store("esm_peptide")
    hla_store = load_store("esm_hla")
    rows = []
    for split_name, split in splits.items():
        for peptide_only, model_name in ((False, "m2"), (True, "m2pep")):
            preds, metrics, n_in = train_one(
                master, split, pep_store, hla_store, peptide_only
            )
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(PRED_DIR / f"{model_name}_{split_name}.parquet", index=False)
            rows.append({"model": model_name, "split": split_name, "n_in": n_in, **metrics})
            print(
                f"{model_name:6} {split_name:8}  "
                f"spearman={metrics['spearman']:.3f}  "
                f"MAE={metrics['mae_hours']:.2f}h  "
                f"epoch={metrics['best_epoch']}"
            )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "m2.csv", index=False)
    return table
