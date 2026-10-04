"""Train M1 and the peptide-only ablation on one split."""

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

PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")
SEED = 42
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _xy(
    frame: pd.DataFrame,
    encoder,
    peptide_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    peptide = encode_peptides(frame["peptide"].tolist())
    if peptide_only:
        x = peptide
    else:
        hla = encode_hla(encoder, frame["hla"].tolist())
        x = np.concatenate([peptide, hla], axis=1)
    y = frame["log_half_life"].to_numpy(dtype=np.float32)
    return x, y


def _loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle)


def train_one(
    master: pd.DataFrame,
    split: pd.DataFrame,
    peptide_only: bool,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    _set_seed(seed)
    device = _device()
    merged = master.merge(split, on="id", how="inner")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]
    encoder = fit_hla_encoder(train["hla"].tolist())

    x_train, y_train = _xy(train, encoder, peptide_only)
    x_val, y_val = _xy(val, encoder, peptide_only)
    model = StabilityMLP(n_in=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad = 0
    best_epoch = 0
    train_loader = _loader(x_train, y_train, shuffle=True)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
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
    x_all, y_all = _xy(merged, encoder, peptide_only)
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


def run_all(master: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, split in splits.items():
        for peptide_only, model_name in ((False, "m1"), (True, "m1pep")):
            preds, metrics, n_in = train_one(master, split, peptide_only=peptide_only)
            path = PRED_DIR / f"{model_name}_{split_name}.parquet"
            preds.to_parquet(path, index=False)
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "n_in": n_in,
                    **metrics,
                }
            )
            print(
                f"{model_name:6} {split_name:8}  "
                f"spearman={metrics['spearman']:.3f}  "
                f"MAE={metrics['mae_hours']:.2f}h  "
                f"RMSE_log={metrics['rmse_log']:.3f}  "
                f"epoch={metrics['best_epoch']}"
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "m1.csv", index=False)
    return table
