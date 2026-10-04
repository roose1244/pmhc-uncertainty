"""Pocket model that predicts a mean and a variance.

Same 860 inputs as the 34-residue model. Early stopping uses validation
negative log-likelihood only. Intervals are still split-conformal on calib.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import HeteroscedasticMLP, gaussian_nll
from src.train_m1 import BATCH, LR, MAX_EPOCHS, PATIENCE, SEED, WEIGHT_DECAY, _device, _set_seed
from src.train_pseudo import _xy, load_pseudo
from src.uncertainty import uncertainty_report

PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")


def train_one(
    master: pd.DataFrame, split: pd.DataFrame, pseudo: dict[str, str]
) -> tuple[pd.DataFrame, dict]:
    _set_seed(SEED)
    device = _device()
    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]
    x_train, y_train = _xy(train, pseudo)
    x_val, y_val = _xy(val, pseudo)
    model = HeteroscedasticMLP(n_in=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
            loss = gaussian_nll(*model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(gaussian_nll(*model(val_x), val_y))
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
    x_all, y_all = _xy(merged, pseudo)
    with torch.no_grad():
        mean, log_var = model(torch.from_numpy(x_all).to(device))
        mean = mean.cpu().numpy()
        sigma = np.exp(0.5 * log_var.cpu().numpy())
    preds = pd.DataFrame(
        {
            "id": merged["id"].to_numpy(),
            "y_true": y_all,
            "y_mean": mean,
            "y_std": sigma,
            "fold": merged["fold"].to_numpy(),
        }
    )
    scored, report = uncertainty_report(preds, "")
    report["best_epoch"] = best_epoch
    report["val_nll"] = best_val
    report["n_in"] = int(x_train.shape[1])
    return scored, report


def run_all(master: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pseudo = load_pseudo()
    rows = []
    for name, split in splits.items():
        scored, report = train_one(master, split, pseudo)
        report["split"] = name
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(PRED_DIR / f"mhetero_{name}.parquet", index=False)
        rows.append(report)
        print(
            f"mhetero {name:8}  spearman={report['spearman']:.3f}  "
            f"err-unc={report['err_unc_spearman_test']:.3f}  "
            f"cover={report['coverage_90']:.3f}  "
            f"MAE100={report['mae_100']:.2f}  MAE50={report['mae_50']:.2f}  "
            f"epoch={report['best_epoch']}"
        )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "mhetero.csv", index=False)
    return table
