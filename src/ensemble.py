"""Five-member M1 deep ensemble. Bootstrap the train fold; never touch calib or test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import encode_hla, encode_peptides, fit_hla_encoder
from src.model import StabilityMLP
from src.train_m1 import BATCH, LR, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, _device
from src.uncertainty import uncertainty_report

N_MEMBERS = 5
PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")


def _fit_member(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> StabilityMLP:
    rng = np.random.default_rng(seed)
    take = rng.choice(len(x_train), size=len(x_train), replace=True)
    torch.manual_seed(seed)
    device = _device()
    model = StabilityMLP(n_in=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train[take]),
            torch.from_numpy(y_train[take]),
        ),
        batch_size=BATCH,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    best_state, best_val, bad = None, float("inf"), 0
    for _epoch in range(1, MAX_EPOCHS + 1):
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
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(model: StabilityMLP, x: np.ndarray) -> np.ndarray:
    device = _device()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy()


def ensemble_split(master: pd.DataFrame, split: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]
    encoder = fit_hla_encoder(train["hla"].tolist())

    def pack(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        peptide = encode_peptides(frame["peptide"].tolist())
        hla = encode_hla(encoder, frame["hla"].tolist())
        x = np.concatenate([peptide, hla], axis=1)
        y = frame["log_half_life"].to_numpy(dtype=np.float32)
        return x, y

    x_train, y_train = pack(train)
    x_val, y_val = pack(val)
    x_all, y_all = pack(merged)
    members = []
    for seed in range(N_MEMBERS):
        model = _fit_member(x_train, y_train, x_val, y_val, seed)
        members.append(_predict(model, x_all))
        print(f"  {name} member {seed} done", flush=True)
    stacked = np.stack(members, axis=1)
    preds = pd.DataFrame({"id": merged["id"].to_numpy(), "y_true": y_all, "fold": merged["fold"].to_numpy()})
    for seed in range(N_MEMBERS):
        preds[f"m{seed}"] = stacked[:, seed]
    preds["y_mean"] = stacked.mean(axis=1)
    preds["y_std"] = stacked.std(axis=1)
    scored, report = uncertainty_report(preds, name)
    return scored, report
