"""Pocket model: BLOSUM50 peptide + BLOSUM50 of the 34 HLA contact residues.

Same MLP and early-stopping rule as M1. The contact string is pseudo_best
from Developer 2's DTU/IMGT table, so a held-out allele still has a sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import regression_metrics
from src.features import encode_chains, encode_peptides
from src.model import StabilityMLP
from src.train_m1 import BATCH, LR, MAX_EPOCHS, PATIENCE, SEED, WEIGHT_DECAY, _device, _set_seed

PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")
PSEUDO_PATH = Path("data/external/hla_pseudo.csv")


def load_pseudo() -> dict[str, str]:
    table = pd.read_csv(PSEUDO_PATH)
    sequences = dict(zip(table["hla"], table["pseudo_best"]))
    bad = [hla for hla, seq in sequences.items() if len(str(seq)) != 34]
    if bad:
        raise ValueError(f"pseudosequences that are not length 34: {bad}")
    return sequences


def _xy(frame: pd.DataFrame, pseudo: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted(set(frame["hla"]) - set(pseudo))
    if missing:
        raise ValueError(f"no pseudosequence for {missing}")
    peptide = encode_peptides(frame["peptide"].tolist())
    pocket = encode_chains([pseudo[hla] for hla in frame["hla"]])
    x = np.concatenate([peptide, pocket], axis=1)
    y = frame["log_half_life"].to_numpy(dtype=np.float32)
    return x, y


def train_one(master: pd.DataFrame, split: pd.DataFrame, pseudo: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    _set_seed(SEED)
    device = _device()
    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]
    x_train, y_train = _xy(train, pseudo)
    x_val, y_val = _xy(val, pseudo)
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
    x_all, y_all = _xy(merged, pseudo)
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
    metrics["n_in"] = int(x_train.shape[1])
    return out, metrics


def ensemble_split(
    master: pd.DataFrame, split: pd.DataFrame, pseudo: dict[str, str], name: str
) -> tuple[pd.DataFrame, dict]:
    """Five pocket members. Same bootstrap and conformal rule as the M1 ensemble."""
    from src.ensemble import N_MEMBERS, _fit_member, _predict
    from src.uncertainty import uncertainty_report

    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]
    x_train, y_train = _xy(train, pseudo)
    x_val, y_val = _xy(val, pseudo)
    x_all, y_all = _xy(merged, pseudo)
    member_preds = []
    for seed in range(N_MEMBERS):
        model = _fit_member(x_train, y_train, x_val, y_val, seed)
        member_preds.append(_predict(model, x_all))
        print(f"  {name} member {seed} done", flush=True)
    stacked = np.stack(member_preds, axis=1)
    preds = pd.DataFrame(
        {
            "id": merged["id"].to_numpy(),
            "y_true": y_all,
            "fold": merged["fold"].to_numpy(),
        }
    )
    for seed in range(N_MEMBERS):
        preds[f"m{seed}"] = stacked[:, seed]
    preds["y_mean"] = stacked.mean(axis=1)
    preds["y_std"] = stacked.std(axis=1)
    return uncertainty_report(preds, name)


def run_ensemble(master: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pseudo = load_pseudo()
    rows = []
    for name, split in splits.items():
        scored, report = ensemble_split(master, split, pseudo, name)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(PRED_DIR / f"mpocketens_{name}.parquet", index=False)
        rows.append(report)
        print(
            f"mpocketens {name:8}  spearman={report['spearman']:.3f}  "
            f"err-unc={report['err_unc_spearman_test']:.3f}  "
            f"cover={report['coverage_90']:.3f}  "
            f"MAE100={report['mae_100']:.2f}  MAE50={report['mae_50']:.2f}"
        )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "mpocketens.csv", index=False)
    return table


def run_all(master: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pseudo = load_pseudo()
    rows = []
    for name, split in splits.items():
        preds, metrics = train_one(master, split, pseudo)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PRED_DIR / f"mpocket_{name}.parquet", index=False)
        rows.append({"model": "mpocket", "split": name, **metrics})
        print(
            f"mpocket {name:8}  spearman={metrics['spearman']:.3f}  "
            f"MAE={metrics['mae_hours']:.2f}h  epoch={metrics['best_epoch']}"
        )
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "mpocket.csv", index=False)
    return table
