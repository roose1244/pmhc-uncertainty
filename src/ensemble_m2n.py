"""Five-member deep ensemble of M2n, to test the hypothesis on a model that can see the allele.

H2/H3 ask whether predictive uncertainty identifies unreliable predictions
under peptide and HLA distribution shift. Every uncertainty number in the
project so far comes from an ensemble of M1, and M1 cannot answer the HLA half
of that question even in principle: it one-hot encodes the allele NAME, so
every allele outside training is the same zero vector, all five members
receive identical input, and their disagreement cannot depend on which allele
it is. Measured, that shows up as novelty-vs-y_std of -0.055 while
novelty-vs-error is +0.546.

M2n reads an ESM embedding of the groove instead, so members CAN disagree
about an allele they have never seen -- there is something allele-specific in
the input for them to disagree about. It is also the better model on unseen
alleles: +0.090 against M1's -0.113 over five seeds, non-overlapping ranges.

So this is the ensemble the hypothesis needs. It follows Dev 1's recipe
exactly -- bootstrap the train fold, five seeds, same early stopping, the same
split-conformal calibration from src/uncertainty.py -- so the comparison
against m1ens isolates the base model and nothing else.

The one addition is standardisation. ESM mean-pools carry per-dimension
offsets up to 6.16 against spreads of 0.046 to 0.308, which costs about 0.06
Spearman when left raw. Statistics are fitted on the bootstrap sample of each
member, so no member sees val, calib or test when computing them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import StabilityMLP
from src.train_m1 import BATCH, LR, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, _device
from src.train_m2_scaled import load_store
from src.uncertainty import uncertainty_report

N_MEMBERS = 5
EPS = 1e-6
PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")


def _fit_member(x_train, y_train, x_val, y_val, seed: int) -> StabilityMLP:
    """One member: bootstrap resample plus a different seed, as in src/ensemble.py."""
    rng = np.random.default_rng(seed)
    take = rng.choice(len(x_train), size=len(x_train), replace=True)
    torch.manual_seed(seed)
    device = _device()

    model = StabilityMLP(n_in=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train[take]), torch.from_numpy(y_train[take])),
        batch_size=BATCH, shuffle=True,
    )
    vx, vy = torch.from_numpy(x_val).to(device), torch.from_numpy(y_val).to(device)

    best_state, best_val, bad = None, float("inf"), 0
    for _ in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(vx), vy))
        if v < best_val - 1e-4:
            best_val, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
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


def ensemble_split(master: pd.DataFrame, split: pd.DataFrame, name: str,
                   pep_store=None, hla_store=None) -> tuple[pd.DataFrame, dict]:
    pep_store = pep_store or load_store("esm_peptide")
    hla_store = hla_store or load_store("esm_hla")

    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]

    def raw(frame):
        return np.concatenate(
            [np.stack([pep_store[p] for p in frame["peptide"]], 0),
             np.stack([hla_store[h] for h in frame["hla"]], 0)], axis=1
        ).astype(np.float32)

    x_train_raw, x_val_raw, x_all_raw = raw(train), raw(val), raw(merged)
    y_train = train["log_half_life"].to_numpy(np.float32)
    y_val = val["log_half_life"].to_numpy(np.float32)
    y_all = merged["log_half_life"].to_numpy(np.float32)

    members = []
    for seed in range(N_MEMBERS):
        # Standardise on this member's own bootstrap sample, so the statistics
        # vary with the member and never see held-out folds.
        rng = np.random.default_rng(seed)
        take = rng.choice(len(x_train_raw), size=len(x_train_raw), replace=True)
        mu = x_train_raw[take].mean(0, keepdims=True)
        sd = np.maximum(x_train_raw[take].std(0, keepdims=True), EPS)
        scale = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731

        model = _fit_member(scale(x_train_raw), y_train, scale(x_val_raw), y_val, seed)
        members.append(_predict(model, scale(x_all_raw)))
        print(f"  {name} member {seed} done", flush=True)

    stacked = np.stack(members, axis=1)
    preds = pd.DataFrame({"id": merged["id"].to_numpy(), "y_true": y_all,
                          "fold": merged["fold"].to_numpy()})
    for seed in range(N_MEMBERS):
        preds[f"m{seed}"] = stacked[:, seed]
    preds["y_mean"] = stacked.mean(axis=1)
    preds["y_std"] = stacked.std(axis=1)

    return uncertainty_report(preds, name)
