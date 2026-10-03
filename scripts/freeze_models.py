"""Train the deployable models once and save everything needed to serve them.

Nothing in the project persists a model: Dev 1's scripts write predictions and
throw the weights away, so a demo has no way to answer a query that is not
already in a results file. This trains the two models the router needs, on the
full dataset, and saves the weights together with the fitted encoder and
scaler -- which matter as much as the weights, since a model fed differently
from how it was trained produces confident nonsense.

Deployment differs from evaluation in one way that is deliberate. The split
files exist to hold data back; a shipped model should see everything. So both
models train on all 75 alleles, using the random split's val fold only for
early stopping. The honest consequence is that the saved models have no
untouched test set -- their quality is the measured quality of the same
recipes under proper evaluation, reported in results/tables/, and the shipped
artefact should never be re-scored on data it trained on.

Saves to models/:
    m1.pt, m2n.pt      weights
    encoder.pkl        the fitted one-hot encoder (train alleles, in order)
    scaler.npz         ESM standardisation statistics
    manifest.json      what was trained, on what, and how to read the outputs

Run:  python scripts/freeze_models.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import encode_hla, encode_peptides, fit_hla_encoder  # noqa: E402
from src.model import StabilityMLP  # noqa: E402
from src.train_m2_scaled import load_store  # noqa: E402

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
MODELS = Path("models")

SEED = 42
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12
EPS = 1e-6


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def fit(x_tr, y_tr, x_va, y_va, label: str):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    model = StabilityMLP(n_in=x_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH, shuffle=True,
    )
    vx = torch.from_numpy(x_va).to(device)
    vy = torch.from_numpy(y_va).to(device)

    best_state, best_val, bad, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(vx), vy))
        if v < best_val - 1e-4:
            best_val, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"   {label:5s} n_in={x_tr.shape[1]:5d}  best_epoch={best_epoch:3d}  "
          f"val_mse={best_val:.4f}")
    return model, best_epoch, best_val


def main() -> int:
    master = pd.read_parquet(MASTER)
    rnd = pd.read_parquet(SPLIT_DIR / "random.parquet")
    d = master.merge(rnd, on="id")
    # Everything except the val fold is training data for the shipped model.
    train = d[d.fold != "val"]
    val = d[d.fold == "val"]
    print(f"deployable fit: {len(train)} train rows, {len(val)} val rows, "
          f"{train.hla.nunique()} alleles\n")

    MODELS.mkdir(exist_ok=True)
    pep_store = load_store("esm_peptide")
    hla_store = load_store("esm_hla")

    # --- M1: BLOSUM peptide + one-hot allele -----------------------------
    encoder = fit_hla_encoder(train["hla"].tolist())

    def m1_x(frame):
        return np.concatenate(
            [encode_peptides(frame["peptide"].tolist()),
             encode_hla(encoder, frame["hla"].tolist())], axis=1
        ).astype(np.float32)

    m1, e1, v1 = fit(m1_x(train), train.log_half_life.to_numpy(np.float32),
                     m1_x(val), val.log_half_life.to_numpy(np.float32), "m1")
    torch.save(m1.state_dict(), MODELS / "m1.pt")

    # --- M2n: standardised ESM peptide + ESM allele -----------------------
    def m2_raw(frame):
        return np.concatenate(
            [np.stack([pep_store[p] for p in frame.peptide], 0),
             np.stack([hla_store[h] for h in frame.hla], 0)], axis=1
        ).astype(np.float32)

    raw_tr = m2_raw(train)
    mu = raw_tr.mean(0, keepdims=True)
    sd = np.maximum(raw_tr.std(0, keepdims=True), EPS)
    m2, e2, v2 = fit(((raw_tr - mu) / sd).astype(np.float32),
                     train.log_half_life.to_numpy(np.float32),
                     ((m2_raw(val) - mu) / sd).astype(np.float32),
                     val.log_half_life.to_numpy(np.float32), "m2n")
    torch.save(m2.state_dict(), MODELS / "m2n.pt")

    with open(MODELS / "encoder.pkl", "wb") as fh:
        pickle.dump(encoder, fh)
    np.savez(MODELS / "scaler.npz", mu=mu, sd=sd)

    manifest = {
        "trained_on": "random split, all folds except val",
        "n_train_rows": int(len(train)),
        "n_alleles": int(train.hla.nunique()),
        "known_alleles": sorted(train.hla.unique().tolist()),
        "target": "log10(half_life_hours + 0.1)",
        "routing": "allele in known_alleles -> m1, else m2n",
        "m1": {"best_epoch": e1, "val_mse": round(v1, 5),
               "features": "BLOSUM50 9-mer (180) + one-hot HLA"},
        "m2n": {"best_epoch": e2, "val_mse": round(v2, 5),
                "features": "standardised ESM-2 35M mean-pool, peptide + HLA (960)"},
        "warning": "trained on all folds except val; do not re-score on the "
                   "split test folds. Measured quality is in results/tables/.",
    }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nsaved to {MODELS}/: m1.pt, m2n.pt, encoder.pkl, scaler.npz, manifest.json")
    print(f"known alleles: {manifest['n_alleles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
