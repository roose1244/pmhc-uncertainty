"""Restrict the HLA representation to the 34 peptide-contacting residues.

M2 mean-pools all 182 groove residues and M3 attends over all 182. Roughly 148
of those never touch the peptide: they are the beta-sheet floor and the outer
faces of the helices, which differ between alleles for reasons that have
nothing to do with which peptides bind. With 24,837 training rows, that is a
lot of capacity spent on positions that cannot carry the signal.

The 34 contact positions are not a guess. They are NetMHCpan's pseudo-sequence
positions, selected because they contact the peptide in solved crystal
structures, so this is structural knowledge distilled into an index -- the same
information a folding model would have to rediscover, at zero compute cost.
They were recovered here by matching MHC_pseudo.dat back onto the groove
sequences (scripts-level derivation, 75/75 alleles round-trip exactly, and all
34 agree with the published positions).

Four HLA representations are compared, everything else held at Dev 1's
protocol and all ESM features standardised on train, since that was worth
+0.04 to +0.08 on its own:

    mean182   mean-pool over the whole groove          (M2, the incumbent)
    mean34    mean-pool over the contact residues only
    attn182   peptide residues attend over the groove  (M3)
    attn34    peptide residues attend over contacts only

mean34 against mean182 asks whether the irrelevant positions were diluting the
average. attn34 against attn182 asks the same of the attention, which is the
likelier win: attention over 182 positions has to learn where to look from
24k rows, while attention over 34 starts with the search narrowed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import regression_metrics
from src.model import StabilityMLP

FEATURES_DIR = Path("data/features")
CONTACTS = Path("data/external/contact_positions.json")
PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")

SEED = 42
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12
EPS = 1e-6
D_MODEL = 128
N_HEADS = 4


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_residue(stem: str) -> tuple[np.ndarray, dict[str, int]]:
    arr = np.load(FEATURES_DIR / f"{stem}_residue.npy")
    idx = pd.read_csv(FEATURES_DIR / f"{stem}_index.csv")
    return arr, {str(r.string): int(r.row) for r in idx.itertuples()}


def contact_positions() -> list[int]:
    return json.loads(CONTACTS.read_text())


class AttnModel(nn.Module):
    """Peptide residues as queries over whichever HLA positions are supplied."""

    def __init__(self, d_in: int = 480, d: int = D_MODEL, heads: int = N_HEADS,
                 dropout: float = 0.2):
        super().__init__()
        self.pep = nn.Linear(d_in, d)
        self.hla = nn.Linear(d_in, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, pep, hla):
        q = self.pep(pep)
        kv = self.hla(hla)
        a, _ = self.attn(q, kv, kv)
        joint = self.norm(a + q)
        return self.head(torch.cat([joint.mean(1), q.mean(1)], -1)).squeeze(-1)


def train_one(master, split, rep: str, pep_arr, pep_lu, hla_arr, hla_lu, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    pos = contact_positions()

    hla_sel = hla_arr[:, pos, :] if rep.endswith("34") else hla_arr
    attention = rep.startswith("attn")

    merged = master.merge(split, on="id")
    train = merged[merged.fold == "train"]

    def idx(frame):
        return (np.array([pep_lu[p] for p in frame.peptide], dtype=np.int64),
                np.array([hla_lu[h] for h in frame.hla], dtype=np.int64),
                frame.log_half_life.to_numpy(np.float32))

    # Standardise on train only; mean-pool first when the model needs vectors.
    if attention:
        pep_store = torch.from_numpy(np.ascontiguousarray(pep_arr)).to(device)
        hla_store = torch.from_numpy(np.ascontiguousarray(hla_sel)).to(device)

        def batch(pi, hi):
            return (pep_store.index_select(0, torch.as_tensor(pi, device=device)),
                    hla_store.index_select(0, torch.as_tensor(hi, device=device)))
        model = AttnModel().to(device)
    else:
        pep_mean, hla_mean = pep_arr.mean(1), hla_sel.mean(1)
        tr_pi, tr_hi, _ = idx(train)
        x_tr = np.concatenate([pep_mean[tr_pi], hla_mean[tr_hi]], 1)
        mu, sd = x_tr.mean(0, keepdims=True), np.maximum(x_tr.std(0, keepdims=True), EPS)

        def batch(pi, hi):
            x = np.concatenate([pep_mean[pi], hla_mean[hi]], 1).astype(np.float32)
            return torch.from_numpy(((x - mu) / sd).astype(np.float32)).to(device), None
        model = StabilityMLP(n_in=pep_mean.shape[1] + hla_mean.shape[1]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    tr_pi, tr_hi, tr_y = idx(train)
    va_pi, va_hi, va_y = idx(merged[merged.fold == "val"])

    loader = DataLoader(
        TensorDataset(torch.from_numpy(tr_pi), torch.from_numpy(tr_hi),
                      torch.from_numpy(tr_y)),
        batch_size=BATCH, shuffle=True,
    )

    def forward(pi, hi):
        a, b = batch(pi, hi)
        return model(a, b) if attention else model(a)

    best_state, best_val, bad, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for pi, hi, yb in loader:
            opt.zero_grad()
            loss_fn(forward(pi.numpy(), hi.numpy()), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for s in range(0, len(va_y), 512):
                p = forward(va_pi[s:s + 512], va_hi[s:s + 512])
                yb = torch.from_numpy(va_y[s:s + 512]).to(device)
                tot += float(loss_fn(p, yb)) * len(yb); n += len(yb)
            val = tot / max(n, 1)
        if val < best_val - 1e-4:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_pi, all_hi, all_y = idx(merged)
    preds = np.empty(len(all_y), np.float32)
    with torch.no_grad():
        for s in range(0, len(all_y), 512):
            preds[s:s + 512] = forward(all_pi[s:s + 512], all_hi[s:s + 512]).cpu().numpy()

    out = pd.DataFrame({"id": merged.id.to_numpy(), "y_true": all_y,
                        "y_mean": preds, "y_std": np.nan,
                        "fold": merged.fold.to_numpy()})
    test = out[out.fold == "test"]
    m = regression_metrics(test.y_true.to_numpy(), test.y_mean.to_numpy())
    m["best_epoch"] = best_epoch
    m["val_mse"] = best_val
    return out, m
