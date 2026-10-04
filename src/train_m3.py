"""M3: peptide residues cross-attend the HLA groove, instead of mean-pooling it.

Rung 3 of the model ladder, and the one the project's central claim depends on.

M1 encodes the allele as a one-hot of its *name*, so an unseen allele is a
zero vector and the model cannot generalise across alleles at all. M2 replaced
that with a mean-pooled ESM embedding of the 182-residue groove, which washes
the signal out: B*15:01 and B*15:02 differ at 5 of 182 positions, about 3% of
the vector, and M2 scores 0.037 on the held-out allele.

M3 keeps the groove as 182 separate residues and lets each peptide residue
attend over them. A five-residue difference in the binding pockets can then
change the representation substantially, because attention can concentrate on
exactly those positions rather than averaging them away.

Same protocol as M1 and M2 -- same MLP head, Adam, MSE on log half-life, early
stopping on val, seed 42 -- so the comparison is like for like and any
difference is the interaction mechanism rather than the training recipe.

The peptide-only twin required by the contract is M2's (ESM peptide alone):
with no HLA there is nothing to cross-attend to, so m3pep would be m2pep.

Run:  python scripts/train_m3.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import regression_metrics

FEATURES_DIR = Path("data/features")
PRED_DIR = Path("results/predictions")
TABLE_DIR = Path("results/tables")

SEED = 42
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12

D_MODEL = 128
N_HEADS = 4


class CrossAttentionStability(nn.Module):
    """Peptide residues as queries, HLA groove residues as keys and values.

    Both sides are projected from ESM's 480 dimensions down to 128 first,
    which keeps the attention matrix small (9 x 182 per example) and the
    parameter count comparable to the MLPs used for M1 and M2.
    """

    def __init__(self, d_in: int = 480, d: int = D_MODEL, heads: int = N_HEADS,
                 dropout: float = 0.2):
        super().__init__()
        self.pep_proj = nn.Linear(d_in, d)
        self.hla_proj = nn.Linear(d_in, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        # The head sees the attended peptide and the plain peptide side by
        # side, so it can fall back on peptide-only signal when the groove
        # contributes nothing -- which is what M2's failure suggests happens
        # for alleles with no close relative in training.
        self.head = nn.Sequential(
            nn.Linear(d * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, pep: torch.Tensor, hla: torch.Tensor) -> torch.Tensor:
        q = self.pep_proj(pep)              # (B, 9, d)
        kv = self.hla_proj(hla)             # (B, 182, d)
        attended, _ = self.attn(q, kv, kv)  # (B, 9, d)
        joint = self.norm(attended + q)     # residual, so attention adds to
        pooled = torch.cat(                 # rather than replaces the peptide
            [joint.mean(dim=1), q.mean(dim=1)], dim=-1
        )
        return self.head(pooled).squeeze(-1)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_residue_store(stem: str) -> tuple[np.ndarray, dict[str, int]]:
    arr = np.load(FEATURES_DIR / f"{stem}_residue.npy")
    index = pd.read_csv(FEATURES_DIR / f"{stem}_index.csv")
    lookup = {str(r.string): int(r.row) for r in index.itertuples()}
    return arr, lookup


def _rows(frame, pep_lookup, hla_lookup):
    """Index arrays rather than materialised features.

    Stacking the residue tensors for every row would be 28,166 x 182 x 480
    floats for the HLA side alone, about 9 GB. Indices are gathered per batch
    instead, which costs a little time and keeps memory flat.
    """
    pi = np.array([pep_lookup[p] for p in frame["peptide"]], dtype=np.int64)
    hi = np.array([hla_lookup[h] for h in frame["hla"]], dtype=np.int64)
    y = frame["log_half_life"].to_numpy(dtype=np.float32)
    return pi, hi, y


def to_device_store(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Park the whole residue array on the device once.

    There are only 5,633 unique peptides and 75 unique alleles, so the stores
    are 97 MB and 26 MB. Gathering per batch on the host instead would move
    about 89 MB per batch for the HLA side alone -- roughly 9 GB an epoch --
    which dominates the run. Indexing on device makes the gather almost free.
    """
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _batch(pep_store, hla_store, pi, hi, device):
    pi_t = torch.as_tensor(pi, dtype=torch.long, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.long, device=device)
    return pep_store.index_select(0, pi_t), hla_store.index_select(0, hi_t)


def train_one(master, split, pep_arr, pep_lookup, hla_arr, hla_lookup,
              split_name: str = ""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = _device()
    pep_store = to_device_store(pep_arr, device)
    hla_store = to_device_store(hla_arr, device)

    merged = master.merge(split, on="id")
    train = merged.loc[merged["fold"] == "train"]
    val = merged.loc[merged["fold"] == "val"]

    tr_pi, tr_hi, tr_y = _rows(train, pep_lookup, hla_lookup)
    va_pi, va_hi, va_y = _rows(val, pep_lookup, hla_lookup)

    model = CrossAttentionStability().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(tr_pi), torch.from_numpy(tr_hi),
                      torch.from_numpy(tr_y)),
        batch_size=BATCH, shuffle=True,
    )

    best_state, best_val, bad, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for pi_b, hi_b, yb in loader:
            pep, hla = _batch(pep_store, hla_store, pi_b.numpy(), hi_b.numpy(), device)
            yb = yb.to(device)
            opt.zero_grad()
            loss_fn(model(pep, hla), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses, n = 0.0, 0
            for s in range(0, len(va_y), 512):
                pep, hla = _batch(pep_store, hla_store, va_pi[s:s + 512],
                                  va_hi[s:s + 512], device)
                yb = torch.from_numpy(va_y[s:s + 512]).to(device)
                losses += float(loss_fn(model(pep, hla), yb)) * len(yb)
                n += len(yb)
            val_loss = losses / max(n, 1)

        print(f"    {split_name:8s} epoch {epoch:3d}  val_mse={val_loss:.5f}"
              f"{'  *' if val_loss < best_val - 1e-4 else ''}", flush=True)
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

    all_pi, all_hi, all_y = _rows(merged, pep_lookup, hla_lookup)
    preds = np.empty(len(all_y), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(all_y), 512):
            pep, hla = _batch(pep_store, hla_store, all_pi[s:s + 512],
                              all_hi[s:s + 512], device)
            preds[s:s + 512] = model(pep, hla).cpu().numpy()

    out = pd.DataFrame({
        "id": merged["id"].to_numpy(),
        "y_true": all_y,
        "y_mean": preds,
        "y_std": np.nan,
        "fold": merged["fold"].to_numpy(),
    })
    test = out.loc[out["fold"] == "test"]
    metrics = regression_metrics(test["y_true"].to_numpy(), test["y_mean"].to_numpy())
    metrics["best_epoch"] = best_epoch
    metrics["val_mse"] = best_val
    return out, metrics, model


def run_all(master, splits, save_weights: Path | None = None):
    pep_arr, pep_lookup = load_residue_store("esm_peptide")
    hla_arr, hla_lookup = load_residue_store("esm_hla")
    print(f"peptide residues {pep_arr.shape}  HLA residues {hla_arr.shape}  "
          f"device {_device()}\n")

    rows = []
    for split_name, split in splits.items():
        preds, metrics, model = train_one(
            master, split, pep_arr, pep_lookup, hla_arr, hla_lookup, split_name
        )
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PRED_DIR / f"m3_{split_name}.parquet", index=False)
        if save_weights:
            save_weights.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_weights / f"m3_{split_name}.pt")
        rows.append({"model": "m3", "split": split_name, **metrics})
        print(f"m3  {split_name:8}  spearman={metrics['spearman']:+.3f}  "
              f"MAE={metrics['mae_hours']:.2f}h  epoch={metrics['best_epoch']}")

    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "m3.csv", index=False)
    return table
