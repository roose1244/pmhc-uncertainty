"""Does a larger protein language model help, or is 35M simply too small?

The challenge names several foundation models and flags compute as a real
constraint, so this tests the scale axis within one family rather than
sampling four architectures shallowly. ESM-2 35M and 150M share an API and an
embedding recipe, which means the comparison isolates capacity: anything that
changes is the model being bigger, not a different tokenizer, pooling rule or
training objective.

The pipeline is otherwise Dev 1's unchanged -- mean-pool the peptide and the
HLA groove, standardise on the train fold, feed the same MLP, early stop on
val, seed 42 -- so the result is directly comparable with every M2 number
already recorded.

Run:  python scripts/scale_esm.py [model_name]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MASTER = Path("data/processed/master.parquet")
SPLIT_DIR = Path("data/splits")
FEATURES = Path("data/features")
TABLE = Path("results/tables/esm_scale.csv")

SPLITS = ["random", "peptide", "cluster", "hla"]
# Measured already with esm2_t12_35M_UR50D, standardised, seed 42.
BASELINE_35M = {"random": 0.712, "peptide": 0.563, "cluster": 0.538, "hla": 0.112}
M1 = {"random": 0.765, "peptide": 0.642, "cluster": 0.629, "hla": -0.210}


def load(name: str):
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import esm

    fn = getattr(esm.pretrained, name)
    model, alphabet = fn()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.eval().to(device), alphabet, device


@torch.no_grad()
def embed(seqs: list[str], model, alphabet, device, layer: int,
          batch_size: int = 16) -> np.ndarray:
    """Mean-pooled final-layer representation, BOS/EOS dropped."""
    conv = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = [(f"s{j}", s) for j, s in enumerate(seqs[i:i + batch_size])]
        _, _, toks = conv(chunk)
        toks = toks.to(device)
        rep = model(toks, repr_layers=[layer])["representations"][layer]
        for k, (_, s) in enumerate(chunk):
            out.append(rep[k, 1:len(s) + 1].mean(0).float().cpu().numpy())
        print(f"   embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    return np.stack(out)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "esm2_t30_150M_UR50D"
    layer = int(name.split("_t")[1].split("_")[0])
    print(f"model {name}, final layer {layer}\n")

    master = pd.read_parquet(MASTER)
    seqs = pd.read_csv("data/external/hla_sequences.csv").set_index("hla")

    peptides = sorted(master.peptide.unique())
    alleles = sorted(master.hla.unique())
    grooves = [seqs.loc[a, "groove_seq"] for a in alleles]

    model, alphabet, device = load(name)
    print(f"on {device}\n")

    print(f"peptides ({len(peptides)}):")
    pep_arr = embed(peptides, model, alphabet, device, layer, batch_size=128)
    print(f"\nHLA grooves ({len(alleles)}):")
    hla_arr = embed(grooves, model, alphabet, device, layer, batch_size=8)

    FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES / f"{name}_peptide.npy", pep_arr)
    np.save(FEATURES / f"{name}_hla.npy", hla_arr)
    pep_lu = {p: i for i, p in enumerate(peptides)}
    hla_lu = {a: i for i, a in enumerate(alleles)}

    # --- train the same head on the new features -------------------------
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.evaluate import regression_metrics
    from src.model import StabilityMLP

    rows = []
    for split_name in SPLITS:
        split = pd.read_parquet(SPLIT_DIR / f"{split_name}.parquet")
        merged = master.merge(split, on="id")
        tr = merged[merged.fold == "train"]
        va = merged[merged.fold == "val"]

        def raw(frame):
            return np.concatenate(
                [pep_arr[[pep_lu[p] for p in frame.peptide]],
                 hla_arr[[hla_lu[h] for h in frame.hla]]], axis=1).astype(np.float32)

        x_tr = raw(tr)
        mu, sd = x_tr.mean(0, keepdims=True), np.maximum(x_tr.std(0, keepdims=True), 1e-6)
        scale = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
        x_tr = scale(x_tr)
        x_va = scale(raw(va))
        y_tr = tr.log_half_life.to_numpy(np.float32)
        y_va = va.log_half_life.to_numpy(np.float32)

        torch.manual_seed(42)
        np.random.seed(42)
        net = StabilityMLP(n_in=x_tr.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lf = nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                            batch_size=256, shuffle=True)
        vx, vy = torch.from_numpy(x_va).to(device), torch.from_numpy(y_va).to(device)

        best_state, best, bad, best_ep = None, float("inf"), 0, 0
        for ep in range(1, 81):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                lf(net(xb.to(device)), yb.to(device)).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lf(net(vx), vy))
            if v < best - 1e-4:
                best, best_ep, bad = v, ep, 0
                best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 12:
                    break
        net.load_state_dict(best_state)
        net.eval()

        te = merged[merged.fold == "test"]
        with torch.no_grad():
            pred = net(torch.from_numpy(scale(raw(te))).to(device)).cpu().numpy()
        m = regression_metrics(te.log_half_life.to_numpy(), pred)
        rows.append({"model": name, "split": split_name, "dim": x_tr.shape[1],
                     **m, "best_epoch": best_ep})
        print(f"\n  {split_name:8s} spearman={m['spearman']:+.3f}  "
              f"MAE={m['mae_hours']:.2f}h  epoch={best_ep}", flush=True)

    t = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLE, index=False)

    print(f"\nscale comparison (test-fold Spearman)")
    print(f"   {'split':9s} {'ESM-2 35M':>10s} {'this model':>11s} {'delta':>8s} {'M1':>8s}")
    for _, r in t.iterrows():
        base = BASELINE_35M[r.split]
        print(f"   {r.split:9s} {base:10.3f} {r.spearman:11.3f} "
              f"{r.spearman - base:+8.3f} {M1[r.split]:8.3f}")
    print(f"\nwrote {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
