"""Serve a prediction for one peptide-HLA pair, routing to the right model.

Loads the frozen models once and answers queries locally, so the demo does not
depend on a deployed endpoint. The same code is what a Modal endpoint would
wrap, so moving to a served version later changes where this runs, not what it
does.

ROUTING. M1 one-hot encodes the allele name, which is a perfect lookup for an
allele it has measured and an all-zero vector for one it has not. M2n reads an
ESM embedding of the groove, so it degrades gracefully instead of collapsing.
Measured on held-out alleles over five seeds, M1 scores -0.113 and M2n +0.090,
with non-overlapping ranges; on familiar alleles M1 is the stronger of the two.
So the allele's presence in the training set decides which model answers. That
feature needs no labels and is known at query time.

UNCERTAINTY. The returned std is ensemble-free -- there is one model per
regime, not five -- so it is a placeholder scaled by the model's measured
residual spread, NOT a calibrated uncertainty. The reliability verdict the app
shows comes from novelty (app/reliability.py), which tracks error across
alleles at +0.546 where ensemble spread does not (-0.055). Reported separately
on purpose: conflating them is the error this project exists to catch.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features import encode_hla, encode_peptides
from src.model import StabilityMLP

MODELS = Path("models")
FEATURES = Path("data/features")

# Residual spread of each model on its own regime, from results/tables/.
# Used only to give the interval a sensible width; not a calibrated quantile.
RESIDUAL_SD = {"m1": 0.50, "m2n": 0.62}
Z90 = 1.645


@dataclass
class Result:
    mean: float
    std: float
    lo: float
    hi: float
    model: str
    allele_known: bool

    @property
    def hours(self) -> float:
        return max(10.0**self.mean - 0.1, 0.0)

    @property
    def hours_interval(self) -> tuple[float, float]:
        return (max(10.0**self.lo - 0.1, 0.0), max(10.0**self.hi - 0.1, 0.0))


@lru_cache(maxsize=1)
def _bundle():
    manifest = json.loads((MODELS / "manifest.json").read_text())
    known = set(manifest["known_alleles"])

    with open(MODELS / "encoder.pkl", "rb") as fh:
        encoder = pickle.load(fh)
    sc = np.load(MODELS / "scaler.npz")

    pep_arr = np.load(FEATURES / "esm_peptide.npy")
    pep_idx = pd.read_csv(FEATURES / "esm_peptide_index.csv")
    pep_lu = {str(r.string): int(r.row) for r in pep_idx.itertuples()}
    hla_arr = np.load(FEATURES / "esm_hla.npy")
    hla_idx = pd.read_csv(FEATURES / "esm_hla_index.csv")
    hla_lu = {str(r.string): int(r.row) for r in hla_idx.itertuples()}

    n_in_m1 = 180 + len(encoder.categories_[0])
    m1 = StabilityMLP(n_in=n_in_m1)
    m1.load_state_dict(torch.load(MODELS / "m1.pt", map_location="cpu"))
    m1.eval()

    m2 = StabilityMLP(n_in=pep_arr.shape[1] + hla_arr.shape[1])
    m2.load_state_dict(torch.load(MODELS / "m2n.pt", map_location="cpu"))
    m2.eval()

    return dict(known=known, encoder=encoder, mu=sc["mu"], sd=sc["sd"],
                pep_arr=pep_arr, pep_lu=pep_lu, hla_arr=hla_arr, hla_lu=hla_lu,
                m1=m1, m2=m2)


def available() -> bool:
    return (MODELS / "manifest.json").exists() and (MODELS / "m1.pt").exists()


def predict(peptide: str, hla: str) -> Result | None:
    """None when the query cannot be served (no weights, or no embedding)."""
    if not available():
        return None
    b = _bundle()
    peptide, hla = peptide.strip().upper(), hla.strip()
    known = hla in b["known"]

    frame = pd.DataFrame({"peptide": [peptide], "hla": [hla]})

    if known:
        x = np.concatenate(
            [encode_peptides(frame.peptide.tolist()),
             encode_hla(b["encoder"], frame.hla.tolist())], axis=1
        ).astype(np.float32)
        model, tag = b["m1"], "m1"
    else:
        # M2n needs an ESM vector for both sides. Peptides outside the training
        # table have none precomputed, so the query cannot be served rather
        # than being answered from a wrong vector.
        if peptide not in b["pep_lu"] or hla not in b["hla_lu"]:
            return None
        raw = np.concatenate(
            [b["pep_arr"][b["pep_lu"][peptide]], b["hla_arr"][b["hla_lu"][hla]]]
        )[None, :]
        x = ((raw - b["mu"]) / b["sd"]).astype(np.float32)
        model, tag = b["m2"], "m2n"

    with torch.no_grad():
        mean = float(model(torch.from_numpy(x))[0])

    std = RESIDUAL_SD[tag]
    return Result(mean=mean, std=std, lo=mean - Z90 * std, hi=mean + Z90 * std,
                  model=tag, allele_known=known)
