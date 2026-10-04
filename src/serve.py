"""Local prediction from the frozen peptide-split ensemble.

This fit saw every allele in the stability table and did not see the peptide-split
test peptides. An allele outside that table is not a calibrated query: ensemble
spread does not warn about a new HLA.
"""

from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import encode_hla, encode_peptides
from src.model import StabilityMLP
from src.hla import normalise_hla
from src.target import from_log

def models_dir() -> Path:
    return Path(os.environ.get("PMHC_MODELS", "models"))


FEATURES = [
    "y_std",
    "hla_novelty_peptide",
    "hla_novelty_top3_peptide",
    "hla_novelty_pseudo_peptide",
    "hla_n_train_peptide",
    "peptide_novelty_peptide",
]


def _mismatch(a: str, b: str) -> float:
    return sum(x != y for x, y in zip(a, b)) / len(a)


@lru_cache(maxsize=1)
def _bundle() -> dict:
    root = models_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    with open(root / "encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)
    with open(root / "error_model.pkl", "rb") as handle:
        error_model = pickle.load(handle)
    members = []
    for seed in range(manifest["n_members"]):
        model = StabilityMLP(n_in=manifest["n_in"])
        state = torch_load(root / f"member_{seed}.pt")
        model.load_state_dict(state)
        model.eval()
        members.append(model)
    alleles = pd.read_parquet(root / "allele_table.parquet")
    peptides = pd.read_parquet(root / "train_peptides.parquet")
    return {
        "manifest": manifest,
        "encoder": encoder,
        "error_model": error_model,
        "members": members,
        "alleles": alleles.set_index("hla"),
        "train_peptides": peptides["peptide"].tolist(),
        "known": set(alleles["hla"]),
    }


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _novelty(peptide: str, hla: str, bundle: dict) -> dict[str, float]:
    alleles = bundle["alleles"]
    known = hla in bundle["known"]
    if known:
        hla_dist = 0.0
        top3 = 0.0
        pseudo = 0.0
        n_train = float(alleles.loc[hla, "n_train"])
    elif hla in alleles.index:
        hla_dist = float(alleles.loc[hla, "groove_dist"])
        top3 = float(alleles.loc[hla, "groove_top3"])
        pseudo = float(alleles.loc[hla, "pseudo_dist"])
        n_train = 0.0
    else:
        hla_dist = top3 = pseudo = float("nan")
        n_train = 0.0
    train_peps = bundle["train_peptides"]
    ham = min(_mismatch(peptide, other) for other in train_peps) if train_peps else 1.0
    return {
        "hla_novelty_peptide": hla_dist,
        "hla_novelty_top3_peptide": top3,
        "hla_novelty_pseudo_peptide": pseudo,
        "hla_n_train_peptide": n_train,
        "peptide_novelty_peptide": ham,
    }


def _verdict(allele_known: bool, pred_err: float, threshold: float, peptide_novel: bool) -> str:
    if not allele_known:
        return (
            "Unseen HLA. Ensemble spread is not a warning for a new allele. "
            "Do not treat the interval as calibrated."
        )
    bits = ["Allele was in training."]
    if peptide_novel:
        bits.append("Peptide was not in training.")
    else:
        bits.append("Peptide was in training, so this is not a generalisation test.")
    if pred_err <= threshold:
        bits.append("Predicted error is in the lower half of the calibration set.")
    else:
        bits.append("Predicted error is in the upper half of the calibration set.")
    return " ".join(bits)


def predict(peptide: str, hla: str) -> dict:
    peptide = peptide.strip().upper()
    hla = normalise_hla(hla)
    if len(peptide) != 9 or any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in peptide):
        raise ValueError("peptide must be a 9-mer of standard amino acids")
    bundle = _bundle()
    encoder = bundle["encoder"]
    x = np.concatenate(
        [encode_peptides([peptide]), encode_hla(encoder, [hla])],
        axis=1,
    )
    import torch

    with torch.no_grad():
        member_preds = [float(model(torch.from_numpy(x))[0]) for model in bundle["members"]]
    mean = float(np.mean(member_preds))
    std = float(np.std(member_preds))
    manifest = bundle["manifest"]
    sigma = max(std, manifest["sigma_floor"])
    lo = mean - manifest["q_norm"] * sigma
    hi = mean + manifest["q_norm"] * sigma
    novelty = _novelty(peptide, hla, bundle)
    features = pd.DataFrame([{**novelty, "y_std": std}])
    pred_err = float(bundle["error_model"].predict(features[FEATURES])[0])
    allele_known = hla in bundle["known"]
    peptide_seen = novelty["peptide_novelty_peptide"] == 0.0
    return {
        "peptide": peptide,
        "hla": hla,
        "allele_known": allele_known,
        "peptide_seen": peptide_seen,
        "log_half_life": mean,
        "half_life_hours": from_log(mean),
        "lo": lo,
        "hi": hi,
        "lo_hours": from_log(lo),
        "hi_hours": from_log(hi),
        "y_std": std,
        "member_hours": [from_log(value) for value in member_preds],
        "pred_err": pred_err,
        "verdict": _verdict(allele_known, pred_err, manifest["pred_err_median"], not peptide_seen),
    }
