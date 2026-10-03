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


@lru_cache(maxsize=1)
def _esm():
    """ESM-2 35M, loaded once and only if an unseen sequence actually arrives.

    The macOS python.org build ships no CA bundle, so torch.hub's download of
    the ESM weights fails with CERTIFICATE_VERIFY_FAILED and the cached file
    ends up being an HTML error page that then fails to unpickle. Point
    OpenSSL at certifi's bundle before any download is attempted.
    """
    import os

    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

    from src.embed import load_esm

    return load_esm()


@lru_cache(maxsize=1)
def _prewarmed() -> dict[str, np.ndarray]:
    """Vectors written by scripts/prewarm_cache.py, if that has been run.

    Checked before any on-demand embedding, so a demo query never waits on an
    ESM load in a freshly started process. Absent file is not an error: it
    just means every novel query embeds itself.
    """
    path = FEATURES / "prewarmed.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return {}


@lru_cache(maxsize=256)
def embed_on_demand(sequence: str) -> np.ndarray | None:
    """Mean-pooled ESM vector for a sequence that was not precomputed."""
    try:
        from src.embed import embed_sequences

        model, alphabet, device = _esm()
        mean, _ = embed_sequences([sequence], model, alphabet, device, batch_size=1)
        return mean[0]
    except Exception:
        return None


@lru_cache(maxsize=256)
def embed_allele(hla: str) -> np.ndarray | None:
    """Groove sequence for any IMGT allele, then its ESM vector.

    The local IMGT protein FASTA holds every deposited allele, so an allele
    absent from the training table is still resolvable: look up its sequence,
    take the mature alpha1-alpha2 groove by the same rule used to build the
    training features, and embed it.
    """
    try:
        import sys
        from pathlib import Path as _P

        sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
        from scripts.verify_hla_seq import read_fasta, rebuild

        fasta = _P("data/raw/hla_prot.fasta")
        if not fasta.exists():
            return None
        groove, _, _ = rebuild(read_fasta(fasta), hla)
        if groove is None:
            return None
        return embed_on_demand(groove)
    except Exception:
        return None


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
        # M2n needs an ESM vector for both sides. Anything outside the training
        # table has none precomputed, so it is embedded on demand -- which is
        # the entire point of a sequence-based model: an allele it has never
        # seen still has a sequence, and that sequence can be read.
        warm = _prewarmed()
        if peptide in b["pep_lu"]:
            pep_vec = b["pep_arr"][b["pep_lu"][peptide]]
        else:
            pep_vec = warm.get(f"pep::{peptide}")
            if pep_vec is None:
                pep_vec = embed_on_demand(peptide)
        if hla in b["hla_lu"]:
            hla_vec = b["hla_arr"][b["hla_lu"][hla]]
        else:
            hla_vec = warm.get(f"hla::{hla}")
            if hla_vec is None:
                hla_vec = embed_allele(hla)
        if pep_vec is None or hla_vec is None:
            return None
        raw = np.concatenate([pep_vec, hla_vec])[None, :]
        x = ((raw - b["mu"]) / b["sd"]).astype(np.float32)
        model, tag = b["m2"], "m2n"

    with torch.no_grad():
        mean = float(model(torch.from_numpy(x))[0])

    std = RESIDUAL_SD[tag]
    return Result(mean=mean, std=std, lo=mean - Z90 * std, hi=mean + Z90 * std,
                  model=tag, allele_known=known)
