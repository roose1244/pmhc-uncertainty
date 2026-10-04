"""The single call that talks to the model.

Everything model-shaped lives behind predict(), so wiring Dev 1's endpoint in
is one function body and no UI change. Until GUARDIAN_URL is set, predict()
returns a clearly-labelled placeholder: `source` says where each result came
from, and the UI shows that, so a demo can never silently present a dummy
number as a real one.

Response contract, agreed shape:

    {"mean": float,   # log10(half_life + 0.1)
     "std": float,    # ensemble std, same units
     "lo": float,     # 90% interval, same units
     "hi": float}

Half-life in hours is 10**x - 0.1, applied to each endpoint separately, so the
interval is asymmetric in hours. That is correct and should not be symmetrised
for display.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CACHE_FILE = Path(__file__).parent / "cache" / "demo_predictions.json"
TIMEOUT = 90
DEFAULT_URL = "https://sarah04menla--pmhc-guardian-guardian-predict.modal.run"


@dataclass
class Prediction:
    mean: float
    std: float
    lo: float
    hi: float
    source: str  # "endpoint" | "cache" | "placeholder" | "endpoint-error"
    verdict: str = ""
    error: str = ""

    @property
    def half_life_hours(self) -> float:
        return to_hours(self.mean)

    @property
    def interval_hours(self) -> tuple[float, float]:
        return to_hours(self.lo), to_hours(self.hi)


def to_hours(log_value: float) -> float:
    """Inverse of log10(half_life + 0.1), floored at 0."""
    return max(10.0**log_value - 0.1, 0.0)


def endpoint() -> str:
    return os.environ.get("GUARDIAN_URL", DEFAULT_URL).strip()


def _cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _placeholder(peptide: str, hla: str) -> Prediction:
    """Deterministic stand-in, so the UI can be built and demoed offline.

    Derived from a stable digest rather than random, so the same query always
    gives the same card and screenshots stay reproducible. Python's built-in
    hash() is salted per process and would change between runs, so it is not
    usable here. These numbers mean nothing.
    """
    digest = hashlib.md5(f"{hla}|{peptide}".encode()).hexdigest()
    h = int(digest[:8], 16) % 1000 / 1000.0
    mean = -0.7 + 1.9 * h
    std = 0.12 + 0.10 * ((h * 7) % 1)
    return Prediction(mean, std, mean - 2.2 * std, mean + 2.2 * std, "placeholder")


def _local(peptide: str, hla: str) -> Prediction | None:
    """Frozen models on disk, routed by whether the allele was in training.

    Tried before the network: the demo should work with no endpoint deployed
    and no connectivity, and these are the same weights an endpoint would
    serve. Returns None when there is nothing to serve from.
    """
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.predict import predict as local_predict

        r = local_predict(peptide, hla)
        if r is None:
            return None
        return Prediction(r.mean, r.std, r.lo, r.hi, f"local:{r.model}")
    except Exception:
        return None


def predict(peptide: str, hla: str, timeout: int = TIMEOUT) -> Prediction:
    """Model call: endpoint, then local weights, then cache, then placeholder.

    Never raises: a demo that dies on a network blip is worse than one that
    says where its numbers came from.
    """
    url = endpoint()
    if url:
        try:
            import requests

            r = requests.post(
                url, json={"peptide": peptide, "hla": hla}, timeout=timeout
            )
            r.raise_for_status()
            d = r.json()
            if "mean" not in d:
                return Prediction(0, 0, 0, 0, "endpoint-error", error=str(d.get("error", d)))
            return Prediction(
                float(d["mean"]), float(d["std"]),
                float(d["lo"]), float(d["hi"]), "endpoint",
                verdict=str(d.get("verdict", "")),
            )
        except Exception:
            pass  # fall through to local weights, cache, then placeholder

    got = _local(peptide, hla)
    if got is not None:
        return got

    hit = _cache().get(f"{hla}|{peptide}")
    if hit:
        return Prediction(
            float(hit["mean"]), float(hit["std"]),
            float(hit["lo"]), float(hit["hi"]), "cache",
        )

    return _placeholder(peptide, hla)
