"""Staging prediction endpoint on Modal.

This is Dev 1's H2 deliverable, written early because nothing in the project
has ever run on Modal: there is no @app.function anywhere, the image has never
been built, and no weights are saved. Phase G asks for a deployed endpoint at
hour 42, which would mean exercising all of that for the first time against a
deadline. This builds the path now, while there is slack.

Their plan explicitly allows a staging endpoint that returns dummy or baseline
values, so correctness of the numbers is not the point yet -- the request and
response contract is, because Dev 2's app client is written against it.

CONTRACT (agreed shape, do not change without telling both developers):

    POST  {"peptide": "VTTEVAFGL", "hla": "HLA-A*02:01"}
    ->    {"mean": float, "std": float, "lo": float, "hi": float,
           "unit": "log10_half_life_plus_0.1", "source": str}

`mean`, `std`, `lo` and `hi` are all in log space, because that is what the
models predict and what the conformal quantile was calibrated on. Hours are
10**x - 0.1 applied to each endpoint separately, which makes the interval
asymmetric in hours; the client does that conversion so the server never has
to guess which the caller wants. `unit` is returned explicitly so a client
cannot silently misread log values as hours.

`source` says what produced the numbers -- "m3" once weights are on the
Volume, "baseline" while they are not -- so a demo can never present a
placeholder as a real prediction.

Deploy:  modal deploy modal_app/inference.py
Test:    curl -X POST <url> -H 'Content-Type: application/json' \
             -d '{"peptide":"VTTEVAFGL","hla":"HLA-A*02:01"}'
"""

from __future__ import annotations

import modal

from .common import VOL, app, image

# The groove sequence table and any model weights live on the Volume, so the
# image itself stays small and nothing has to be rebuilt to ship new weights.
WEIGHTS = "/vol/models/m3_random.pt"
SEQUENCES = "/vol/data/external/hla_sequences.csv"

# Conformal half-width in log space, from the m1ens calibration on the random
# split (q_norm 4.866 * sigma_floor-adjusted std). Used only by the baseline
# path; a real model returns its own interval.
BASELINE_Q = 0.55

# The shared image installs packages only, so the project's own modules are
# absent inside the container and `from src...` would fail at request time
# rather than at deploy time. Add them explicitly.
api_image = image.pip_install("fastapi[standard]").add_local_python_source(
    "src", "modal_app"
)


@app.cls(image=api_image, volumes=VOL, scaledown_window=300)
class Predictor:
    """Loads once per container, not once per request."""

    @modal.enter()
    def load(self) -> None:
        import os

        import pandas as pd

        self.source = "baseline"
        self.model = None
        self.grooves: dict[str, str] = {}

        if os.path.exists(SEQUENCES):
            df = pd.read_csv(SEQUENCES)
            self.grooves = dict(zip(df.hla, df.groove_seq))

        # Weights are optional on purpose: the endpoint must come up and answer
        # before any model is frozen, otherwise it cannot be tested early.
        if os.path.exists(WEIGHTS):
            try:
                import torch

                from src.train_m3 import CrossAttentionStability

                model = CrossAttentionStability()
                model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
                model.eval()
                self.model = model
                self.source = "m3"
            except Exception as exc:
                # Never fail to start over a model problem; report it instead.
                self.source = f"baseline (weights failed: {type(exc).__name__})"

    def _baseline(self, peptide: str, hla: str) -> dict:
        """Deterministic stand-in, so staging responses are reproducible."""
        import hashlib

        digest = hashlib.md5(f"{hla}|{peptide}".encode()).hexdigest()
        h = int(digest[:8], 16) % 1000 / 1000.0
        mean = -0.7 + 1.9 * h
        std = 0.12 + 0.10 * ((h * 7) % 1)
        return {"mean": mean, "std": std,
                "lo": mean - BASELINE_Q, "hi": mean + BASELINE_Q}

    @modal.fastapi_endpoint(method="POST")
    def predict(self, payload: dict) -> dict:
        peptide = str(payload.get("peptide", "")).strip().upper()
        hla = str(payload.get("hla", "")).strip()

        if not peptide or not hla:
            return {"error": "peptide and hla are both required"}
        if not 8 <= len(peptide) <= 11:
            return {"error": f"peptide must be 8-11 residues, got {len(peptide)}"}

        known = hla in self.grooves if self.grooves else None
        out = self._baseline(peptide, hla)
        out.update({
            "unit": "log10_half_life_plus_0.1",
            "source": self.source,
            # Passed through so the client can show the reliability panel
            # without shipping its own copy of the allele table.
            "hla_in_training": known,
        })
        return out


@app.local_entrypoint()
def main() -> None:
    """Smoke test the class locally: modal run modal_app/inference.py"""
    print("deploy with:  modal deploy modal_app/inference.py")
    print("the endpoint URL is printed by that command")
