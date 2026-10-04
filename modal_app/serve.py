"""Deploy the frozen peptide-split ensemble.

Contract, shared with Developer 2. Do not rename these keys:

    POST {"peptide": "GILGFVFTL", "hla": "A*02:01"}
    ->   mean, std, lo, hi     all in log10(half-life + 0.1)
         unit = "log10_half_life_plus_0.1"
         source = "m1ens_peptide"

Hours are 10**x - 0.1 at each endpoint. Extra keys (verdict, allele_known,
pred_err, half_life_hours) are safe for a client that only reads the contract.

This model was trained on 9-mers only. Other lengths return an error rather
than a made-up prediction.

    modal deploy modal_app/serve.py
"""

from __future__ import annotations

import os

import modal

from modal_app.common import VOL, app, image
import modal_app.web  # noqa: F401  registers the public demo page on this app

api_image = image.pip_install("fastapi[standard]").add_local_python_source("src", "modal_app")


@app.cls(image=api_image, volumes=VOL, timeout=300, scaledown_window=300)
class Guardian:
    @modal.enter()
    def load(self) -> None:
        os.environ["PMHC_MODELS"] = "/vol/models"
        from src.serve import _bundle

        _bundle()
        self.ready = True

    @modal.fastapi_endpoint(method="POST")
    def predict(self, payload: dict) -> dict:
        from src.serve import predict as local_predict

        peptide = str(payload.get("peptide", "")).strip().upper()
        hla = str(payload.get("hla", "")).strip()
        if not peptide or not hla:
            return {"error": "peptide and hla are both required"}
        try:
            result = local_predict(peptide, hla)
        except ValueError as exc:
            return {"error": str(exc)}
        return {
            "mean": result["log_half_life"],
            "std": result["y_std"],
            "lo": result["lo"],
            "hi": result["hi"],
            "unit": "log10_half_life_plus_0.1",
            "source": "m1ens_peptide",
            "hla_in_training": result["allele_known"],
            "peptide_seen": result["peptide_seen"],
            "pred_err": result["pred_err"],
            "half_life_hours": result["half_life_hours"],
            "lo_hours": result["lo_hours"],
            "hi_hours": result["hi_hours"],
            "member_hours": result.get("member_hours", []),
            "verdict": result["verdict"],
        }


@app.local_entrypoint()
def main() -> None:
    print("deploy with: modal deploy modal_app/serve.py")
