"""Shared Modal objects. Import these; do not redefine them."""

import modal

app = modal.App("pmhc-guardian")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "fair-esm",
        "pandas",
        "pyarrow",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "biopython",
    )
)
vol = modal.Volume.from_name("pmhc-data", create_if_missing=True)
VOL = {"/vol": vol}
