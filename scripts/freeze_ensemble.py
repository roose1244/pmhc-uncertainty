"""Freeze the peptide-split ensemble and its calibration-fold error model.

Every allele in the stability table is in this training fold. B*15:02 is
therefore known to the frozen model. The held-out HLA experiment was a
different fit. An allele absent from the table is reported as unseen.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import MASTER_PATH
from src.ensemble import train_members
from src.serve import FEATURES
from src.splits import SPLITS_DIR

MODELS = Path("models")


def _nearest(query: str, pool: list[tuple[str, str]]) -> tuple[float, float]:
    scored = sorted(
        (sum(a != b for a, b in zip(query, other)) / len(query), name)
        for name, other in pool
        if len(other) == len(query)
    )
    top3 = float(sum(dist for dist, _ in scored[:3]) / min(3, len(scored)))
    return scored[0][0], top3


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    split = pd.read_parquet(SPLITS_DIR / "peptide.parquet")
    novelty = pd.read_parquet("data/features/novelty.parquet")
    models, encoder, scored, report = train_members(master, split, "peptide")
    MODELS.mkdir(parents=True, exist_ok=True)
    import torch

    for seed, model in enumerate(models):
        torch.save(model.state_dict(), MODELS / f"member_{seed}.pt")
    with open(MODELS / "encoder.pkl", "wb") as handle:
        pickle.dump(encoder, handle)

    merged = scored.merge(novelty, on="id").merge(master[["id", "peptide", "hla"]], on="id")
    merged["abs_err"] = (merged["y_true"] - merged["y_mean"]).abs()
    calib = merged.loc[merged["fold"] == "calib"]
    error_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        random_state=42,
    )
    error_model.fit(calib[FEATURES], calib["abs_err"])
    with open(MODELS / "error_model.pkl", "wb") as handle:
        pickle.dump(error_model, handle)

    train = merged.loc[merged["fold"] == "train"]
    train[["peptide"]].drop_duplicates().to_parquet(MODELS / "train_peptides.parquet", index=False)
    grooves = master.groupby("hla").hla_seq.first().to_dict()
    pseudo = pd.read_csv("data/external/hla_pseudo.csv").set_index("hla")["pseudo_best"].to_dict()
    train_alleles = sorted(train["hla"].unique())
    groove_pool = [(hla, grooves[hla]) for hla in train_alleles]
    pseudo_pool = [(hla, pseudo[hla]) for hla in train_alleles if hla in pseudo]
    counts = train.groupby("hla").size().to_dict()
    rows = []
    for hla, seq in grooves.items():
        if hla in counts:
            groove_dist, groove_top3, pseudo_dist = 0.0, 0.0, 0.0
        else:
            groove_dist, groove_top3 = _nearest(seq, groove_pool)
            pseudo_dist, _ = _nearest(pseudo.get(hla, ""), pseudo_pool) if hla in pseudo else (float("nan"), 0.0)
        rows.append(
            {
                "hla": hla,
                "hla_seq": seq,
                "n_train": int(counts.get(hla, 0)),
                "groove_dist": groove_dist,
                "groove_top3": groove_top3,
                "pseudo_dist": pseudo_dist,
            }
        )
    pd.DataFrame(rows).to_parquet(MODELS / "allele_table.parquet", index=False)
    manifest = {
        "split": "peptide",
        "n_members": len(models),
        "n_in": int(models[0].net[0].in_features),
        "q_norm": report["q_norm"],
        "q_const": report["q_const"],
        "sigma_floor": report["sigma_floor"],
        "known_allele_count": int(train["hla"].nunique()),
        "note": (
            "Trained on the peptide split. Every stability-table allele is known. "
            "B*15:02 was held out only in the separate HLA-split experiment."
        ),
    }
    # median of predicted error on calib, used as the high/low cut
    manifest["pred_err_median"] = float(pd.Series(error_model.predict(calib[FEATURES])).median())
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("q_norm", "sigma_floor", "pred_err_median", "known_allele_count")}, indent=2))
    print(f"test spearman of this fit: {report['spearman']:.3f}")


if __name__ == "__main__":
    main()
