"""Pre-embed demo alleles and peptides so no query waits on ESM.

An allele outside the training table is embedded on demand, which costs a
one-off ESM-2 load of a second or two in a fresh process. That is fine in
normal use and a bad surprise in a live demo, where the first click is the one
people are watching and Streamlit reloads the process on every code change.

This writes the vectors to disk, so src/predict.py can serve them without
loading ESM at all. Anything not listed here still works -- it falls back to
on-demand embedding -- so this is purely a latency fix, not a restriction on
what can be queried.

The default allele set is every common class I allele absent from the training
table. That turns out to be the whole HLA-C locus: all 75 training alleles are
A or B, so the model has never seen a C allele at all, which is the clearest
demonstration available of a genuinely novel input.

Run:  python scripts/prewarm_cache.py [extra alleles...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.verify_hla_seq import read_fasta, rebuild  # noqa: E402

FASTA = Path("data/raw/hla_prot.fasta")
MANIFEST = Path("models/manifest.json")
OUT = Path("data/features/prewarmed.npz")

# Common class I alleles that are NOT in the training table.
DEMO_ALLELES = [
    "HLA-C*01:02", "HLA-C*03:04", "HLA-C*04:01", "HLA-C*05:01",
    "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02", "HLA-C*08:02",
    "HLA-C*12:03", "HLA-C*16:01", "HLA-B*44:02",
]

# Peptides worth having instant. Known epitopes a demo is likely to reach for.
DEMO_PEPTIDES = [
    "VTTEVAFGL",   # the app's default
    "GILGFVFTL",   # influenza M1, the textbook A*02:01 epitope
    "NLVPMVATV",   # CMV pp65
    "GLCTLVAML",   # EBV BMLF1
    "SLYNTVATL",   # HIV gag
    "KRWIILGLNK",  # HIV gag, 10-mer
    "WWWWWWWWW",   # deliberate nonsense, for the red case
]


def main() -> int:
    extra = sys.argv[1:]
    alleles = DEMO_ALLELES + [a for a in extra if a not in DEMO_ALLELES]

    known = set(json.loads(MANIFEST.read_text())["known_alleles"]) if MANIFEST.exists() else set()
    seqs = read_fasta(FASTA)

    # Import late: loading ESM is the expensive part and pointless if the
    # sequence lookups fail first.
    import certifi
    import os

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from src.embed import embed_sequences, load_esm

    grooves, skipped = {}, []
    for hla in alleles:
        if hla in known:
            skipped.append((hla, "already in training, served by M1"))
            continue
        groove, _, kind = rebuild(seqs, hla)
        if groove is None or len(groove) != 182:
            skipped.append((hla, f"no usable groove ({kind})"))
            continue
        grooves[hla] = groove

    print(f"alleles to embed: {len(grooves)}")
    for hla, why in skipped:
        print(f"   skipped {hla}: {why}")

    model, alphabet, device = load_esm()
    print(f"\nESM-2 35M on {device}")

    store: dict[str, np.ndarray] = {}
    if grooves:
        names = list(grooves)
        mean, _ = embed_sequences([grooves[n] for n in names], model, alphabet,
                                  device, batch_size=8)
        for n, v in zip(names, mean):
            store[f"hla::{n}"] = v
        print(f"embedded {len(names)} alleles")

    peps = [p for p in DEMO_PEPTIDES]
    mean, _ = embed_sequences(peps, model, alphabet, device, batch_size=8)
    for p, v in zip(peps, mean):
        store[f"pep::{p}"] = v
    print(f"embedded {len(peps)} peptides")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **store)
    size = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT}  ({len(store)} vectors, {size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
