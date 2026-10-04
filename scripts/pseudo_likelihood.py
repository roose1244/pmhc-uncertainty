"""ESM pseudo-likelihood of the peptide, as a feature the embeddings do not carry.

The challenge brief notes that foundation models "produce embeddings, log
likelihoods/perplexities, and both internal and external confidence metrics"
and can "be run with different inputs masked". Everything in this project so
far uses only the first of those. This adds the second.

The quantity is the masked pseudo-log-likelihood: mask each residue of the
peptide in turn, ask the model how probable the true residue is given the rest,
and sum the log probabilities. It measures how ordinary a sequence looks to a
model trained on natural proteins, which is a different question from where the
sequence sits in embedding space -- two peptides can have near-identical mean
pooled embeddings and very different likelihoods.

Two variants are computed, because the interesting one is the difference:

    plain     the peptide scored alone
    in_hla    the peptide scored with the HLA groove prepended as context

If the context version carries signal the plain one does not, the model is
using the allele to judge the peptide, which is exactly the interaction the
stability target depends on. If they are identical, ESM-2 is not conditioning
on the groove at all, and that is worth knowing before building on it.

Run:  python scripts/pseudo_likelihood.py [n_peptides]
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
SEQS = Path("data/external/hla_sequences.csv")
OUT = Path("data/features/pseudo_likelihood.parquet")
TABLE = Path("results/tables/pseudo_likelihood.csv")

BATCH = 64


def load_model():
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from src.embed import load_esm

    return load_esm()


@torch.no_grad()
def masked_pll(sequences: list[str], model, alphabet, device,
               context: dict[str, str] | None = None) -> np.ndarray:
    """Sum of log P(true residue | rest) over the peptide's own positions.

    When `context` is given, the HLA groove is prepended and only the peptide
    positions are masked and scored, so the number stays comparable between
    the two variants -- the same positions are being judged either way, only
    the surrounding context changes.
    """
    converter = alphabet.get_batch_converter()
    out = np.zeros(len(sequences), dtype=np.float32)

    for start in range(0, len(sequences), BATCH):
        chunk = sequences[start:start + BATCH]
        for j, pep in enumerate(chunk):
            prefix = (context or {}).get(pep, "")
            full = prefix + pep
            offset = len(prefix)

            # One masked copy per peptide position.
            rows = []
            for i in range(len(pep)):
                masked = list(full)
                masked[offset + i] = "<mask>"
                rows.append((f"m{i}", "".join(
                    c if c != "<mask>" else "<mask>" for c in masked)))

            # Build the batch manually: the converter needs plain strings, so
            # masking is applied after tokenisation instead.
            _, _, tokens = converter([(f"s{i}", full) for i in range(len(pep))])
            tokens = tokens.to(device)
            mask_idx = alphabet.mask_idx
            true_tokens = tokens.clone()
            for i in range(len(pep)):
                tokens[i, offset + i + 1] = mask_idx  # +1 for the BOS token

            logits = model(tokens)["logits"]
            logprobs = torch.log_softmax(logits, dim=-1)
            total = 0.0
            for i in range(len(pep)):
                pos = offset + i + 1
                total += float(logprobs[i, pos, true_tokens[i, pos]])
            out[start + j] = total
    return out


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    master = pd.read_parquet(MASTER)
    seqs = pd.read_csv(SEQS).set_index("hla")

    pairs = master[["id", "peptide", "hla", "log_half_life"]].copy()
    if limit:
        pairs = pairs.sample(limit, random_state=42)
    print(f"scoring {len(pairs)} peptide-HLA pairs")

    model, alphabet, device = load_model()
    print(f"ESM-2 on {device}\n")

    uniq_pep = sorted(pairs.peptide.unique())
    print(f"unique peptides: {len(uniq_pep)}")

    plain = masked_pll(uniq_pep, model, alphabet, device)
    plain_map = dict(zip(uniq_pep, plain))
    print("plain pseudo-likelihood done", flush=True)

    # Context version: one pass per (peptide, allele) pair, so restrict to the
    # pairs actually present rather than the full cross product.
    ctx_vals = {}
    todo = pairs[["peptide", "hla"]].drop_duplicates()
    print(f"context-conditioned passes: {len(todo)}", flush=True)
    for n, (_, r) in enumerate(todo.iterrows()):
        groove = seqs.loc[r.hla, "groove_seq"] if r.hla in seqs.index else ""
        if not groove:
            continue
        v = masked_pll([r.peptide], model, alphabet, device,
                       context={r.peptide: groove})
        ctx_vals[(r.peptide, r.hla)] = float(v[0])
        if n and n % 250 == 0:
            print(f"   {n}/{len(todo)}", flush=True)

    pairs["pll_plain"] = pairs.peptide.map(plain_map)
    pairs["pll_in_hla"] = [ctx_vals.get((p, h), np.nan)
                           for p, h in zip(pairs.peptide, pairs.hla)]
    pairs["pll_delta"] = pairs.pll_in_hla - pairs.pll_plain

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs[["id", "pll_plain", "pll_in_hla", "pll_delta"]].to_parquet(OUT, index=False)

    from scipy.stats import spearmanr

    rows = []
    d = pairs.dropna(subset=["pll_in_hla"])
    for col in ["pll_plain", "pll_in_hla", "pll_delta"]:
        rho = float(spearmanr(d[col], d.log_half_life).statistic)
        rows.append({"feature": col, "n": len(d),
                     "spearman_vs_half_life": round(rho, 4)})
        print(f"   {col:12s} vs half-life: {rho:+.4f}")

    same = float(np.corrcoef(d.pll_plain, d.pll_in_hla)[0, 1])
    print(f"\n   correlation between plain and context-conditioned: {same:.4f}")
    print("   (near 1.0 would mean ESM-2 is barely conditioning on the groove)")

    pd.DataFrame(rows).to_csv(TABLE, index=False)
    print(f"\nwrote {OUT} and {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
