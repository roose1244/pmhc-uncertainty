"""Build data/external/hla_sequences.csv: hla, hla_seq, groove_seq, supertype.

The file contract asks for one row per allele in master.parquet with the full
heavy chain, the alpha1-alpha2 groove, and an HLA supertype label.

Supertypes come from Sidney et al., BMC Immunology 2008;9:1 (the standard
nine-supertype classification). That table covers 46 of the 75 alleles here.
The remaining 29 are rarer four-digit alleles the paper does not list, so they
are assigned to the supertype of their nearest classified neighbour measured
over the 34 peptide-contacting residues -- the same positions NetMHCstabpan
uses as its pseudo-sequence.

Nearest-neighbour on the pocket residues is the right fallback rather than a
guess from the allele's family name, because supertypes are *defined* by B and
F pocket specificity, not by numbering: A*68:02 sits in A02 while A*68:01 sits
in A03, so "same family" reasoning would get it wrong.

The method is checked by leave-one-out over the 46 alleles that do have a
published label, and the accuracy is printed. Every row records where its
label came from, so an inferred label is never mistaken for a published one.

Run:  python scripts/build_hla_sequences.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.verify_hla_seq import lookup, read_fasta, rebuild  # noqa: E402

MASTER = Path("data/processed/master.parquet")
FASTA = Path("data/raw/hla_prot.fasta")
PSEUDO = Path("data/external/hla_pseudo.csv")
OUT = Path("data/external/hla_sequences.csv")

# Sidney J, Peters B, Frahm N, Brander C, Sette A. HLA class I supertypes:
# a revised and updated classification. BMC Immunol. 2008;9:1.
SUPERTYPES: dict[str, str] = {
    "A01": "A*01:01 A*26:01 A*26:02 A*26:03 A*30:02 A*30:03 A*30:04 A*32:01",
    "A02": (
        "A*02:01 A*02:02 A*02:03 A*02:04 A*02:05 A*02:06 A*02:07 A*02:14 "
        "A*02:17 A*68:02 A*69:01"
    ),
    "A03": "A*03:01 A*11:01 A*31:01 A*33:01 A*33:03 A*66:01 A*68:01 A*74:01",
    "A24": "A*23:01 A*24:02",
    "B07": (
        "B*07:02 B*07:03 B*07:05 B*35:01 B*35:03 B*42:01 B*51:01 B*51:02 "
        "B*51:03 B*53:01 B*54:01 B*55:01 B*55:02 B*56:01 B*67:01 B*78:01"
    ),
    "B27": (
        "B*14:02 B*15:03 B*15:09 B*15:10 B*15:18 B*27:02 B*27:03 B*27:04 "
        "B*27:05 B*27:06 B*27:07 B*27:08 B*27:09 B*38:01 B*39:01 B*39:02 "
        "B*39:09 B*48:01 B*73:01"
    ),
    "B44": "B*18:01 B*40:01 B*40:02 B*40:06 B*44:02 B*44:03 B*45:01",
    "B58": "B*15:16 B*15:17 B*57:01 B*57:02 B*58:01 B*58:02",
    "B62": "B*15:01 B*15:02 B*15:12 B*15:13 B*46:01 B*52:01",
}

ENGINEERED = re.compile(r"\(.*\)$")


def published_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    for supertype, names in SUPERTYPES.items():
        for name in names.split():
            out[f"HLA-{name}"] = supertype
    return out


def parent_of(hla: str) -> str:
    """Engineered variants inherit their parent allele's pocket identity."""
    return ENGINEERED.sub("", hla)


def pocket_distance(a: str, b: str) -> float:
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def nearest_label(
    hla: str, pseudo: dict[str, str], labelled: dict[str, str], exclude: str = ""
) -> tuple[str, str, float]:
    """Supertype of the closest labelled allele over the contact residues."""
    mine = pseudo.get(hla)
    if not mine:
        return "", "", float("nan")
    best = (float("inf"), "", "")
    for other, label in labelled.items():
        if other == exclude or other == hla:
            continue
        seq = pseudo.get(other)
        if not seq or len(seq) != len(mine):
            continue
        d = pocket_distance(mine, seq)
        if d < best[0]:
            best = (d, other, label)
    return best[2], best[1], best[0]


# The leave-one-out check fails on 8 of 46 published alleles, and 6 of those 8
# are A01 mistaken for A03 or the reverse: those two supertypes have genuinely
# overlapping pocket preferences, so pocket distance cannot separate them. Any
# inferred A01/A03 label is therefore low confidence regardless of how close
# its neighbour is -- A*26:03 and A*66:01 sit 0.059 apart and belong to
# different supertypes.
CONFUSABLE = {"A01", "A03"}


def confidence(source: str, label: str, dist: float) -> str:
    if source == "sidney2008":
        return "published"
    if label in CONFUSABLE:
        return "low"
    if dist != dist or dist > 0.15:
        return "low"
    return "high" if dist <= 0.06 else "medium"


def main() -> int:
    master = pd.read_parquet(MASTER)
    alleles = sorted(master.hla.unique())
    seqs = read_fasta(FASTA)
    pseudo = dict(
        zip(*pd.read_csv(PSEUDO)[["hla", "pseudo_best"]].to_numpy().T.tolist())
    )
    published = published_lookup()

    # --- validate the fallback before relying on it ----------------------
    known = {h: published[parent_of(h)] for h in alleles if parent_of(h) in published}
    correct = 0
    for hla, truth in known.items():
        guess, _, _ = nearest_label(hla, pseudo, known, exclude=hla)
        correct += guess == truth
    print(
        f"leave-one-out check of the pocket-distance fallback: "
        f"{correct}/{len(known)} ({100 * correct / len(known):.0f}%) "
        f"of published alleles recover their own supertype"
    )

    rows = []
    for hla in alleles:
        parent = parent_of(hla)
        entry, full = lookup(seqs, parent)
        groove, _, _ = rebuild(seqs, hla)

        if parent in published:
            label, source, nn, dist = published[parent], "sidney2008", "", float("nan")
        else:
            label, nn, dist = nearest_label(hla, pseudo, known)
            source = "nearest_pocket"

        rows.append(
            {
                "hla": hla,
                "hla_seq": full or "",
                "groove_seq": groove or "",
                "supertype": label,
                "supertype_confidence": confidence(source, label, dist),
                "supertype_source": source,
                "supertype_nn": nn,
                "supertype_nn_dist": round(dist, 4) if dist == dist else "",
                "imgt_entry": entry or "",
                "n_rows": int((master.hla == hla).sum()),
            }
        )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}  alleles={len(out)}")
    print(f"  groove_seq present : {(out.groove_seq.str.len() == 182).sum()}/{len(out)}")
    print(f"  hla_seq present    : {(out.hla_seq.str.len() > 0).sum()}/{len(out)}")
    print(f"  supertype assigned : {(out.supertype != '').sum()}/{len(out)}")
    print("\n  by source:")
    print(out.supertype_source.value_counts().to_string())
    print("\n  by supertype:")
    print(out.supertype.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
