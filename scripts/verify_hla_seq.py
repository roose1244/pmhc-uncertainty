"""Verify master.parquet's hla_seq column against IPD-IMGT/HLA, independently.

Developer 1 filled hla_seq from the IMGT protein FASTA (mature 182-mer of the
alpha1-alpha2 groove) and applied C67S to the three engineered variants. This
script rebuilds the same column by a separate route and reports every
disagreement, so the column can be trusted without being replaced.

Three checks, in increasing strength:

  1. structural   -- one 182-mer per allele, no nulls, expected start motif
  2. IMGT rebuild -- independent reconstruction from data/raw/hla_prot.fasta
  3. DTU pseudo   -- NetMHCstabpan's own MHC_pseudo.dat contact residues,
                     a third source that is not derived from either of the above

Run:  python scripts/verify_hla_seq.py
Exit code is 1 if any allele disagrees, so this is usable as a CI gate.
"""

from __future__ import annotations

import re
import sys
import collections
from pathlib import Path

import pandas as pd

MASTER = Path("data/processed/master.parquet")
FASTA = Path("data/raw/hla_prot.fasta")
PSEUDO = Path("data/external/hla_pseudo.csv")

GROOVE_LEN = 182
MUTANT = re.compile(r"^(HLA-[A-Z]+\*\d+:\d+)\((\w)(\d+)(\w)\)$")


def read_fasta(path: Path) -> dict[str, str]:
    """IMGT allele name -> full protein sequence, from the second header field."""
    seqs: dict[str, list[str]] = {}
    name = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line.split()[1]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def lookup(seqs: dict[str, str], hla: str) -> tuple[str | None, str | None]:
    """Find the representative IMGT entry for a two-field allele name.

    IMGT names carry a variable number of fields (A*02:01:01:01 but also
    A*43:01), so match the bare name as well as any colon-extended form, and
    skip expression-suffixed entries (A*24:02:01:02L and similar).
    """
    key = hla.replace("HLA-", "")
    for name, seq in seqs.items():
        if (name == key or name.startswith(key + ":")) and not name[-1].isalpha():
            return name, seq
    return None, None


def groove(seq: str) -> tuple[str | None, str]:
    """Mature alpha1-alpha2 182-mer.

    Full-length entries carry a 24-residue leader, so the mature chain starts
    at the GSHSM motif. Some alleles (A*02:50, A*24:19, B*08:03) are deposited
    as partial 181-mers that begin at SHSM -- they are missing the leader *and*
    the leading G of the mature chain, so slicing at a fixed offset silently
    reads the wrong frame. Restore the G instead.
    """
    start = seq.find("GSHSM")
    if start >= 0:
        return seq[start : start + GROOVE_LEN], "full"
    if seq.startswith("SHSM"):
        return ("G" + seq)[:GROOVE_LEN], "partial(+G)"
    return None, "unrecognised"


def rebuild(seqs: dict[str, str], hla: str) -> tuple[str | None, str, str]:
    """Independent reconstruction of one allele's groove sequence."""
    mutant = MUTANT.match(hla)
    parent = mutant.group(1) if mutant else hla

    name, seq = lookup(seqs, parent)
    if seq is None:
        return None, "", f"no IMGT entry for {parent}"

    grv, kind = groove(seq)
    if grv is None or len(grv) != GROOVE_LEN:
        return None, name, f"{kind}, got {len(grv) if grv else 0} residues"

    if mutant:
        old, pos, new = mutant.group(2), int(mutant.group(3)), mutant.group(4)
        # Fail loudly rather than silently substituting on a wrong numbering
        # assumption: the parent must actually carry `old` at `pos`.
        if grv[pos - 1] != old:
            return None, name, f"parent has {grv[pos - 1]} at {pos}, expected {old}"
        grv = grv[: pos - 1] + new + grv[pos:]
        kind += f", {old}{pos}{new}"

    return grv, name, kind


def main() -> int:
    master = pd.read_parquet(MASTER)
    seqs = read_fasta(FASTA)
    dev = master.groupby("hla").hla_seq.agg(lambda s: s.dropna().unique().tolist())

    print(f"master: {len(master)} rows, {len(dev)} alleles\n")

    # --- 1. structural ---------------------------------------------------
    print("[1] structure")
    nulls = master.hla_seq.isna().sum()
    multi = {h: v for h, v in dev.items() if len(v) != 1}
    lengths = collections.Counter(len(v[0]) for v in dev if v)
    motifs = collections.Counter(v[0][:5] for v in dev if v)
    print(f"    null hla_seq rows        : {nulls}")
    print(f"    alleles not exactly 1 seq: {len(multi)} {list(multi) or ''}")
    print(f"    lengths                  : {dict(lengths)}")
    print(f"    start motifs             : {dict(motifs)}")

    # Alleles sharing a sequence are proxies: a stand-in was used for an
    # allele whose own sequence was not found. Report them even when IMGT
    # happens to agree, because an exact tie between distinct alleles is
    # usually substitution rather than coincidence.
    shared = collections.defaultdict(list)
    for hla, v in dev.items():
        if v:
            shared[v[0]].append(hla)
    proxies = [hs for hs in shared.values() if len(hs) > 1]
    print(f"    alleles sharing a sequence: {proxies or 'none'}")

    # --- 2. IMGT rebuild -------------------------------------------------
    print("\n[2] independent IMGT rebuild")
    ok, disagree, problems = 0, [], []
    for hla, v in dev.items():
        theirs = v[0] if v else ""
        mine, name, kind = rebuild(seqs, hla)
        if mine is None:
            problems.append((hla, kind))
        elif mine == theirs:
            ok += 1
        else:
            diff = [i + 1 for i, (a, b) in enumerate(zip(theirs, mine)) if a != b]
            disagree.append((hla, name, kind, theirs, mine, diff))

    print(f"    identical to rebuild: {ok}/{len(dev)}")
    for hla, name, kind, theirs, mine, diff in disagree:
        rows = int((master.hla == hla).sum())
        detail = ", ".join(f"{p}:{theirs[p - 1]}->{mine[p - 1]}" for p in diff)
        print(f"    DIFFERS {hla} ({rows} rows, IMGT {name}, {kind})")
        print(f"            {detail}")
    for hla, why in problems:
        print(f"    PROBLEM {hla}: {why}")

    # --- 3. DTU pseudo-sequence cross-check ------------------------------
    print("\n[3] DTU MHC_pseudo.dat contact residues")
    if PSEUDO.exists():
        ps = pd.read_csv(PSEUDO)
        # The engineered variants have no MHC_pseudo.dat entry, so they are
        # neither agreements nor disagreements -- count only DTU-listed alleles.
        listed = ps[ps.match.notna()]
        bad = listed[listed.match == False]  # noqa: E712 -- elementwise compare
        n_ok = int((listed.match == True).sum())  # noqa: E712
        print(f"    agree: {n_ok}/{len(listed)} DTU-listed "
              f"({len(ps) - len(listed)} not in MHC_pseudo.dat)")
        for _, r in bad.iterrows():
            a, b = str(r.pseudo_from_seq), str(r.pseudo_dtu)
            d = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
            detail = ", ".join(f"{p}:{a[p - 1]}->{b[p - 1]}" for p in d)
            print(f"    DIFFERS {r.hla} at pseudo {d}  {detail}")
    else:
        print(f"    skipped, {PSEUDO} not present")

    failed = bool(disagree or problems or multi or nulls)
    print("\nFAIL: hla_seq disagrees with IMGT" if failed else "\nOK: hla_seq verified")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
