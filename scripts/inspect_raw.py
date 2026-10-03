"""Print the facts we locked in the README. Run after download_stability.py."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hla import normalise_hla
from src.target import to_log

RAW = Path("data/raw/Stability.txt")


def main() -> None:
    rows: list[tuple[str, str, float]] = []
    with RAW.open() as handle:
        header = handle.readline().split()
        if header[:3] != ["HLA", "Pep", "Thalf"]:
            raise SystemExit(f"Unexpected header: {header}")
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            hla = normalise_hla(parts[0])
            peptide = parts[1].upper()
            half_life = float(parts[2])
            rows.append((hla, peptide, half_life))

    n = len(rows)
    alleles = {hla for hla, _, _ in rows}
    peptides = {pep for _, pep, _ in rows}
    pairs = {(hla, pep) for hla, pep, _ in rows}
    zeros = sum(1 for _, _, t in rows if t == 0.0)
    lengths = Counter(len(pep) for _, pep, _ in rows)
    engineered = sorted(a for a in alleles if "(" in a)
    values = sorted(t for _, _, t in rows)

    print(f"rows                {n}")
    print(f"unique HLA          {len(alleles)}")
    print(f"unique peptides     {len(peptides)}")
    print(f"unique pairs        {len(pairs)}")
    print(f"peptide lengths     {dict(sorted(lengths.items()))}")
    print(f"exact zeros         {zeros} ({zeros / n:.1%})")
    print(f"min / median / max  {values[0]} / {values[n // 2]} / {values[-1]}")
    print(f"engineered alleles  {engineered}")
    print(f"log10(0 + 0.1)      {to_log(0.0):.3f}")
    print(f"log10(1.1 + 0.1)    {to_log(1.1):.3f}")
    print(f"B*15:02 rows        {sum(1 for h, _, _ in rows if h == 'HLA-B*15:02')}")
    print(f"B*27:02 rows        {sum(1 for h, _, _ in rows if h == 'HLA-B*27:02')}")
    print(f"A*02:01 rows        {sum(1 for h, _, _ in rows if h == 'HLA-A*02:01')}")


if __name__ == "__main__":
    main()
