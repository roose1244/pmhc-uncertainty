"""Print one frozen-ensemble prediction. Example: GILGFVFTL A*02:01"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serve import predict


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/predict_one.py PEPTIDE HLA")
    print(json.dumps(predict(sys.argv[1], sys.argv[2]), indent=2))


if __name__ == "__main__":
    main()
