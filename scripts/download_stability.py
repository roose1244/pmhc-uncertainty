"""Download the Rasmussen 2016 / NetMHCstabpan stability table.

Source:
https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

URL = "https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt"
DEST = Path("data/raw/Stability.txt")


def _download() -> None:
    curl = shutil.which("curl")
    headers = ["-H", "User-Agent: pMHC-Guardian/0.1 (hackathon research)"]
    if curl:
        subprocess.run(
            [curl, "-fsSL", *headers, "-o", str(DEST), URL],
            check=True,
        )
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "pMHC-Guardian/0.1"})
    with urllib.request.urlopen(request) as response, DEST.open("wb") as handle:
        handle.write(response.read())


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    _download()
    size = DEST.stat().st_size
    print(f"Wrote {DEST} ({size} bytes)")


if __name__ == "__main__":
    main()
