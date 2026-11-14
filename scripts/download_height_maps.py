# SPDX-License-Identifier: BSD-3-Clause
"""Download the public NASA height maps that the real-terrain surface needs.

Usage::

    python scripts/download_height_maps.py                 # the 2 m Apollo 17 product (952 MB)
    python scripts/download_height_maps.py --with-5m       # also the 5 m product (463 MB)
    python scripts/download_height_maps.py --target /data/terrain/apollo17

The files go to ``assets/terrain/apollo17`` (git-ignored) unless ``--target`` is given; point
``height_map.directory`` in ``configs/surface/apollo17_taurus_littrow.yaml`` there in that case.
A file that is already complete (same size as on the server) is not downloaded again.
Source: LROC NAC DTM products, Arizona State University, PDS volume LROLRC_2001.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM"
PRODUCTS = {
    "2m": ("APOLLO17_2M", ("NAC_DTM_APOLLO17_2M.LBL", "NAC_DTM_APOLLO17_2M.TIF")),
    "5m": ("APOLLO17", ("NAC_DTM_APOLLO17.LBL", "NAC_DTM_APOLLO17.TIF")),
}
CHUNK_BYTES = 8 * 1024 * 1024


def remote_size(url: str) -> int | None:
    """Return the size of the file on the server in bytes, or None if the server does not say."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` through a temporary file, printing progress."""
    size = remote_size(url)
    if destination.is_file() and size is not None and destination.stat().st_size == size:
        print(f"already complete: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as file:
        received = 0
        while chunk := response.read(CHUNK_BYTES):
            file.write(chunk)
            received += len(chunk)
            if size:
                print(f"\r{destination.name}: {received / 1e6:8.1f} of {size / 1e6:.1f} MB", end="", flush=True)
    print()
    if size is not None and partial.stat().st_size != size:
        raise RuntimeError(f"{destination.name}: got {partial.stat().st_size} bytes, expected {size}")
    partial.replace(destination)
    print(f"saved {destination}")


def main() -> int:
    """Parse the command line and download the chosen products."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=Path, default=REPOSITORY_ROOT / "assets" / "terrain" / "apollo17")
    parser.add_argument("--with-5m", action="store_true", help="also download the 5 m product")
    options = parser.parse_args()
    options.target.mkdir(parents=True, exist_ok=True)
    chosen = ["2m"] + (["5m"] if options.with_5m else [])
    for key in chosen:
        folder, files = PRODUCTS[key]
        for name in files:
            download(f"{SERVER}/{folder}/{name}", options.target / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
