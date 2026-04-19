"""Download raw corpora used for pre-training and the four downstream tasks.

This script is intentionally lightweight: it only *stages* datasets from
publicly-hosted mirrors and does not perform any heavy preprocessing.
Preprocessing (triplet construction) is done by
:func:`contrabin.data.triplet_builder.TripletBuilder`.

Supported corpora:

* ``anghabench-sample`` (small tar of 1k C functions for quick experiments)
* ``poj104``            (POJ-104 algorithmic functionality benchmark)

Use the ``--dataset`` flag to pick one; see ``--help`` for the full list.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

DATASETS = {
    "anghabench-sample": (
        "https://zenodo.org/record/5137614/files/sample.tar.gz",
        "data/raw/anghabench-sample",
    ),
    "poj104": (
        "https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU",  # placeholder
        "data/raw/poj104",
    ),
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract_tar(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as tar:
        tar.extractall(out_dir)  # noqa: S202 - trust our own mirrors
    print(f"[extract] {path} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out-dir", default=".", type=Path)
    args = parser.parse_args()

    url, rel = DATASETS[args.dataset]
    out_dir = (args.out_dir / rel).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = out_dir / Path(url).name
    download(url, archive)
    if archive.suffix in {".gz", ".tgz"} or archive.name.endswith(".tar.gz"):
        extract_tar(archive, out_dir)
    print(f"[done] {args.dataset} staged under {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
