"""Download the 7 MoleculeNet CSVs from DeepChem's S3 bucket into data/raw/ (the files DGT reads).

The files on S3 have no version, so each one is checked against the sha256 of the file used for
the paper; the script stops if any file differs, because the split files index its rows."""

from __future__ import annotations

import gzip
import hashlib
import io
import pathlib
import sys
import urllib.request

import paths as PT

BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
DATA = PT.RAW

# name -> (remote file, local file)
FILES = {
    "esol": ("delaney-processed.csv", "esol.csv"),
    "freesolv": ("SAMPL.csv", "freesolv.csv"),
    "lipophilicity": ("Lipophilicity.csv", "lipophilicity.csv"),
    "bbbp": ("BBBP.csv", "bbbp.csv"),
    "bace": ("bace.csv", "bace.csv"),
    "clintox": ("clintox.csv.gz", "clintox.csv"),
    "sider": ("sider.csv.gz", "sider.csv"),
}
SHA256 = {
    "esol.csv": "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
    "freesolv.csv": "ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72",
    "lipophilicity.csv": "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
    "bbbp.csv": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
    "bace.csv": "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737",
    "clintox.csv": "9999816e760dd838358b5d88e81cea2fc062be4458ffa412ceecdba4f88a67b6",
    "sider.csv": "71efc6ac4ca82d6545bc512509863281be0f17234afd7adaf71a71899988302e",
}


def fetch(remote: str, local: pathlib.Path) -> None:
    url = f"{BASE}/{remote}"
    print(f"  GET {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
    if remote.endswith(".gz"):
        raw = gzip.decompress(raw)
    local.write_bytes(raw)
    n_lines = len(io.BytesIO(raw).readlines())
    print(f"  -> {local.name}  ({len(raw):,} bytes, {n_lines - 1:,} rows)", flush=True)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (remote, local_name) in FILES.items():
        local = DATA / local_name
        if local.exists():
            print(f"[skip] {name}: {local.name} already present")
        else:
            print(f"[get ] {name}")
            fetch(remote, local)
        got = hashlib.sha256(local.read_bytes()).hexdigest()
        if got != SHA256[local_name]:
            raise SystemExit(f"{local}: sha256 {got} differs from the file used for the paper "
                             f"({SHA256[local_name]}); the split files would point to other rows")
    print("all 7 files match the files used for the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
