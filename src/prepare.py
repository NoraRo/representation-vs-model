"""Build the rows, the 5 representations and the 11 scaffold splits once, so every model sees the
same molecules. Run after download_data.py. Writes (folders in paths.py):
  data/moleculenet/{dataset}_rows.csv  row index, SMILES, labels (molecules RDKit can parse)
  features/{dataset}_{feat}.npy        feature matrix before cleaning
  splits/{dataset}_splits.json         train/valid/test row indices: "det" (the published split)
                                       and seeded splits "0"-"9"
  splits/manifest.json                 sha256 of every file, RDKit version, split sizes
The splits in the repository were built with RDKit 2026.03.5.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import rdkit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import datasets as D  # noqa: E402
import featurize as F  # noqa: E402
import paths as PT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATS = ["rdkit2d", "rdkit2d_214", "ecfp4_2048", "morgan512", "maccs"]
SPLITS: list[int | None] = [None] + list(range(10))


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def det_split_dgt(smiles: list[str]) -> tuple[list[int], list[int], list[int]]:
    """The published split, built as DGT's code builds it (github.com/zhangsy-ryan/DGT,
    graphgps/loader/split_generator.py): scaffold groups largest first; a group goes to
    training while training stays <= 80% of n, then to validation while training + validation
    stays <= 90%, otherwise to test."""
    groups = sorted(D.scaffold_groups(smiles).values(), key=lambda g: (len(g), g[0]), reverse=True)
    n, tr, va, te = len(smiles), [], [], []
    for g in groups:
        if len(tr) + len(g) > 0.8 * n:
            if len(tr) + len(va) + len(g) > 0.9 * n:
                te += g
            else:
                va += g
        else:
            tr += g
    return tr, va, te


def main() -> int:
    for d in (PT.ROWS, PT.SPLITS, PT.FEATURES):
        d.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"rdkit": rdkit.__version__, "files": {}, "splits": {}}

    for name in D.LOADERS:
        ds = D.load(name)
        rows = pd.DataFrame({"row": range(len(ds.smiles)), "smiles": ds.smiles})
        for j, t in enumerate(ds.task_names):
            rows[t] = ds.y[:, j]
        paths = [PT.ROWS / f"{name}_rows.csv"]
        rows.to_csv(paths[0], index=False)

        for feat in FEATS:
            p = PT.FEATURES / f"{name}_{feat}.npy"
            np.save(p, F.featurize(ds.smiles, feat))
            paths.append(p)

        splits = {}
        for seed in SPLITS:
            if seed is None:
                tr, va, te = (np.array(x) for x in det_split_dgt(ds.smiles))
            else:
                tr, va, te = D.scaffold_split(ds.smiles, seed=seed)
            tag = "det" if seed is None else str(seed)
            if len(set(tr) | set(va) | set(te)) != len(ds.smiles):
                raise RuntimeError(f"{name} split {tag} does not cover every row")
            splits[tag] = {"train": tr.tolist(), "valid": va.tolist(), "test": te.tolist()}
            manifest["splits"][f"{name}/{tag}"] = {
                "n_train_valid": len(tr) + len(va),
                "n_test": len(te),
            }
        p = PT.SPLITS / f"{name}_splits.json"
        p.write_text(json.dumps(splits))
        paths.append(p)

        for p in paths:
            manifest["files"][p.name] = sha256(p)
        print(f"{ds}  features={FEATS}  splits={len(splits)}", flush=True)

    (PT.SPLITS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"RDKit {rdkit.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
