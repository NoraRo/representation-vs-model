"""Every folder the scripts read or write. Data, features and model weights are not in the
repository; README.md says how to get them."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"                    # MoleculeNet CSVs from download_data.py
ROWS = DATA / "moleculenet"           # {dataset}_rows.csv from prepare.py: row index, SMILES, labels
EXT_DATA = DATA / "external"          # external test rows from external_test.py rows; raw/ = source files
EXT_LOCAL = EXT_DATA / "local"        # external SMILES, labels and fingerprints

SPLITS = ROOT / "splits"              # {dataset}_splits.json, manifest.json
FEATURES = ROOT / "features"          # feature matrices from prepare.py
MODELS = ROOT / "models"              # TabFM and TabPFN weights

PRED = ROOT / "predictions"           # one folder per model and representation:
                                      #   raw_results.csv + {dataset}__{task}__{split}.npy
RES = ROOT / "results"                # stats/, external/, identity/


def arm_dirs() -> list[str]:
    """Every prediction folder; fails if there is none."""
    found = sorted(p.name for p in PRED.iterdir() if (p / "raw_results.csv").exists()) if PRED.exists() else []
    if not found:
        raise SystemExit(f"no prediction folders in {PRED}")
    return found


_MODEL_DIR = {"RF": "rf", "CatBoost": "catboost", "TabPFN": "tabpfn", "TabFM": "tabfm",
              "TabFMplus": "tabfmplus", "Chemprop": "chemprop", "ChempropRefit": "chemprop_refit",
              "ChempropDesc": "chemprop", "DGT": "dgt"}
_FEAT_DIR = {"rdkit2d": "rdkit2d", "rdkit2d_214": "rdkit2d_214", "ecfp4_2048": "ecfp4",
             "morgan512": "morgan512", "maccs": "maccs", "graph": "graph", "graph+rdkit2d": "graph_rdkit2d",
             "graph+rdkit2d_214": "graph_rdkit2d_214"}


def arm_dir(model: str, features: str) -> pathlib.Path:
    """Prediction folder of one model on one representation, e.g. predictions/tabfm_maccs."""
    return PRED / f"{_MODEL_DIR[model]}_{_FEAT_DIR[features]}"
