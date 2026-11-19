"""The 5 representations, computed with RDKit from each molecule's SMILES:

  rdkit2d       all RDKit 2D descriptors (Descriptors._descList)
  rdkit2d_214   rdkit2d without MolLogP, MolMR and TPSA
  ecfp4_2048    binary Morgan fingerprint, radius 2, 2,048 bits
  morgan512     binary Morgan fingerprint, radius 2, 512 bits
  maccs         the 167 MACCS keys
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rdfp

RDLogger.DisableLog("rdApp.*")

DROPPED_214 = ("MolLogP", "MolMR", "TPSA")

_MORGAN2048 = rdfp.GetMorganGenerator(radius=2, fpSize=2048)
_MORGAN512 = rdfp.GetMorganGenerator(radius=2, fpSize=512)

_DESC_214 = [(n, fn) for n, fn in Descriptors._descList if n not in DROPPED_214]


def _descriptors(desc_list, mol: Chem.Mol) -> np.ndarray:
    out = np.empty(len(desc_list), dtype=np.float64)
    for i, (_, fn) in enumerate(desc_list):
        try:
            v = fn(mol)
        except Exception:
            v = np.nan
        out[i] = v
    return out


def _rdkit2d(mol: Chem.Mol) -> np.ndarray:
    return _descriptors(Descriptors._descList, mol)


def _rdkit2d_214(mol: Chem.Mol) -> np.ndarray:
    return _descriptors(_DESC_214, mol)


def _ecfp4_2048(mol: Chem.Mol) -> np.ndarray:
    return np.array(_MORGAN2048.GetFingerprintAsNumPy(mol), dtype=np.float64)


def _morgan512(mol: Chem.Mol) -> np.ndarray:
    return np.array(_MORGAN512.GetFingerprintAsNumPy(mol), dtype=np.float64)


def _maccs(mol: Chem.Mol) -> np.ndarray:
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float64)


FEATURIZERS = {"rdkit2d": _rdkit2d, "rdkit2d_214": _rdkit2d_214, "ecfp4_2048": _ecfp4_2048,
               "morgan512": _morgan512, "maccs": _maccs}


def featurize(smiles: list[str], kind: str) -> np.ndarray:
    """Return the (n_molecules, n_features) matrix of one representation."""
    fn = FEATURIZERS[kind]
    return np.vstack([fn(Chem.MolFromSmiles(smi)) for smi in smiles])
