"""The 3 representations, computed with RDKit from each molecule's SMILES:

  rdkit2d     all RDKit 2D descriptors (Descriptors._descList)
  ecfp4_2048  binary Morgan fingerprint, radius 2, 2,048 bits
  maccs       the 167 MACCS keys
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rdfp

RDLogger.DisableLog("rdApp.*")

_MORGAN2048 = rdfp.GetMorganGenerator(radius=2, fpSize=2048)


def _rdkit2d(mol: Chem.Mol) -> np.ndarray:
    out = np.empty(len(Descriptors._descList), dtype=np.float64)
    for i, (_, fn) in enumerate(Descriptors._descList):
        try:
            v = fn(mol)
        except Exception:
            v = np.nan
        out[i] = v
    return out


def _ecfp4_2048(mol: Chem.Mol) -> np.ndarray:
    return np.array(_MORGAN2048.GetFingerprintAsNumPy(mol), dtype=np.float64)


def _maccs(mol: Chem.Mol) -> np.ndarray:
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float64)


FEATURIZERS = {"rdkit2d": _rdkit2d, "ecfp4_2048": _ecfp4_2048, "maccs": _maccs}


def featurize(smiles: list[str], kind: str) -> np.ndarray:
    """Return the (n_molecules, n_features) matrix of one representation."""
    fn = FEATURIZERS[kind]
    return np.vstack([fn(Chem.MolFromSmiles(smi)) for smi in smiles])
