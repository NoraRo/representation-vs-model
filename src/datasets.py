"""Loading of the 7 MoleculeNet datasets (data/raw/) and the seeded Bemis-Murcko scaffold split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

import paths as PT  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = PT.RAW


@dataclass
class Dataset:
    name: str
    domain: str  # physchem | physiology | biophysics
    kind: str  # regression | classification
    smiles: list[str]
    y: np.ndarray  # (n_mols, n_tasks); NaN = missing label
    task_names: list[str]
    unit: str = ""

    @property
    def n_tasks(self) -> int:
        return self.y.shape[1]

    def __repr__(self) -> str:
        return (
            f"Dataset({self.name}, {self.kind}, n={len(self.smiles)}, "
            f"tasks={self.n_tasks})"
        )


# ---------------------------------------------------------------- loaders


def _read(fname: str) -> pd.DataFrame:
    return pd.read_csv(DATA / fname)


def load_esol() -> Dataset:
    df = _read("esol.csv")
    return Dataset(
        name="ESOL",
        domain="physchem",
        kind="regression",
        smiles=df["smiles"].tolist(),
        y=df[["measured log solubility in mols per litre"]].to_numpy(float),
        task_names=["logS"],
        unit="log mol/L",
    )


def load_freesolv() -> Dataset:
    df = _read("freesolv.csv")
    return Dataset(
        name="FreeSolv",
        domain="physchem",
        kind="regression",
        smiles=df["smiles"].tolist(),
        y=df[["expt"]].to_numpy(float),
        task_names=["dGhyd"],
        unit="kcal/mol",
    )


def load_lipophilicity() -> Dataset:
    df = _read("lipophilicity.csv")
    return Dataset(
        name="Lipophilicity",
        domain="physchem",
        kind="regression",
        smiles=df["smiles"].tolist(),
        y=df[["exp"]].to_numpy(float),
        task_names=["logD"],
        unit="log D",
    )


def load_bbbp() -> Dataset:
    df = _read("bbbp.csv")
    return Dataset(
        name="BBBP",
        domain="physiology",
        kind="classification",
        smiles=df["smiles"].tolist(),
        y=df[["p_np"]].to_numpy(float),
        task_names=["p_np"],
    )


def load_bace() -> Dataset:
    df = _read("bace.csv")
    return Dataset(
        name="BACE",
        domain="biophysics",
        kind="classification",
        smiles=df["mol"].tolist(),
        y=df[["Class"]].to_numpy(float),
        task_names=["Class"],
    )


def load_clintox() -> Dataset:
    df = _read("clintox.csv")
    tasks = ["FDA_APPROVED", "CT_TOX"]
    return Dataset(
        name="ClinTox",
        domain="physiology",
        kind="classification",
        smiles=df["smiles"].tolist(),
        y=df[tasks].to_numpy(float),
        task_names=tasks,
    )


def load_sider() -> Dataset:
    df = _read("sider.csv")
    tasks = [c for c in df.columns if c != "smiles"]
    return Dataset(
        name="SIDER",
        domain="physiology",
        kind="classification",
        smiles=df["smiles"].tolist(),
        y=df[tasks].to_numpy(float),
        task_names=tasks,
    )


LOADERS = {
    "esol": load_esol,
    "freesolv": load_freesolv,
    "lipophilicity": load_lipophilicity,
    "bbbp": load_bbbp,
    "bace": load_bace,
    "clintox": load_clintox,
    "sider": load_sider,
}


def sanitize(ds: Dataset) -> Dataset:
    """Drop rows RDKit cannot parse. Reports how many were lost."""
    keep, n_bad = [], 0
    for i, smi in enumerate(ds.smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            n_bad += 1
            continue
        keep.append(i)
    if n_bad:
        print(f"    [{ds.name}] dropped {n_bad} unparseable SMILES")
    return Dataset(
        name=ds.name,
        domain=ds.domain,
        kind=ds.kind,
        smiles=[ds.smiles[i] for i in keep],
        y=ds.y[keep],
        task_names=ds.task_names,
        unit=ds.unit,
    )


def load(name: str) -> Dataset:
    return sanitize(LOADERS[name]())


# ---------------------------------------------------------- scaffold split


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES; empty string for acyclic molecules."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(murcko_scaffold(smi), []).append(i)
    return groups


def scaffold_split(
    smiles: list[str],
    seed: int,
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded scaffold split, 80:10:10, in the style of Chemprop's scaffold_balanced: scaffold
    groups larger than half the validation size go to training first, the other groups follow
    in a seeded random order. Every scaffold stays within one part.
    (The published split is prepare.det_split_dgt.)
    """
    n = len(smiles)
    n_train, n_valid = int(frac[0] * n), int(frac[1] * n)
    rng = np.random.default_rng(seed)
    big, small = [], []
    for g in scaffold_groups(smiles).values():
        (big if len(g) > n_valid / 2 else small).append(g)
    rng.shuffle(small)
    groups = big + small

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for g in groups:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(valid) + len(g) <= n_valid:
            valid += g
        else:
            test += g
    return np.array(train), np.array(valid), np.array(test)
