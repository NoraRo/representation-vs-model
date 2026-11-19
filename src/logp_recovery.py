"""Can MolLogP be rebuilt from the other RDKit descriptors?

Ridge regression on the published split: fit on training + validation rows, R^2 on the test rows.
The target is MolLogP, the inputs are rdkit2d without MolLogP, MolMR and TPSA (214 columns), and
again without the 28 more columns that RDKit derives from Crippen's atom table (186 columns).
Also the R^2 of MolLogP against each label alone.

Writes results/stats/logp_recovery.csv.

  python src/logp_recovery.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import featurize as F  # noqa: E402
import paths as PT  # noqa: E402

SETS = ("esol", "lipophilicity", "freesolv")
TARGET = "MolLogP"
CRIPPEN_DERIVED = tuple(
    [f"SlogP_VSA{i}" for i in range(1, 13)] + [f"SMR_VSA{i}" for i in range(1, 11)]
    + ["BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "qed", "LabuteASA"])
DROPPED = {"214": F.DROPPED_214, "186": F.DROPPED_214 + CRIPPEN_DERIVED}
ALPHA = 1.0


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray):
    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=ALPHA).fit(sc.transform(X[tr]), y[tr])
    return r2(y[te], m.predict(sc.transform(X[te]))), m


def main() -> int:
    names = [n for n, _ in Descriptors._descList]
    keep = {k: [i for i, c in enumerate(names) if c not in drop] for k, drop in DROPPED.items()}
    for k, drop in DROPPED.items():
        if len(keep[k]) != len(names) - len(drop):
            raise SystemExit(f"{k}: some of the dropped descriptors are not in this RDKit build")
    target = names.index(TARGET)

    rows = []
    for ds in SETS:
        X = np.load(PT.FEATURES / f"{ds}_rdkit2d.npy")
        sp = json.loads((PT.SPLITS / f"{ds}_splits.json").read_text())["det"]
        tr, te = np.array(sp["train"] + sp["valid"]), np.array(sp["test"])
        finite = np.isfinite(X).all(axis=0)
        y = X[:, target]
        label = pd.to_numeric(pd.read_csv(PT.ROWS / f"{ds}_rows.csv").iloc[:, -1], errors="coerce").to_numpy()
        ok = np.isfinite(y) & np.isfinite(label)
        row = {"dataset": ds, "n": len(label),
               "MolLogP_vs_label_r2": float(np.corrcoef(y[ok], label[ok])[0, 1] ** 2)}
        for k in DROPPED:
            cols = [i for i in keep[k] if finite[i]]
            row[f"recover_from_{k}"], model = fit(X[:, cols], y, tr, te)
            if k == "214":
                top = sorted(zip([names[i] for i in cols], np.abs(model.coef_)), key=lambda kv: -kv[1])[:10]
                row["crippen_in_top10"] = sum(c in CRIPPEN_DERIVED for c, _ in top)
                row["top3"] = ", ".join(c for c, _ in top[:3])
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(PT.RES / "stats" / "logp_recovery.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
