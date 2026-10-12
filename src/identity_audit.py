"""The same molecule on both sides of a split.

Two keys per molecule (they are not nested, so both are used):
  inchikey   standard InChIKey of the standardised parent
             (largest fragment -> Normalizer -> Uncharger)
  tautomer   canonical tautomer of that parent, stereo kept
             (RDKit's TautomerEnumerator erases stereo by default; both flags set)

Per dataset and split: test molecules whose key also appears in training + validation, and how
many of them disagree on a label; also duplicates inside each dataset. The splits are not
changed; the shared test rows are written out so stats.py can rescore without them (no refit).

Writes results/identity/{summary.json, leaked_test_rows.csv, duplicates.csv}.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

import paths as PT  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = PT.RES / "identity"
DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace", "clintox", "sider"]
REGRESSION = {"esol", "freesolv", "lipophilicity"}

_FRAG = rdMolStandardize.LargestFragmentChooser()
_NORM = rdMolStandardize.Normalizer()
_UNCH = rdMolStandardize.Uncharger()
_TAUT = rdMolStandardize.TautomerEnumerator()
_TAUT.SetRemoveSp3Stereo(False)
_TAUT.SetRemoveBondStereo(False)


def keys(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable after sanitize: {smiles}")
    parent = _UNCH.uncharge(_NORM.normalize(_FRAG.choose(mol)))
    ik = Chem.MolToInchiKey(parent)
    if not ik:  # '' not None: '*' placeholders, some metal complexes. Counted and listed.
        ik = "NOINCHI:" + Chem.MolToSmiles(parent)
    taut = Chem.MolToSmiles(_TAUT.Canonicalize(parent))
    return ik, taut


def label_conflict(y: np.ndarray, a: int, others: list[int], regression: bool) -> float:
    """Classification: 1.0 if any task label differs from any match, else 0.0.
    Regression: largest |difference| to any match."""
    if regression:
        return float(np.max(np.abs(y[others, 0] - y[a, 0])))
    ya, yo = y[a], y[others]
    both = ~np.isnan(ya) & ~np.isnan(yo)
    return float(np.any((ya != yo) & both))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"rdkit": rdkit.__version__, "datasets": {}}
    leaked_rows, dup_rows = [], []

    for name in DATASETS:
        rows = pd.read_csv(PT.ROWS / f"{name}_rows.csv")
        tasks = [c for c in rows.columns if c not in ("row", "smiles")]
        y = rows[tasks].to_numpy(float)
        k = [keys(s) for s in rows.smiles]
        no_ik = [i for i, (a, _) in enumerate(k) if a.startswith("NOINCHI:")]
        ds = {"n": len(rows), "no_inchikey": len(no_ik), "no_inchikey_rows": no_ik}
        key_cols = {"inchikey": [a for a, _ in k], "tautomer": [b for _, b in k]}
        reg = name in REGRESSION

        for kname, kv in key_cols.items():
            groups: dict[str, list[int]] = {}
            for i, v in enumerate(kv):
                groups.setdefault(v, []).append(i)
            dups = [g for g in groups.values() if len(g) > 1]
            n_conf = sum(label_conflict(y, g[0], g[1:], reg) > 0 for g in dups)
            ds[f"dup_groups_{kname}"] = len(dups)
            ds[f"dup_rows_{kname}"] = sum(len(g) for g in dups)
            ds[f"dup_groups_label_differs_{kname}"] = int(n_conf)
            for g in dups:
                dup_rows.append(dict(dataset=name, key=kname, rows=" ".join(map(str, g))))

        splits = json.loads((PT.SPLITS / f"{name}_splits.json").read_text())
        per_split = {}
        for sp, s in splits.items():
            ctx, te = s["train"] + s["valid"], s["test"]
            rec = {"n_test": len(te)}
            for kname, kv in key_cols.items():
                where: dict[str, list[int]] = {}
                for i in ctx:
                    where.setdefault(kv[i], []).append(i)
                hits = [(t, where[kv[t]]) for t in te if kv[t] in where]
                conf = [label_conflict(y, t, m, reg) for t, m in hits]
                rec[f"leaked_{kname}"] = len(hits)
                rec[f"leaked_label_differs_{kname}"] = (
                    int(sum(c > 0 for c in conf)) if not reg
                    else int(sum(c > 1e-9 for c in conf)))
                if reg and conf:
                    rec[f"leaked_max_abs_dy_{kname}"] = round(max(conf), 4)
                for (t, m), c in zip(hits, conf):
                    leaked_rows.append(dict(dataset=name, split=sp, key=kname, test_row=t,
                                            context_rows=" ".join(map(str, m)),
                                            label_diff=c))
            per_split[sp] = rec
        ds["splits"] = per_split
        summary["datasets"][name] = ds
        det = per_split["det"]
        print(f"{name:<13} n={len(rows):>5}  dup groups ik/taut={ds['dup_groups_inchikey']}/"
              f"{ds['dup_groups_tautomer']}  det test leaked ik/taut="
              f"{det['leaked_inchikey']}/{det['leaked_tautomer']} of {det['n_test']} "
              f"(label differs {det['leaked_label_differs_inchikey']}/"
              f"{det['leaked_label_differs_tautomer']})", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(leaked_rows).to_csv(OUT / "leaked_test_rows.csv", index=False)
    pd.DataFrame(dup_rows).to_csv(OUT / "duplicates.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
