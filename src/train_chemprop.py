"""Train Chemprop (D-MPNN, library default settings) and save its test predictions.

Same rows and splits as the tabular models. Three variants:

  plain    trained on the training rows, Chemprop picks the best epoch on the validation rows;
           also writes best_epochs.csv (0-based, read from the checkpoint name).
  --refit  trained once on training + validation rows for (best epoch + 1) epochs of the plain
           run, without validation; the test rows are predicted with the last checkpoint.
           Needs the plain run first.
  --descriptors rdkit2d
           the plain variant with each molecule's RDKit descriptor vector as extra input:
           cleaned on the training rows (clean.py), then a quantile transform fitted on the
           training rows maps each column to [0, 1]; Chemprop's own descriptor scaling stays on.

ClinTox (2 tasks) and SIDER (27 tasks) are trained multi-task. Seed per split: published split
-> 0, seeded split k -> k. CPU only, one thread, no data-loader workers, so that runs repeat.
Writes predictions/<chemprop variant>/{dataset}__{task}__{split}.npy (test row index, prediction)
and raw_results.csv; datasets and splits already there are skipped.

  OMP_NUM_THREADS=1 python src/train_chemprop.py
  OMP_NUM_THREADS=1 python src/train_chemprop.py --refit
  OMP_NUM_THREADS=1 python src/train_chemprop.py --descriptors rdkit2d
"""

from __future__ import annotations

import argparse
import json
import os
import re
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import models as M  # noqa: E402
from clean import clean_matrix  # noqa: E402
from train_tabular import COLUMNS, KIND  # noqa: E402
import paths as PT  # noqa: E402

WARMUP_EPOCHS = 2  # Chemprop's default; it refuses to train for fewer than WARMUP_EPOCHS + 1 epochs


def run(cmd: list[str], log: pathlib.Path) -> None:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"chemprop failed (exit {p.returncode}):\n{log.read_text()[-2000:]}")


def best_epoch(out_dir: pathlib.Path) -> int:
    """0-based best epoch from Chemprop's checkpoint name 'best-epoch=K-...'."""
    found = sorted(out_dir.rglob("best-epoch=*.ckpt"))
    if len(found) != 1:
        raise RuntimeError(f"expected one best-epoch checkpoint in {out_dir}, found {found}")
    m = re.match(r"best-epoch=(\d+)-", found[0].name)
    if not m:
        raise RuntimeError(f"cannot read the epoch from {found[0].name}")
    return int(m.group(1))


def descriptor_frames(X: np.ndarray, s: dict) -> dict[str, pd.DataFrame]:
    """Clean on the training rows, then a quantile transform to [0, 1] fitted on the training rows."""
    from sklearn.preprocessing import QuantileTransformer
    tr, va, te = (X[s[k]] for k in ("train", "valid", "test"))
    tr, va, te = clean_matrix(tr, va, te)
    qt = QuantileTransformer(n_quantiles=min(1000, len(tr)), output_distribution="uniform",
                             random_state=0).fit(tr)
    out = {}
    for part, m in (("train", tr), ("valid", va), ("test", te)):
        m = qt.transform(m)
        if not np.isfinite(m).all() or m.min() < 0 or m.max() > 1:
            raise RuntimeError(f"quantile-transformed {part} descriptors are outside [0, 1]")
        out[part] = pd.DataFrame(m, columns=[f"d{i:03d}" for i in range(m.shape[1])])
    return out


def descriptor_width(ckpt: pathlib.Path) -> int:
    """Extra-descriptor width the saved model was built for: FFN input minus graph output."""
    from chemprop.models import MPNN
    model = MPNN.load_from_file(str(ckpt), map_location="cpu")
    return int(model.predictor.input_dim - model.message_passing.output_dim)


def fit_predict(name: str, sp: str, rows: pd.DataFrame, tasks: list[str], s: dict,
                work: pathlib.Path, refit_epochs: int | None = None,
                X_desc: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame, int | None]:
    """Train and predict one dataset and split. Returns the test rows, one prediction column per
    task, and the best epoch (plain and descriptor variants only)."""
    kind = KIND[name]
    desc = descriptor_frames(X_desc, s) if X_desc is not None else None
    dcols = list(desc["train"].columns) if desc is not None else []
    paths = {}
    for part, key in (("train", "train"), ("val", "valid"), ("test", "test")):
        paths[part] = work / f"{part}.csv"
        df = rows.iloc[s[key]][["smiles"] + tasks].reset_index(drop=True)
        if desc is not None:
            df = pd.concat([df, desc[key]], axis=1)
        df.to_csv(paths[part], index=False)
    dflag = ["--descriptors-columns", *dcols] if dcols else []
    seed = "0" if sp == "det" else sp
    chemprop = shutil.which("chemprop") or str(pathlib.Path(sys.executable).parent / "chemprop")
    log = work / "chemprop.log"
    out_dir = work / "model"
    common = ["--smiles-columns", "smiles", "--target-columns", *tasks,
              "-t", "regression" if kind == "regression" else "classification",
              "--pytorch-seed", seed, "--data-seed", seed,
              "--num-workers", "0", "--accelerator", "cpu", "--output-dir", str(out_dir), *dflag]
    if refit_epochs is None:
        run([chemprop, "train", "--data-path", str(paths["train"]), str(paths["val"]), str(paths["test"]),
             *common], log)
        ckpts = sorted(out_dir.rglob("best*.pt")) or sorted(out_dir.rglob("*.ckpt"))
        epoch = best_epoch(out_dir)
    else:
        full = work / "train_full.csv"
        rows.iloc[list(s["train"]) + list(s["valid"])][["smiles"] + tasks].to_csv(full, index=False)
        run([chemprop, "train", "--data-path", str(full), "--split-sizes", "1.0", "0.0", "0.0",
             "--epochs", str(refit_epochs), *common], log)
        ckpts = sorted(out_dir.rglob("last.ckpt"))
        epoch = None
    if len(ckpts) != 1:
        raise RuntimeError(f"expected one checkpoint in {out_dir}, found {ckpts}")
    width = descriptor_width(ckpts[0])
    if width != len(dcols):
        raise RuntimeError(f"{name}/{sp}: model descriptor width {width}, expected {len(dcols)}")
    pred_csv = work / "test_preds.csv"
    run([chemprop, "predict", "--test-path", str(paths["test"]), "--smiles-columns", "smiles",
         "--model-path", str(ckpts[0]), "--preds-path", str(pred_csv),
         "--num-workers", "0", "--accelerator", "cpu", *dflag], log)
    preds = pd.read_csv(pred_csv)
    te = np.array(s["test"])
    if len(preds) != len(te):
        raise RuntimeError(f"{name}/{sp}: {len(preds)} predictions for {len(te)} test rows")
    if list(preds.smiles) != list(rows.smiles.iloc[te]):
        raise RuntimeError(f"{name}/{sp}: prediction rows are not in test order")
    # predicting from a .ckpt keeps the true targets and adds pred_0..pred_{k-1}; from a
    # .pt the targets are replaced by predictions. Normalise to one column per task.
    pcols = [f"pred_{i}" for i in range(len(tasks))]
    if all(c in preds.columns for c in pcols):
        preds = preds[["smiles"] + pcols].set_axis(["smiles"] + tasks, axis=1)
    elif refit_epochs is not None:
        raise RuntimeError(f"{name}/{sp}: refit predictions lack {pcols}: {list(preds.columns)}")
    for task in tasks:
        if np.allclose(preds[task].to_numpy(float), rows[task].to_numpy(float)[te], equal_nan=True):
            raise RuntimeError(f"{name}/{sp}/{task}: predictions equal the true values")
    return te, preds, epoch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(KIND))
    ap.add_argument("--splits", nargs="+", default=["det"] + [str(i) for i in range(10)])
    ap.add_argument("--refit", action="store_true",
                    help="train on training + validation rows for the plain run's best epochs")
    ap.add_argument("--descriptors", choices=["rdkit2d"], default=None,
                    help="add the cleaned, quantile-transformed rdkit2d vector as extra input")
    args = ap.parse_args()
    if args.descriptors and args.refit:
        raise SystemExit("--descriptors and --refit cannot be combined")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1")
    model_name = "ChempropRefit" if args.refit else "ChempropDesc" if args.descriptors else "Chemprop"
    feat_name = "graph+rdkit2d" if args.descriptors else "graph"
    out = PT.arm_dir(model_name, feat_name)
    epochs = None
    if args.refit:
        be = PT.arm_dir("Chemprop", "graph") / "best_epochs.csv"
        if not be.exists():
            raise SystemExit(f"{be} is missing: run the plain variant first")
        e = pd.read_csv(be, dtype={"split": str})
        epochs = {(r.dataset, r.split): int(r.best_epoch) + 1 for r in e.itertuples()}

    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_results.csv"
    done = set()
    if raw.exists():
        d = pd.read_csv(raw, dtype={"split": str})
        done = set(zip(d.dataset, d.split))

    for name in args.datasets:
        kind = KIND[name]
        rows = pd.read_csv(PT.ROWS / f"{name}_rows.csv")
        tasks = [c for c in rows.columns if c not in ("row", "smiles")]
        splits = json.loads((PT.SPLITS / f"{name}_splits.json").read_text())
        X_desc = None
        if args.descriptors:
            X_desc = np.load(PT.FEATURES / f"{name}_{args.descriptors}.npy")
            if len(X_desc) != len(rows):
                raise RuntimeError(f"{name}: {len(X_desc)} descriptor rows for {len(rows)} molecules")
        for sp in args.splits:
            if (name, sp) in done:
                continue
            n_ep = None
            if epochs is not None:
                if (name, sp) not in epochs:
                    raise RuntimeError(f"{name}/{sp}: no best epoch in the plain run")
                n_ep = max(epochs[(name, sp)], WARMUP_EPOCHS + 1)
            with tempfile.TemporaryDirectory() as tmp:
                te, preds, ep = fit_predict(name, sp, rows, tasks, splits[sp], pathlib.Path(tmp), n_ep, X_desc)
            if ep is not None and model_name == "Chemprop":
                be = out / "best_epochs.csv"
                pd.DataFrame([dict(dataset=name, split=sp, best_epoch=ep)]).to_csv(
                    be, mode="a", header=not be.exists(), index=False)
            out_rows = []
            for task in tasks:
                y, p = rows[task].to_numpy(float)[te], preds[task].to_numpy(float)
                if kind == "classification" and len(np.unique(y)) < 2:
                    continue  # same rule as train_tabular.py: AUC needs both classes
                metric, value = (("RMSE", M.rmse(y, p)) if kind == "regression"
                                 else ("AUC-ROC", M.auc(y, p)))
                pf = f"{name}__{task}__{sp}.npy"
                np.save(out / pf, np.column_stack([te, p]))
                out_rows.append(dict(dataset=name, kind=kind, task=task, model=model_name,
                                     features=feat_name, split=sp, metric=metric, value=value,
                                     pred_file=pf))
            pd.DataFrame(out_rows, columns=COLUMNS).to_csv(raw, mode="a", header=not raw.exists(),
                                                           index=False)
            print(f"{name:<13} {sp:<3} {model_name} {len(out_rows)} tasks  "
                  f"mean {out_rows[0]['metric']}={np.mean([r['value'] for r in out_rows]):.4f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
