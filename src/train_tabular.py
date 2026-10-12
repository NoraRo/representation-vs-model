"""Train the tabular models (RF, CatBoost, TabPFN, TabFM, TabFM+) and save their test predictions.

Reads the rows, splits and feature matrices written by prepare.py. For every dataset, task,
representation and split it cleans the matrix on the training rows (clean.py), fits the model and
writes predictions/<model>_<representation>/{dataset}__{task}__{split}.npy (test row index,
prediction) plus one line in that folder's raw_results.csv. Rows already in raw_results.csv are
skipped, so an interrupted run can be restarted.

  OMP_NUM_THREADS=1 python src/train_tabular.py --models RF CatBoost --features rdkit2d ecfp4_2048 maccs --device cpu
  OMP_NUM_THREADS=1 python src/train_tabular.py --models TabPFN --features rdkit2d ecfp4_2048 maccs --device mps
  OMP_NUM_THREADS=1 python src/train_tabular.py --models TabFM --features rdkit2d maccs --device cuda

TabFM and TabPFN weights are read from models/ (see README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import clean as F  # noqa: E402
import models as M  # noqa: E402
import paths as PT  # noqa: E402

KIND = {
    "esol": "regression", "freesolv": "regression", "lipophilicity": "regression",
    "bbbp": "classification", "bace": "classification",
    "clintox": "classification", "sider": "classification",
}
TABFM_DIR = PT.MODELS / "tabfm"  # classification/ and regression/, each with model.safetensors
TABPFN_CKPT = {
    "classification": PT.MODELS / "tabpfn" / "tabpfn-v3-classifier-v3_default.ckpt",
    "regression": PT.MODELS / "tabpfn" / "tabpfn-v3-regressor-v3_default.ckpt",
}
N_ESTIMATORS = 8  # ensemble members of TabFM and TabPFN
CATBOOST_THREADS = 6

COLUMNS = ["dataset", "kind", "task", "model", "features", "split", "metric", "value", "pred_file"]

_TABFM: dict[str, object] = {}


def tabfm_model(kind: str, device: str):
    from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF

    sub = "regression" if kind == "regression" else "classification"
    if sub not in _TABFM:
        _TABFM.clear()
        d = TABFM_DIR / sub
        if not (d / "model.safetensors").exists():
            raise FileNotFoundError(f"TabFM {sub} weights missing: {d}")
        _TABFM[sub] = TabFM_HF.from_pretrained(str(d), local_files_only=True).to(device).eval()
    return _TABFM[sub]


def fit_predict(model: str, kind: str, Xc, yc, Xva, yva, Xte, seed: int, device: str):
    """Return test predictions (regression values or P(class 1))."""
    if model == "RF":
        return M.run_rf(Xc, yc, Xva, yva, Xte, kind, seed=seed)
    # every other model is trained on (or reads as context) the training and validation rows
    X = np.vstack([Xc, Xva])
    y = np.concatenate([yc, yva])
    if model == "CatBoost":
        # library defaults, no early stopping, fixed thread count for repeatable results
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if kind == "regression" else CatBoostClassifier
        est = cls(random_seed=seed, thread_count=CATBOOST_THREADS, verbose=0,
                  allow_writing_files=False)
        est.fit(X, y)
        if kind == "regression":
            return np.asarray(est.predict(Xte), dtype=float)
        return np.asarray(est.predict_proba(Xte), dtype=float)[:, 1]
    if model in ("TabFM", "TabFMplus"):
        from tabfm import TabFMClassifier, TabFMRegressor

        cls = TabFMRegressor if kind == "regression" else TabFMClassifier
        if kind == "regression":
            y = y.astype(np.float32)  # Apple GPU has no float64
        if model == "TabFM":
            est = cls(model=tabfm_model(kind, device), n_estimators=N_ESTIMATORS,
                      max_num_rows=None, max_num_features=500, random_state=seed)
        else:
            # TabFM+: the package's .ensemble() preset (32 members, feature crosses and SVD
            # features, NNLS stacking, Platt calibration for binary tasks)
            est = cls.ensemble(tabfm_model(kind, device), max_num_rows=None,
                               max_num_features=500, random_state=seed)
    elif model == "TabPFN":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        cls = TabPFNRegressor if kind == "regression" else TabPFNClassifier
        ckpt = TABPFN_CKPT[kind]
        if not ckpt.exists():
            raise FileNotFoundError(f"TabPFN checkpoint missing: {ckpt}")
        est = cls(model_path=str(ckpt), device=device, n_estimators=N_ESTIMATORS,
                  n_preprocessing_jobs=1, random_state=seed,
                  # ECFP4 keeps more than 2,000 columns after cleaning, over TabPFN's feature check
                  ignore_pretraining_limits=X.shape[1] > 2000)
    else:
        raise ValueError(model)
    est.fit(X, y)
    if kind == "regression":
        return np.asarray(est.predict(Xte), dtype=float)
    proba = np.asarray(est.predict_proba(Xte), dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise RuntimeError(f"{model} predict_proba shape {proba.shape}, expected (n, 2)")
    return proba[:, 1]


def check_not_collapsed(pred: np.ndarray, Xte: np.ndarray, where: str) -> None:
    """Fail if a foundation model gives the same output for many different inputs.

    TabFM on the Apple GPU (mps) with a large context, such as Lipophilicity, can return the same
    value for most test rows without raising an error; run it on cpu or cuda instead.
    Identical input rows may share a prediction (MACCS), so compare with those."""
    n_pred = len(np.unique(np.round(pred, 6)))
    n_rows = len(np.unique(np.round(Xte, 6), axis=0))
    if n_pred < 0.5 * n_rows:
        raise RuntimeError(f"{where}: {n_pred} distinct predictions for {n_rows} distinct test "
                           "rows; the model output collapsed (see check_not_collapsed)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=["TabFM", "TabFMplus", "RF", "TabPFN", "CatBoost"])
    ap.add_argument("--features", nargs="+", required=True, choices=["rdkit2d", "ecfp4_2048", "maccs"])
    ap.add_argument("--datasets", nargs="+", default=list(KIND))
    ap.add_argument("--splits", nargs="+", default=["det"] + [str(i) for i in range(10)])
    ap.add_argument("--device", default="cpu", help="cpu, cuda or mps (used by TabFM and TabPFN)")
    args = ap.parse_args()

    # results depend on the thread count, so it must be fixed: 1, or any fixed value for
    # TabFM on the cpu, where one thread is too slow
    threads = os.environ.get("OMP_NUM_THREADS", "")
    if threads != "1" and not (args.device == "cpu" and args.models in (["TabFM"], ["TabFMplus"]) and threads.isdigit()):
        raise SystemExit("set OMP_NUM_THREADS=1 (any fixed value allowed only for TabFM on cpu)")

    for name in args.datasets:
        kind = KIND[name]
        rows = pd.read_csv(PT.ROWS / f"{name}_rows.csv")
        tasks = [c for c in rows.columns if c not in ("row", "smiles")]
        splits = json.loads((PT.SPLITS / f"{name}_splits.json").read_text())
        feats = {f: np.load(PT.FEATURES / f"{name}_{f}.npy") for f in args.features}
        for sp in args.splits:
            s = splits[sp]
            tr, va, te = (np.array(s[k]) for k in ("train", "valid", "test"))
            seed = 0 if sp == "det" else int(sp)
            for task in tasks:
                y = rows[task].to_numpy(float)
                if kind == "classification" and len(np.unique(y[te])) < 2:
                    print(f"  skip {name}/{task}/{sp}: single-class test", flush=True)
                    continue
                for feat, X in feats.items():
                    Xtr, Xva, Xte = F.clean_matrix(X[tr], X[va], X[te])
                    for model in args.models:
                        out = PT.arm_dir(model, feat)
                        raw = out / "raw_results.csv"
                        pf = f"{name}__{task}__{sp}.npy"
                        if raw.exists() and pf in set(pd.read_csv(raw).pred_file):
                            continue
                        pred = fit_predict(model, kind, Xtr, y[tr], Xva, y[va], Xte, seed, args.device)
                        if model not in ("RF", "CatBoost"):  # tree models may tie
                            check_not_collapsed(pred, Xte, f"{name}/{task}/{model}/{feat}/{sp}")
                        metric, value = (("RMSE", M.rmse(y[te], pred)) if kind == "regression"
                                         else ("AUC-ROC", M.auc(y[te], pred)))
                        out.mkdir(parents=True, exist_ok=True)
                        np.save(out / pf, np.column_stack([te, pred]))
                        row = dict(dataset=name, kind=kind, task=task, model=model, features=feat,
                                   split=sp, metric=metric, value=value, pred_file=pf)
                        pd.DataFrame([row], columns=COLUMNS).to_csv(
                            raw, mode="a", header=not raw.exists(), index=False)
                        print(f"{name:<13} {task[:20]:<20} {sp:<3} {model:<9} {feat:<10} "
                              f"{metric}={value:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
