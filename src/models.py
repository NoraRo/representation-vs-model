"""Metrics (RMSE, ROC-AUC), the random forest and CatBoost, each with a 4-setting tuning grid."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC, or NaN when the fold is single-class (undefined)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


RF_GRID = [
    {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 3},
    {"n_estimators": 500, "max_features": 0.3, "min_samples_leaf": 1},
    {"n_estimators": 500, "max_features": 0.1, "min_samples_leaf": 1},
]

CATBOOST_GRID = [
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0, "iterations": 1000},
    {"depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "iterations": 1000},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3.0, "iterations": 1000},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 10.0, "iterations": 1000},
]
CATBOOST_THREADS = 6


def _tune_and_refit(make, grid, X_tr, y_tr, X_va, y_va, X_te, kind: str) -> np.ndarray:
    """Pick the setting with the best validation score, refit it on training + validation rows
    and return the test predictions (values for regression, P(class 1) for classification)."""
    def predict(m, X):
        if kind == "regression":
            return np.asarray(m.predict(X), dtype=float)
        return np.asarray(m.predict_proba(X), dtype=float)[:, 1]

    scores = []
    for cfg in grid:
        m = make(cfg).fit(X_tr, y_tr)
        if kind == "regression":
            scores.append(-rmse(y_va, predict(m, X_va)))
        else:
            s = auc(y_va, predict(m, X_va))
            scores.append(0.5 if np.isnan(s) else s)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"non-finite validation score in {scores}")
    best = grid[int(np.argmax(scores))]
    m = make(best).fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    return predict(m, X_te)


def run_rf(X_tr, y_tr, X_va, y_va, X_te, kind: str, seed: int) -> np.ndarray:
    if kind == "regression":
        def make(cfg):
            return RandomForestRegressor(random_state=seed, n_jobs=-1, **cfg)
    else:
        def make(cfg):
            return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight="balanced", **cfg)
    return _tune_and_refit(make, RF_GRID, X_tr, y_tr, X_va, y_va, X_te, kind)


def run_catboost(X_tr, y_tr, X_va, y_va, X_te, kind: str, seed: int) -> np.ndarray:
    from catboost import CatBoostClassifier, CatBoostRegressor

    cls = CatBoostRegressor if kind == "regression" else CatBoostClassifier

    def make(cfg):
        return cls(random_seed=seed, thread_count=CATBOOST_THREADS, verbose=0,
                   allow_writing_files=False, **cfg)
    return _tune_and_refit(make, CATBOOST_GRID, X_tr, y_tr, X_va, y_va, X_te, kind)
