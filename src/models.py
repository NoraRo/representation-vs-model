"""Metrics (RMSE, ROC-AUC) and the random forest with its small tuning grid."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

# --------------------------------------------------------------- metrics


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC, or NaN when the fold is single-class (undefined)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# --------------------------------------------------------------- random forest

# 4 settings; the one with the best validation score is refit on training + validation rows
RF_GRID = [
    {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 3},
    {"n_estimators": 500, "max_features": 0.3, "min_samples_leaf": 1},
    {"n_estimators": 500, "max_features": 0.1, "min_samples_leaf": 1},
]


def run_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    kind: str,
    seed: int,
) -> np.ndarray:
    """Pick the setting on the validation rows, refit on training + validation rows, predict test."""
    best_cfg, best_score = None, None
    for cfg in RF_GRID:
        if kind == "regression":
            m = RandomForestRegressor(random_state=seed, n_jobs=-1, **cfg)
            m.fit(X_tr, y_tr)
            score = -rmse(y_va, m.predict(X_va))  # higher is better
        else:
            m = RandomForestClassifier(
                random_state=seed, n_jobs=-1, class_weight="balanced", **cfg
            )
            m.fit(X_tr, y_tr)
            score = auc(y_va, m.predict_proba(X_va)[:, 1])
            if np.isnan(score):
                score = 0.5
        if best_score is None or score > best_score:
            best_cfg, best_score = cfg, score

    X_full = np.vstack([X_tr, X_va])
    y_full = np.concatenate([y_tr, y_va])
    if kind == "regression":
        m = RandomForestRegressor(random_state=seed, n_jobs=-1, **best_cfg)
        m.fit(X_full, y_full)
        return m.predict(X_te)
    m = RandomForestClassifier(
        random_state=seed, n_jobs=-1, class_weight="balanced", **best_cfg
    )
    m.fit(X_full, y_full)
    return m.predict_proba(X_te)[:, 1]
