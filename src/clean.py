"""Cleaning of a feature matrix, with every column statistic taken from the training rows only."""

from __future__ import annotations

import numpy as np


def clean_matrix(
    X_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Make descriptor matrices model-safe.

    Replaces inf with NaN, drops columns that are all-NaN or constant on the
    training fold, then imputes remaining NaN with the training median. Column
    statistics come from the TRAINING fold only - no test leakage.
    """
    mats = [np.asarray(m, dtype=np.float64).copy() for m in (X_train, *others)]
    for m in mats:
        m[~np.isfinite(m)] = np.nan

    tr = mats[0]
    finite_any = ~np.all(np.isnan(tr), axis=0)
    with np.errstate(invalid="ignore"):
        col_min = np.nanmin(np.where(np.isnan(tr), np.nan, tr), axis=0)
        col_max = np.nanmax(np.where(np.isnan(tr), np.nan, tr), axis=0)
    non_const = ~np.isclose(np.nan_to_num(col_max), np.nan_to_num(col_min))
    keep = finite_any & non_const
    if keep.sum() == 0:  # pathological; keep everything rather than crash
        keep = np.ones(tr.shape[1], dtype=bool)

    mats = [m[:, keep] for m in mats]
    med = np.nanmedian(mats[0], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for m in mats:
        idx = np.where(np.isnan(m))
        m[idx] = np.take(med, idx[1])

    # Some RDKit descriptors (Ipc above all) exceed the float32 range on larger molecules,
    # and scikit-learn casts to float32. Clip every column to a wide band around its
    # training quantiles, and to the float32 range.
    lo = np.quantile(mats[0], 0.001, axis=0)
    hi = np.quantile(mats[0], 0.999, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    lo, hi = lo - 10.0 * span, hi + 10.0 * span
    f32_max = float(np.finfo(np.float32).max) / 10.0
    lo = np.clip(lo, -f32_max, f32_max)
    hi = np.clip(hi, -f32_max, f32_max)
    mats = [np.clip(m, lo, hi) for m in mats]
    return tuple(mats)
