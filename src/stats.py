"""Statistics from the saved predictions (predictions/) and the data rows (data/moleculenet/).

Writes results/stats/:
  dataset_scores.csv     score of every model, representation, dataset and split
  comparisons_det.csv    each pair in COMPARISONS on the published split ("det"): difference,
                         95% interval, bootstrap p, Holm-corrected p
  comparisons_seeded.csv the same pairs on the 10 seeded splits: mean difference, SD, wins
  leak_sensitivity.csv   the published-split differences without the test molecules that also
                         occur in training (results/identity/leaked_test_rows.csv), no refit
  signtest.csv           exact two-sided sign test over the 7 published-split differences
  parts_det.csv          square quantities (representation effect, model effect, their
  parts_seeded.csv       difference, interaction) on the published and the seeded splits

Rules:
  * Dataset score: RMSE (regression) or mean ROC-AUC over the tasks both compared models scored.
  * Published split: paired bootstrap that resamples whole test scaffolds, 10,000 draws; percentile
    95% interval; two-sided bootstrap p; Holm across the 7 datasets for each comparison or
    quantity. Classification draws in which a task has a single class are redrawn and counted.
  * Seeded splits: mean, SD and win count; no p-value. Published and seeded splits are never pooled.
  * Squares (SQUARES): for models m0 -> m1 and representations i0 -> rdkit2d, the representation
    effect is the representation step averaged over both models, the model effect is the model
    step averaged over both representations, and the interaction is the model step on rdkit2d
    minus the model step on i0. Published split: one joint scaffold bootstrap of all GRID_ARMS
    per dataset.
  * Sign: a positive difference means the FIRST model of the pair is better (AUC: a - b; RMSE: b - a).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import datasets as D  # noqa: E402
import paths as PT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = PT.RES
OUT = RES / "stats"
DET_ONLY = {"DGT|graph"}  # run on the fixed split only (seed 0, as every arm there); no seeded splits
DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace", "clintox", "sider"]
SEEDED = [str(i) for i in range(10)]
N_BOOT, BOOT_SEED = 10_000, 0
BOOT_SEED_PARTS = 1

# Every pair is (reference, option); a positive difference means the REFERENCE is better.
# One bootstrap random-number stream is shared by all pairs in this order, so the order fixes each
# pair's draws: add new pairs at the end, never in between.
COMPARISONS = {
    "C1_model": ("RF|rdkit2d", "TabFM|rdkit2d"),
    "C2_input": ("RF|rdkit2d", "RF|ecfp4_2048"),
    "C3_graph": ("Chemprop|graph", "TabFM|rdkit2d"),
    "C4_tfm": ("TabPFN|rdkit2d", "TabFM|rdkit2d"),
    "S_rf_maccs": ("RF|rdkit2d", "RF|maccs"),
    "S_tabpfn_ecfp4": ("TabPFN|rdkit2d", "TabPFN|ecfp4_2048"),
    "S_tabpfn_maccs": ("TabPFN|rdkit2d", "TabPFN|maccs"),
    "S_tabfm_maccs": ("TabFM|rdkit2d", "TabFM|maccs"),
    "P_catboost": ("CatBoost|rdkit2d", "TabFM|rdkit2d"),
    "P_catboost_input": ("CatBoost|rdkit2d", "CatBoost|ecfp4_2048"),
    "P_model_maccs_fm": ("RF|maccs", "TabFM|maccs"),
    "P_model_maccs_pfn": ("RF|maccs", "TabPFN|maccs"),
    "P_model_ecfp4_pfn": ("RF|ecfp4_2048", "TabPFN|ecfp4_2048"),
    "P_model_ecfp4_cat": ("RF|ecfp4_2048", "CatBoost|ecfp4_2048"),
    "P_model_rdkit_cat": ("RF|rdkit2d", "CatBoost|rdkit2d"),
    "P_tabfmplus": ("RF|rdkit2d", "TabFMplus|rdkit2d"),
    "P_plus_vs_fm": ("TabFM|rdkit2d", "TabFMplus|rdkit2d"),
    "P_graph_rf_ecfp4": ("RF|ecfp4_2048", "Chemprop|graph"),
    "P_graph_rf_rdkit": ("RF|rdkit2d", "Chemprop|graph"),
    "P_catboost_maccs": ("CatBoost|rdkit2d", "CatBoost|maccs"),
    "P_model_maccs_cat": ("RF|maccs", "CatBoost|maccs"),
    "P_dgt_tabfm": ("DGT|graph", "TabFM|rdkit2d"),
    "P_dgt_rf_rdkit": ("DGT|graph", "RF|rdkit2d"),
    "P_dgt_rf_ecfp4": ("RF|ecfp4_2048", "DGT|graph"),
    "P_plus_vs_pfn": ("TabPFN|rdkit2d", "TabFMplus|rdkit2d"),
    "P_refit_tabfm": ("ChempropRefit|graph", "TabFM|rdkit2d"),
    "P_refit_vs_chemprop": ("Chemprop|graph", "ChempropRefit|graph"),
    "P_dgt_tabpfn": ("DGT|graph", "TabPFN|rdkit2d"),
    "P_dgt_plus": ("DGT|graph", "TabFMplus|rdkit2d"),
    "P_dgt_tabpfn_ecfp4": ("DGT|graph", "TabPFN|ecfp4_2048"),
    "P_dgt_tabfm_maccs": ("DGT|graph", "TabFM|maccs"),
    "P_dgt_tabpfn_maccs": ("DGT|graph", "TabPFN|maccs"),
    "P_desc_gain": ("Chemprop|graph", "ChempropDesc|graph+rdkit2d"),
    "P_desc_rf_rdkit": ("RF|rdkit2d", "ChempropDesc|graph+rdkit2d"),
    "P_desc_tabfm": ("TabFM|rdkit2d", "ChempropDesc|graph+rdkit2d"),
    # morgan512 and rdkit2d_214: the ablation per model, the input pairs of rdkit2d repeated on
    # rdkit2d_214, and the model pairs on each new input
    "N_abl_rf": ("RF|rdkit2d", "RF|rdkit2d_214"),
    "N_abl_catboost": ("CatBoost|rdkit2d", "CatBoost|rdkit2d_214"),
    "N_abl_tabpfn": ("TabPFN|rdkit2d", "TabPFN|rdkit2d_214"),
    "N_abl_tabfm": ("TabFM|rdkit2d", "TabFM|rdkit2d_214"),
    "N_rf_morgan512": ("RF|rdkit2d", "RF|morgan512"),
    "N_tabpfn_morgan512": ("TabPFN|rdkit2d", "TabPFN|morgan512"),
    "N_tabfm_morgan512": ("TabFM|rdkit2d", "TabFM|morgan512"),
    "N_catboost_morgan512": ("CatBoost|rdkit2d", "CatBoost|morgan512"),
    "N214_rf_ecfp4": ("RF|rdkit2d_214", "RF|ecfp4_2048"),
    "N214_rf_maccs": ("RF|rdkit2d_214", "RF|maccs"),
    "N214_tabpfn_ecfp4": ("TabPFN|rdkit2d_214", "TabPFN|ecfp4_2048"),
    "N214_tabpfn_maccs": ("TabPFN|rdkit2d_214", "TabPFN|maccs"),
    "N214_tabfm_maccs": ("TabFM|rdkit2d_214", "TabFM|maccs"),
    "N214_catboost_ecfp4": ("CatBoost|rdkit2d_214", "CatBoost|ecfp4_2048"),
    "N214_catboost_maccs": ("CatBoost|rdkit2d_214", "CatBoost|maccs"),
    "N214_rf_morgan512": ("RF|rdkit2d_214", "RF|morgan512"),
    "N214_tabpfn_morgan512": ("TabPFN|rdkit2d_214", "TabPFN|morgan512"),
    "N214_tabfm_morgan512": ("TabFM|rdkit2d_214", "TabFM|morgan512"),
    "N214_catboost_morgan512": ("CatBoost|rdkit2d_214", "CatBoost|morgan512"),
    "N_model_morgan512_fm": ("RF|morgan512", "TabFM|morgan512"),
    "N_model_morgan512_pfn": ("RF|morgan512", "TabPFN|morgan512"),
    "N_model_morgan512_cat": ("RF|morgan512", "CatBoost|morgan512"),
    "N_model_214_fm": ("RF|rdkit2d_214", "TabFM|rdkit2d_214"),
    "N_model_214_pfn": ("RF|rdkit2d_214", "TabPFN|rdkit2d_214"),
    "N_model_214_cat": ("RF|rdkit2d_214", "CatBoost|rdkit2d_214"),
}


# ------------------------------------------------------------------ loading


def load_rows() -> pd.DataFrame:
    frames = []
    for d in PT.arm_dirs():
        f = pd.read_csv(PT.PRED / d / "raw_results.csv", dtype={"split": str})
        f["dir"] = d
        frames.append(f)
    if not frames:
        raise SystemExit("no results found")
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df.model + "|" + df.features
    dup = df.duplicated(["dataset", "task", "arm", "split"])
    if dup.any():
        raise RuntimeError(f"{dup.sum()} duplicate result rows")
    want = {"det", *SEEDED}
    arms = sorted({x for pair in COMPARISONS.values() for x in pair})
    gaps = [(arm, ds) for arm in arms for ds in DATASETS
            if set(df[(df.arm == arm) & (df.dataset == ds)].split) != ({"det"} if arm in DET_ONLY else want)]
    if gaps:
        raise SystemExit(f"incomplete arms (arm, dataset): {gaps}")
    return df


_LABELS: dict[str, pd.DataFrame] = {}


def labels(dataset: str, task: str) -> np.ndarray:
    """True labels of one task, indexed by row (data/moleculenet/, written by prepare.py)."""
    if dataset not in _LABELS:
        p = PT.ROWS / f"{dataset}_rows.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}: run download_data.py and prepare.py first")
        r = pd.read_csv(p)
        if not np.array_equal(r["row"].to_numpy(), np.arange(len(r))):
            raise SystemExit(f"{p}: rows are not numbered 0..n-1")
        _LABELS[dataset] = r
    return _LABELS[dataset][task].to_numpy(float)


def preds(df_row, task: str) -> np.ndarray:
    """(n_test, 3): test row index, y_true, prediction. Saved files hold the index and the
    prediction; the label comes from the data rows."""
    a = np.load(PT.PRED / df_row.dir / df_row.pred_file)
    if a.ndim != 2 or a.shape[1] != 2:
        raise SystemExit(f"{df_row.dir}/{df_row.pred_file}: expected (n_test, 2), got {a.shape}")
    return np.column_stack([a[:, 0], labels(df_row.dataset, task)[a[:, 0].astype(int)], a[:, 1]])


# ------------------------------------------------------------------ metrics


def auc_fast(y: np.ndarray, s: np.ndarray) -> float:
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(kind: str, per_task: dict[str, np.ndarray], idx: np.ndarray | None = None) -> float:
    """Dataset score: RMSE, or mean AUC over tasks. per_task[t] = (n, 3) aligned on test rows."""
    vals = []
    for arr in per_task.values():
        a = arr if idx is None else arr[idx]
        vals.append(rmse(a[:, 1], a[:, 2]) if kind == "regression" else auc_fast(a[:, 1], a[:, 2]))
    return float(np.mean(vals))


def diff(kind: str, a: float, b: float) -> float:
    return (b - a) if kind == "regression" else (a - b)


# ------------------------------------------------------------------ per-split data


def arm_split(df, name, arm, sp, tasks) -> dict[str, np.ndarray] | None:
    sub = df[(df.dataset == name) & (df.arm == arm) & (df.split == sp)].set_index("task")
    if not set(tasks) <= set(sub.index):
        return None
    out = {}
    for t in tasks:
        a = preds(sub.loc[t], t)
        out[t] = a[np.argsort(a[:, 0], kind="stable")]
    return out


def shared_tasks(df, name, arms, sp) -> list[str]:
    sets = [set(df[(df.dataset == name) & (df.arm == a) & (df.split == sp)].task) for a in arms]
    return sorted(set.intersection(*sets)) if sets else []


# ------------------------------------------------------------------ bootstrap


def scaffold_clusters(name: str, test_rows: np.ndarray) -> list[np.ndarray]:
    rows = pd.read_csv(PT.ROWS / f"{name}_rows.csv")
    pos = {r: i for i, r in enumerate(test_rows)}  # position within the sorted test block
    groups: dict[str, list[int]] = {}
    for r in test_rows:
        groups.setdefault(D.murcko_scaffold(rows.smiles[int(r)]), []).append(pos[r])
    return [np.array(g) for g in groups.values()]


def bootstrap(kind, A, B, clusters, rng) -> tuple[np.ndarray, int]:
    reps, redraws = np.empty(N_BOOT), 0
    k = len(clusters)
    ys = [a[:, 1] for a in A.values()]
    i = 0
    while i < N_BOOT:
        idx = np.concatenate([clusters[j] for j in rng.integers(0, k, k)])
        if kind == "classification" and any(len(np.unique(y[idx])) < 2 for y in ys):
            redraws += 1
            continue
        reps[i] = diff(kind, score(kind, A, idx), score(kind, B, idx))
        i += 1
    return reps, redraws


def gain(kind: str, before: float, after: float) -> float:
    """Improvement from `before` to `after`: RMSE down or AUC up is positive."""
    return diff(kind, after, before)


# Models and representations used by the squares; RF is the reference model in every square.
GRID_ARMS = ["RF|ecfp4_2048", "RF|maccs", "RF|rdkit2d",
             "CatBoost|ecfp4_2048", "CatBoost|maccs", "CatBoost|rdkit2d",
             "TabPFN|ecfp4_2048", "TabPFN|maccs", "TabPFN|rdkit2d",
             "TabFM|maccs", "TabFM|rdkit2d"]
GRID_ARMS += [f"{m}|{i}" for i in ("morgan512", "rdkit2d_214") for m in ("RF", "CatBoost", "TabPFN", "TabFM")]
# square name: (reference model, compared model, starting input, rdkit2d)
SQUARES = {
    "tabfm_maccs": ("RF", "TabFM", "maccs"),
    "tabpfn_maccs": ("RF", "TabPFN", "maccs"),
    "catboost_maccs": ("RF", "CatBoost", "maccs"),
    "tabpfn_ecfp4": ("RF", "TabPFN", "ecfp4_2048"),
    "catboost_ecfp4": ("RF", "CatBoost", "ecfp4_2048"),
    "tabfm_morgan512": ("RF", "TabFM", "morgan512"),
    "tabpfn_morgan512": ("RF", "TabPFN", "morgan512"),
    "catboost_morgan512": ("RF", "CatBoost", "morgan512"),
}
TARGET_INPUT = "rdkit2d"
GRID_INPUTS = ("ecfp4_2048", "maccs", "morgan512", "rdkit2d_214")


def square_quantities(kind: str, S: dict) -> dict:
    """S maps arm -> score (scalar or array of draws). Gains: improvement, positive = better.

    For models m0 (reference) -> m1 and inputs i0 -> i1 (rdkit2d):
      input part = mean of the input gain within m0 and within m1   (average over both orders)
      model part = mean of the model gain on i0 and on i1
      input part + model part = total gain m0|i0 -> m1|i1, exactly
      interaction = model gain on i1 - model gain on i0 (= input gain in m1 - input gain in m0)
    """
    out = {}
    for sq, (m0, m1, i0) in SQUARES.items():
        i1 = TARGET_INPUT
        a, b, c, d = (S[f"{m}|{i}"] for m, i in ((m0, i0), (m0, i1), (m1, i0), (m1, i1)))
        in_m0, in_m1 = gain(kind, a, b), gain(kind, c, d)
        mo_i0, mo_i1 = gain(kind, a, c), gain(kind, b, d)
        q = {"input_ref": in_m0, "input_cmp": in_m1, "model_start": mo_i0, "model_rdkit": mo_i1,
             "input_avg": (in_m0 + in_m1) / 2, "model_avg": (mo_i0 + mo_i1) / 2,
             "total": gain(kind, a, d)}
        q["input_minus_model"] = q["input_avg"] - q["model_avg"]
        q["interaction"] = mo_i1 - mo_i0
        for k, v in q.items():
            out[(sq, k)] = v
    # grid-wide input effect: rdkit2d vs each other input, averaged over every model that has both
    for i0 in GRID_INPUTS:
        ms = [m for m in ("RF", "CatBoost", "TabPFN", "TabFM") if f"{m}|{i0}" in S]
        out[("all_models", f"input_from_{i0.split('_')[0]}")] = sum(
            gain(kind, S[f"{m}|{i0}"], S[f"{m}|{TARGET_INPUT}"]) for m in ms) / len(ms)
    # the published-baseline gap: TabFM|rdkit2d vs RF|ecfp4, and what RF|rdkit2d alone closes
    out[("baseline_gap", "total")] = gain(kind, S["RF|ecfp4_2048"], S["TabFM|rdkit2d"])
    out[("baseline_gap", "closed_by_rf_rdkit")] = gain(kind, S["RF|ecfp4_2048"], S["RF|rdkit2d"])
    return out


def parts(df, kinds, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed split: one joint scaffold bootstrap of all GRID_ARMS per dataset (same draws for
    every arm), then every quantity per draw. Seeded splits: the same quantities per split."""
    rows, seed_rows = [], []
    for name in DATASETS:
        kind = kinds[name]
        tasks = shared_tasks(df, name, GRID_ARMS, "det")
        P = {a: arm_split(df, name, a, "det", tasks) for a in GRID_ARMS}
        if not tasks or any(v is None for v in P.values()):
            raise RuntimeError(f"parts/{name}: an arm is missing")
        te = next(iter(P[GRID_ARMS[0]].values()))[:, 0]
        if any(not np.array_equal(x[:, 0], te) for A in P.values() for x in A.values()):
            raise RuntimeError(f"parts/{name}: arms scored different test rows")
        obs = square_quantities(kind, {a: score(kind, A) for a, A in P.items()})
        clusters = scaffold_clusters(name, te)
        k = len(clusters)
        ys = [x[:, 1] for x in P[GRID_ARMS[0]].values()]
        draws, redraws, i = {a: np.empty(N_BOOT) for a in GRID_ARMS}, 0, 0
        while i < N_BOOT:
            idx = np.concatenate([clusters[j] for j in rng.integers(0, k, k)])
            if kind == "classification" and any(len(np.unique(y[idx])) < 2 for y in ys):
                redraws += 1
                continue
            for a, A in P.items():
                draws[a][i] = score(kind, A, idx)
            i += 1
        reps = square_quantities(kind, draws)
        for (sq, q), r in reps.items():
            kk = min((r <= 0).sum(), (r >= 0).sum())
            rows.append(dict(square=sq, quantity=q, dataset=name, n_tasks=len(tasks), est=obs[(sq, q)],
                             ci_lo=np.quantile(r, 0.025), ci_hi=np.quantile(r, 0.975),
                             p_boot=min(1.0, 2 * (kk + 1) / (N_BOOT + 1)), redraws=redraws))
        # seeded splits: dataset scores per split (tasks every grid arm scored on that split)
        per_split = []
        for sp in SEEDED:
            t = shared_tasks(df, name, GRID_ARMS, sp)
            if not t:
                raise RuntimeError(f"parts/{name}: seeded split {sp} missing")
            sub = df[(df.dataset == name) & (df.split == sp) & df.task.isin(t)]
            per_split.append(square_quantities(kind, sub.groupby("arm").value.mean().to_dict()))
        for key in obs:
            v = np.array([x[key] for x in per_split])
            seed_rows.append(dict(square=key[0], quantity=key[1], dataset=name, n_splits=len(v),
                                  mean=v.mean(), sd=v.std(ddof=1), n_positive=int((v > 0).sum())))
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby(["square", "quantity"]).p_boot.transform(holm)
    out["resolved"] = (out.p_holm < 0.05) & ((out.ci_lo > 0) | (out.ci_hi < 0))
    return out, pd.DataFrame(seed_rows)


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, run, out = len(p), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return pd.Series(out)[p.index]


# ------------------------------------------------------------------ main


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    kinds = df.groupby("dataset").kind.first().to_dict()
    leaks = pd.read_csv(RES / "identity" / "leaked_test_rows.csv", dtype={"split": str})
    rng = np.random.default_rng(BOOT_SEED)

    # dataset-level scores per arm and split (tasks = what that arm scored)
    ds_rows = []
    for (name, arm, sp), g in df.groupby(["dataset", "arm", "split"]):
        ds_rows.append(dict(dataset=name, arm=arm, split=sp, n_tasks=len(g),
                            score=g.value.mean(), metric=g.metric.iloc[0]))
    pd.DataFrame(ds_rows).to_csv(OUT / "dataset_scores.csv", index=False)

    det_rows, seed_rows, leak_rows = [], [], []
    for cname, (a, b) in COMPARISONS.items():
        for name in DATASETS:
            kind = kinds.get(name)
            if kind is None:
                continue
            # --- det: bootstrap CI
            tasks = shared_tasks(df, name, [a, b], "det")
            A, B = (arm_split(df, name, x, "det", tasks) for x in (a, b)) if tasks else (None, None)
            if A and B:
                te = next(iter(A.values()))[:, 0]
                for x in (*A.values(), *B.values()):
                    if not np.array_equal(x[:, 0], te):
                        raise RuntimeError(f"{cname}/{name}: arms scored different test rows")
                obs = diff(kind, score(kind, A), score(kind, B))
                reps, redraws = bootstrap(kind, A, B, scaffold_clusters(name, te), rng)
                # 2(k+1)/(N+1), capped at 1: never exactly 0 (minimum 2/10,001)
                k = min((reps <= 0).sum(), (reps >= 0).sum())
                p = min(1.0, 2 * (k + 1) / (N_BOOT + 1))
                det_rows.append(dict(comparison=cname, a=a, b=b, dataset=name, n_tasks=len(tasks),
                                     score_a=score(kind, A), score_b=score(kind, B), diff=obs,
                                     ci_lo=np.quantile(reps, 0.025), ci_hi=np.quantile(reps, 0.975),
                                     p_boot=p, redraws=redraws))
                # --- leak sensitivity: drop leaked test rows, rescore, no refit
                bad = set(leaks[(leaks.dataset == name) & (leaks.split == "det")].test_row)
                keep = np.array([r not in bad for r in te])
                if kind == "classification" and any(
                        len(np.unique(x[keep, 1])) < 2 for x in A.values()):
                    raise RuntimeError(f"{name}: leak-free test set is single-class for a task")
                leak_rows.append(dict(comparison=cname, dataset=name, n_dropped=int((~keep).sum()),
                                      diff=obs, diff_leak_free=diff(kind, score(kind, A, keep),
                                                                    score(kind, B, keep))))
            # --- seeded: split spread
            d = []
            for sp in SEEDED:
                t = shared_tasks(df, name, [a, b], sp)
                if not t:
                    continue
                sa = df[(df.dataset == name) & (df.arm == a) & (df.split == sp) & df.task.isin(t)].value.mean()
                sb = df[(df.dataset == name) & (df.arm == b) & (df.split == sp) & df.task.isin(t)].value.mean()
                d.append(diff(kind, sa, sb))
            if d:
                seed_rows.append(dict(comparison=cname, a=a, b=b, dataset=name, n_splits=len(d),
                                      diff_mean=np.mean(d), split_spread_sd=np.std(d, ddof=1) if len(d) > 1 else np.nan,
                                      wins_a=int(sum(x > 0 for x in d)), wins_b=int(sum(x < 0 for x in d))))

    det = pd.DataFrame(det_rows)
    if len(det):
        det["p_holm"] = det.groupby("comparison").p_boot.transform(holm)
        det["a_better"] = (det.ci_lo > 0) & (det.p_holm < 0.05)
    det.to_csv(OUT / "comparisons_det.csv", index=False)
    seeded = pd.DataFrame(seed_rows)
    seeded.to_csv(OUT / "comparisons_seeded.csv", index=False)
    pd.DataFrame(leak_rows).to_csv(OUT / "leak_sensitivity.csv", index=False)
    pdet, pseed = parts(df, kinds, np.random.default_rng(BOOT_SEED_PARTS))
    pdet.to_csv(OUT / "parts_det.csv", index=False)
    pseed.to_csv(OUT / "parts_seeded.csv", index=False)

    sgn = []
    for cname, g in det.groupby("comparison") if len(det) else []:
        n_pos, n_neg = int((g["diff"] > 0).sum()), int((g["diff"] < 0).sum())
        p = binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if len(g) == 7 and n_pos + n_neg else np.nan
        sgn.append(dict(comparison=cname, n=len(g), n_positive=n_pos, n_negative=n_neg, p=p))
    sgn = pd.DataFrame(sgn)  # secondary; no correction across comparisons
    sgn.to_csv(OUT / "signtest.csv", index=False)
    pd.set_option("display.width", 200)
    if len(det):
        print(det[["comparison", "dataset", "diff", "ci_lo", "ci_hi", "p_holm", "redraws"]].round(4).to_string(index=False))
    if len(seeded):
        print(seeded[["comparison", "dataset", "n_splits", "diff_mean", "split_spread_sd", "wins_a"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
