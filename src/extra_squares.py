"""The 7 other squares: does the split into representation and model effect depend on the RF
as reference model and on RDKit descriptors as final representation? No training: every model
is in stats.GRID_ARMS. The published-split bootstrap uses the same random-number stream and
order as stats.parts(), so the 5 main squares come out exactly as in parts_det.csv (guard 1).

Writes results/stats/extra_squares_det.csv and extra_squares_seeded.csv.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import stats as S  # noqa: E402

OUT = S.OUT
TOL = 1e-12

# square: (reference model, compared model, start input, end input). Direction rule:
# reference = earlier in RF -> CatBoost -> TabPFN -> TabFM; start = earlier in ECFP4 -> MACCS -> rdkit2d.
MAIN = {
    "tabfm_maccs": ("RF", "TabFM", "maccs", "rdkit2d"),
    "tabpfn_maccs": ("RF", "TabPFN", "maccs", "rdkit2d"),
    "catboost_maccs": ("RF", "CatBoost", "maccs", "rdkit2d"),
    "tabpfn_ecfp4": ("RF", "TabPFN", "ecfp4_2048", "rdkit2d"),
    "catboost_ecfp4": ("RF", "CatBoost", "ecfp4_2048", "rdkit2d"),
}
NEW = {
    "rf_tabpfn_e2m": ("RF", "TabPFN", "ecfp4_2048", "maccs"),
    "rf_catboost_e2m": ("RF", "CatBoost", "ecfp4_2048", "maccs"),
    "cat_tabpfn_m2r": ("CatBoost", "TabPFN", "maccs", "rdkit2d"),
    "cat_tabpfn_e2r": ("CatBoost", "TabPFN", "ecfp4_2048", "rdkit2d"),
    "cat_tabpfn_e2m": ("CatBoost", "TabPFN", "ecfp4_2048", "maccs"),
    "cat_tabfm_m2r": ("CatBoost", "TabFM", "maccs", "rdkit2d"),
    "pfn_tabfm_m2r": ("TabPFN", "TabFM", "maccs", "rdkit2d"),
}
SQUARES = {**MAIN, **NEW}
NEW_TO_RDKIT = [s for s, v in NEW.items() if v[3] == "rdkit2d"]
NEW_E2M = [s for s, v in NEW.items() if v[3] == "maccs"]
QUANTS = ["input_ref", "input_cmp", "model_start", "model_end", "input_avg", "model_avg", "total",
          "input_minus_model", "interaction", "abs_input_minus_abs_model"]
REG = ["esol", "freesolv", "lipophilicity"]
CLS = ["bbbp", "bace", "clintox", "sider"]


class GuardError(RuntimeError):
    pass


# ------------------------------------------------------------------ quantities


def quantities(kind: str, sc: dict, squares=SQUARES) -> dict:
    """sc maps arm -> score (scalar or draws). Same arithmetic, same order as stats.square_quantities."""
    out = {}
    for sq, (m0, m1, i0, i1) in squares.items():
        a, b, c, d = (sc[f"{m}|{i}"] for m, i in ((m0, i0), (m0, i1), (m1, i0), (m1, i1)))
        in_m0, in_m1 = S.gain(kind, a, b), S.gain(kind, c, d)
        mo_i0, mo_i1 = S.gain(kind, a, c), S.gain(kind, b, d)
        q = {"input_ref": in_m0, "input_cmp": in_m1, "model_start": mo_i0, "model_end": mo_i1,
             "input_avg": (in_m0 + in_m1) / 2, "model_avg": (mo_i0 + mo_i1) / 2,
             "total": S.gain(kind, a, d)}
        q["input_minus_model"] = q["input_avg"] - q["model_avg"]
        q["interaction"] = mo_i1 - mo_i0
        q["abs_input_minus_abs_model"] = np.abs(q["input_avg"]) - np.abs(q["model_avg"])
        for k, v in q.items():
            out[(sq, k)] = v
    return out


def both_orders(q: dict, sq: str) -> bool:
    i_r, i_c = q[(sq, "input_ref")], q[(sq, "input_cmp")]
    m = max(abs(q[(sq, "model_start")]), abs(q[(sq, "model_end")]))
    return bool(np.sign(i_r) == np.sign(i_c) != 0 and min(abs(i_r), abs(i_c)) > m)


# ------------------------------------------------------------------ scores


def compute_scores(df, kinds) -> dict:
    """Per dataset: observed score per arm, bootstrap draws per arm, redraws, n_tasks, and the
    seeded split scores. Same RNG (BOOT_SEED_PARTS), order and draws as stats.parts()."""
    rng = np.random.default_rng(S.BOOT_SEED_PARTS)
    res = {}
    for name in S.DATASETS:
        kind = kinds[name]
        tasks = S.shared_tasks(df, name, S.GRID_ARMS, "det")
        P = {a: S.arm_split(df, name, a, "det", tasks) for a in S.GRID_ARMS}
        if not tasks or any(v is None for v in P.values()):
            raise GuardError(f"{name}: an arm is missing")
        te = next(iter(P[S.GRID_ARMS[0]].values()))[:, 0]
        if any(not np.array_equal(x[:, 0], te) for A in P.values() for x in A.values()):
            raise GuardError(f"{name}: arms scored different test rows")
        obs = {a: S.score(kind, A) for a, A in P.items()}
        clusters = S.scaffold_clusters(name, te)
        k = len(clusters)
        ys = [x[:, 1] for x in P[S.GRID_ARMS[0]].values()]
        draws, redraws, i = {a: np.empty(S.N_BOOT) for a in S.GRID_ARMS}, 0, 0
        while i < S.N_BOOT:
            idx = np.concatenate([clusters[j] for j in rng.integers(0, k, k)])
            if kind == "classification" and any(len(np.unique(y[idx])) < 2 for y in ys):
                redraws += 1
                continue
            for a, A in P.items():
                draws[a][i] = S.score(kind, A, idx)
            i += 1
        seeded = []
        for sp in S.SEEDED:
            t = S.shared_tasks(df, name, S.GRID_ARMS, sp)
            if not t:
                raise GuardError(f"{name}: seeded split {sp} missing")
            sub = df[(df.dataset == name) & (df.split == sp) & df.task.isin(t)]
            sc = sub.groupby("arm").value.mean().to_dict()
            if missing := [a for a in S.GRID_ARMS if a not in sc]:
                raise GuardError(f"{name}: split {sp} lacks {missing}")
            seeded.append({a: sc[a] for a in S.GRID_ARMS})
        res[name] = dict(kind=kind, n_tasks=len(tasks), obs=obs, draws=draws, redraws=redraws,
                         seeded=seeded)
        print(f"  {name}: {len(tasks)} tasks, {k} scaffolds, {redraws} redraws", flush=True)
    return res


# ------------------------------------------------------------------ tables


def tables(res, squares=SQUARES, check_identities=True):
    rows, seed_rows = [], []
    for name in S.DATASETS:
        r = res[name]
        kind = r["kind"]
        obs = quantities(kind, r["obs"], squares)
        reps = quantities(kind, r["draws"], squares)
        if check_identities:
            guard2(reps, name, squares)
        for (sq, q), v in reps.items():
            kk = min((v <= 0).sum(), (v >= 0).sum())
            rows.append(dict(square=sq, quantity=q, dataset=name, n_tasks=r["n_tasks"], est=obs[(sq, q)],
                             ci_lo=np.quantile(v, 0.025), ci_hi=np.quantile(v, 0.975),
                             p_boot=min(1.0, 2 * (kk + 1) / (S.N_BOOT + 1)), redraws=r["redraws"],
                             both_orders=both_orders(obs, sq)))
        per_split = [quantities(kind, s, squares) for s in r["seeded"]]
        for key in obs:
            v = np.array([x[key] for x in per_split])
            seed_rows.append(dict(square=key[0], quantity=key[1], dataset=name, n_splits=len(v),
                                  mean=v.mean(), sd=v.std(ddof=1), n_positive=int((v > 0).sum())))
    det = pd.DataFrame(rows)
    det["p_holm"] = det.groupby(["square", "quantity"]).p_boot.transform(S.holm)
    det["resolved"] = (det.p_holm < 0.05) & ((det.ci_lo > 0) | (det.ci_hi < 0))
    return det, pd.DataFrame(seed_rows)


# ------------------------------------------------------------------ guards


def guard1(det, seeded):
    """The five main squares, recomputed here, match parts_det.csv / parts_seeded.csv."""
    ref_d = pd.read_csv(OUT / "parts_det.csv")
    ref_s = pd.read_csv(OUT / "parts_seeded.csv")
    ren = {"model_rdkit": "model_end"}
    ref_d, ref_s = (x[x.square.isin(MAIN)].replace({"quantity": ren}) for x in (ref_d, ref_s))
    key = ["square", "quantity", "dataset"]
    bad = []
    for ours, ref, num, exact in ((det, ref_d, ["est", "ci_lo", "ci_hi", "p_boot", "p_holm"],
                                   ["resolved", "n_tasks", "redraws"]),
                                  (seeded, ref_s, ["mean", "sd"], ["n_positive", "n_splits"])):
        m = ref.merge(ours[ours.square.isin(MAIN)], on=key, how="left", suffixes=("_ref", ""),
                      indicator=True)
        if (m._merge != "both").any() or len(m) != len(ref):
            bad.append(f"{(m._merge != 'both').sum()} main rows not recomputed")
        for c in num:
            dev = (m[c] - m[f"{c}_ref"]).abs().max()
            if not dev <= TOL:
                bad.append(f"{c} max deviation {dev:.3g}")
        for c in exact:
            n = int((m[c].astype(m[f"{c}_ref"].dtype) != m[f"{c}_ref"]).sum())
            if n:
                bad.append(f"{c}: {n} rows differ")
    if bad:
        raise GuardError("guard 1 (main squares match stats.py): " + "; ".join(bad))
    return len(ref_d), len(ref_s)


def guard2(reps, name, squares=SQUARES):
    """Per draw: input_avg + model_avg = total; interaction = input_cmp - input_ref."""
    for sq in squares:
        g = lambda q: reps[(sq, q)]  # noqa: E731
        e1 = np.abs(g("input_avg") + g("model_avg") - g("total")).max()
        e2 = np.abs(g("interaction") - (g("input_cmp") - g("input_ref"))).max()
        e3 = np.abs(g("interaction") - (g("model_end") - g("model_start"))).max()
        if not max(e1, e2, e3) <= TOL:
            raise GuardError(f"guard 2 ({name}/{sq}): parts do not add up "
                             f"(sum {e1:.3g}, interaction {e2:.3g} / {e3:.3g})")


def guard3(det, seeded):
    """All 12 squares x all quantities x 7 datasets present; no NaN."""
    want = len(SQUARES) * len(QUANTS) * len(S.DATASETS)
    bad = []
    for name, t, cols in (("det", det, ["est", "ci_lo", "ci_hi", "p_boot", "p_holm"]),
                          ("seeded", seeded, ["mean", "sd"])):
        if len(t) != want or set(t.square) != set(SQUARES) or set(t.dataset) != set(S.DATASETS):
            bad.append(f"{name}: {len(t)} rows, expected {want}")
        if (n := int(t[cols].isna().sum().sum())):
            bad.append(f"{name}: {n} NaN")
    if bad:
        raise GuardError("guard 3 (complete, no NaN): " + "; ".join(bad))


# ------------------------------------------------------------------ main


def main() -> int:
    df = S.load_rows()
    kinds = df.groupby("dataset").kind.first().to_dict()
    res = compute_scores(df, kinds)
    det, seeded = tables(res)          # guard 2 runs inside, per draw
    guard1(det, seeded)
    guard3(det, seeded)
    det.to_csv(OUT / "extra_squares_det.csv", index=False)
    seeded.to_csv(OUT / "extra_squares_seeded.csv", index=False)
    print(f"wrote {OUT / 'extra_squares_det.csv'} and extra_squares_seeded.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
