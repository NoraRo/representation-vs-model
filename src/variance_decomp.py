"""Two-way analysis of variance of the seeded scores: representation, model, interaction.

Per dataset and grid (GRIDS), the balanced layout s ~ input + model + input:model + split (one
score per cell and split; the residual is split x treatment). Sums of squares come straight from
the means. Shares within split = SS_term / (SS_total - SS_split). Bootstrap over the 10 splits
(2,000 draws, the same draws for every cell). Verdict from the interval of
input share - model share.

Reads results/stats/dataset_scores.csv and parts_seeded.csv. Writes results/stats/
variance_decomp.csv, variance_direction.csv and variance_marginals.csv.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import paths as PT  # noqa: E402


STATS = PT.RES / "stats"

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace", "clintox", "sider"]
REGRESSION = ["esol", "freesolv", "lipophilicity"]
SPLITS = [str(i) for i in range(10)]
N_BOOT, BOOT_SEED_VAR = 2_000, 4
GRIDS = {
    "A": (["RF", "CatBoost", "TabPFN"], ["rdkit2d", "ecfp4_2048", "maccs"]),
    "B": (["RF", "CatBoost", "TabPFN", "TabFM"], ["rdkit2d", "maccs"]),
    "C": (["RF", "CatBoost"], ["rdkit2d", "ecfp4_2048", "maccs"]),
    # without ECFP4 and TabFM; its rows come after Grids A-C in the CSV
    "D": (["RF", "CatBoost", "TabPFN"], ["rdkit2d", "maccs"]),
    # morgan512 as an extra input level, and rdkit2d_214 in place of rdkit2d; rows after Grid D
    "Am": (["RF", "CatBoost", "TabPFN"], ["rdkit2d", "ecfp4_2048", "maccs", "morgan512"]),
    "Bm": (["RF", "CatBoost", "TabPFN", "TabFM"], ["rdkit2d", "maccs", "morgan512"]),
    "Cm": (["RF", "CatBoost"], ["rdkit2d", "ecfp4_2048", "maccs", "morgan512"]),
    "Dm": (["RF", "CatBoost", "TabPFN"], ["rdkit2d", "maccs", "morgan512"]),
    "A214": (["RF", "CatBoost", "TabPFN"], ["rdkit2d_214", "ecfp4_2048", "maccs"]),
    "B214": (["RF", "CatBoost", "TabPFN", "TabFM"], ["rdkit2d_214", "maccs"]),
    "C214": (["RF", "CatBoost"], ["rdkit2d_214", "ecfp4_2048", "maccs"]),
    "D214": (["RF", "CatBoost", "TabPFN"], ["rdkit2d_214", "maccs"]),
}
TERMS = ["input", "model", "interaction", "split", "residual"]
DIRECTION = [  # (marginal, kind, reference level, other level, squares)
    ("rdkit2d - ECFP4", "input", "rdkit2d", "ecfp4_2048", ["tabpfn_ecfp4", "catboost_ecfp4"], "input_avg"),
    ("rdkit2d - MACCS", "input", "rdkit2d", "maccs", ["tabpfn_maccs", "catboost_maccs"], "input_avg"),
    ("TabPFN - RF", "model", "TabPFN", "RF", ["tabpfn_ecfp4", "tabpfn_maccs"], "model_avg"),
    ("CatBoost - RF", "model", "CatBoost", "RF", ["catboost_ecfp4", "catboost_maccs"], "model_avg"),
]


def load() -> pd.DataFrame:
    d = pd.read_csv(STATS / "dataset_scores.csv", dtype={"split": str})
    d = d[d.split.isin(SPLITS)]
    dup = d.duplicated(["dataset", "arm", "split"])
    if dup.any():
        raise SystemExit(f"duplicated (dataset, arm, split): {d[dup].head().to_dict('records')}")
    nt = d.groupby(["dataset", "split"]).n_tasks.nunique()
    if (nt > 1).any():
        raise SystemExit(f"n_tasks differs between arms: {nt[nt > 1].index.tolist()}")
    return d


def cube(d: pd.DataFrame, name: str, models, inputs) -> np.ndarray:
    """Scores as an array (model, input, split). Stops if any (arm, split) is missing."""
    sub = d[d.dataset == name].set_index(["arm", "split"]).score
    Y = np.empty((len(models), len(inputs), len(SPLITS)))
    for a, m in enumerate(models):
        for b, i in enumerate(inputs):
            for k, s in enumerate(SPLITS):
                key = (f"{m}|{i}", s)
                if key not in sub.index:
                    raise SystemExit(f"missing {name} {key}")
                Y[a, b, k] = sub[key]
    return Y


def decompose(Y: np.ndarray) -> dict:
    """Y: (..., M, I, S). Returns SS per term, shape (...)."""
    M, I, S = Y.shape[-3:]
    g = Y.mean(axis=(-3, -2, -1), keepdims=True)
    cell = Y.mean(axis=-1, keepdims=True)
    mod = Y.mean(axis=(-2, -1), keepdims=True)
    inp = Y.mean(axis=(-3, -1), keepdims=True)
    spl = Y.mean(axis=(-3, -2), keepdims=True)
    ss = {
        "input": M * S * ((inp - g) ** 2).sum(axis=(-3, -2, -1)),
        "model": I * S * ((mod - g) ** 2).sum(axis=(-3, -2, -1)),
        "interaction": S * ((cell - mod - inp + g) ** 2).sum(axis=(-3, -2, -1)),
        "split": M * I * ((spl - g) ** 2).sum(axis=(-3, -2, -1)),
        "residual": ((Y - cell - spl + g) ** 2).sum(axis=(-3, -2, -1)),
    }
    ss["total"] = ((Y - g) ** 2).sum(axis=(-3, -2, -1))
    gap = np.abs(sum(ss[t] for t in TERMS) - ss["total"]).max()
    if gap > 1e-9:
        raise SystemExit(f"sums of squares do not add up: {gap:.3e}")
    return ss


def within(ss: dict, term: str):
    return ss[term] / (ss["total"] - ss["split"])


def main() -> int:
    d = load()
    rng = np.random.default_rng(BOOT_SEED_VAR)
    draws = rng.integers(0, len(SPLITS), size=(N_BOOT, len(SPLITS)))  # same draws for every cell
    rows, rows_d, rows_new, verdicts, verdicts_d = [], [], [], {}, {}
    for name in DATASETS:
        for grid, (models, inputs) in GRIDS.items():
            dest = rows_d if grid == "D" else rows_new if grid not in ("A", "B", "C") else rows
            M, I, S = len(models), len(inputs), len(SPLITS)
            Y = cube(d, name, models, inputs)
            ss = decompose(Y)
            ssb = decompose(Y[..., draws].transpose(2, 0, 1, 3))
            df = {"input": I - 1, "model": M - 1, "interaction": (I - 1) * (M - 1), "split": S - 1,
                  "residual": (M * I - 1) * (S - 1)}
            ms_res = ss["residual"] / df["residual"]
            wtot = ss["total"] - ss["split"]
            for t in TERMS:
                lo = hi = np.nan
                if t != "split":
                    lo, hi = np.percentile(within(ssb, t), [2.5, 97.5])
                dest.append(dict(
                    dataset=name, grid=grid, term=t, df=df[t], ss=ss[t], ms=ss[t] / df[t],
                    share_within=within(ss, t) if t != "split" else np.nan,
                    share_total=ss[t] / ss["total"],
                    omega2=(ss[t] - df[t] * ms_res) / (wtot + ms_res) if t not in ("split", "residual") else np.nan,
                    lo=lo, hi=hi))
            diff = within(ss, "input") - within(ss, "model")
            dlo, dhi = np.percentile(within(ssb, "input") - within(ssb, "model"), [2.5, 97.5])
            dest.append(dict(dataset=name, grid=grid, term="input_minus_model", share_within=diff, lo=dlo, hi=dhi))
            if grid == "D":
                verdicts_d[name] = ("input larger" if dlo > 0 else "model larger" if dhi < 0 else "unresolved",
                                    diff, dlo, dhi)
            if grid == "A":
                v = "input larger" if dlo > 0 else "model larger" if dhi < 0 else "unresolved"
                inter_top = within(ss, "interaction") > max(within(ss, "input"), within(ss, "model"))
                verdicts[name] = (v, diff, dlo, dhi, inter_top)

    out = pd.DataFrame(rows + rows_d + rows_new)[["dataset", "grid", "term", "df", "ss", "ms", "share_within", "share_total",
                              "omega2", "lo", "hi"]]
    out.to_csv(STATS / "variance_decomp.csv", index=False)

    print("Grid A verdicts (within-split input share - model share, 95% interval):")
    for name, (v, diff, lo, hi, top) in verdicts.items():
        print(f"  {name:14s} {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  {v}" + ("  (interaction largest)" if top else ""))
    reg = [verdicts[n][0] for n in REGRESSION]
    n_in = reg.count("input larger")
    story = ("stay" if n_in == 3 else
             "change" if n_in == 0 or "model larger" in reg else "narrow")
    print(f"Story rule (regression, Grid A): {story}  ({n_in}/3 input larger)")
    print("Grid D verdicts (qualifier only; read through omega2):")
    for name, (v, diff, lo, hi) in verdicts_d.items():
        print(f"  {name:14s} {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  {v}")
    marginals(d)

    # direction check: Grid A marginals (higher = better) vs seeded square parts
    parts = pd.read_csv(STATS / "parts_seeded.csv")
    drows = []
    for name in DATASETS:
        models, inputs = GRIDS["A"]
        Y = cube(d, name, models, inputs)
        if d[d.dataset == name].metric.iloc[0] == "RMSE":
            Y = -Y
        for label, kind, ref, oth, squares, q in DIRECTION:
            if kind == "input":
                marg = Y[:, inputs.index(ref)].mean() - Y[:, inputs.index(oth)].mean()
            else:
                marg = Y[models.index(ref)].mean() - Y[models.index(oth)].mean()
            for sq in squares:
                p = parts[(parts.square == sq) & (parts.quantity == q) & (parts.dataset == name)]
                if len(p) != 1:
                    raise SystemExit(f"parts_seeded row missing: {sq} {q} {name}")
                pm = float(p["mean"].iloc[0])
                drows.append(dict(dataset=name, marginal=label, marginal_mean=marg, square=sq, quantity=q,
                                  square_mean=pm, agree=bool(np.sign(marg) == np.sign(pm))))
    dd = pd.DataFrame(drows)
    dd.to_csv(STATS / "variance_direction.csv", index=False)
    mis = dd[~dd.agree]
    print(f"Direction check: {len(dd) - len(mis)}/{len(dd)} signs agree")
    if len(mis):
        print(mis.to_string(index=False))

    return 0


def marginals(d: pd.DataFrame) -> None:
    """Grid A mean scores over splits 0-9, in score units: per input, per model, per cell."""
    models, inputs = GRIDS["A"]
    rows = []
    for name in DATASETS:
        Y = cube(d, name, models, inputs)
        metric = d[d.dataset == name].metric.iloc[0]
        for b, i in enumerate(inputs):
            rows.append(dict(dataset=name, metric=metric, level="input", model="all", input=i, mean=Y[:, b].mean()))
        for a, m in enumerate(models):
            rows.append(dict(dataset=name, metric=metric, level="model", model=m, input="all", mean=Y[a].mean()))
        for a, m in enumerate(models):
            for b, i in enumerate(inputs):
                rows.append(dict(dataset=name, metric=metric, level="cell", model=m, input=i, mean=Y[a, b].mean()))
    pd.DataFrame(rows).to_csv(STATS / "variance_marginals.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
