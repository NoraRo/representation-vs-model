"""TabFM (rdkit2d) vs DGT, one paired test for each DGT training seed 0-3, on the published split.

Same test as P_dgt_tabfm in stats.py (scaffold-cluster bootstrap, 10,000 draws, Holm over the 7
datasets of one seed). Seed 0 is the GitHub run of predictions/dgt_graph/ and takes its interval
and p-values from comparisons_det.csv. Seeds 1-3 are in predictions/dgt_graph/seeds/
({dataset}__{task}__seed{seed}__{config}.npy: test row index, prediction). They are GitHub runs
except ClinTox and SIDER, which ran with the settings of DGT's Zenodo release (config zenodo).

Writes results/stats/dgt_seeds_det.csv, dgt_seeds_summary.csv and dgt_seeds_per_seed.csv.
Needs stats.py to have run first.

  python src/dgt_seed_tests.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import paths as PT  # noqa: E402
import stats as S  # noqa: E402

OUT = S.OUT
SEED_RNG = 20261118
SEEDS = range(4)
ZENODO_SETS = ("clintox", "sider")
ARM = "TabFM|rdkit2d"


def load_run(name: str, seed: int, config: str, tasks: list[str]) -> dict[str, np.ndarray]:
    """Per task (row, label, prediction) of one DGT run, checked against the published split."""
    te = np.sort(json.loads((PT.SPLITS / f"{name}_splits.json").read_text())["det"]["test"])
    out = {}
    if seed == 0 and config == "github":
        base = pd.read_csv(PT.PRED / "dgt_graph" / "raw_results.csv", dtype={"split": str})
        files = base[base.dataset == name].set_index("task").pred_file
        paths = {t: PT.PRED / "dgt_graph" / files[t] for t in tasks}
    else:
        paths = {t: PT.PRED / "dgt_graph" / "seeds" / f"{name}__{t}__seed{seed}__{config}.npy" for t in tasks}
    for t, path in paths.items():
        a = np.load(path)
        a = a[np.argsort(a[:, 0], kind="stable")]
        if not np.array_equal(a[:, 0].astype(int), te):
            raise RuntimeError(f"{path.name}: test rows differ from the published split")
        out[t] = np.column_stack([a[:, 0], S.labels(name, t)[a[:, 0].astype(int)], a[:, -1]])
    return out


def plan() -> list[tuple[str, int, str]]:
    """(dataset, seed, config) of every test."""
    return [(ds, seed, "zenodo" if ds in ZENODO_SETS and seed > 0 else "github")
            for seed in SEEDS for ds in S.DATASETS]


def run_tests(df: pd.DataFrame, kinds: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED_RNG)
    rows = []
    for ds, seed, config in plan():
        kind = kinds[ds]
        tasks = S.shared_tasks(df, ds, [ARM], "det")
        A = S.arm_split(df, ds, ARM, "det", tasks)
        B = load_run(ds, seed, config, tasks)
        te = next(iter(A.values()))[:, 0]
        for x in (*A.values(), *B.values()):
            if not np.array_equal(x[:, 0], te):
                raise RuntimeError(f"{ds}/seed{seed}: arms scored different test rows")
        gain = S.diff(kind, S.score(kind, A), S.score(kind, B))  # > 0: TabFM is better
        reps, redraws = S.bootstrap(kind, A, B, S.scaffold_clusters(ds, te), rng)
        k = min((reps <= 0).sum(), (reps >= 0).sum())
        rows.append(dict(seed=seed, dataset=ds, config=config, metric="RMSE" if kind == "regression" else "AUC",
                         tabfm=S.score(kind, A), dgt=S.score(kind, B), tabfm_gain=gain,
                         ci_lo=np.quantile(reps, 0.025), ci_hi=np.quantile(reps, 0.975),
                         p_boot=min(1.0, 2 * (k + 1) / (S.N_BOOT + 1)), redraws=redraws))
    d = pd.DataFrame(rows)
    d["p_holm"] = d.groupby("seed").p_boot.transform(S.holm)
    return d


def seed0_reference() -> pd.DataFrame:
    ref = pd.read_csv(OUT / "comparisons_det.csv")
    return ref[ref.comparison == "P_dgt_tabfm"].set_index("dataset")


def check_seed0(d: pd.DataFrame) -> None:
    """Seed 0 must reproduce the scores and differences of P_dgt_tabfm in comparisons_det.csv."""
    ref, mine = seed0_reference(), d[d.seed == 0].set_index("dataset")
    if set(ref.index) != set(mine.index):
        raise RuntimeError(f"seed 0 covers {sorted(mine.index)}, P_dgt_tabfm {sorted(ref.index)}")
    for ds in mine.index:
        if abs(mine.loc[ds, "tabfm_gain"] + ref.loc[ds, "diff"]) > 1e-9:
            raise RuntimeError(f"seed 0 {ds}: difference differs from P_dgt_tabfm")
        if abs(mine.loc[ds, "dgt"] - ref.loc[ds, "score_a"]) > 1e-9 or abs(mine.loc[ds, "tabfm"] - ref.loc[ds, "score_b"]) > 1e-9:
            raise RuntimeError(f"seed 0 {ds}: scores differ from P_dgt_tabfm")


def use_seed0_intervals(d: pd.DataFrame) -> pd.DataFrame:
    """Seed 0 takes its interval and p-values from comparisons_det.csv, where DGT is the first
    model of the pair, so the interval is mirrored."""
    ref, d = seed0_reference(), d.copy()
    for i in d.index[d.seed == 0]:
        r = ref.loc[d.at[i, "dataset"]]
        d.loc[i, ["ci_lo", "ci_hi", "p_boot", "p_holm"]] = [-r.ci_hi, -r.ci_lo, r.p_boot, r.p_holm]
    d["tabfm_better"] = (d.ci_lo > 0) & (d.p_holm < 0.05)
    d["dgt_better"] = (d.ci_hi < 0) & (d.p_holm < 0.05)
    return d


def main() -> int:
    df = S.load_rows()
    kinds = df.groupby("dataset").kind.first().to_dict()
    d = run_tests(df, kinds)
    check_seed0(d)
    d = use_seed0_intervals(d)
    summary = d.groupby("dataset").agg(
        seeds=("seed", "count"), tabfm_better=("tabfm_better", "sum"), dgt_better=("dgt_better", "sum"),
        configs=("config", lambda x: "+".join(sorted(set(x)))), mean_gain=("tabfm_gain", "mean")).reindex(S.DATASETS)
    per_seed = d.groupby("seed").agg(tabfm_better=("tabfm_better", "sum"), dgt_better=("dgt_better", "sum"))
    d.to_csv(OUT / "dgt_seeds_det.csv", index=False)
    summary.to_csv(OUT / "dgt_seeds_summary.csv")
    per_seed.to_csv(OUT / "dgt_seeds_per_seed.csv")
    pd.set_option("display.width", 200)
    print(d.round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
