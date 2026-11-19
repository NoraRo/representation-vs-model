"""Near and far test molecules. No training.

Is the representation still the larger effect on the test molecules least similar to the
training data? Per dataset and split, each test molecule gets its maximum Tanimoto similarity
(ECFP4, 2,048 bits) to the training + validation rows. The test set is cut into two equal
halves by that similarity: far = the lower half (ties broken by row index), near = the rest.
Labels and predictions are not used for the cut.

Per half: the four steps of every square (stats.square_quantities) and whether both
representation steps beat both model steps. Published split, regression datasets: one joint
scaffold-cluster bootstrap of the whole test set; each draw is scored within each half, so near,
far and far - near come from the same draws. Classification datasets: point estimates only.
Seeded splits: verdicts per split. Second measure: the same verdicts with MACCS Tanimoto.

Writes results/stats/similarity_groups.csv, similarity_parts.csv, similarity_verdicts.csv.
Checks: the halves cover the test set once; their sizes differ by at most 1; the full test set
reproduces parts_det.csv.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import extra_squares as X  # noqa: E402
import stats as S  # noqa: E402  (read-only use of its helpers)
import paths as PT  # noqa: E402
from train_tabular import KIND  # noqa: E402

RES = S.RES
OUT = S.OUT
BOOT_SEED_SIM = 2
REGRESSION = [d for d in S.DATASETS if KIND[d] == "regression"]
SINGLE = ["input_ref", "input_cmp", "model_start", "model_rdkit"]
# stats.SQUARES end at rdkit2d; these end at rdkit2d_214 and come from extra_squares, which calls
# the fourth quantity model_end where stats calls it model_rdkit
R214 = X.R214
SQUARES = list(S.SQUARES) + list(R214)


def all_quantities(kind: str, sc: dict) -> dict:
    q = S.square_quantities(kind, sc)
    for (sq, k), v in X.quantities(kind, sc, squares=R214).items():
        q[(sq, "model_rdkit" if k == "model_end" else k)] = v
    return q


def load_grid() -> pd.DataFrame:
    frames = []
    for d in PT.arm_dirs():
        f = pd.read_csv(PT.PRED / d / "raw_results.csv", dtype={"split": str})
        f["dir"] = d
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df.model + "|" + df.features
    df = df[df.arm.isin(S.GRID_ARMS)]
    if df.duplicated(["dataset", "task", "arm", "split"]).any():
        raise RuntimeError("duplicate grid rows")
    return df


def max_tanimoto(X: np.ndarray, test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    A = (X[test] > 0).astype(np.float32)
    B = (X[ref] > 0).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return sim.max(1)


def halves(sim: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """True = far (lower half by similarity, ties by row id)."""
    order = np.lexsort((rows, sim))
    far = np.zeros(len(sim), bool)
    far[order[: len(sim) // 2]] = True
    return far


def verdict(q: dict, sq: str) -> str:
    ir, ic, ms, mr = (q[(sq, k)] for k in SINGLE)
    if min(ir, ic) > max(ms, mr):
        return "input"
    if min(ms, mr) > max(ir, ic):
        return "model"
    return "depends"


def main() -> int:
    df = load_grid()
    rng = np.random.default_rng(BOOT_SEED_SIM)
    ref_det = pd.read_csv(OUT / "parts_det.csv").set_index(["square", "quantity", "dataset"]).est
    ref_x = pd.read_csv(OUT / "extra_squares_det.csv").set_index(["square", "quantity", "dataset"]).est
    groups, parts, verdicts = [], [], []
    for name in S.DATASETS:
        kind = KIND[name]
        splits = json.loads((PT.SPLITS / f"{name}_splits.json").read_text())
        fps = {"ecfp4": np.load(PT.FEATURES / f"{name}_ecfp4_2048.npy"), "maccs": np.load(PT.FEATURES / f"{name}_maccs.npy")}
        for sp in ["det", *S.SEEDED]:
            tasks = S.shared_tasks(df, name, S.GRID_ARMS, sp)
            P = {a: S.arm_split(df, name, a, sp, tasks) for a in S.GRID_ARMS}
            if not tasks or any(v is None for v in P.values()):
                raise RuntimeError(f"{name}/{sp}: a grid arm is missing")
            te = next(iter(P[S.GRID_ARMS[0]].values()))[:, 0].astype(int)
            if any(not np.array_equal(x[:, 0], te) for A in P.values() for x in A.values()):
                raise RuntimeError(f"{name}/{sp}: arms scored different test rows")
            if not np.array_equal(te, np.sort(splits[sp]["test"])):
                raise RuntimeError(f"{name}/{sp}: test rows differ from the split file")
            ref = np.array(splits[sp]["train"] + splits[sp]["valid"])
            full = all_quantities(kind, {a: S.score(kind, A) for a, A in P.items()})
            for sq in SQUARES:
                verdicts.append(dict(dataset=name, kind=kind, split=sp, measure="none", group="all",
                                     square=sq, n=len(te), verdict=verdict(full, sq)))
            if sp == "det":
                for (sq, q), v in full.items():
                    r = (ref_x[(sq, "model_end" if q == "model_rdkit" else q, name)]
                         if sq in R214 else ref_det[(sq, q, name)])
                    if abs(v - r) > 1e-12:
                        raise RuntimeError(f"{name}: full test set does not reproduce {sq}/{q}")
            for measure, fp in fps.items():
                sim = max_tanimoto(fp, te, ref)
                far = halves(sim, te)
                if abs(far.sum() - (~far).sum()) > 1:
                    raise RuntimeError(f"{name}/{sp}/{measure}: unequal halves")
                if measure == "ecfp4":
                    groups += [dict(dataset=name, split=sp, row=r, similarity=s, group="far" if f else "near")
                               for r, s, f in zip(te, sim, far)]
                for g, mask in (("near", ~far), ("far", far)):
                    idx = np.flatnonzero(mask)
                    q = all_quantities(kind, {a: S.score(kind, A, idx) for a, A in P.items()})
                    for sq in SQUARES:
                        verdicts.append(dict(dataset=name, kind=kind, split=sp, measure=measure, group=g,
                                             square=sq, n=len(idx), verdict=verdict(q, sq)))
                    if sp == "det" and measure == "ecfp4":
                        for (sq, qq), v in q.items():
                            if sq in SQUARES:
                                parts.append(dict(dataset=name, kind=kind, group=g, square=sq, quantity=qq,
                                                  n=len(idx), est=v))
            # fixed split, regression: joint bootstrap, scored within each half
            if sp != "det" or kind != "regression":
                continue
            far = halves(max_tanimoto(fps["ecfp4"], te, ref), te)
            clusters = S.scaffold_clusters(name, te)
            k = len(clusters)
            draws = {g: {a: np.empty(S.N_BOOT) for a in S.GRID_ARMS} for g in ("near", "far")}
            for i in range(S.N_BOOT):
                idx = np.concatenate([clusters[j] for j in rng.integers(0, k, k)])
                for g, sel in (("near", idx[~far[idx]]), ("far", idx[far[idx]])):
                    for a, A in P.items():
                        draws[g][a][i] = S.score(kind, A, sel)
            reps = {g: all_quantities(kind, draws[g]) for g in draws}
            for row in parts:
                if row["dataset"] != name or "ci_lo" in row:
                    continue
                key = (row["square"], row["quantity"])
                r = reps[row["group"]][key]
                kk = min((r <= 0).sum(), (r >= 0).sum())
                row.update(ci_lo=np.quantile(r, 0.025), ci_hi=np.quantile(r, 0.975),
                           p_boot=min(1.0, 2 * (kk + 1) / (S.N_BOOT + 1)))
            for sq in SQUARES:  # far - near of (input part - model part)
                key = (sq, "input_minus_model")
                obs = (next(p["est"] for p in parts if p["dataset"] == name and p["group"] == "far"
                            and p["square"] == sq and p["quantity"] == "input_minus_model")
                       - next(p["est"] for p in parts if p["dataset"] == name and p["group"] == "near"
                              and p["square"] == sq and p["quantity"] == "input_minus_model"))
                r = reps["far"][key] - reps["near"][key]
                kk = min((r <= 0).sum(), (r >= 0).sum())
                parts.append(dict(dataset=name, kind=kind, group="far_minus_near", square=sq,
                                  quantity="input_minus_model", n=len(te), est=obs,
                                  ci_lo=np.quantile(r, 0.025), ci_hi=np.quantile(r, 0.975),
                                  p_boot=min(1.0, 2 * (kk + 1) / (S.N_BOOT + 1))))
        print(f"{name} done", flush=True)

    pt = pd.DataFrame(parts)
    reg = pt.kind == "regression"
    pt.loc[reg, "p_holm"] = pt[reg].groupby(["group", "square", "quantity"]).p_boot.transform(S.holm)
    pt["resolved"] = (pt.p_holm < 0.05) & ((pt.ci_lo > 0) | (pt.ci_hi < 0))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(groups).to_csv(OUT / "similarity_groups.csv", index=False)
    pt.to_csv(OUT / "similarity_parts.csv", index=False)
    vd = pd.DataFrame(verdicts)
    vd.to_csv(OUT / "similarity_verdicts.csv", index=False)


    r = vd[(vd.kind == "regression")]
    n_cells = len(REGRESSION) * len(SQUARES)
    print(f"\nFixed split, regression: 'input' verdicts out of {n_cells}")
    print(r[r.split == "det"].groupby(["measure", "group"]).verdict.apply(lambda v: (v == "input").sum()).to_string())
    print(f"\nSeeded splits: mean 'input' verdicts out of {n_cells} per split")
    s = r[r.split != "det"].groupby(["measure", "group", "split"]).verdict.apply(lambda v: (v == "input").sum())
    print(s.groupby(["measure", "group"]).agg(["mean", "min", "max"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
