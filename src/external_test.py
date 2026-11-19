"""External datasets: the grid models of the published split, tested on molecules from another
source. ESOL -> AqSolDBc (Llompart et al. 2024), Lipophilicity -> T-data of RTlogD (Wang et al. 2023).
Put the source files in data/external/raw/ (README.md).

Four stages, each its own command:

  rows        build the external test rows: parse, keep molecules whose elements all occur in
              the training rows, remove every molecule also in the MoleculeNet dataset (InChIKey,
              its first block, or canonical tautomer), merge duplicates whose labels differ by at most
              DUP_TOL (drop the others), check the label range;
              compute the 5 representations. Needs RDKit 2026.03.5, as used for the MoleculeNet
              features.
  predict     fit each model as train_tabular.py does (training + validation rows of the
              published split) and predict the MoleculeNet test rows and the external rows in
              one call. The MoleculeNet predictions must match the saved ones (exact for RF and
              CatBoost, within 1e-5 otherwise, or within 1e-3 test RMSE), else the run stops;
              results/external/gate.csv records this. External predictions are saved without labels.
  stats       join the external labels: square quantities and verdicts for every square of
              extra_squares.py, a joint scaffold-cluster bootstrap (seed 3), Holm across the 2
              datasets.
  similarity  median maximum ECFP4 similarity of the external rows and of the published test
              rows to the training + validation rows (no labels).

  OMP_NUM_THREADS=1 python src/external_test.py rows
  OMP_NUM_THREADS=1 python src/external_test.py predict --models RF CatBoost --features rdkit2d rdkit2d_214 ecfp4_2048 morgan512 maccs --sets esol lipophilicity --device cpu
  OMP_NUM_THREADS=1 python src/external_test.py predict --models TabPFN --features rdkit2d rdkit2d_214 ecfp4_2048 morgan512 maccs --sets esol lipophilicity --device mps
  OMP_NUM_THREADS=1 python src/external_test.py predict --models TabFM --features rdkit2d rdkit2d_214 morgan512 maccs --sets esol --device mps
  OMP_NUM_THREADS=6 python src/external_test.py predict --models TabFM --features rdkit2d rdkit2d_214 morgan512 maccs --sets lipophilicity --device cpu
  OMP_NUM_THREADS=1 python src/external_test.py stats
  OMP_NUM_THREADS=1 python src/external_test.py similarity

Writes data/external/ (rows; SMILES, labels and fingerprints in data/external/local/),
results/external/ (gate.csv, preds/) and results/stats/external_*.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clean as C  # noqa: E402  (RDKit-free)
import models as M  # noqa: E402
import paths as PT  # noqa: E402

ROOT = HERE.parent
RES = PT.RES
EXT = PT.RES / "external"      # gate.csv, preds/
EXT_ROWS = PT.EXT_DATA        # outside test rows
LOCAL = PT.EXT_LOCAL
SEARCH = PT.EXT_DATA          # raw/ holds the source files

# MoleculeNet set -> (outside name, file, smiles column, label column, plausible label range)
SETS = {
    "esol": ("aqsoldbc", SEARCH / "raw/llompart/AqSolDBc.csv", "SmilesCurated", "ExperimentalLogS", (-15.0, 3.0)),
    "lipophilicity": ("rtlogd_tdata", SEARCH / "raw/rtlogd/T-data_chembl32.csv", "smiles", "standard_value", (-5.0, 8.0)),
}
FEATS = ["rdkit2d", "rdkit2d_214", "ecfp4_2048", "morgan512", "maccs"]
DUP_TOL = 0.5
GATE_TOL, GATE_RMSE_TOL = 1e-5, 1e-3
BOOT_SEED_EXT = 3


# ------------------------------------------------------------------ stage 1: rows


def stage_rows() -> int:
    from rdkit import Chem, RDLogger
    import rdkit

    RDLogger.DisableLog("rdApp.*")
    from identity_audit import keys
    import featurize as F

    if rdkit.__version__ != "2026.03.5":
        raise SystemExit(f"RDKit {rdkit.__version__}; the MoleculeNet features were built with 2026.03.5")
    EXT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    report = {}
    for mn, (name, path, scol, ycol, (lo, hi)) in SETS.items():
        rows = pd.read_csv(PT.ROWS / f"{mn}_rows.csv")
        det = json.loads((PT.SPLITS / f"{mn}_splits.json").read_text())["det"]
        trv = list(det["train"]) + list(det["valid"])

        elements = {a.GetSymbol() for s in rows.smiles[trv] for a in Chem.MolFromSmiles(s).GetAtoms()}
        mn_keys = [keys(s) for s in rows.smiles]
        mn_full = {k[0] for k in mn_keys}
        mn_14 = {k[0][:14] for k in mn_keys if not k[0].startswith("NOINCHI")}
        mn_taut = {k[1] for k in mn_keys}

        df = pd.read_csv(path)
        n0 = len(df)
        smi = df[scol].astype(str).str.strip()
        mols = [Chem.MolFromSmiles(s) for s in smi]
        ok = np.array([m is not None and m.GetNumAtoms() > 0 for m in mols])
        el_ok = np.array([ok[i] and {a.GetSymbol() for a in mols[i].GetAtoms()} <= elements
                          for i in range(n0)])
        k = [keys(s) if el_ok[i] else (None, None) for i, s in enumerate(smi)]
        full = pd.Series([x[0] for x in k])
        taut = pd.Series([x[1] for x in k])
        ik14 = full.str[:14]
        shared = full.isin(mn_full) | ik14.isin(mn_14) | taut.isin(mn_taut)
        keep = el_ok & ~shared.to_numpy()
        cand = pd.DataFrame({"src_row": np.flatnonzero(keep), "smiles": smi[keep].to_numpy(),
                             "full": full[keep].to_numpy(), "ik14": ik14[keep].to_numpy(),
                             "taut": taut[keep].to_numpy()})
        # duplicate rule (labels read only here and in the unit check; nothing printed)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()[cand.src_row]
        if np.isnan(y).any():
            raise SystemExit(f"{name}: {int(np.isnan(y).sum())} labels are not numbers")
        cand["y"] = y
        out, n_dup_groups, n_dropped_groups = [], 0, 0
        for _, g in cand.groupby("ik14", sort=False):
            if len(g) == 1:
                out.append(g.iloc[0].to_dict())
                continue
            n_dup_groups += 1
            if g.y.max() - g.y.min() > DUP_TOL:
                n_dropped_groups += 1
                continue
            r = g.iloc[0].to_dict()
            r["y"] = float(g.y.mean())
            r["n_merged"] = len(g)
            out.append(r)
        t = pd.DataFrame(out).sort_values("src_row").reset_index(drop=True)
        t["n_merged"] = t.get("n_merged", pd.Series(1, index=t.index)).fillna(1).astype(int)
        n_out = int(((t.y < lo) | (t.y > hi)).sum())
        if n_out:
            raise SystemExit(f"{name}: {n_out} labels outside [{lo}, {hi}]; unit check failed")
        t.insert(0, "ext_row", np.arange(len(t)))
        cols = ["ext_row", "src_row", "smiles", "full", "ik14", "taut", "n_merged"]
        labels = t[["ext_row", "y"]]
        if name == "aqsoldbc":  # Etalab 2.0: labels may be shared
            t[cols + ["y"]].to_csv(EXT_ROWS / f"{name}_rows.csv", index=False)
        else:  # RTlogD states no licence: InChIKeys only in the repo; SMILES, tautomer SMILES and labels stay local
            t[[c for c in cols if c not in ("smiles", "taut")]].to_csv(EXT_ROWS / f"{name}_rows.csv", index=False)
            t[["ext_row", "smiles", "taut"]].to_csv(LOCAL / f"{name}_smiles.csv", index=False)
        labels.to_csv(LOCAL / f"{name}_labels.csv", index=False)
        for feat in FEATS:
            np.save(LOCAL / f"{name}_{feat}.npy", F.featurize(list(t.smiles), feat))
        report[name] = dict(moleculenet=mn, source_rows=n0, unparsed=int((~ok).sum()),
                            element_rule_dropped=int((ok & ~el_ok).sum()),
                            shared_dropped=int((el_ok & shared.to_numpy()).sum()),
                            duplicate_groups=n_dup_groups, duplicate_groups_dropped=n_dropped_groups,
                            test_rows=len(t), unit_check_outside=n_out,
                            training_elements=sorted(elements), rdkit=rdkit.__version__)
        print(name, report[name], flush=True)
    (EXT_ROWS / "rows_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


# ------------------------------------------------------------------ stage 2: predict


def saved_det(mn: str, model: str, feat: str) -> np.ndarray:
    import stats as S

    hits = []
    for d in PT.arm_dirs():
        p = PT.PRED / d / "raw_results.csv"
        if not p.exists():
            continue
        f = pd.read_csv(p, dtype={"split": str})
        f = f[(f.dataset == mn) & (f.model == model) & (f.features == feat) & (f.split == "det")]
        hits += [r for r in f.assign(dir=d).itertuples()]
    if len(hits) != 1:
        raise SystemExit(f"{mn}/{model}/{feat}: expected one saved det prediction, found {len(hits)}")
    return S.preds(hits[0], hits[0].task)


def stage_predict(args) -> int:
    from train_tabular import KIND, fit_predict, check_not_collapsed

    threads = os.environ.get("OMP_NUM_THREADS", "")
    if threads != "1" and not (args.device == "cpu" and args.models == ["TabFM"] and threads.isdigit()):
        raise SystemExit("set OMP_NUM_THREADS=1 (any fixed value only for TabFM on cpu)")
    if "TabFM" in args.models and args.device == "mps" and "lipophilicity" in args.sets:
        raise SystemExit("TabFM on the Apple GPU (mps) collapses on Lipophilicity; use cpu or cuda")
    (EXT / "preds").mkdir(parents=True, exist_ok=True)
    gate_csv = EXT / "gate.csv"
    done = set()
    if gate_csv.exists():
        g = pd.read_csv(gate_csv)
        done = set(zip(g.dataset, g.model, g.features))

    for mn in args.sets:
        name = SETS[mn][0]
        kind = KIND[mn]
        rows = pd.read_csv(PT.ROWS / f"{mn}_rows.csv")
        s = json.loads((PT.SPLITS / f"{mn}_splits.json").read_text())["det"]
        tr, va, te = (np.array(s[k]) for k in ("train", "valid", "test"))
        y = rows["logS" if mn == "esol" else "logD"].to_numpy(float)
        for feat in args.features:
            X = np.load(PT.FEATURES / f"{mn}_{feat}.npy")
            Xo = np.load(LOCAL / f"{name}_{feat}.npy")
            if Xo.shape[1] != X.shape[1]:
                raise SystemExit(f"{name}/{feat}: {Xo.shape[1]} columns vs {X.shape[1]} in prep")
            Xtr, Xva, Xte_all = C.clean_matrix(X[tr], X[va], np.vstack([X[te], Xo]))
            for model in args.models:
                if (mn, model, feat) in done:
                    continue
                if model == "TabFM" and feat == "ecfp4_2048":
                    continue  # not a grid arm (500-feature cap)
                t0 = time.time()
                pred = fit_predict(model, kind, Xtr, y[tr], Xva, y[va], Xte_all, 0, args.device)
                secs = time.time() - t0
                if model not in ("RF", "CatBoost"):
                    check_not_collapsed(pred, Xte_all, f"external/{mn}/{model}/{feat}")
                p_mn, p_out = pred[: len(te)], pred[len(te):]
                saved = saved_det(mn, model, feat)
                if not np.array_equal(saved[:, 0], te):
                    raise SystemExit(f"{mn}/{model}/{feat}: saved test rows differ from the split")
                max_diff = float(np.abs(p_mn - saved[:, 2]).max())
                rmse_new, rmse_saved = M.rmse(y[te], p_mn), M.rmse(y[te], saved[:, 2])
                exact = bool(np.array_equal(p_mn, saved[:, 2]))
                strict = exact if model in ("RF", "CatBoost") else max_diff <= GATE_TOL
                passed = strict or abs(rmse_new - rmse_saved) <= GATE_RMSE_TOL
                rec = dict(dataset=mn, model=model, features=feat, exact=exact,
                           max_abs_diff=max_diff, rmse_saved=rmse_saved, rmse_new=rmse_new,
                           rmse_diff=rmse_new - rmse_saved, passed_strict=strict, passed=passed,
                           n_outside=len(p_out))
                pd.DataFrame([rec]).to_csv(gate_csv, mode="a", header=not gate_csv.exists(), index=False)
                print(f"{mn:<13} {model:<8} {feat:<10} gate {'PASS' if passed else 'FAIL'} "
                      f"(strict {strict}, max diff {max_diff:.2e}, RMSE diff {rmse_new - rmse_saved:+.2e}) "
                      f"{secs:.0f}s", flush=True)
                if not passed:
                    raise SystemExit(f"gate failed for {mn}/{model}/{feat}; see {gate_csv}")
                np.save(EXT / "preds" / f"{name}__{model}__{feat}.npy",
                        np.column_stack([np.arange(len(p_out)), p_out]))
    return 0


# ------------------------------------------------------------------ stage 3: stats

X_KEEP = ["input_ref", "input_cmp", "model_start", "model_end", "input_avg", "model_avg",
          "input_minus_model", "interaction"]


def square_group(sq: str) -> str:
    """main = the 5 main designs, other = the 7 other designs on ecfp4_2048, maccs and rdkit2d,
    extra_input_designs = the 36 designs that involve rdkit2d_214 or morgan512."""
    import extra_squares as X
    return "main" if sq in X.MAIN else "other" if sq in X.NEW else "extra_input_designs"


def verdict_end(q: dict, sq: str) -> str:
    """similarity_check.verdict for squares whose end representation is not always rdkit2d."""
    ir, ic, ms, me = (q[(sq, k)] for k in ("input_ref", "input_cmp", "model_start", "model_end"))
    if min(ir, ic) > max(ms, me):
        return "input"
    if min(ms, me) > max(ir, ic):
        return "model"
    return "depends"


def stage_stats() -> int:
    import stats as S
    import datasets as D
    import extra_squares as X
    from similarity_check import verdict

    gate = pd.read_csv(EXT / "gate.csv")
    if not gate.passed.all() or gate.duplicated(["dataset", "model", "features"]).any():
        raise SystemExit("gate.csv has a failed or duplicated cell")
    if len(gate) != 2 * len(S.GRID_ARMS):
        raise SystemExit(f"gate.csv has {len(gate)} cells, expected {2 * len(S.GRID_ARMS)}")
    rng = np.random.default_rng(BOOT_SEED_EXT)
    ref_det = pd.read_csv(S.OUT / "parts_det.csv").set_index(["square", "quantity", "dataset"]).est
    part_rows, verdicts, scores = [], [], []
    x_rows, x_verdicts = [], []  # every square of extra_squares.py, scored on the same bootstrap draws
    for mn, (name, *_rest) in SETS.items():
        rows = pd.read_csv(EXT_ROWS / f"{name}_rows.csv")
        if "smiles" not in rows:  # T-data: SMILES kept local
            smi = pd.read_csv(LOCAL / f"{name}_smiles.csv")
            if not np.array_equal(smi.ext_row, rows.ext_row):
                raise SystemExit(f"{name}: local SMILES and rows are not aligned")
            rows["smiles"] = smi.smiles
        lab = pd.read_csv(LOCAL / f"{name}_labels.csv")
        if not np.array_equal(lab.ext_row, rows.ext_row):
            raise SystemExit(f"{name}: labels and rows are not aligned")
        y = lab.y.to_numpy()
        P = {}
        for arm in S.GRID_ARMS:
            m, f = arm.split("|")
            p = np.load(EXT / "preds" / f"{name}__{m}__{f}.npy")
            if not np.array_equal(p[:, 0], rows.ext_row):
                raise SystemExit(f"{name}/{arm}: prediction rows are not aligned")
            P[arm] = {"t": np.column_stack([rows.ext_row, y, p[:, 1]])}
        obs = S.square_quantities("regression", {a: S.score("regression", A) for a, A in P.items()})
        g = gate[gate.dataset == mn].set_index(["model", "features"])
        for arm, A in P.items():
            m, f = arm.split("|")
            scores.append(dict(dataset=mn, outside=name, arm=arm, n=len(y),
                               rmse_outside=S.score("regression", A),
                               rmse_moleculenet=float(g.loc[(m, f), "rmse_saved"])))
        for sq in S.SQUARES:
            verdicts.append(dict(dataset=mn, outside=name, square=sq, verdict=verdict(obs, sq),
                                 verdict_moleculenet=verdict({k: ref_det[(k[0], k[1], mn)]
                                                              for k in obs if k[0] == sq}, sq)))
        obs_x = X.quantities("regression", {a: S.score("regression", A) for a, A in P.items()})
        for sq in X.SQUARES:
            x_verdicts.append(dict(dataset=mn, outside=name, square=sq, group=square_group(sq),
                                   verdict=verdict_end(obs_x, sq)))
        groups: dict[str, list[int]] = {}
        for i, smi in enumerate(rows.smiles):
            groups.setdefault(D.murcko_scaffold(smi), []).append(i)
        clusters = [np.array(v) for v in groups.values()]
        k = len(clusters)
        draws = {a: np.empty(S.N_BOOT) for a in S.GRID_ARMS}
        for i in range(S.N_BOOT):
            idx = np.concatenate([clusters[j] for j in rng.integers(0, k, k)])
            for a, A in P.items():
                draws[a][i] = S.score("regression", A, idx)
        reps = S.square_quantities("regression", draws)
        for (sq, q), r in reps.items():
            kk = min((r <= 0).sum(), (r >= 0).sum())
            part_rows.append(dict(square=sq, quantity=q, dataset=mn, outside=name, n=len(y),
                                  n_clusters=k, largest_cluster=max(len(c) for c in clusters),
                                  est=obs[(sq, q)], est_moleculenet=ref_det.get((sq, q, mn), np.nan),
                                  ci_lo=np.quantile(r, 0.025), ci_hi=np.quantile(r, 0.975),
                                  p_boot=min(1.0, 2 * (kk + 1) / (S.N_BOOT + 1))))
        for (sq, q), r in X.quantities("regression", draws).items():
            if q not in X_KEEP:
                continue
            kk = min((r <= 0).sum(), (r >= 0).sum())
            x_rows.append(dict(square=sq, group=square_group(sq), quantity=q, dataset=mn, outside=name,
                               n=len(y), est=obs_x[(sq, q)],
                               ci_lo=np.quantile(r, 0.025), ci_hi=np.quantile(r, 0.975),
                               p_boot=min(1.0, 2 * (kk + 1) / (S.N_BOOT + 1))))
    pt = pd.DataFrame(part_rows)
    pt["p_holm"] = pt.groupby(["square", "quantity"]).p_boot.transform(S.holm)
    pt["resolved"] = (pt.p_holm < 0.05) & ((pt.ci_lo > 0) | (pt.ci_hi < 0))
    vd, sc = pd.DataFrame(verdicts), pd.DataFrame(scores)
    pt.to_csv(S.OUT / "external_parts.csv", index=False)
    vd.to_csv(S.OUT / "external_verdicts.csv", index=False)
    sc.to_csv(S.OUT / "external_scores.csv", index=False)
    xp = pd.DataFrame(x_rows)
    xp["p_holm"] = xp.groupby(["square", "quantity"]).p_boot.transform(S.holm)
    xp["resolved"] = (xp.p_holm < 0.05) & ((xp.ci_lo > 0) | (xp.ci_hi < 0))
    if len(xp) != 2 * len(X.SQUARES) * len(X_KEEP) or len(x_verdicts) != 2 * len(X.SQUARES):
        raise SystemExit(f"other squares: {len(xp)} part rows, {len(x_verdicts)} verdicts")
    xp.to_csv(S.OUT / "external_extra_parts.csv", index=False)
    pd.DataFrame(x_verdicts).to_csv(S.OUT / "external_extra_verdicts.csv", index=False)

    n_in = int((vd.verdict == "input").sum())
    n_mn = int((vd.verdict_moleculenet == "input").sum())
    imm = pt[(pt.quantity == "input_minus_model") & pt.square.isin(S.SQUARES)]
    n_sig = int(imm.resolved.sum())
    print(sc.round(4).to_string(index=False))
    print(vd.to_string(index=False))
    print(imm[["dataset", "square", "est", "est_moleculenet", "ci_lo", "ci_hi", "p_holm", "resolved"]]
          .round(4).to_string(index=False))
    print(f"\n'input' verdicts: outside {n_in}/{len(vd)}, MoleculeNet {n_mn}/{len(vd)}; "
          f"input - model resolved: {n_sig}/{len(imm)}")
    return 0


# ------------------------------------------------------------------ stage 4: similarity


def stage_similarity() -> int:
    """Median max ECFP4 Tanimoto (2,048 bits) of the outside test rows and of the published-split
    test rows to the published split's train + valid rows. No labels, no models. Writes
    results/stats/external_similarity.csv."""
    from similarity_check import max_tanimoto

    out = []
    for mn, (name, *_rest) in SETS.items():
        det = json.loads((PT.SPLITS / f"{mn}_splits.json").read_text())["det"]
        trv = np.array(list(det["train"]) + list(det["valid"]))
        test = np.array(list(det["test"]))
        X = np.load(PT.FEATURES / f"{mn}_ecfp4_2048.npy")
        f_ext = LOCAL / f"{name}_ecfp4_2048.npy"
        if not f_ext.exists():
            raise SystemExit(f"{f_ext} is missing; run the rows stage first")
        E = np.load(f_ext)
        if E.shape[1] != X.shape[1]:
            raise SystemExit(f"{name}: {E.shape[1]} fingerprint bits, MoleculeNet has {X.shape[1]}")
        n_rows = len(pd.read_csv(EXT_ROWS / f"{name}_rows.csv"))
        if len(E) != n_rows:
            raise SystemExit(f"{name}: {len(E)} fingerprints for {n_rows} test rows")
        both = np.vstack([X[trv], E])
        ref = np.arange(len(trv))
        sim_ext = max_tanimoto(both, np.arange(len(trv), len(both)), ref)
        sim_mn = max_tanimoto(X, test, trv)
        out.append(dict(dataset=mn, outside=name, n_outside=len(E), median_outside=np.median(sim_ext),
                        n_moleculenet_test=len(test), median_moleculenet_test=np.median(sim_mn)))
        print(f"{name}: median {np.median(sim_ext):.4f} (n={len(E)}); {mn} test: "
              f"{np.median(sim_mn):.4f} (n={len(test)})")
    pd.DataFrame(out).to_csv(RES / "stats" / "external_similarity.csv", index=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("rows")
    p = sub.add_parser("predict")
    p.add_argument("--models", nargs="+", required=True, choices=["RF", "CatBoost", "TabPFN", "TabFM"])
    p.add_argument("--features", nargs="+", required=True, choices=FEATS)
    p.add_argument("--sets", nargs="+", required=True, choices=list(SETS))
    p.add_argument("--device", required=True)
    sub.add_parser("stats")
    sub.add_parser("similarity")
    args = ap.parse_args()
    if args.stage == "rows":
        return stage_rows()
    if args.stage == "predict":
        return stage_predict(args)
    if args.stage == "similarity":
        return stage_similarity()
    return stage_stats()


if __name__ == "__main__":
    sys.exit(main())
