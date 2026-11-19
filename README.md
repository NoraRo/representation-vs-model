# Representation and Model Contributions on Small Molecular Benchmarks

Code, split files, saved predictions and statistics for the paper *Representation and Model
Contributions on Small Molecular Benchmarks: Controlled Comparisons and External Evaluation*.
The supplementary material (S1–S7) is `docs/supplement.pdf`.

The saved predictions are enough to recompute every statistic without training a model
(step 3 below). Retraining is described in step 4.

## Contents

| Folder | What it holds |
|---|---|
| `src/` | All code (listed below) |
| `splits/` | `{dataset}_splits.json`: training, validation and test row indices of the published split (`det`) and the 10 seeded scaffold splits (`0`–`9`); `manifest.json`: sha256 of every rebuilt file, RDKit version and split sizes |
| `predictions/` | One folder per model and representation (e.g. `tabfm_maccs/`): `raw_results.csv` (one line per dataset, task and split, with the test score) and `{dataset}__{task}__{split}.npy` (test row index, prediction); `dgt_graph/seeds/` holds the DGT runs with seeds 1–3 |
| `results/stats/` | Every statistic in the paper and the supplement, as CSV |
| `results/identity/` | Test molecules that also occur in training (by InChIKey or tautomer), and duplicates inside each dataset |
| `results/external/` | Predictions for the external datasets (no labels) and `gate.csv`, the check that the refitted models reproduce the saved MoleculeNet predictions |
| `docs/` | `supplement.pdf` |

No data files and no model weights are included. The scripts download or rebuild them (steps 2
and 4).

## 1. Install

Python 3.13.

```
pip install -r requirements.txt
```

`requirements.txt` lists the exact version of every package. All models and analyses ran with
these versions.

## 2. Data

```
python src/download_data.py   # the 7 MoleculeNet CSVs from DeepChem -> data/raw/
python src/prepare.py         # rows, the 5 representations and the 11 splits
```

DeepChem's files have no version, so `download_data.py` checks each one against the sha256 of
the file used for the paper and stops if any differs. `prepare.py` writes `data/moleculenet/`, `features/` and `splits/`, and records the sha256 of
every file in `splits/manifest.json`. With RDKit 2026.03.5 the rebuilt files match the
manifest in this repository exactly.

The five representations: `rdkit2d` (all RDKit 2D descriptors), `ecfp4_2048` (binary Morgan
fingerprint, radius 2, 2,048 bits) and `maccs` (167 MACCS keys) are the three main inputs.
`morgan512` (the same fingerprint with 512 bits) and `rdkit2d_214` (`rdkit2d` without MolLogP,
MolMR and TPSA) are reported next to them.

## 3. Statistics from the saved predictions

```
python src/identity_audit.py    # results/identity/
python src/stats.py             # comparisons, factorial designs, sign tests, leak sensitivity
python src/extra_squares.py     # the 7 other factorial designs and the designs with the extra inputs
python src/variance_decomp.py   # two-way analysis of variance of the seeded scores
python src/similarity_check.py  # high- and low-similarity halves of each test set
python src/dgt_seed_tests.py    # TabFM against DGT for each DGT seed 0-3
python src/logp_recovery.py     # MolLogP rebuilt from the other descriptors
```

Each writes CSV files to `results/stats/`. Run them in this order; `stats.py` must run before
the rest, and `extra_squares.py` before `similarity_check.py`.

| File in `results/stats/` | Supplement |
|---|---|
| `parts_det.csv`, `parts_seeded.csv`, `extra_squares_*.csv` | S3 Results for all factorial designs |
| `logp_recovery.csv` | S3 (the check behind the descriptors without MolLogP, MolMR and TPSA) |
| `leak_sensitivity.csv` (with `results/identity/`) | S2 Molecular identity check across splits |
| `variance_*.csv` | S4 Full factorial analysis of variance |
| `similarity_*.csv` | S5 High- and low-similarity test molecules |
| `comparisons_*.csv`, `signtest.csv`, `dataset_scores.csv`, `dgt_seeds_*.csv` | S6 Model comparisons and DGT reproduction |
| `external_*.csv` | S1 Datasets and statistical tests, and the external-dataset results |

S7 (run time and memory) is in the supplement only.

## 4. Training

### Model weights

TabFM and TabPFN read their weights from `models/`:

| Model | Source (Hugging Face) | Revision | Put in | sha256 |
|---|---|---|---|---|
| TabFM | `google/tabfm-1.0.0-pytorch` | `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83` | `models/tabfm/classification/model.safetensors` | `928cb350becdc77cdb7a9e8c36deda88917bfd14a3091894a2dc516db58a2085` |
| | | | `models/tabfm/regression/model.safetensors` | `bd5a615b0322a8f04a895038de6df6fbd71430eca750e1d792f31048654674a9` |
| TabPFN-3 | `Prior-Labs/tabpfn_3` (gated: accept the licence first) | `24a16a89d245878b846555110985634aa2e656d7` | `models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt` | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |
| | | | `models/tabpfn/tabpfn-v3-regressor-v3_default.ckpt` | `311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301` |

```
hf download google/tabfm-1.0.0-pytorch --revision 77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83 --local-dir models/tabfm
hf download Prior-Labs/tabpfn_3 tabpfn-v3-classifier-v3_default.ckpt tabpfn-v3-regressor-v3_default.ckpt \
    --revision 24a16a89d245878b846555110985634aa2e656d7 --local-dir models/tabpfn
```

The weights have their own licences (TabPFN-3: non-commercial).

### Tabular models and Chemprop

```
OMP_NUM_THREADS=1 python src/train_tabular.py --models RF CatBoost --features rdkit2d rdkit2d_214 ecfp4_2048 morgan512 maccs --device cpu
OMP_NUM_THREADS=1 python src/train_tabular.py --models TabPFN --features rdkit2d ecfp4_2048 maccs --device mps
OMP_NUM_THREADS=1 python src/train_tabular.py --models TabPFN --features rdkit2d_214 morgan512 --device cuda
OMP_NUM_THREADS=1 python src/train_tabular.py --models TabFM TabFMplus --features rdkit2d --device cuda
OMP_NUM_THREADS=1 python src/train_tabular.py --models TabFM --features maccs rdkit2d_214 morgan512 --device cuda
OMP_NUM_THREADS=1 python src/train_chemprop.py --accelerator gpu
OMP_NUM_THREADS=1 python src/train_chemprop.py --accelerator gpu --refit
OMP_NUM_THREADS=1 python src/train_chemprop.py --accelerator gpu --descriptors rdkit2d
OMP_NUM_THREADS=1 python src/train_chemprop.py --accelerator gpu --descriptors rdkit2d_214
```

Both scripts write to `predictions/` and skip what is already there, so move or delete a
folder to retrain it. `--device` is `cpu`, `cuda` or `mps` (Apple GPU); Chemprop's `--accelerator` is `cpu` or `gpu`.

### External datasets

Download the two source files to `data/external/raw/`:

| Dataset | Source | Save as | sha256 |
|---|---|---|---|
| AqSolDBc | Recherche Data Gouv, doi:10.57745/CZVZIA, version 2.0, file `AqSolDBc.tab` in its original format | `llompart/AqSolDBc.csv` | `d41e512b43d66d838c8d3644df6ea99d37e0b60ee7705f0f1631da4d37011fc1` |
| T-data (RTlogD) | github.com/WangYitian123/RTlogD, commit `23b90f5615d38513a9547033952eb1ef062a32c3`, file `T-data_predictions(chembl32_logD).csv` | `rtlogd/T-data_chembl32.csv` | `34ab491bdfa77b174946bbd32071767d08671838d63029399f814c23566083b8` |

Then follow the commands at the top of `src/external_test.py` (stages `rows`, `predict`,
`stats`, `similarity`). The `stats` stage needs `stats.py` and `extra_squares.py` to have run.

### DGT

DGT was run with the code and settings of its release (github.com/zhangsy-ryan/DGT, commit
`2f5a4c3e`), seed 0, on the published split from `splits/`, on an NVIDIA A100. On ClinTox, rows
981 and 982 were left out of training because DGT cannot featurize their dative bonds. Its
test predictions are in `predictions/dgt_graph/`; `other_runs.csv` there lists the scores of
the further runs (seeds 1–3, and the settings of DGT's Zenodo release); the seeds 1–3 predictions are in `predictions/dgt_graph/seeds/`.

## Settings

Model settings are in the code: the RF and CatBoost grids in `src/models.py`, TabPFN, TabFM and
TabFM+ in `src/train_tabular.py`, Chemprop in `src/train_chemprop.py`, and feature cleaning in
`src/clean.py`. What the code alone does not show:

- **Rows used for training.** The RF and CatBoost each pick one of their 4 settings on the
  validation rows and are then refit on training + validation rows; TabFM and TabPFN read them as
  context.
- **Chemprop settings.** Chemprop is trained with 4 settings (batch normalisation and dropout
  each on or off, everything else at Chemprop's defaults). Each keeps its best epoch on the
  validation rows, and the setting with the best validation score predicts the test rows. The
  refit variant trains that setting again on training + validation rows. `best_epochs.csv` in
  `predictions/chemprop_graph/` lists every setting's best epoch and validation score.
- **Hardware.** RF and CatBoost ran on a CPU; Chemprop, TabFM, TabFM+ and TabPFN on GPUs (an Apple
  GPU and an NVIDIA A100); DGT on an NVIDIA A100. The foundation models can give slightly different
  predictions on different devices.
- **TabFM on the Apple GPU** returned the same value for most test molecules on Lipophilicity,
  without an error. Those runs used the CPU or the A100. `train_tabular.py` stops when a model
  gives far fewer distinct predictions than there are distinct test rows.
- **TabPFN on ECFP4.** ECFP4 keeps more than 2,000 columns after cleaning, so TabPFN's
  feature-count check is turned off for it.
- **TabFM+** is the package's `.ensemble()` preset: 32 members, added features, stacking and
  calibration.
- **Chemprop + RDKit descriptors.** The descriptors are cleaned as for the other models, then
  mapped to [0, 1] by a quantile transform fitted on the training rows.

## Code

| File | Job |
|---|---|
| `paths.py` | Every folder the scripts use |
| `download_data.py`, `datasets.py` | Download and load the MoleculeNet datasets |
| `prepare.py`, `featurize.py` | Rows, representations and splits |
| `clean.py` | Feature cleaning on the training rows |
| `models.py` | Metrics and the random forest |
| `train_tabular.py`, `train_chemprop.py` | Training and test predictions |
| `identity_audit.py` | The same molecule in training and test |
| `stats.py`, `extra_squares.py`, `variance_decomp.py`, `similarity_check.py`, `dgt_seed_tests.py`, `logp_recovery.py` | Statistics |
| `external_test.py` | External datasets |

## Licence

The code is released under the MIT licence (`LICENSE`). The datasets and model weights keep the
licences of their sources.
