# Leakage-Free Is Not Confounder-Free

[![arXiv](https://img.shields.io/badge/arXiv-2611.27585-b31b1b.svg)](https://arxiv.org/abs/2611.27585)

Code and stored results for the paper *Leakage-Free Is Not Confounder-Free: What a
Patient-Level Split Cannot Certify About a Medical Imaging Benchmark*
([arXiv:2611.27585](https://arxiv.org/abs/2611.27585)).

The study audits two public breast-thermography benchmarks. Splitting by patient removes
leakage between partitions; it does not remove a confounder that travels with the label. On
DMR-IR the acquisition date alone predicts malignancy almost perfectly, the date is legible
in the pixels, and what an image model appears to know has to be re-read once the date is
controlled. This repository contains the analysis code, the exact experiment
configurations, the initial model weights, and the statistics behind every reported
number; the patient-level predictions are regenerated from the public datasets
(Section 1).

**The patient is the experimental unit.** Every view and every frame of a patient stays in
one partition, and frame predictions are averaged within a patient before any interval is
computed.

## 1. What is distributed, and what is not

**Neither the datasets nor any per-patient record derived from them is redistributed.**
DMR-IR is distributed by its authors on registration and without a licence, and its
Hugging Face mirror states none; the second cohort is released under CC BY-NC 3.0. A
manifest, a fold assignment or a stored prediction carries, for each patient, an
identifier, a diagnosis and often a date and an age, so none of them is included here.

What is included:

* the code, with the exact command lines of the batch runs (`jobs/`);
* for every run, its configuration, input hashes, metrics, bootstrap intervals and fold
  log, and every aggregate statistic the paper reports (`results/`);
* the initial weights every training run starts from -- the pretrained encoders with the
  seeded classifier head -- published as assets of the
  [`weights-v1` release](https://github.com/ignaziogallo/breast_thermography_reliable_ai/releases/tag/weights-v1)
  (Section 8);
* the scripts that rebuild every per-patient file from your own copy of the datasets: the
  manifests and folds on a CPU (Section 5), the predictions on a GPU (Section 8), after
  which the analyses of Section 7 can be recomputed.

Three levels of verification follow from that:

1. **Check the reported numbers against the stored statistics** (seconds, nothing to
   download): `python scripts/36_consistency_sweep.py --strict`.
2. **Re-run the experiments** from the datasets, which needs a GPU (Section 8) and
   regenerates the per-patient predictions. They are not distributed, so this level comes
   before the next one.
3. **Recompute the analyses** once the manifests (Section 5) and the regenerated
   predictions exist on your machine: `python scripts/reproduce_paper.py`. CPU only.

## 2. Quick check, and the full recomputation

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # analysis only: no torch, no GPU
python scripts/36_consistency_sweep.py --strict   # reported numbers vs stored statistics
python scripts/reproduce_paper.py                 # needs manifests and regenerated predictions
```

The last step of `reproduce_paper.py` is the same verification gate. On success it prints:

```
18/18 steps ok in N.N min
Every number in the manuscript still matches the files these steps wrote.
```

A few minutes on a laptop CPU; the bootstrap and permutation loops dominate. No step needs
a network connection. `--list` prints the plan without running it, `--only 34 39` runs
selected steps, and each step's output is kept under `results/reproduce_logs/`.

Steps run **in place**, overwriting the statistics under `results/` with freshly computed
ones. That is the point: the stored tree and the recomputed tree should agree, and the
consistency check is what says so. Copy `results/` first if you want a before/after diff.

The verification gate checks a hand-written map from each claim of the manuscript to the file, row and column
of `results/` that produces it. When the manuscript source is present under `paper/`
(`main.tex`, as distributed by arXiv, plus `supplementary.tex` if a version has one) it
additionally looks up every numeric literal it prints; without it the check runs the map
alone. Cohort counts are recomputed from the manifests
and are reported as not distributed when those are absent.

The leakage regression tests are part of the evidence. They run on whatever prediction
files are present and skip the ones that are not:

```bash
python -m pytest -q
```

`tests/test_no_prediction_leakage.py` re-checks every `results/experiment_*/predictions.csv`
against the fold definition that run recorded in its own `config.json`: a scored patient may
never appear in the train or validation split of its own fold.

Scripts are run from the repository root, as above. The repository is not an installable
package.

## 3. Repository layout

```
.
├── README.md
├── LICENSE                    Apache License 2.0
├── requirements.txt           analysis only, CPU
├── requirements-training.txt  adds torch, torchvision, timm
├── environment.yml            conda equivalent of requirements-training.txt
├── pyproject.toml             pytest and lint configuration
├── scripts/                   every entry point (Section 4)
├── src/                       the library those scripts import
├── configs/                   the declared design of the second cohort's arms
├── tests/                     split-integrity and prediction-leakage regression tests
├── jobs/                      the GPU command lines exactly as they were run
├── results/                   per-run configurations, metrics and aggregate statistics
├── reports/                   the second cohort's dataset audit
└── data/                      not tracked; link the two datasets here to re-run training
```

`results/` is the substance. Every run directory is named
`experiment_<cohort>_<id>_<arm>` and holds:

| file | contents |
|---|---|
| `config.json` | the full configuration, the input file hashes, the quality-gate outcome, the patients dropped and why |
| `predictions.csv` | one row per patient per repeat: `patient_id, repeat, fold, y_true, y_prob, model` (written by a run; not distributed) |
| `metrics.csv`, `metrics_per_repeat.csv` | point estimates |
| `bootstrap_ci.csv` | patient-level bootstrap percentiles |
| `fold_log.csv` | per fold, the selected epoch and its validation AUROC |
| `contamination.csv` | where a split-unit arm was run, the fraction of test frames whose patient is also in train |

`config.json` is the authoritative record of what a run did; the YAML files in `configs/`
are the design as it was declared in advance, and nothing reads them at run time.
`_FAILED_QUALITY_GATE` in a directory name marks a configuration quarantined by the
pre-specified prediction-spread gate and reported in the paper as a failed configuration,
not a result.

The library is organised by what a module is responsible for:

| package | responsibility |
|---|---|
| `src/data` | manifests, fold construction, the DMR-IR tables (`dmr.py`, no torch) and the torch datasets (`dmr_datasets.py`, `dataset.py`) |
| `src/models` | the pretrained encoder, the multi-view classifier, the tabular baselines |
| `src/training` | the fold loop for image models (`trainer.py`) and for tabular models (`tabular_cv.py`) |
| `src/evaluation` | metrics, patient-level bootstrap, run summaries, the quality gate, and the rank-based resampling tests every shortcut-usage column shares (`resampling.py`) |
| `src/confounding` | discrimination restricted to confounder-comparable pairs |
| `src/matching` | confounder-controlled cohort construction, from metadata only |
| `src/preprocessing` | augmentation and the bilateral-asymmetry descriptors |
| `src/utils` | hashing and guarded reads, seeding, figure conventions |

## 4. Scripts at a glance

Scripts keep the number they were written under, because job files, logs, stored
configurations and the paper's appendices cite them by that name. By role:

| stage | script | what it does |
|---|---|---|
| **Dataset preparation** (CPU) | `00_audit_dataset.py` | build the second cohort's manifests and audit the dataset |
| | `01_create_manifest.py` | the same manifests without the audit, e.g. over the radiometric tree |
| | `02_create_patient_splits.py` | build and store the patient-level folds |
| | `17_download_dmr_ir.py` | download the DMR-IR mirror into a patient-organised tree |
| | `18_audit_dmr_ir.py` | audit the downloaded DMR-IR cohort |
| | `19_build_dmr_manifest.py` | DMR-IR manifests, with the cohort decisions of Section 5.1 |
| | `20_extract_radiometric.py` | recover absolute temperature from the second cohort's FLIR JPEGs |
| | `21_audit_dmr_design.py` | which patient-level designs DMR-IR supports |
| | `22_build_dmr_view_table.py` | one row per patient with a column per view |
| | `29_build_matched_cohorts.py` | the confounder-controlled cohort ladder and its size controls |
| **Training** (GPU) | `03_leakage_experiment.py` | all-frames arms; patient-level against image-level split |
| | `04_train_single_view.py` | single-view arms on either cohort, frozen or fine-tuned |
| | `15_train_breast_level.py` | breast-level model from the oblique views |
| | `24_epoch_control.py` | acquisition-era decoders within a diagnosis; score-versus-date analysis |
| | `27_transfer_eval.py` | train on one cohort, evaluate on the other |
| **Non-image channels** (CPU) | `09_metadata_baseline.py` | clinical covariates alone |
| | `09b_asymmetry_baseline.py` | bilateral thermal asymmetry descriptors |
| | `25_dmr_metadata_leakage.py` | where a metadata-only DMR-IR score comes from |
| | `26_dmr_text_probe.py` | the clinical text prompt, clause by clause |
| | `16_incremental_value.py` | what the image adds once the confounder is known |
| **Analyses of stored predictions** (CPU) | `14_compare_models.py` | paired comparison of two arms on identical folds |
| | `30_conditional_evaluation.py` | discrimination within confounder-comparable pairs |
| | `32_matched_ladder_report.py` | each matched rung against its size controls |
| | `34_exploitation_interval.py` | does an arm's score track the acquisition era (DMR-IR) |
| | `35_caliper_multiplicity.py` | Holm correction over the caliper grid |
| | `37_verification_confounder.py` | the same shortcut-usage statistic on the second cohort, for age |
| | `38_matching_balance_report.py` | standardised mean differences of every rung |
| | `39_signature_source.py` | where in the image the era is written |
| | `40_finetuned_era_probe.py` | era legibility in a fine-tuned representation |
| | `41_cross_label_support.py` | cross-label acquisition support against a permutation null |
| | `42_age_direction.py` | age scored directly on both cohorts |
| | `43_selection_diagnostics.py` | why the epoch-selecting run was discarded |
| **Tables, figures, verification** | `28_generate_dmr_tables.py` | prints the LaTeX tables |
| | `31_figure_confounder_control.py` | the matched-ladder and caliper figures |
| | `33_paper_figures.py` | the remaining figures |
| | `36_consistency_sweep.py` | checks every reported number against `results/` |
| | `reproduce_paper.py` | runs the analysis path end to end, then the check above |

Every script documents itself: `python scripts/<name>.py --help` prints its rationale, its
options and an example.

A few docstrings, stored `config.json` fields and job files still name experiments by the
label of the plan they were run under: *E7* is the metadata baseline, *E7b* the asymmetry
baseline, *E1* the single-view image model. Job-file prefixes mean: `e1_allframes_*` the
matched ladder on the all-frames arms, `f2_*` the age-band probes on the second cohort,
`f3_e5*` the fine-tuned era probe and its frozen control, `f3_o1_*` the signature-source
extraction.

## 5. Dataset preparation

Neither dataset is included. Both are public; obtain them from their distributors and place
them where the code expects. All paths below are relative to the repository root, and every
script takes `--data-root`, so a link to a dataset elsewhere works:

```bash
ln -s /path/to/datasets/DMR-IR              data/DMR-IR
ln -s /path/to/datasets/breast-thermography data/Breast-Thermal-Image
```

Manifests store image paths **relative to `--data-root`**, so a manifest built on one
machine stays valid on another.

### 5.1 DMR-IR (primary cohort)

The Database for Mastology Research with Infrared image, Silva et al., *A new database for
breast research with infrared image*, J. Medical Imaging and Health Informatics 4(1),
92–100, 2014. The code consumes the Hugging Face mirror `SemilleroCV/DMR-IR`, whose value
here is that it carries a `patient_id` column (the flattened mirrors do not, and cannot be
split at patient level) and ships raw-temperature TIFFs rather than an 8-bit render.

```bash
pip install datasets huggingface_hub     # needed by this script only
python scripts/17_download_dmr_ir.py --out data/DMR-IR
```

writes, with image bytes copied verbatim:

```
data/DMR-IR/
├── images/<patient_id>/<record>_<protocol>_<view>_<seq>.tiff
├── masks/<patient_id>/<...>.png          breast segmentation masks, where present
├── metadata.csv                          one row per image, paths relative to data/DMR-IR
├── download_report.md                    counts, label consistency, split-overlap audit
└── dataset_info.json
```

Then the audit and the manifests:

```bash
python scripts/18_audit_dmr_ir.py                     # roles, ages, duplicate pixels, protocols
python scripts/19_build_dmr_manifest.py --root data/DMR-IR
python scripts/02_create_patient_splits.py \
    --manifest results/dmr_patient_manifest.csv \
    --output   results/dmr_patient_splits.csv --folds 5 --repeats 5
python scripts/22_build_dmr_view_table.py --views frontal \
    --output results/dmr_view_table_frontal.csv
python scripts/21_audit_dmr_design.py                 # which designs the cohort supports
```

**Fields the analyses need**, all present in the released metadata and carried into
`results/dmr_patient_manifest.csv`: `patient_id` (grouping variable),
`patient_binary_label` (0 benign / 1 malignant, derived from `label_name`), `visit_date`
(the acquisition date, and the paper's dominant confounder), `age_at_visit`,
`age_current`, `body_temperature`, `menarche`, `biopsy`, `mammography`, `mastectomy`,
`role`, `protocol` and `view` (per image, in `results/dmr_image_manifest.csv`).

**Cohort decisions, each made by `scripts/19_build_dmr_manifest.py` and each reversible by
a flag**, taking the released pool of 287 patients to the 236 the paper reports:

* **Healthy volunteers are excluded** (`--keep-volunteers` to keep them). DMR-IR mixes
  volunteers with mastology-clinic patients, 3.1% vs 40.7% malignant, so volunteer status
  alone scores AUROC 0.585: a recruitment-channel shortcut. Dropping 32 volunteers leaves a
  better-balanced cohort than keeping them.
* **Patients entered twice are collapsed** (`--keep-duplicate-patients`). Patients sharing
  byte-identical images are one person under two ids; left in place they sit in train and
  test simultaneously. Grouping is by shared image hash, cached in
  `results/dmr_image_hashes.csv`.
* **`--min-images`** (default 1) drops patients below a frame count.
* **The published `train`/`validation`/`test` split is ignored.** It is kept in
  `metadata.csv` as `hf_split` for provenance only; it is not patient-disjoint, which is
  the mechanism behind the >0.97 accuracies in this literature.

**Static and dynamic frames.** DMR-IR records five static view slots plus a dynamic
recovery sequence per visit: medians 5 and 22 frames per patient, 6051 thermograms over the
236 patients. `--protocol all` (an *all-frames* arm) reads every frame of a visit;
`--views frontal` on the static view table (a *frontal* arm) reads the one static anterior
image. Five of the 236 patients hold no static frontal view, which is where the 231 of the
frontal arms comes from.

### 5.2 Second cohort

Rodríguez-Guerrero et al., *Dataset of breast thermography images for the detection of
benign and malignant masses*, Data in Brief 54:110503, 2024. Obtain it from that
publication and check its licence terms before redistributing anything. 119 patients, 357
thermograms, one centre and one camera.

```
data/Breast-Thermal-Image/
├── Benign/IIR0001/IIR0001_anterior.jpg
│                  IIR0001_oblleft.jpg
│                  IIR0001_oblright.jpg
├── Malignant/IIR0019/...
└── Diagnostics.xlsx
```

```bash
python scripts/00_audit_dataset.py --data-root data/Breast-Thermal-Image
python scripts/02_create_patient_splits.py --folds 5 --repeats 5
```

`Diagnostics.xlsx` supplies per-breast labels (`N` / `PB` / `PM`) and the covariates
`age`, `weight`, `height`, `temperature`; BMI is computed. The patient-level label rule is
**malignant if at least one breast is `PM`, otherwise non-malignant**, and it is recorded
in `results/patient_manifest.csv` as `patient_binary_label`. In `IIRxxxx_oblleft.jpg`,
*left* denotes the patient's anatomical left breast, so the breast-level `Left`/`Right`
labels map onto `oblleft`/`oblright`; the `anterior` view contains both breasts and is not
assigned to a side. No subject is bilaterally normal, so the contrast is malignancy versus
benign breast disease in referred women, and malignancy is exclusively unilateral.

**The render is auto-ranged; the files are radiometric.** The distributed greyscale JPEGs
are stretched to each frame's own dynamic range (351/357 saturate at 255), so the same grey
level means temperatures up to 11.6 °C apart in different files. The raw 16-bit thermal
array and the camera calibration are still in the FLIR APP1 segments of all 357 files, and

```bash
python scripts/20_extract_radiometric.py \
    --data-root data/Breast-Thermal-Image \
    --out       data/Breast-Thermal-Image-Radiometric
```

recovers absolute temperature into float32 TIFFs in °C, mirroring the input tree, plus
`radiometric_manifest.csv`. Per-image standardisation is therefore a modelling choice, not
a constraint. DMR-IR's TIFFs are already temperatures.

**Never pool the two cohorts into one cross-validation pool.** Camera, country, protocol
and label semantics differ, so cohort membership would itself be a near-perfect confounder.
Train on one, evaluate on the other (`scripts/27_transfer_eval.py`).

## 6. Paper terms in the code

| Paper term | In the code |
|---|---|
| ConvNeXt-T | `--encoder convnext_tiny` |
| Swin-T | `--encoder swin_tiny_patch4_window7_224` |
| ResNet-50 | `--encoder resnet50` |
| frozen probe | `--freeze-backbone` (a linear head on frozen ImageNet features) |
| fine-tuned | no `--freeze-backbone`; `--backbone-lr-scale 0.01 --warmup-epochs 3` |
| frontal | `scripts/04_train_single_view.py --view frontal` on `results/dmr_view_table_frontal.csv` |
| all frames | `scripts/03_leakage_experiment.py --protocol all --arms patient_level` |
| DMR-IR | `--data-root data/DMR-IR`; run dirs `experiment_dmr_*`; ids `DMR####` |
| second cohort | `--data-root data/Breast-Thermal-Image`; run dirs `experiment_ver_*` and `experiment_0*`; ids `IIR####`. `--dataset verification` selects it in `scripts/29`, and the zero-shot transfer runs are `experiment_transfer_dmr_to_cali_*`, named after the city the published dataset was collected in |
| patient-disjoint split | `results/*_patient_splits.csv`, built by `scripts/02_create_patient_splits.py` |
| image-level split (the contaminated reference) | `scripts/03_leakage_experiment.py --arms image_level` |
| matched cohort / size control / null control | rungs under `results/matching/<cohort>/`, built by `scripts/29_build_matched_cohorts.py` |

## 7. Reproducing the paper results

Every row below reads only manifests and predictions -- no pixels, no GPU -- once the
predictions have been regenerated (Section 8). Step ids are the ones
`scripts/reproduce_paper.py --list` prints, so `--only 34` runs just that row. Table,
figure and appendix numbers are those of the arXiv preprint (v1); the result named in each
row is the stable reference if a later version renumbers them.

| Paper result | Step / command | Inputs | Output |
|---|---|---|---|
| **Tab. 1**, every information channel | `scripts/28_generate_dmr_tables.py --table channels` | `experiment_dmr_04_metadata_leakage/`, `_05_text_probe/`, `_01_leakage_combined/`, `_11_*/`, `_02_frontal_static/` | LaTeX on stdout |
| **Tab. 2** + **App. C** (Tab. 5–6), shortcut exploitation | `34` | nine arms' `predictions.csv`, `dmr_patient_manifest.csv` | `confounders/exploitation_interval.csv`, `exploitation_view_contrast.csv` |
| **Tab. 3**, incremental value | `scripts/28_generate_dmr_tables.py --table incremental` | `experiment_dmr_08_epoch_frontal/paired.csv`, `_08_epoch_allframes/paired.csv`, `_13_incremental_swin_allframes/paired.csv` | LaTeX on stdout |
| **Fig. 1**, the audit diagram | `33` | `experiment_dmr_04_*`, `_12_epoch_convnext_tiny_within_benign/`, `exploitation_interval.csv`, `_13_*/paired.csv`, `_10_matched_overlap/`, `dmr_patient_manifest.csv` | `paper/figures/fig_audit_questions.pdf` |
| **Fig. 2**, era decodability vs discrimination | `33` | `experiment_dmr_11_*/`, `experiment_dmr_12_*/` | `paper/figures/fig_shortcut_tradeoff.pdf` |
| **Fig. 3** + **App. F–G** (Tab. 9–10), matched ladder | `32`, then `31` | `matching/{dmr,verification}/ladder.csv`, `experiment_dmr_10_matched_*/`, `_14_matched_allframes_*/`, `experiment_ver_10_matched_*/` | `matching/*/rung_vs_control_*.csv`, `paper/figures/fig_matching_ladder.pdf` |
| **Fig. 4** + **App. H** (Tab. 11), conditional discrimination | `30a`, `30c`, then `31` | the five DMR-IR arms and three cohort-2 arms' `predictions.csv`, both manifests | `confounders/{dmr_session,verification}_conditional/caliper_series.csv`, `paper/figures/fig_conditional_caliper.pdf` |
| **Tab. 11** daggers, Holm over 25 cells | `30b`, then `35` | `confounders/dmr_encoders_conditional/patient_scores.csv` | `confounders/caliper_multiplicity.csv` |
| **Fig. 5** + **App. I** (Tab. 12), cross-label acquisition support | `41` | `dmr_patient_manifest.csv` only | `confounders/cross_label_support.csv`, `paper/figures/fig_cross_label_support.pdf` |
| **App. B** (Tab. 4), encoder robustness | `scripts/28_generate_dmr_tables.py --table encoders` | `experiment_dmr_11_encoder_*/`, `_11_allframes_*/`, `_12_epoch_*/` | LaTeX on stdout |
| **App. D** (Tab. 7, Fig. 6), split protocol × unit of analysis | `33` | `experiment_dmr_01_leakage_combined/metrics.csv`, `contamination.csv` | `paper/figures/fig_split_leakage.pdf` |
| **App. E** (Tab. 8), the *alone* rows | — (read from the same `paired.csv` as Tab. 3) | `experiment_dmr_08_*/paired.csv`, `_13_*/paired.csv` | — |
| **App. J** (Tab. 13), metadata and text attribution | `scripts/28_generate_dmr_tables.py --table attribution` | `experiment_dmr_04_metadata_leakage/`, `_05_text_probe/` | LaTeX on stdout |
| **App. K**, cross-cohort transfer | `scripts/28_generate_dmr_tables.py --table transfer` | `experiment_transfer_dmr_to_cali_{fixed,perimage}/metrics.csv` | LaTeX on stdout |
| **App. L**, model-selection failure | `43` | `experiment_02_single_view_earlystop_auroc/{fold_log,predictions}.csv` | `confounders/selection_diagnostics.csv` |
| **App. M**, age on DMR-IR | `42` | both patient manifests | `confounders/age_direction.csv` |
| **App. N** (Tab. 14), shortcut usage on the second cohort | `37` | `experiment_ver_11_*/`, `experiment_02_single_view_*/`, `patient_manifest.csv` | `confounders/verification_exploitation.csv` |
| **App. O** (Tab. 15–16), matching diagnostics | `38` | `matching/{dmr,verification}/*/balance.csv` | `matching/smd_all_rungs.csv` |
| **App. Q** (Tab. 17), where the era signature lives | `39` | `dmr_signature_features.csv` (descriptors extracted from your copy of DMR-IR), `dmr_patient_manifest.csv` | `confounders/signature_source.csv` |
| **App. R** (Tab. 18), era legibility under fine-tuning | `40a`, `40b` | the penultimate-layer activations of `experiment_dmr_15_features_*/` (written with `--save-features`) | `confounders/finetuned_era_probe_{convnext,swin}.csv` |
| **every number, checked** | `36` | all of `results/` (and the manuscript source, when present) | `results/paper_consistency.md` |

Two entries need a word, because their per-patient inputs are not distributed either.
`39` reads frame descriptors cached in `results/dmr_signature_features.csv`, one row per
patient; build the cache from your copy of DMR-IR first with
`python scripts/39_signature_source.py --extract --data-root data/DMR-IR` (CPU, about two
minutes). `40` refits its probe inside each fold from penultimate-layer activations, which
the four `experiment_dmr_15_features_*` runs write when re-run with `--save-features`
(`jobs/f3_e5_finetuned_features.txt`, `jobs/f3_e5b_frozen_control.txt`); the activations
(~20 MB per arm) and the per-fold probe predictions are written to the run directories.

The shortcut-usage statistics of `34`, `37` and `40` share one implementation of the
bootstrap interval and the permutation test (`src/evaluation/resampling.py`), so the
columns they produce on the two cohorts and on the fine-tuned representation mean exactly
the same thing.

## 8. Re-running training

Training needs `pip install -r requirements-training.txt`, a GPU, both datasets prepared as
in Section 5, and the initial weights below. `mkdir -p logs` first: the job files redirect
there.

### Initial weights

Every training script starts each fold from a cached state, `results/pretrained/<encoder>.pt`:
the ImageNet-pretrained encoder from `timm` together with its randomly initialised
classifier head, cached on first use so that every fold of every run starts from the same
head. The files the paper's runs started from are published as assets of the
[`weights-v1` release](https://github.com/ignaziogallo/breast_thermography_reliable_ai/releases/tag/weights-v1),
with their SHA-256 in `SHA256SUMS`. Download them before training:

```bash
mkdir -p results/pretrained && cd results/pretrained
for f in convnext_tiny.pt swin_tiny_patch4_window7_224.pt resnet50.pt SHA256SUMS; do
  curl -fLO "https://github.com/ignaziogallo/breast_thermography_reliable_ai/releases/download/weights-v1/$f"
done
sha256sum -c SHA256SUMS          # macOS: shasum -a 256 -c SHA256SUMS
cd ../..
```

or, with the GitHub CLI,
`gh release download weights-v1 --repo ignaziogallo/breast_thermography_reliable_ai --dir results/pretrained`.

With these files a fresh clone reproduces the stored statistics: the frontal ConvNeXt-T
frozen probe (`experiment_dmr_11_encoder_convnext_tiny_probe`) returned a pooled AUROC of
0.8108 against the stored 0.8108, with current library versions. Without them `timm`
downloads the weights afresh and a new head is drawn; the same run then returned 0.8216 --
inside the stored interval [0.755, 0.864], but not the same predictions. Frozen probes are
the most sensitive to this, since the head is the only part they train.

### Command lines

`jobs/*.txt` hold the command lines of the GPU batch runs exactly as they were run, one per
line, grouped by the GPU they were dispatched to. The four main imaging arms:

```bash
# all frames, ConvNeXt-T frozen probe, both split units in one run (~1 h on one A100)
python scripts/03_leakage_experiment.py --data-root data/DMR-IR --protocol all \
    --arms patient_level image_level --freeze-backbone --epochs 15 --batch-size 32 \
    --output-dir results/experiment_dmr_01_leakage_combined

# all frames, Swin-T frozen probe (~1 h)
python scripts/03_leakage_experiment.py --data-root data/DMR-IR --protocol all \
    --arms patient_level --encoder swin_tiny_patch4_window7_224 --freeze-backbone \
    --epochs 15 --batch-size 32 \
    --output-dir results/experiment_dmr_11_allframes_swin_tiny_patch4_window7_224

# frontal, ConvNeXt-T frozen probe (~20 min)
python scripts/04_train_single_view.py --data-root data/DMR-IR \
    --manifest results/dmr_view_table_frontal.csv --splits results/dmr_patient_splits.csv \
    --view frontal --encoder convnext_tiny --freeze-backbone --epochs 60 --batch-size 16 \
    --output-dir results/experiment_dmr_11_encoder_convnext_tiny_probe

# frontal, ConvNeXt-T fine-tuned (~1 h)
python scripts/04_train_single_view.py --data-root data/DMR-IR \
    --manifest results/dmr_view_table_frontal.csv --splits results/dmr_patient_splits.csv \
    --view frontal --encoder convnext_tiny --backbone-lr-scale 0.01 --warmup-epochs 3 \
    --no-early-stopping --epochs 30 --batch-size 16 \
    --output-dir results/experiment_dmr_11_encoder_convnext_tiny_finetune
```

The era decoders (`scripts/24_epoch_control.py`, one run per backbone) and the two matched
ladders (`scripts/29_build_matched_cohorts.py` to build the cohorts on CPU, then one
`scripts/04` run per rung and per size control) are the expensive parts: the DMR-IR ladder
alone is 22 frontal runs and the all-frames ladder 18 more. Rung splits are **not** built
by the training scripts: they are read from `results/matching/<cohort>/<rung>/splits.csv`,
which hold one row per patient and are therefore not distributed. Build them before the
ladder runs with `python scripts/29_build_matched_cohorts.py --dataset dmr` and
`--dataset verification` (seconds, metadata only, seed 42; no model score enters cohort
construction). The second cohort's age-legibility probes (`jobs/f2_ver_age_legibility.txt`)
likewise need their cohorts built first with
`python scripts/37_verification_confounder.py --build-probe-cohorts`. Every run's
`config.json` records the SHA-256 of the manifest and splits file it consumed, so a rebuilt
cohort can be checked against the one that was actually run before any GPU time is spent.

`scripts/16_incremental_value.py` also writes a run directory rather than only reading one:
it re-fits each arm's stored patient score as a single-feature logistic model on the same
folds, and combines it with the confounder. It is listed here rather than in Section 7 for
that reason.

## 9. Expected outputs

* **Predictions** -- `results/experiment_*/predictions.csv`, one row per patient per repeat.
  Written by the training scripts and by `scripts/16`.
* **Statistics** -- `results/confounders/*.csv` (exploitation, conditional calipers,
  multiplicity, cross-label support, signature source, era probes, age direction, selection
  diagnostics) and `results/matching/*/rung_vs_control_*.csv`, `smd_all_rungs.csv`.
* **Tables** -- printed as LaTeX on stdout by `scripts/28_generate_dmr_tables.py`. Nothing
  writes into a manuscript, by design: a number cannot reach the paper without existing
  under `results/` first.
* **Figures** -- `paper/figures/fig_*.pdf` and `.png`, written by `scripts/31`, `33`
  and `41`.
* **Verification** -- `results/paper_consistency.md`, the anchor-by-anchor report.
* **Logs** -- `results/reproduce_logs/<step>.log` when run through `reproduce_paper.py`.

## 10. Reproducibility notes

* **Seeds.** Every script defaults to `--seed 42` and seeds Python, NumPy and torch through
  `src/utils/seed.py`. The five cross-validation repeats use seeds 42–46, recorded in the
  `seed` column of every splits file.
* **Fold persistence.** Folds are built once by `scripts/02_create_patient_splits.py` and
  stored; no experiment re-derives them. Each run records the SHA-256 of the manifest and
  splits file it consumed in its `config.json`, and
  `tests/test_no_prediction_leakage.py` checks each run against *those* files rather than
  against a fixed pair.
* **Patient grouping.** Five repetitions of stratified group five-fold cross-validation
  with `patient_id` as the grouping variable. Every frame of a patient stays in one
  partition, and frame predictions are averaged within patient before any interval is
  computed.
* **Bootstrap.** 2000 resamples at patient level, percentile intervals. Paired
  comparisons resample patients once and recompute both arms on the same draw, on identical
  folds.
* **Permutations.** 10 000 relabellings, two-sided, reported as `(hits + 1) / (n + 1)`.
  The caliper grid is Holm-corrected over a family of 25 cells (5 imaging arms × 5
  calipers) declared in advance; the confounder's own row is excluded from the family
  because it is required to fail.
* **Quality gate.** A configuration is quarantined if more than 5 of its 25 folds emit
  predictions with standard deviation below 0.05. The outcome is recorded in
  `config.json` under `quality_gate` and in the directory name. Quarantined runs are
  reported in the paper as failed configurations, never as results.
* **Epoch policy.** Frozen linear probes restore the epoch of best validation AUROC and use
  patience. Every fine-tuned arm and every rung of both matched ladders trains to a fixed
  budget with no validation-based selection (`early_stopping: false` in `config.json`).
  Ladder rungs are compared only against each other and their own size controls.
* **Numerical determinism.** The analysis path is deterministic given its seeds and
  reproduces bit-identically across runs on one machine. Small last-digit differences
  across platforms, BLAS builds or library versions are expected -- the stored statistics
  were produced on Linux -- and the consistency check's tolerance is 5e-4, half a unit in
  the last decimal the manuscript prints. The training path reproduces the stored
  statistics when it starts from the initial weights of Section 8.
* **Platforms.** Paths in a `config.json` are recorded with forward slashes and text files
  are checked out with LF line endings on every platform, so a run recorded on one system
  can be matched to its inputs on another.

## 11. Licence

The code is released under the Apache License, Version 2.0 (see `LICENSE`); `CITATION.cff`
says how to cite it, and the paper is
[arXiv:2611.27585](https://arxiv.org/abs/2611.27585). The two datasets are not part of this repository and keep the terms
of their own distributors.

## 12. Development

```bash
python -m pytest -q          # unit tests and the leakage regression tests
ruff check scripts src tests # lint (configuration in pyproject.toml)
```

This repository is the code release that accompanies the paper: the code, configurations,
job files, tests and aggregate results. It was assembled from the authors' development
repository from an explicit allow-list, so that nothing reaches it by default, and it
withholds every table that holds one row per patient.
