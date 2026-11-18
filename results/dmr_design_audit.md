# DMR-IR design audit

Sources: `results/dmr_patient_manifest.csv`, `results/dmr_image_manifest.csv`.
Reads manifests only -- no pixels -- so it runs without the dataset mounted.

## Cohort

- patients: **236** (140 benign / 96 malignant)
- roles: {'Patient': np.int64(236)}

Labels are **patient-level only**. DMR-IR records no per-breast diagnosis, so a
breast-level task and any left/right supervised asymmetry model are out of scope;
laterality can enter only as an unsupervised feature of a patient-level model.

## Acquisition epoch

`visit_date` alone scores **AUROC 0.981**.

| year | label=0 | label=1 |
|---|---|---|
| 2012 | 65 | 0 |
| 2013 | 71 | 10 |
| 2014 | 2 | 13 |
| 2015 | 2 | 16 |
| 2018 | 0 | 5 |
| 2019 | 0 | 39 |
| 2020 | 0 | 13 |

Restricting to the overlap window shrinks the cohort without clearing the date:

| window | n | benign | malignant | date AUROC |
|---|---|---|---|---|
| full cohort | 236 | 140 | 96 | 0.981 |
| 2013-2015 | 114 | 75 | 39 | 0.914 |
| 2013H2-2015 | 68 | 29 | 39 | 0.778 |

- distinct acquisition dates: **86** (median 2 patients each)
- label-pure sessions: **83/86**, covering **226/236** patients
- mixed sessions: 3, covering 10 patients

Session membership is therefore an almost perfect proxy for the label. No split
scheme repairs this; the confound has to be measured and reported, not removed.

## Clinical metadata

| field | kind | non-null | AUROC | AUROC of missingness |
|---|---|---|---|---|
| age_at_visit | pre | 236/236 | 0.495 | nan |
| menarche | pre | 236/236 | 0.51 | nan |
| menopause | pre | 69/236 | 0.771 | 0.518 |
| body_temperature | pre | 236/236 | 0.628 | nan |
| biopsy | POST-DIAGNOSTIC | 217/236 | 0.854 | 0.59 |
| mastectomy | POST-DIAGNOSTIC | 235/236 | 0.471 | 0.496 |
| mammography | POST-DIAGNOSTIC | 227/236 | 0.465 | 0.547 |

`biopsy` in detail -- a record of what the clinic already suspected:

| biopsy | label=0 | label=1 |
|---|---|---|
| 0.0 | 111 | 9 |
| 1.0 | 5 | 10 |
| 2.0 | 15 | 30 |
| 3.0 | 8 | 29 |
| missing | 1 | 18 |

Data-quality defects to repair before any tabular model:

- `age_at_visit` <= 0 for 1 patient(s): [-4]
- `body_temperature` == 0 for 4 patients (missing recorded as zero, not NaN)

## Views and protocols

| view_code | Dinamic | Static |
|---|---|---|
| 0 | 4386 | 234 |
| 1 | 31 | 224 |
| 2 | 227 | 235 |
| 3 | 34 | 225 |
| 4 | 223 | 232 |

Static views held per patient:

| n static views | patients |
|---|---|
| 1 | 3 |
| 2 | 4 |
| 3 | 5 |
| 4 | 18 |
| 5 | 206 |


- patients with **all 5 static views**: **206** (121 benign / 85 malignant)
- patients with a frontal dynamic sequence of >= 20 frames: **207** (124 benign / 83 malignant)
- sequences whose `seq` is not contiguous 0..n-1: **0**
- patients with **both**: **189** (113 benign / 76 malignant)

The last line is the complete-case cohort for a model using every modality.

Overlap between the static and dynamic protocols (byte-identical files):

| view_code | patients with static image | static frame is a dynamic frame |
|---|---|---|
| 0 | 231 | 97 |
| 1 | 221 | 0 |
| 2 | 227 | 99 |
| 3 | 224 | 0 |
| 4 | 225 | 98 |

For the 97 frontal cases the matched sequence index spans 0-19 (median 9), i.e. the static image is an arbitrary member of the sequence rather than its first frame.

## Duplicates and masks

- duplicate-hash groups: **330** covering 665 images
- **cross-patient** duplicate groups: **0** (patient-level splitting is sufficient)
- images carrying a mask: **6051/6051**

Mask coverage of 100% confirms the masks are TransUNet *predictions* generated over
the whole mirror, not annotations. Because that segmenter was fitted on this cohort,
a mask may carry label information; run the shape-only baseline before using masks.
