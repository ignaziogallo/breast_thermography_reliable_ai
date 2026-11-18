# Dataset Audit

## Core counts

- **n manifest rows:** 119
- **n patients in folders:** 119
- **n patients in diagnostics:** 119
- **n images found:** 357
- **patients with all 3 views:** 119
- **patients missing views:** 0
- **folder diagnostics inconsistencies:** 0
- **patient label counts:** {0: 84, 1: 35}
- **left label counts:** {'N': 52, 'PB': 50, 'PM': 17}
- **right label counts:** {'N': 65, 'PB': 36, 'PM': 18}
- **exact duplicate image rows:** 0
- **exact duplicate hash groups:** 0
- **near duplicate candidate pairs:** 2

## Label rule

The default patient-level binary label is **malignant if at least one breast is PM; otherwise non-malignant (N/PB)**. This rule is explicit and should be reconsidered after inspecting the breast-level label counts.

## Metadata statistics

```
   variable   n  missing       mean       std     median       min        max  mean_label_0  std_label_0  mean_label_1  std_label_1  point_biserial_r  point_biserial_p
        age 119        0  48.714286 13.157548  47.000000  18.00000  81.000000     45.095238    12.273607     57.400000    11.093718          0.427915          0.000001
     weight 119        0  72.264706 16.956163  70.000000  37.10000 159.000000     71.276190    17.058373     74.637143    16.711445          0.090697          0.326607
     height 119        0 157.057143  7.167246 157.500000 134.50000 172.500000    157.652381     6.847029    155.628571     7.801228         -0.129204          0.161378
        bmi 119        0  29.263608  6.398780  29.043581  15.52615  62.893082     28.664593     6.493409     30.701243     6.013019          0.145639          0.114014
temperature 119        0  35.999160  0.431492  36.100000  34.50000  36.800000     35.982143     0.459743     36.040000     0.357442          0.061354          0.507424
```

## Data-quality checks

Exact duplicates are listed in `results/exact_duplicates.csv`. Cross-patient perceptual-hash candidates are listed in `results/near_duplicate_candidates.csv`. These are audit flags and require visual confirmation before exclusion.
