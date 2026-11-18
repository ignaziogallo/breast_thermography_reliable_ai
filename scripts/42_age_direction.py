#!/usr/bin/env python3
"""Is age inverted on DMR-IR, or is it simply at chance?

The main channels table reports "age alone" at AUROC 0.416, which is *below* chance, and the
manuscript glossed that as "age is uninformative". That gloss is wrong as written: a
sub-chance AUROC does not license the word "uninformative": read literally it points at an
inverse association, and reversing the score would give 0.584. That reading is right for a
statistic computed on the covariate itself. It is not right here, and the difference matters
enough to be measured rather than argued.

The channels table scores a logistic model *fitted* to age on the same folds as the image
(that is what makes it comparable to an imaging arm). A single-feature model fitted where
there is no signal produces a coefficient whose sign need not survive to the held-out fold,
so its cross-validated AUROC scatters around 0.5 and can land below it. The quantity that
answers the direction question is the covariate scored directly, with nothing fitted -- the
same estimator the ladder rungs and the conditional statistic already use for age on the
second cohort.

This script computes that raw statistic on both cohorts, with patient-level bootstrap
percentile intervals, so the manuscript can say which of the two readings the data support.

Example
-------
    python scripts/42_age_direction.py
    python scripts/42_age_direction.py --resamples 2000 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path("results")
OUT_CSV = R / "confounders/age_direction.csv"


def bootstrap_auroc(y: np.ndarray, score: np.ndarray, resamples: int,
                    seed: int) -> tuple[float, float, float]:
    """Patient-level percentile interval, the same recipe as every other CI in the paper."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(roc_auc_score(y, score)), float(lo), float(hi)


def dmr() -> pd.DataFrame:
    """Label and age at the visit, for every DMR-IR patient with a recorded age."""
    d = pd.read_csv(R / "dmr_patient_manifest.csv")
    d = d[d["label_name"].isin(["benign", "malignant"])].dropna(subset=["age_at_visit"])
    return pd.DataFrame({"y": (d["label_name"] == "malignant").astype(int),
                         "age": d["age_at_visit"].astype(float)})


def verification() -> pd.DataFrame:
    """Label and age for the second cohort."""
    d = pd.read_csv(R / "patient_manifest.csv").dropna(subset=["age"])
    return pd.DataFrame({"y": d["patient_binary_label"].astype(int),
                         "age": d["age"].astype(float)})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    rows = []
    for cohort, frame in (("dmr_ir", dmr()), ("verification", verification())):
        y, age = frame["y"].values, frame["age"].values
        est, lo, hi = bootstrap_auroc(y, age, a.resamples, a.seed)
        rows.append({
            "cohort": cohort, "statistic": "age_raw_auroc",
            "n_patients": len(y), "n_benign": int((y == 0).sum()),
            "n_malignant": int((y == 1).sum()),
            "mean_age_benign": float(age[y == 0].mean()),
            "mean_age_malignant": float(age[y == 1].mean()),
            "estimate": est, "ci_lower": lo, "ci_upper": hi,
        })
    table = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
