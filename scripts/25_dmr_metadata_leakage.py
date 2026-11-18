#!/usr/bin/env python3
"""Where does a metadata-only DMR-IR score actually come from?

Pinto et al. (IEEE ColCACI 2025), Table III, report 0.918 accuracy from a linear probe on
text embeddings of the clinical metadata alone -- no image. They read this as evidence that
the clinical record is informative. This script tests the alternative: that the record
leaks the diagnosis, and that no language model is needed to see it.

The metadata is partitioned before anything is fitted, because the partition *is* the
experiment:

  pre_diagnostic     fields that exist before anyone suspects a lesion: age at visit,
                     menarche, age at menopause, body temperature.
  post_diagnostic    fields that record what the clinic did *because* a lesion was suspected
                     or found: biopsy, mastectomy, mammography. On this cohort `biopsy`
                     alone scores AUROC 0.854 and its very *absence* scores 0.590 -- 18 of
                     the 19 patients with no biopsy field recorded are malignant.
  age_pair           `age_at_visit` and `age_current` together. Individually they score
                     0.495 and 0.373, near chance. Their difference is years elapsed since
                     the visit (Pearson 0.990 against the true elapsed time, five distinct
                     values, one per acquisition era), so the pair reconstructs the
                     acquisition calendar -- which predicts the label at AUROC 0.981. This
                     is the channel a "just give the model every column" design opens
                     without anybody noticing.
  missingness        indicator columns only, no values at all. What can be predicted from
                     which fields happen to be filled in.
  as_published       every field at once, the setting Table III approximates.

`acquisition_date_only` is carried as the reference: the calendar these channels are proxies
for.

Our post-diagnostic set is a *subset* of theirs -- their prompts also mention radiotherapy,
symptoms, complaints and nipple changes, which `scripts/19_build_dmr_manifest.py` does not
carry from `metadata.csv`. Whatever leakage this script measures is therefore a lower bound
on the leakage available to their Table III.

Two recording defects are repaired here rather than silently imputed, and both are reported:
one `age_at_visit` of -4, and four `body_temperature` of exactly 0, which is a missing value
written as a number.

Example
-------
    python scripts/25_dmr_metadata_leakage.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.summary import summarize_predictions
from src.models.metadata_model import build_pipeline
from src.training.tabular_cv import run_patient_cv
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

PRE_DIAGNOSTIC = ["age_at_visit", "menarche", "menopause", "body_temperature"]
POST_DIAGNOSTIC = ["biopsy", "mastectomy", "mammography"]
AGE_PAIR = ["age_at_visit", "age_current"]
DATE = "acquisition_days"
GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/dmr_patient_splits.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/experiment_dmr_04_metadata_leakage"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clean(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric feature table, with impossible values turned into honest missingness."""
    df = pd.DataFrame({"patient_id": manifest["patient_id"],
                       "patient_binary_label": manifest["patient_binary_label"].astype(int)})
    notes = []

    for col in dict.fromkeys(PRE_DIAGNOSTIC + POST_DIAGNOSTIC + AGE_PAIR):
        if col not in manifest.columns:
            notes.append(f"{col}: absent from the manifest, skipped")
            continue
        s = pd.to_numeric(manifest[col].replace({"Yes": 1, "No": 0, "yes": 1, "no": 0}),
                          errors="coerce")
        # An age of -4 and a body temperature of 0 are not measurements. Imputing them as if
        # they were would push the median of every fold and hide the defect.
        if col in ("age_at_visit", "age_current", "body_temperature"):
            bad = s <= 0
            if bad.any():
                notes.append(f"{col}: {int(bad.sum())} impossible value(s) "
                             f"{sorted(s[bad].unique().tolist())} set to missing")
                s = s.where(~bad)
        df[col] = s
        df[f"{col}_missing"] = s.isna().astype(int)

    date = pd.to_datetime(manifest["visit_date"], errors="coerce")
    df[DATE] = (date - date.min()).dt.days.astype(float)
    return df, notes


def univariate(df: pd.DataFrame) -> pd.DataFrame:
    """Each field on its own, and the predictive value of its mere presence."""
    y = df["patient_binary_label"]
    rows = []
    for col in dict.fromkeys(PRE_DIAGNOSTIC + POST_DIAGNOSTIC + ["age_current", DATE]):
        if col not in df.columns:
            continue
        s, m = df[col], df[col].notna()
        auc = (roc_auc_score(y[m], s[m])
               if m.sum() > 10 and s[m].nunique() > 1 and y[m].nunique() > 1 else np.nan)
        miss = df.get(f"{col}_missing")
        rows.append({
            "field": col,
            "kind": ("pre" if col in PRE_DIAGNOSTIC else
                     "POST-DIAGNOSTIC" if col in POST_DIAGNOSTIC else "date-derived"),
            "non_null": int(m.sum()), "n": len(df),
            "auroc": round(float(auc), 4) if np.isfinite(auc) else np.nan,
            "auroc_of_missingness": (round(float(roc_auc_score(y, miss)), 4)
                                     if miss is not None and miss.nunique() > 1 else np.nan),
        })

    # The pair is the point: neither age column is informative, their difference is a clock.
    if {"age_at_visit", "age_current"} <= set(df.columns):
        gap = df["age_current"] - df["age_at_visit"]
        m = gap.notna()
        rows.append({"field": "age_current MINUS age_at_visit", "kind": "date-derived",
                     "non_null": int(m.sum()), "n": len(df),
                     "auroc": round(float(roc_auc_score(y[m], gap[m])), 4),
                     "auroc_of_missingness": np.nan})
        rows.append({"field": "  same, sign-corrected", "kind": "date-derived",
                     "non_null": int(m.sum()), "n": len(df),
                     "auroc": round(float(roc_auc_score(y[m], -gap[m])), 4),
                     "auroc_of_missingness": np.nan})
    return pd.DataFrame(rows)


def arms(df: pd.DataFrame) -> dict[str, list[str]]:
    """Feature sets, each restricted to columns the manifest actually provided."""
    have = [c for c in PRE_DIAGNOSTIC if c in df.columns]
    post = [c for c in POST_DIAGNOSTIC if c in df.columns]
    pair = [c for c in AGE_PAIR if c in df.columns]
    missing_cols = [c for c in df.columns if c.endswith("_missing")]
    return {
        "pre_diagnostic": have,
        "post_diagnostic": post,
        "biopsy_only": [c for c in ("biopsy",) if c in df.columns],
        "age_pair_only": pair,
        "missingness_only": missing_cols,
        "as_published_all_fields": sorted(set(have + post + pair)),
        "acquisition_date_only": [DATE],
    }


def main() -> int:
    a = parse_args()
    seed_everything(a.seed)

    manifest = read_results_csv(a.manifest).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)
    df, notes = clean(manifest)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    if notes:
        print("Recording defects repaired:")
        for n in notes:
            print(f"  {n}")

    uni = univariate(df)
    uni.to_csv(a.output_dir / "univariate.csv", index=False)
    print("\nEach field on its own:")
    print(uni.to_string(index=False))

    all_preds = []
    for name, features in arms(df).items():
        if not features:
            print(f"\n{name}: no usable field, skipped")
            continue
        print(f"\n{name}: {features}", flush=True)
        all_preds.append(run_patient_cv(df, features, splits, build_pipeline("logistic", a.seed),
                                        GRID, name))

    preds = pd.concat(all_preds, ignore_index=True)
    per_repeat, ci, summary = summarize_predictions(preds, a.n_bootstrap, a.seed)

    preds.to_csv(a.output_dir / "predictions.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": a.output_dir.name,
        "research_question": "does a metadata-only DMR-IR score come from clinical information "
                             "or from records of the diagnostic pathway and the calendar",
        "comparison": "Pinto et al. 2025 Table III reports 0.918 accuracy from text embeddings "
                      "of the metadata alone",
        "arms": {k: v for k, v in arms(df).items() if v},
        "pre_diagnostic_fields": PRE_DIAGNOSTIC,
        "post_diagnostic_fields": POST_DIAGNOSTIC,
        "post_diagnostic_rationale": "biopsy, mastectomy and mammography record what the clinic "
                                     "did because a lesion was suspected or found",
        "age_pair_rationale": "age_current - age_at_visit is years elapsed since the visit "
                              "(Pearson 0.990, five distinct values), so the pair encodes the "
                              "acquisition era, which predicts the label at AUROC 0.981",
        "lower_bound_note": "their prompts also carry radiotherapy, symptoms, complaints and "
                            "nipple changes, which this manifest does not hold; the leakage "
                            "measured here is a lower bound on theirs",
        "recording_defects_repaired": notes,
        "inputs": input_fingerprint(a.manifest, a.splits),
        "n_bootstrap": a.n_bootstrap, "seed": a.seed,
    }, indent=2), encoding="utf-8")

    table = ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]].merge(
        summary[["model", "balanced_accuracy_mean", "accuracy_at_selected_threshold_mean"]],
        on="model", how="left")
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nWrote {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
