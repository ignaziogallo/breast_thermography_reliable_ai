#!/usr/bin/env python3
"""Metadata-only patient-level baseline.

Research question: how much patient-level malignancy is predictable from age, BMI, and
body temperature alone, under the frozen leakage-free patient folds? This establishes the
non-image reference against which every image model must be interpreted, because the data
audit found age to be strongly associated with the target.

The fold loop lives in src/training/tabular_cv.py and is shared with the asymmetry baseline
(`scripts/09b_asymmetry_baseline.py`).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.summary import summarize_predictions
from src.models.metadata_model import MODELS, build_estimator, feature_names
from src.training.tabular_cv import run_patient_cv
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/experiment_07_metadata"))
    p.add_argument("--models", nargs="+", default=sorted(MODELS))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    seed_everything(a.seed)
    manifest = read_results_csv(a.manifest).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)

    unknown = set(splits["patient_id"]) - set(manifest["patient_id"])
    if unknown:
        raise SystemExit(f"Split file references {len(unknown)} patients absent from the manifest: {sorted(unknown)[:5]}")

    preds = pd.concat(
        [run_patient_cv(manifest, feature_names(m), splits, build_estimator(m, a.seed),
                        MODELS[m]["grid"], m) for m in a.models],
        ignore_index=True)
    per_repeat, ci, summary = summarize_predictions(preds, a.n_bootstrap, a.seed)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(a.output_dir / "predictions.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": "experiment_07_metadata",
        "models": a.models,
        "features": {m: feature_names(m) for m in a.models},
        "manifest": str(a.manifest), "splits": str(a.splits),
        "n_bootstrap": a.n_bootstrap, "seed": a.seed,
        "inputs": input_fingerprint(a.manifest, a.splits),
        "threshold_rule": "balanced accuracy maximized on validation patients of each fold",
        "aggregation": "out-of-fold predictions pooled within a repeat; CI bootstrapped on repeat-averaged per-patient probability",
    }, indent=2), encoding="utf-8")

    print(f"Wrote {len(preds)} predictions to {a.output_dir}")
    show = ["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "balanced_accuracy_mean", "brier_score_mean"]
    print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    print("\nPatient-level bootstrap 95% CI (AUROC):")
    print(ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
