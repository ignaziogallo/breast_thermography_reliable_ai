#!/usr/bin/env python3
"""Bilateral thermal asymmetry baseline.

Research question: how much patient-level malignancy is recoverable from the images by a
contralateral subtraction and a handful of interpretable scalars, with no learned
representation at all?

This sits deliberately between the metadata-only baseline and the pretrained CNN. It supplies the
second reference bar: a deep model that beats neither the patient's age nor an image
subtraction has not demonstrated that it extracts thermal evidence. It also separates two
things a CNN result conflates -- whether the modality carries signal, and whether deep
representation learning is what extracts it.

Feature sets evaluated:
  asymmetry            image-derived asymmetry scalars only
  asymmetry_metadata   asymmetry + age/BMI/temperature, to test complementarity
  age_only             repeated from the metadata baseline so the three bars share a table

Features are unsupervised image statistics, so computing them for the whole cohort before
splitting is not leakage; the fold-wise standardisation inside the estimator pipeline is
what must stay within the training partition.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import resolve_image_path
from src.evaluation.summary import summarize_predictions
from src.models.metadata_model import build_pipeline
from src.preprocessing.asymmetry import FEATURE_NAMES, extract_features
from src.training.tabular_cv import run_patient_cv
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

METADATA_FEATURES = ["age", "bmi", "temperature"]

FEATURE_SETS = {
    # Primary, pre-specified: the full descriptor set, no selection of any kind.
    "asymmetry": FEATURE_NAMES,
    "asymmetry_metadata": FEATURE_NAMES + METADATA_FEATURES,
    # Parsimonious variant. Net bilateral asymmetry magnitude is the canonical single
    # measurement of the contralateral reading, so it has an a-priori justification --
    # but it was added AFTER observing that it is the strongest single descriptor on the
    # full cohort. It is therefore EXPLORATORY and must be reported as such: its estimate
    # is optimistic by an unknown amount, because the cohort that suggested it is the
    # cohort it is evaluated on. Do not present it as a pre-registered comparison.
    "asymmetry_net_exploratory": ["asym_net_abs"],
    "age_only": ["age"],
}
GRIDS = {
    "logistic": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "gradient_boosting": {"clf__max_depth": [1, 2], "clf__n_estimators": [50, 150]},
}


def build_feature_table(manifest: pd.DataFrame, data_root: Path, cache: Path | None) -> pd.DataFrame:
    if cache and cache.exists():
        cached = pd.read_csv(cache)
        if set(FEATURE_NAMES).issubset(cached.columns) and \
           set(cached["patient_id"]) == set(manifest["patient_id"]):
            print(f"Reusing cached asymmetry features from {cache}")
            return manifest.merge(cached[["patient_id"] + FEATURE_NAMES], on="patient_id")
        print(f"Cache {cache} is stale; recomputing")

    rows = []
    for i, r in enumerate(manifest.itertuples(index=False), start=1):
        feats = extract_features(
            resolve_image_path(data_root, r.anterior_path),
            resolve_image_path(data_root, r.oblleft_path),
            resolve_image_path(data_root, r.oblright_path),
        )
        rows.append({"patient_id": r.patient_id, **feats})
        if i % 40 == 0:
            print(f"  extracted {i}/{len(manifest)}")
    feats = pd.DataFrame(rows)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(cache, index=False)
        print(f"Wrote asymmetry features to {cache}")
    return manifest.merge(feats, on="patient_id")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/Breast-Thermal-Image"))
    p.add_argument("--manifest", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/experiment_07b_asymmetry"))
    p.add_argument("--feature-cache", type=Path, default=Path("results/asymmetry_features.csv"))
    p.add_argument("--feature-sets", nargs="+", default=sorted(FEATURE_SETS))
    p.add_argument("--estimators", nargs="+", default=["logistic"], choices=sorted(GRIDS))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    seed_everything(a.seed)
    manifest = read_results_csv(a.manifest).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)

    view_cols = [f"{v}_path" for v in ("anterior", "oblleft", "oblright")]
    missing_views = manifest[view_cols].isna().any(axis=1)
    if missing_views.any():
        raise SystemExit(f"{int(missing_views.sum())} patients lack a view; E7b needs all three.")

    # Fail fast and actionably, before opening 357 files.
    absolute = [v for c in view_cols for v in manifest[c].astype(str) if v.startswith("/")]
    if absolute:
        raise SystemExit(
            f"{a.manifest} stores absolute paths (e.g. {absolute[0]}).\n"
            "It was produced by an older audit. Regenerate it first:\n"
            f"    python scripts/00_audit_dataset.py --data-root {a.data_root}")
    probe = resolve_image_path(a.data_root, manifest[view_cols[0]].iloc[0])
    if not probe.exists():
        raise SystemExit(
            f"Dataset not found: {probe} does not exist.\n"
            f"Check that --data-root points at the dataset (a symlink is fine):\n"
            f"    ln -s /path/to/Breast-Thermal-Image {a.data_root}")

    table = build_feature_table(manifest, a.data_root, a.feature_cache)
    nan_counts = table[FEATURE_NAMES].isna().sum()
    if nan_counts.any():
        print(f"Warning: NaN features (median-imputed within each training fold):\n{nan_counts[nan_counts > 0]}")

    preds = []
    for kind in a.estimators:
        for fs in a.feature_sets:
            name = fs if kind == "logistic" else f"{fs}_{kind}"
            preds.append(run_patient_cv(table, FEATURE_SETS[fs], splits,
                                        build_pipeline(kind, a.seed), GRIDS[kind], name))
    preds = pd.concat(preds, ignore_index=True)
    per_repeat, ci, summary = summarize_predictions(preds, a.n_bootstrap, a.seed)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(a.output_dir / "predictions.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": "experiment_07b_asymmetry",
        "feature_sets": {k: FEATURE_SETS[k] for k in a.feature_sets},
        "estimators": a.estimators,
        "data_root": str(a.data_root), "manifest": str(a.manifest), "splits": str(a.splits),
        "n_bootstrap": a.n_bootstrap, "seed": a.seed,
        "inputs": input_fingerprint(a.manifest, a.splits),
        "threshold_rule": "balanced accuracy maximized on validation patients of each fold",
        "scale_invariance": "every image z-scored before feature extraction; exports are per-image auto-ranged",
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(preds)} predictions to {a.output_dir}")
    show = ["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "balanced_accuracy_mean"]
    print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    print("\nPatient-level bootstrap 95% CI (AUROC):")
    print(ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
