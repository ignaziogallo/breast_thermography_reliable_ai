#!/usr/bin/env python3
"""Does the image add information *given* a known confounder?

A multimodal model that contains the confounder will beat it almost by construction, so
"better than the confounder alone" is not evidence that the thermogram contributes. The
question that carries content is whether adding the image score improves on it, evaluated
paired on identical folds.

`--covariate` selects what the image must improve upon, and the choice is cohort-specific:

  age           the Cali cohort's dominant confounder (AUROC 0.758 there).
  visit_date    DMR-IR's. Acquisition epoch scores AUROC 0.981 on that cohort -- benign
                patients were acquired 2012-2015, malignant 2013-2020 -- and label-pure
                acquisition sessions cover 226 of its 236 patients, so no split scheme
                decorrelates the two. On DMR-IR this script asks the only question the data
                can answer: does the thermogram carry anything the calendar does not?

A date is converted to days since its cohort minimum before entering the model; the
transform is affine and monotone, so it changes no AUROC and keeps the coefficient readable.

A subtlety governs the design. The image score is itself an out-of-fold prediction: for a
patient in the test fold of (repeat, fold) it was produced by a model that never saw that
patient, which is clean. But the same is not true of *training* patients: their image
scores came from models fitted on folds that include the current test patients. Any
meta-model fitted on training patients' image scores therefore inherits a dependence on the
test fold. Two variants are reported, and the distinction is not cosmetic:

  fixed_combination   standardises age and image score with training-fold means and
                      standard deviations and averages them. Nothing is fitted to labels,
                      so no label information crosses the partition. This is the primary,
                      leakage-free estimate.

  stacked_logistic    fits a logistic regression on age and image score over the training
                      patients of each fold. More flexible, and optimistic by an unknown
                      amount for the reason above. Reported as exploratory.

Writes predictions in the standard format; use scripts/14_compare_models.py for the paired
comparison against the age-only reference.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.summary import select_threshold, summarize_predictions
from src.models.metadata_model import build_pipeline
from src.training.tabular_cv import run_patient_cv
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

IMAGE_FEATURE = "image_score"


def numeric_covariate(manifest: pd.DataFrame, column: str) -> pd.Series:
    """The covariate as a float column, converting a date to days since the cohort minimum."""
    if column not in manifest.columns:
        raise SystemExit(f"--covariate {column!r} absent from the manifest; available: "
                         f"{sorted(manifest.columns)}")
    raw = manifest[column]
    values = pd.to_numeric(raw, errors="coerce")
    if values.notna().sum() < len(raw) / 2:
        dates = pd.to_datetime(raw, errors="coerce")
        if dates.notna().sum() > values.notna().sum():
            values = (dates - dates.min()).dt.days.astype(float)
    if values.isna().all():
        raise SystemExit(f"--covariate {column!r} is neither numeric nor a parseable date")
    return values


def fixed_combination(table: pd.DataFrame, splits: pd.DataFrame, model_name: str,
                      covariate: str) -> pd.DataFrame:
    """Average of the covariate and image score, each standardised on the training fold only."""
    meta = table.set_index("patient_id")
    rows = []
    for (repeat, fold), g in splits.groupby(["repeat", "fold"]):
        ids = {s: g.loc[g["split"] == s, "patient_id"].tolist() for s in ("train", "val", "test")}
        assert not set(ids["train"]) & set(ids["test"]), f"train/test overlap r{repeat} f{fold}"

        stats = {}
        for col in (covariate, IMAGE_FEATURE):
            tr = meta.loc[ids["train"], col].astype(float)
            stats[col] = (float(tr.mean()), float(tr.std()) or 1.0)

        def score(pids):
            z = [(meta.loc[pids, c].astype(float) - m) / s for c, (m, s) in stats.items()]
            return (sum(z) / len(z)).to_numpy()

        y_val = meta.loc[ids["val"], "patient_binary_label"].astype(int).to_numpy()
        y_test = meta.loc[ids["test"], "patient_binary_label"].astype(int).to_numpy()
        # The combined score is a mean of z-scores, not a probability, and the threshold
        # rule and the Brier score both need one on the unit interval. It is mapped through
        # a logistic, which is monotone -- so AUROC is exactly that of the raw combination --
        # bounded, tie-free, and a function of that patient alone.
        #
        # The previous min-max rescaling took its range over the validation *and test*
        # partitions together. That let the test fold set the scale on which a
        # validation-selected threshold was applied, contaminating every threshold-dependent
        # metric of the fixed-combination arms; AUROC was unaffected because the map is
        # monotone. Every `*_plus_image_fixed` result produced before this fix must be
        # regenerated before its balanced accuracy, sensitivity, specificity or Brier score
        # is reported.
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        p_val, p_test = sigmoid(score(ids["val"])), sigmoid(score(ids["test"]))
        threshold = select_threshold(y_val, p_val)

        for pid, yt, pp in zip(ids["test"], y_test, p_test):
            rows.append({"patient_id": pid, "repeat": int(repeat), "fold": int(fold),
                         "y_true": int(yt), "y_prob": float(pp),
                         "y_pred": int(pp >= threshold), "threshold": threshold,
                         "model": model_name, "config": json.dumps({"rule": "mean of z-scores"})})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-predictions", type=Path, required=True,
                   help="predictions.csv from an image experiment")
    p.add_argument("--image-model", default=None,
                   help="model name inside that file (default: the only one present)")
    p.add_argument("--manifest", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--covariate", default="age",
                   help="what the image must add to: 'age' (Cali) or 'visit_date' (DMR-IR)")
    p.add_argument("--output-dir", type=Path, default=Path("results/experiment_08_multimodal"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    seed_everything(a.seed)
    manifest = read_results_csv(a.manifest).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)
    image = read_results_csv(a.image_predictions)

    names = sorted(image["model"].unique())
    image_model = a.image_model or (names[0] if len(names) == 1 else None)
    if image_model is None:
        raise SystemExit(f"--image-model required; {a.image_predictions} holds {names}")
    image = image[image["model"] == image_model]
    print(f"Image model: {image_model} ({len(image)} predictions)")

    cov = a.covariate
    base = manifest[["patient_id", "patient_binary_label"]].copy()
    base[cov] = numeric_covariate(manifest, cov)
    base = base.dropna(subset=[cov])
    if len(base) < len(manifest):
        print(f"Dropped {len(manifest) - len(base)} patients with no {cov}")

    # An image experiment may legitimately cover fewer patients than the manifest: a
    # view-restricted run scores the 231 DMR-IR patients holding a frontal view, not all
    # 236. Every arm must then be computed on that same set, or the covariate reference
    # would be estimated on a different cohort than the arm it is compared against, and the
    # paired difference would mix a cohort change with a model change.
    scored = set.intersection(*(set(g["patient_id"]) for _, g in image.groupby("repeat")))
    unscored = sorted(set(base["patient_id"]) - scored)
    if unscored:
        print(f"Restricting every arm to the {len(scored)} patients scored in all repeats: "
              f"{len(unscored)} dropped, e.g. {unscored[:5]}")
        base = base[base["patient_id"].isin(scored)].reset_index(drop=True)
    if base.empty:
        raise SystemExit(f"No patient of {a.manifest} carries an image score")
    all_preds = []

    # The image score is out-of-fold within a repeat, so each repeat carries its own feature
    # column and the repeats are processed separately rather than averaged together.
    for repeat, ig in image.groupby("repeat"):
        scores = ig.groupby("patient_id")["y_prob"].mean().rename(IMAGE_FEATURE)
        table = base.merge(scores, on="patient_id", how="inner")
        if len(table) != len(base):
            missing = set(base["patient_id"]) - set(table["patient_id"])
            raise SystemExit(f"repeat {repeat}: no image score for {len(missing)} patients")
        rsplits = splits[splits["repeat"] == repeat]
        rsplits = rsplits[rsplits["patient_id"].isin(set(table["patient_id"]))]

        all_preds.append(fixed_combination(table, rsplits, f"{cov}_plus_image_fixed", cov))
        all_preds.append(run_patient_cv(table, [cov, IMAGE_FEATURE], rsplits,
                                        build_pipeline("logistic", a.seed),
                                        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                                        f"{cov}_plus_image_stacked_exploratory"))
        all_preds.append(run_patient_cv(table, [cov], rsplits,
                                        build_pipeline("logistic", a.seed),
                                        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                                        f"{cov}_only_reference"))
        all_preds.append(run_patient_cv(table, [IMAGE_FEATURE], rsplits,
                                        build_pipeline("logistic", a.seed),
                                        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                                        "image_only_reference"))

    preds = pd.concat(all_preds, ignore_index=True)
    per_repeat, ci, summary = summarize_predictions(preds, a.n_bootstrap, a.seed)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(a.output_dir / "predictions.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": a.output_dir.name,
        "research_question": f"does the image score add information given {cov}",
        "covariate": cov,
        "n_patients_scored": len(base),
        "n_patients_dropped_absent_from_manifest": len(unscored),
        "patients_dropped": unscored,
        "image_model": image_model, "image_predictions": str(a.image_predictions),
        "primary_variant": f"{cov}_plus_image_fixed (no label fitting; leakage-free)",
        "exploratory_variant": f"{cov}_plus_image_stacked_exploratory (meta-model fitted on "
                               "training patients whose image scores depend on the test fold)",
        "caveat": f"a multimodal model containing {cov} beats {cov}-alone by construction; the "
                  f"quantity of interest is the paired difference against {cov}_only_reference",
        "inputs": input_fingerprint(a.manifest, a.splits),
        "n_bootstrap": a.n_bootstrap, "seed": a.seed,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(preds)} predictions to {a.output_dir}")
    print(ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]]
          .round(3).to_string(index=False))
    print("\nThe verdict is the paired difference, not these intervals:")
    print(f"  python scripts/14_compare_models.py --reference {cov}_only_reference \\")
    print(f"    --predictions {a.output_dir}/predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
