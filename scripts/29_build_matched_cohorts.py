#!/usr/bin/env python3
"""How much of the cohort survives controlling the confounder, and what is left of it?

Patient-level splitting removes leakage and is blind to acquisition confounding: every fold
inherits the same alignment between the calendar and the diagnosis. The remedy is to change
the *cohort*, not the split, and this script builds the ladder of cohorts on which an
imaging model can then be trained and scored under the ordinary protocol:

    original          every eligible patient.
    overlap           the period in which both classes occur (DMR-IR: 2013-2015).
    year_balanced     classes down-sampled to a common size within each acquisition year.
    session_balanced  the same within each acquisition date, the actual acquisition batch.
    date_matched_W    1:1 nearest-neighbour matching within a caliper of W days.
    propensity        1:1 matching on a propensity score over pre-diagnostic covariates.

Nothing here sees a prediction. A cohort is a function of the metadata and the label only,
so the selection cannot be tuned -- deliberately or otherwise -- towards a favourable
imaging result, and `tests/test_matching.py` asserts that the public API takes no score.

The output each rung is judged on is written before any model is run:

  * `n_patients`, and the benign/malignant counts, because a rung retaining 28 patients
    settles nothing and the table must say so rather than report its AUROC as if it did;
  * standardized mean differences on the covariates, the balance the rung actually achieved;
  * `confounder_auroc`, the discrimination the confounder itself retains on that rung. This
    is the number that says whether the rung worked: a date-matched cohort on which the date
    still scores 0.85 has not controlled the date.

Each rung also emits a `splits.csv` in the repository's standard schema, so the existing
training scripts consume it unchanged:

    python scripts/04_train_single_view.py --data-root data/DMR-IR \\
        --manifest results/dmr_view_table_frontal.csv \\
        --splits   results/matching/dmr/overlap/splits.csv \\
        --output-dir results/experiment_dmr_10_matched_overlap --freeze-backbone

Matched pairs are not kept together in a fold. A pair is two different patients, so a
partner in the training partition is not leakage, and the out-of-fold protocol still gives
every patient exactly one score per repeat -- which is what a matched-pair test on the
stored predictions needs.

Example
-------
    python scripts/29_build_matched_cohorts.py                 # DMR-IR, acquisition date
    python scripts/29_build_matched_cohorts.py --dataset verification
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import make_repeated_patient_splits, validate_split_integrity
from src.matching.cohorts import (
    LABEL,
    acquisition_days,
    describe_cohort,
    nearest_neighbour_pairs,
    overlap_cohort,
    pairs_to_cohort,
    propensity_scores,
    size_control_cohort,
    standardized_mean_differences,
    stratum_balanced_cohort,
)
from src.utils.io import file_sha256, read_results_csv
from src.utils.seed import seed_everything

# Pre-diagnostic covariates only. `biopsy`, `mastectomy` and `mammography` record what the
# clinic did *because* it already suspected the diagnosis; matching on them would balance
# the classes on the diagnosis itself.
DATASETS = {
    "dmr": {
        "patients": Path("results/dmr_patient_manifest.csv"),
        "date_column": "visit_date",
        "confounder": "acquisition_days",
        "confounder_label": "acquisition date (days since the first visit)",
        "covariates": ["acquisition_days", "age_at_visit", "menarche", "body_temperature"],
        # Two propensity models, because the contrast between them is the point: a
        # multivariable matching that omits the acquisition axis leaves the dominant
        # confounder exactly where it was, however well balanced everything else becomes.
        "propensity_covariates": {
            "propensity_clinical": ["age_at_visit", "menarche", "body_temperature"],
            "propensity_with_acquisition": ["age_at_visit", "menarche", "body_temperature",
                                            "acquisition_days"],
        },
        "calipers": [7.0, 30.0, 90.0, 180.0],
        "output": Path("results/matching/dmr"),
    },
    "verification": {
        "patients": Path("results/patient_manifest.csv"),
        "date_column": None,
        "confounder": "age",
        "confounder_label": "age at visit (years)",
        "covariates": ["age", "bmi", "temperature"],
        "propensity_covariates": {
            "propensity_clinical": ["bmi", "temperature"],
            "propensity_with_age": ["bmi", "temperature", "age"],
        },
        "calipers": [2.0, 5.0, 10.0],
        "output": Path("results/matching/verification"),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="dmr")
    p.add_argument("--patients", type=Path, default=None, help="override the manifest path")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--calipers", type=float, nargs="+", default=None,
                   help="matching calipers, in the confounder's own units")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--min-per-class", type=int, default=10,
                   help="a rung with fewer patients of either class emits no splits; its "
                        "counts are still reported, marked not evaluable")
    p.add_argument("--size-control-seeds", type=int, default=3,
                   help="random subsets of each rung's size, drawn with the confounding left "
                        "intact, so that a fall in imaging discrimination on a matched rung "
                        "can be separated from the fall in sample size that produced it. "
                        "0 disables them")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def confounder_auroc(cohort: pd.DataFrame, confounder: str) -> float:
    """What the confounder alone still discriminates on this rung.

    Univariate and unfitted: the raw covariate is the score. There is no model to select and
    no partition to respect, so this is the honest description of the cohort rather than an
    estimate whose optimism would have to be argued about. The sign is chosen to give the
    larger of the two orientations, because a confounder that predicts the label backwards
    is just as confounded.
    """
    v = pd.to_numeric(cohort[confounder], errors="coerce")
    y = cohort[LABEL].astype(int)
    m = v.notna()
    if m.sum() < 4 or y[m].nunique() < 2:
        return float("nan")
    return float(max(roc_auc_score(y[m], v[m]), roc_auc_score(y[m], -v[m])))


def emit(cohort: pd.DataFrame, name: str, spec: dict, out_root: Path, args: argparse.Namespace,
         note: str) -> dict:
    """Write one rung and return the row describing it."""
    conf = spec["confounder"]
    row = describe_cohort(cohort, name, conf)
    row["confounder_auroc"] = round(confounder_auroc(cohort, conf), 4) if len(cohort) else np.nan
    row["note"] = note

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    balance = standardized_mean_differences(cohort, spec["covariates"])
    balance.insert(0, "cohort", name)
    balance.to_csv(out / "balance.csv", index=False)
    row["max_abs_smd"] = (round(float(balance["smd"].abs().max()), 4)
                          if balance["smd"].notna().any() else np.nan)

    evaluable = min(row["n_benign"], row["n_malignant"]) >= max(args.min_per_class, args.folds)
    row["evaluable"] = bool(evaluable)
    if evaluable:
        splits = make_repeated_patient_splits(cohort, args.folds, args.repeats)
        errors = validate_split_integrity(splits)
        if errors:
            raise SystemExit(f"Leakage in the {name} splits:\n" + "\n".join(errors))
        splits.to_csv(out / "splits.csv", index=False)
    else:
        # No split file at all, rather than an unusable one: a rung with three malignant
        # patients must not be runnable by accident and then reported as a result.
        (out / "NOT_EVALUABLE.md").write_text(
            f"# {name}: not evaluable\n\n"
            f"{row['n_benign']} benign and {row['n_malignant']} malignant patients remain, "
            f"below the pre-specified minimum of {max(args.min_per_class, args.folds)} per "
            f"class. No split file is emitted, so no experiment can be run on this rung by "
            f"accident. The counts are reported in the ladder because the collapse of the "
            f"cohort under this level of control is itself the finding.\n", encoding="utf-8")

    (out / "config.json").write_text(json.dumps({
        "cohort": name, "note": note, "dataset": args.dataset,
        "confounder": conf, "confounder_units": spec["confounder_label"],
        "covariates_reported": spec["covariates"],
        "selection_inputs": "metadata and the diagnostic label only; no model score of any "
                            "kind enters cohort construction",
        "counts": {k: v for k, v in row.items() if k.startswith("n_")},
        "confounder_auroc_after_control": row["confounder_auroc"],
        "max_abs_smd": row["max_abs_smd"],
        "evaluable": row["evaluable"],
        "folds": args.folds, "repeats": args.repeats, "seed": args.seed,
        "source_manifest": str(spec["patients"]),
        "source_manifest_sha256": file_sha256(spec["patients"]),
    }, indent=2, default=str), encoding="utf-8")
    return row


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    spec = dict(DATASETS[args.dataset])
    if args.patients:
        spec["patients"] = args.patients
    if args.calipers:
        spec["calipers"] = list(args.calipers)
    out_root = args.output_dir or spec["output"]
    out_root.mkdir(parents=True, exist_ok=True)

    patients = read_results_csv(spec["patients"]).dropna(subset=[LABEL])
    patients[LABEL] = patients[LABEL].astype(int)
    conf = spec["confounder"]

    if spec["date_column"]:
        patients[conf] = acquisition_days(patients[spec["date_column"]])
        patients["acquisition_year"] = pd.to_datetime(patients[spec["date_column"]],
                                                      errors="coerce").dt.year
        patients["acquisition_session"] = patients[spec["date_column"]].astype(str)
    patients = patients.dropna(subset=[conf]).reset_index(drop=True)
    print(f"{args.dataset}: {len(patients)} patients with a {conf} value "
          f"({int((patients[LABEL] == 0).sum())} benign / {int((patients[LABEL] == 1).sum())} malignant)")

    rows = [emit(patients, "original", spec, out_root, args,
                 "the cohort as distributed, after this study's exclusions")]

    rows.append(emit(overlap_cohort(patients, conf), "overlap", spec, out_root, args,
                     f"restricted to the {conf} range both classes occupy"))

    for stratum, name, note in (("acquisition_year", "year_balanced",
                                 "classes down-sampled to a common size within each year"),
                                ("acquisition_session", "session_balanced",
                                 "the same within each acquisition date, the actual batch")):
        if stratum in patients.columns:
            rows.append(emit(stratum_balanced_cohort(patients, stratum, args.seed), name,
                             spec, out_root, args, note))

    for caliper in spec["calipers"]:
        pairs = nearest_neighbour_pairs(patients, conf, caliper)
        cohort = pairs_to_cohort(patients, pairs)
        name = f"{conf}_matched_{caliper:g}"
        row = emit(cohort, name, spec, out_root, args,
                   f"1:1 nearest-neighbour matching within +/-{caliper:g}")
        row["n_pairs"] = int(len(pairs))
        if len(pairs):
            pairs.to_csv(out_root / name / "pairs.csv", index=False)
        rows.append(row)

    for name, prop in spec["propensity_covariates"].items():
        usable = [c for c in prop if c in patients.columns]
        if not usable:
            continue
        scored = patients.copy()
        scored["propensity"] = propensity_scores(scored, usable, args.seed)
        # Caliper of 0.2 standard deviations of the propensity score: the conventional
        # choice, fixed here rather than tuned, since tuning it against the resulting
        # balance would be a selection on the outcome of the matching.
        caliper = 0.2 * float(scored["propensity"].std())
        pairs = nearest_neighbour_pairs(scored, "propensity", caliper)
        cohort = pairs_to_cohort(scored, pairs)
        row = emit(cohort, name, spec, out_root, args,
                   f"1:1 propensity matching on {usable}, caliper 0.2 SD ({caliper:.4f})")
        row["n_pairs"] = int(len(pairs))
        if len(pairs):
            pairs.to_csv(out_root / name / "pairs.csv", index=False)
        rows.append(row)

    # Size-matched controls, one family per evaluable rung. Built last, from the rungs that
    # were actually produced, so that a control always matches a real rung's class counts.
    # Deduplicated by class counts: two rungs that retained 14 benign and 14 malignant
    # patients need one control family between them, not two identical ones, and every GPU
    # job saved here is one the cohort could not have afforded anyway.
    by_size: dict[tuple[int, int], list[str]] = {}
    for row in [r for r in rows if r["evaluable"] and r["cohort"] != "original"]:
        by_size.setdefault((row["n_benign"], row["n_malignant"]), []).append(row["cohort"])

    controls = []
    for (n_benign, n_malignant), rungs in sorted(by_size.items()):
        for k in range(1, args.size_control_seeds + 1):
            cohort = size_control_cohort(patients, n_benign, n_malignant,
                                         args.seed + 1000 * k)
            name = f"sizecontrol_{n_benign}b{n_malignant}m_s{k}"
            control = emit(cohort, name, spec, out_root, args,
                           f"random subset of {n_benign}+{n_malignant} patients, confounding "
                           f"intact; the size control for {', '.join(rungs)}")
            control["control_for"] = ";".join(rungs)
            controls.append(control)

    ladder = pd.DataFrame(rows + controls)
    ladder.to_csv(out_root / "ladder.csv", index=False)

    show = ["cohort", "n_patients", "n_benign", "n_malignant", "confounder_auroc",
            "max_abs_smd", "evaluable"]
    print("\nConfounder-control ladder (confounder_auroc is what the confounder still "
          "discriminates after the control):")
    print(ladder[[c for c in show if c in ladder.columns]].to_string(index=False))
    print(f"\nWrote {out_root}/ladder.csv and one directory per rung.")
    not_eval = ladder[~ladder["evaluable"]]
    if len(not_eval):
        print(f"\n{len(not_eval)} rung(s) retained too few patients to be evaluated: "
              f"{', '.join(not_eval['cohort'])}. Their collapse is reported, not their AUROC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
