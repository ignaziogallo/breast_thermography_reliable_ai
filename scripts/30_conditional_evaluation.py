#!/usr/bin/env python3
"""Does the thermogram still discriminate between patients the confounder cannot tell apart?

An AUROC counts the benign/malignant patient pairs a score ranks correctly. On DMR-IR most
of those pairs are a 2012 benign against a 2019 malignant, so any feature tracking the
calendar orders them without reading pathology. This script recomputes the same statistic
over the pairs whose acquisition dates differ by at most a caliper:

    c_W = P(score_malignant > score_benign | |date_malignant - date_benign| <= W)

Two things make this the primary confounder-controlled estimator on these cohorts rather
than the matched cohorts of `scripts/29_build_matched_cohorts.py`.

*   It is far better powered. 1:1 date matching on DMR-IR retains 13 pairs at a 90-day
    caliper; the conditional statistic retains 276, because a benign patient may appear in
    every admissible pair instead of being consumed by one match. Both are reported, and the
    matched rung is reported precisely so that its collapse is visible.
*   It needs no retraining. It re-reads predictions an experiment has already made, so the
    confounder control can be applied to every arm this study has run -- and to any arm a
    future study runs -- at the cost of a CPU minute.

What it cannot do is remove a shortcut from a model that already learned one. A model fitted
on the full confounded cohort may still key on acquisition; the caliper only guarantees that
*within an admissible pair* the calendar has almost no room to help. Training on a matched
cohort is the complementary experiment, and the two answer different questions.

The control arm is what makes the output readable: the confounder's own value is scored
through the identical statistic, and it must fall towards 0.5 as the caliper narrows. That
is arithmetic rather than a finding, and its absence would mean the caliper is too wide to
be controlling anything. Its orientation is fixed once on the full cohort, never per
caliper, so the control cannot be flattered by re-orienting it where that helps.

Example
-------
    python scripts/30_conditional_evaluation.py \\
        --predictions results/experiment_dmr_01_leakage/predictions.csv \\
                      results/experiment_dmr_02_frontal_static/predictions.csv \\
        --manifest results/dmr_patient_manifest.csv --confounder visit_date \\
        --output-dir results/confounders/dmr_conditional
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confounding.conditional import caliper_series, paired_conditional_difference
from src.evaluation.predictions import pool_repeats
from src.matching.cohorts import LABEL, acquisition_days
from src.utils.io import file_sha256, posix, read_results_csv, write_json
from src.utils.seed import seed_everything

CONTROL = "confounder_alone_control"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    p.add_argument("--manifest", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--confounder", default="visit_date",
                   help="manifest column; a date is converted to days since the cohort minimum")
    p.add_argument("--calipers", type=float, nargs="+", default=[180.0, 90.0, 30.0, 7.0],
                   help="in the confounder's units; the unrestricted statistic is always added")
    p.add_argument("--stratum", default=None,
                   help="manifest column the two patients of a pair must share exactly, e.g. "
                        "visit_date. A date caliper does not control the acquisition session: "
                        "two patients photographed days apart usually belong to different "
                        "sessions, and a label-pure session is a shortcut a caliper leaves open")
    p.add_argument("--reference", default=CONTROL,
                   help=f"arm the paired differences are taken against (default: {CONTROL})")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/confounders/dmr_conditional"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def confounder_values(manifest: pd.DataFrame, column: str) -> pd.Series:
    """The confounder as a float, converting a date to days since the cohort minimum."""
    if column not in manifest.columns:
        raise SystemExit(f"--confounder {column!r} absent; available: {sorted(manifest.columns)}")
    values = pd.to_numeric(manifest[column], errors="coerce")
    if values.notna().sum() < len(manifest) / 2:
        dates = pd.to_datetime(manifest[column], errors="coerce")
        if dates.notna().sum() > values.notna().sum():
            return acquisition_days(manifest[column])
    if values.isna().all():
        raise SystemExit(f"--confounder {column!r} is neither numeric nor a parseable date")
    return values


def patient_table(prediction_files: list[Path], manifest: pd.DataFrame,
                  confounder: str, stratum: str | None = None
                  ) -> tuple[pd.DataFrame, dict[str, str], dict]:
    """One row per patient: the confounder, the label, and one column per model.

    Repeats are averaged within a model, giving one out-of-fold score per patient, which is
    the unit every interval in this repository is computed on. Arms are then intersected on
    the patients they all scored: a paired comparison between a 236-patient arm and a
    231-patient arm would mix a change of cohort with a change of model.
    """
    raw = []
    for path in prediction_files:
        preds = read_results_csv(path)
        for model, g in preds.groupby("model"):
            raw.append((path, str(model), pool_repeats(g)))

    # Several experiments emit the same model name -- every all-frames arm writes
    # `patient_level` whatever encoder produced it. Qualify a repeated name by its
    # experiment directory rather than refusing the comparison, and leave a name that is
    # already unique alone so that existing output stays readable.
    seen = Counter(model for _, model, _ in raw)
    scores, columns, provenance = {}, {}, {}
    for path, model, pooled in raw:
        name = model if seen[model] == 1 else f"{path.parent.name}::{model}"
        if name in scores:
            raise SystemExit(f"two arms resolve to the same name {name!r}; pass their "
                             "prediction files in separate runs")
        scores[name] = pooled
        columns[name] = f"score__{name}"
        provenance[name] = posix(path)

    common = set.intersection(*(set(s.index) for s in scores.values()))
    common &= set(manifest["patient_id"])
    if not common:
        raise SystemExit("no patient carries a score from every arm and a manifest row")

    keep = ["patient_id", LABEL] + ([stratum] if stratum and stratum != confounder else [])
    missing = [c for c in keep if c not in manifest.columns]
    if missing:
        raise SystemExit(f"manifest has no column(s) {missing}")
    table = manifest[manifest["patient_id"].isin(common)][keep].copy()
    table[confounder] = confounder_values(manifest, confounder).loc[table.index].to_numpy()
    table = table.set_index("patient_id")
    for model, pooled in scores.items():
        table[columns[model]] = pooled.loc[table.index, "y_prob"]
        mismatch = (pooled.loc[table.index, "y_true"].astype(int)
                    != table[LABEL].astype(int)).sum()
        if mismatch:
            raise SystemExit(f"{model}: {mismatch} patients disagree with the manifest label")
    dropped = {model: sorted(set(s.index) - common) for model, s in scores.items()}
    return table.reset_index().dropna(subset=[confounder]), columns, {
        "provenance": provenance,
        "n_common_patients": len(common),
        "patients_dropped_per_arm": {k: len(v) for k, v in dropped.items()},
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    manifest = read_results_csv(args.manifest).dropna(subset=[LABEL])
    manifest[LABEL] = manifest[LABEL].astype(int)
    table, columns, prov = patient_table(args.predictions, manifest, args.confounder,
                                         args.stratum)

    # The control arm: the confounder itself, oriented once on the full cohort so that a
    # narrow caliper cannot be handed the more favourable of two orientations.
    y = table[LABEL].astype(int).to_numpy()
    c = table[args.confounder].astype(float).to_numpy()
    sign = 1.0 if roc_auc_score(y, c) >= 0.5 else -1.0
    table[f"score__{CONTROL}"] = sign * c
    columns[CONTROL] = f"score__{CONTROL}"

    print(f"{len(table)} patients scored by every arm "
          f"({int((y == 0).sum())} benign / {int((y == 1).sum())} malignant)")
    print(f"arms: {', '.join(sorted(columns))}")

    calipers: list[float | None] = [None] + sorted(args.calipers, reverse=True)
    series = caliper_series(table, columns, args.confounder, calipers,
                            args.n_bootstrap, args.seed, stratum_column=args.stratum)

    if args.reference not in columns:
        raise SystemExit(f"--reference {args.reference!r} is not among {sorted(columns)}")
    diffs = []
    for caliper in calipers:
        for name, column in columns.items():
            if name == args.reference:
                continue
            d = paired_conditional_difference(
                y, table[column].to_numpy(dtype=float),
                table[columns[args.reference]].to_numpy(dtype=float), c, caliper,
                args.n_bootstrap, args.seed)
            diffs.append({"score": name, "reference": args.reference,
                          "caliper": "none" if caliper is None else caliper, **d})
    diffs = pd.DataFrame(diffs)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    series.to_csv(out / "caliper_series.csv", index=False)
    diffs.to_csv(out / "paired_differences.csv", index=False)
    table.to_csv(out / "patient_scores.csv", index=False)
    write_json({
        "experiment": out.name,
        "research_question": "does a score still discriminate between patients the confounder "
                             "cannot separate",
        "estimator": "c_W = P(score_malignant > score_benign | |C_mal - C_ben| <= W), ties 0.5; "
                     "caliper 'none' recovers the ordinary AUROC exactly",
        "confounder": args.confounder,
        "stratum": args.stratum,
        "stratum_rationale": "a caliper bounds the distance on a continuous confounder; it "
                             "does not hold an acquisition session constant, and a "
                             "label-pure session is a shortcut the caliper leaves open",
        "control_arm": CONTROL,
        "control_orientation": f"fixed once on the full cohort (sign = {int(sign):+d})",
        "control_rationale": "the confounder's own c_W must fall towards 0.5 as the caliper "
                             "narrows; that it does is what shows the caliper controls it",
        "calipers": ["none" if x is None else x for x in calipers],
        "reference_for_differences": args.reference,
        "decision_rule": "improvement demonstrated only when delta_ci_lower > 0; an interval "
                         "containing zero means not demonstrated, not equivalent",
        "bootstrap": "patients resampled within class, admissible pairs recomputed per draw",
        "caveat": "the caliper controls what the *evaluation* pairs can differ by; it does not "
                  "remove a shortcut a model trained on the full cohort may already have "
                  "learned. Training on a matched cohort is the complementary experiment.",
        "inputs": {"manifest_path": posix(args.manifest),
                   "manifest_sha256": file_sha256(args.manifest),
                   "prediction_files": dict(prov["provenance"])},
        **{k: v for k, v in prov.items() if k != "provenance"},
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
    }, out / "config.json", default=str)

    view = series.copy()
    view["c [95% CI]"] = [f"{r.c_statistic:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
                          if np.isfinite(r.c_statistic) else "--" for r in series.itertuples()]
    print("\nConditional discrimination by caliper "
          f"({args.confounder}; 'none' is the ordinary AUROC):")
    print(view.pivot(index="score", columns="caliper", values="c [95% CI]").to_string())
    print("\nAdmissible pairs per caliper:")
    print(series.drop_duplicates("caliper")[["caliper", "n_pairs", "n_positive_in_pairs",
                                             "n_negative_in_pairs"]].to_string(index=False))
    print("\nDiscrimination against chance (the readout that carries content once the "
          "confounder is controlled):")
    print(series[["score", "caliper", "n_pairs", "c_statistic", "ci_lower", "ci_upper",
                  "exceeds_chance"]].round(4).to_string(index=False))
    print(f"\nPaired difference against {args.reference} "
          "(improvement requires delta_ci_lower > 0):")
    print(diffs[["score", "caliper", "n_pairs", "delta", "delta_ci_lower", "delta_ci_upper",
                 "improvement_demonstrated"]].round(4).to_string(index=False))
    control = series[series["score"] == args.reference]
    below = control[(control["caliper"] != "none") & (control["c_statistic"] < 0.5)]
    if len(below):
        print(f"\nNote: at caliper(s) {', '.join(str(c) for c in below['caliper'])} the "
              f"reference arm {args.reference!r} itself sits below chance, so beating it is "
              "not evidence of discrimination. Read the exceeds_chance column there.")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
