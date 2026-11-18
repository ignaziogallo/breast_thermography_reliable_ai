#!/usr/bin/env python3
"""Does the thermogram encode *when* it was taken?

On DMR-IR `visit_date` alone scores AUROC 0.981 for malignancy: benign patients were
acquired 2012-2015, malignant 2013-2020, and 83 of 86 acquisition sessions are label-pure.
A malignancy model that keyed on an acquisition signature -- camera drift, a protocol
change, a different room -- would look excellent and generalise to nothing. This script
asks whether such a signature is legible at all, and it does so where the label cannot
answer for it.

The design point is the *within-label* contrast. Asking "can the image predict epoch?" over
the whole cohort is vacuous, because epoch is nearly the label: a perfect malignancy model
would score perfectly on it. The question that carries content is whether the image
separates two acquisition eras among patients who share a diagnosis:

  within_benign      2012 vs 2013 (roughly 64 / 69 patients). The clean contrast: both eras
                     are well populated and the disease is held constant.
  within_malignant   2013-2015 vs 2018-2020 (roughly 39 / 55). Informative but not
                     attributable: no benign patient was acquired after 2015, so a positive
                     result cannot separate a protocol change from a shift in case mix.
  all                the whole cohort, reported only as the vacuous reference above.

An epoch decoder that succeeds within benign patients is direct evidence that the encoder
can read acquisition era from pixels, and therefore that the malignancy result is
confounded. One that fails is evidence against a generic batch artifact -- which is what the
score/date correlation of the existing frontal model already suggests, at Spearman -0.007
among benign patients against +0.388 among malignant ones.

`--image-predictions` adds that correlation analysis over an existing malignancy experiment,
so both halves of the epoch control live in one place and one report.

Each arm restricts the cohort differently and defines its own target, so each writes a
*separate* experiment directory `<prefix>_<arm>/` holding its own manifest, folds,
predictions and config. Pooling them into one predictions.csv would leave
`tests/test_no_prediction_leakage.py` checking every arm against a single arm's folds and
reporting a cohort mismatch for the others.

Example
-------
    CUDA_VISIBLE_DEVICES=4 python scripts/24_epoch_control.py \\
        --data-root data/DMR-IR --freeze-backbone \\
        --image-predictions results/experiment_dmr_02_frontal_static/predictions.csv \\
        --output-prefix results/experiment_dmr_03_epoch
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dmr import ALL_VIEWS, ERA_CONTRASTS, STATIC
from src.evaluation.summary import select_threshold, summarize_predictions
from src.models.single_view import DEFAULT_ENCODER
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

# torch and everything that needs it are loaded inside the functions that train. The
# score/date analysis reads stored predictions and a manifest, so `--score-vs-date-only`
# has to run on a workstation with no deep-learning stack installed -- which is where the
# manuscript is written, and where a number that goes into it must be reproducible. The
# modules imported above are torch-free for that reason.

# arm -> (label restriction, early years, late years). None keeps both labels.
ARMS = ERA_CONTRASTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/DMR-IR"))
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--images", type=Path, default=Path("results/dmr_image_manifest.csv"))
    p.add_argument("--output-prefix", type=Path, default=Path("results/experiment_dmr_03_epoch"),
                   help="each arm writes <prefix>_<arm>/, a self-contained experiment directory")
    p.add_argument("--arms", nargs="+", default=["within_benign", "within_malignant"],
                   choices=sorted(ARMS))
    p.add_argument("--image-predictions", type=Path, default=None,
                   help="a malignancy experiment's predictions.csv, for the score/date analysis")
    p.add_argument("--score-vs-date-only", action="store_true",
                   help="run only the score/date analysis over --image-predictions and exit. "
                        "It reads stored predictions and needs no pixels and no GPU, so the "
                        "question 'does this trained model track the calendar' can be asked "
                        "of any arm without retraining an era decoder for it")
    p.add_argument("--score-vs-date-output", type=Path,
                   default=Path("results/dmr_epoch_score_vs_date.csv"),
                   help="the score/date analysis is over an existing experiment, not a new one")
    p.add_argument("--protocol", default=STATIC)
    p.add_argument("--views", nargs="+", default=["frontal"], choices=list(ALL_VIEWS))
    p.add_argument("--normalisation", choices=["per_image", "fixed"], default="per_image")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--weights-cache", type=Path, default=Path("results/pretrained"))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-folds", type=int, default=0)
    return p.parse_args()


def epoch_cohort(patients: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Patients of one arm, relabelled so that 1 = the later acquisition era."""
    label, early, late = ARMS[arm]
    df = patients.copy()
    df["year"] = pd.to_datetime(df["visit_date"], errors="coerce").dt.year
    if label is not None:
        df = df[df["patient_binary_label"].astype(int) == label]
    df = df[df["year"].isin(early + late)].copy()
    # The target is the era, not the diagnosis. Keeping the column name lets the existing
    # split builder, fold loop and leakage tests operate unchanged.
    df["diagnosis_label"] = df["patient_binary_label"].astype(int)
    df["patient_binary_label"] = df["year"].isin(late).astype(int)
    return df.reset_index(drop=True)


def score_vs_date(predictions: Path, patients: pd.DataFrame) -> pd.DataFrame:
    """Does an existing malignancy score track acquisition date, holding diagnosis fixed?

    Needs no training: a confounded model betrays itself in predictions it has already
    made. Computed within each diagnosis, because across the whole cohort date and label
    are nearly the same variable and any correlation would be uninformative.
    """
    preds = read_results_csv(predictions)
    pooled = preds.groupby("patient_id").agg(score=("y_prob", "mean"),
                                             y_true=("y_true", "first"))
    meta = patients.set_index("patient_id")
    common = pooled.index.intersection(meta.index)
    pooled, meta = pooled.loc[common], meta.loc[common]
    date = pd.to_datetime(meta["visit_date"], errors="coerce")

    rows = []
    for name, mask in (("all", pd.Series(True, index=common)),
                       ("benign", pooled["y_true"] == 0),
                       ("malignant", pooled["y_true"] == 1)):
        g, d = pooled[mask], date[mask]
        if len(g) < 10 or d.nunique() < 2:
            continue
        days = (d - d.min()).dt.days
        rho, pval = spearmanr(g["score"], days)
        rows.append({"stratum": name, "n": len(g), "spearman_score_date": round(float(rho), 4),
                     "p_value": float(pval)})

    # The same question as a discrimination: can the malignancy score tell the two eras
    # apart among patients sharing a diagnosis?
    for arm in ("within_benign", "within_malignant"):
        label, early, late = ARMS[arm]
        year = date.dt.year
        m = (pooled["y_true"] == label) & year.isin(early + late)
        if m.sum() < 20 or year[m].isin(late).nunique() < 2:
            continue
        auc = roc_auc_score(year[m].isin(late).astype(int), pooled.loc[m, "score"])
        rows.append({"stratum": f"{arm}_auroc_of_malignancy_score", "n": int(m.sum()),
                     "spearman_score_date": round(float(auc), 4), "p_value": np.nan})
    return pd.DataFrame(rows)


def run_arm(arm: str, cohort: pd.DataFrame, frames: pd.DataFrame, splits: pd.DataFrame,
            args: argparse.Namespace, state: dict, device) -> pd.DataFrame:
    from src.data.dmr import Normalisation
    from src.data.dmr_datasets import DmrFrameDataset
    from src.models.single_view import build_model_from_state
    from src.preprocessing.augmentations import build_train_augmentation
    from src.training.trainer import TrainConfig, train_fold

    augment = build_train_augmentation()
    norm = Normalisation(args.normalisation) if args.normalisation == "per_image" else None
    labels = cohort.set_index("patient_id")["patient_binary_label"].astype(int)
    pool = frames[frames["patient_id"].isin(set(cohort["patient_id"]))].copy()
    pool["patient_binary_label"] = pool["patient_id"].map(labels)

    rows = []
    for (repeat, fold), g in splits.groupby(["repeat", "fold"]):
        if args.limit_folds and fold > args.limit_folds:
            continue
        ids = {s: set(g.loc[g["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
        part = {s: pool[pool["patient_id"].isin(ids[s])] for s in ids}
        if any(p.empty for p in part.values()):
            print(f"  [{arm}] r{repeat} f{fold}: empty partition, skipped", flush=True)
            continue

        if args.normalisation == "fixed":
            from src.data.dmr import cohort_statistics
            mean, std = cohort_statistics(part["train"]["image_path"], args.data_root,
                                          seed=args.seed)
            fold_norm = Normalisation("fixed", mean, std)
        else:
            fold_norm = norm

        def ds(split, aug=None):
            return DmrFrameDataset(part[split], args.data_root, args.image_size, fold_norm,
                                   augment=aug, id_column="patient_id")

        seed_everything(args.seed + repeat * 100 + fold)
        model = build_model_from_state(state, args.encoder, freeze_backbone=args.freeze_backbone)
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, patience=args.patience,
                          mixed_precision=not args.no_amp, num_workers=args.num_workers,
                          freeze_backbone=args.freeze_backbone)
        t0 = time.time()
        res = train_fold(model, ds("train", augment), ds("val"), ds("test"), cfg, device)
        threshold = select_threshold(res["val_y"], res["val_prob"])
        print(f"  [{arm}] r{repeat} f{fold}: {len(part['train'])}/{len(part['test'])} frames, "
              f"val_auroc {res['best_val_auroc']:.3f}, {time.time() - t0:.0f}s", flush=True)

        for pid, yt, pp in zip(res["test_ids"], res["test_y"], res["test_prob"]):
            rows.append({"patient_id": pid, "repeat": int(repeat), "fold": int(fold),
                         "y_true": int(yt), "y_prob": float(pp),
                         "y_pred": int(pp >= threshold), "threshold": threshold,
                         "model": f"epoch_decoder_{arm}", "config": json.dumps({"arm": arm})})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    patients = read_results_csv(args.patients).dropna(subset=["patient_binary_label"])

    if args.image_predictions is not None:
        analysis = score_vs_date(args.image_predictions, patients)
        args.score_vs_date_output.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(args.score_vs_date_output, index=False)
        print("Does an existing malignancy score track acquisition date?")
        print(analysis.to_string(index=False))
        print(f"Wrote {args.score_vs_date_output}\n")
        if args.score_vs_date_only:
            return 0

    import torch

    from src.data.dmr import aggregate_to_patient, restrict_to_available, select_frames
    from src.data.splits import make_repeated_patient_splits, validate_split_integrity
    from src.models.single_view import (
        build_model_from_state,
        count_parameters,
        fetch_pretrained_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = read_results_csv(args.images)
    frames = select_frames(images, protocol=args.protocol, views=tuple(args.views))

    state = fetch_pretrained_state(args.encoder, args.weights_cache / f"{args.encoder}.pt")
    n_params = count_parameters(build_model_from_state(state, args.encoder,
                                                       freeze_backbone=args.freeze_backbone))
    summaries = []

    for arm in args.arms:
        cohort = epoch_cohort(patients, arm)
        # Before the folds are built, not after: a patient with no frame of the selected
        # view would be assigned to a fold and then never scored.
        cohort, dropped = restrict_to_available(cohort, frames)
        counts = cohort["patient_binary_label"].value_counts()
        n_early, n_late = int(counts.get(0, 0)), int(counts.get(1, 0))
        print(f"=== {arm}: {len(cohort)} patients, {n_early} early / {n_late} late ===", flush=True)
        if dropped:
            print(f"  {len(dropped)} dropped for holding no {args.views} frame: {dropped}",
                  flush=True)
        if min(n_early, n_late) < args.folds:
            print(f"  skipped: fewer than {args.folds} patients in an era", flush=True)
            continue

        splits = make_repeated_patient_splits(cohort, args.folds, args.repeats)
        errors = validate_split_integrity(splits)
        if errors:
            raise SystemExit("Leakage in the epoch splits:\n" + "\n".join(errors))

        out = Path(f"{args.output_prefix}_{arm}")
        out.mkdir(parents=True, exist_ok=True)
        manifest_path, splits_path = out / "manifest.csv", out / "splits.csv"
        cohort.to_csv(manifest_path, index=False)
        splits.to_csv(splits_path, index=False)

        preds = run_arm(arm, cohort, frames, splits, args, state, device)
        if preds.empty:
            continue
        preds = aggregate_to_patient(preds)
        preds.to_csv(out / "predictions.csv", index=False)

        _, ci, summary = summarize_predictions(preds, args.n_bootstrap, args.seed)
        summary = summary.merge(
            ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]],
            on="model", how="left")
        summary.to_csv(out / "metrics.csv", index=False)
        ci.to_csv(out / "bootstrap_ci.csv", index=False)
        summaries.append(summary)

        (out / "config.json").write_text(json.dumps({
            "experiment": out.name, "arm": arm,
            "research_question": "can the thermogram identify its acquisition era among "
                                 "patients sharing a diagnosis",
            "target": "1 = later acquisition era; the diagnosis is held constant within an arm",
            "diagnosis_held_constant": ARMS[arm][0],
            "early_years": list(ARMS[arm][1]), "late_years": list(ARMS[arm][2]),
            "n_patients": len(cohort), "n_early": n_early, "n_late": n_late,
            "n_patients_dropped_absent_from_manifest": len(dropped),
            "patients_dropped": dropped,
            "interpretation": "above chance within_benign shows the encoder reads acquisition "
                              "era from pixels, so the malignancy result is confounded; at "
                              "chance it is evidence against a generic batch artifact. "
                              "within_malignant is NOT attributable: no benign patient was "
                              "acquired after 2015, so a protocol change and a change of case "
                              "mix cannot be separated",
            "encoder": args.encoder, "freeze_backbone": args.freeze_backbone,
            "protocol": args.protocol, "views": args.views,
            "normalisation": args.normalisation, "epochs": args.epochs,
            "trainable_parameters": n_params,
            "inputs": input_fingerprint(manifest_path, splits_path),
            "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {out}", flush=True)

    if not summaries:
        raise SystemExit("No arm produced predictions.")
    table = pd.concat(summaries, ignore_index=True)
    cols = ["model", "roc_auc_mean", "estimate", "ci_lower", "ci_upper", "balanced_accuracy_mean"]
    print("\n" + table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print("\nChance is 0.5. A within_benign interval excluding 0.5 means the encoder reads "
          "acquisition era, not pathology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
