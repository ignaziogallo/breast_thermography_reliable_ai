#!/usr/bin/env python3
"""How much of a reported DMR-IR accuracy is produced by the split?

Pinto et al. (IEEE ColCACI 2025) report 0.975 accuracy on DMR-IR, and 0.995 on the frontal
view, from an *image-level* 85/15 partition with no test set. DMR-IR holds up to 27
correlated frames per patient: five static views plus a 20-frame dynamic recovery sequence,
of which 97 static frontals are byte-identical to a frame of the same patient's sequence.
A random frame split therefore puts near-copies of the same breast on both sides of the
partition.

This experiment holds the model, the encoder, the epoch budget, the normalisation and the
fold count fixed, and varies only what a fold is made of:

  patient_level   frames grouped by patient; a patient is wholly in train, val or test.
  image_level     frames partitioned at random, the published protocol.

Both arms are additionally scored at two units of analysis, because the split unit and the
evaluation unit are separate choices that the literature conflates:

  per_frame       one row per frame, as image-level work reports it.
  per_patient     frame probabilities averaged within a patient, one row per patient.

The 2x2 separates two distinct inflations -- contamination across the partition, and
counting 27 correlated frames as 27 independent observations -- which a single headline
number confounds. `results/<out>/contamination.csv` records, per fold, the fraction of test
frames whose patient was also seen in training: the exposure that produces the gap.

The image-level arm is deliberately leaky, so its predictions are written to
`predictions_image_level_LEAKY.csv`. Only the patient-level arm writes `predictions.csv`,
the filename `tests/test_no_prediction_leakage.py` collects -- the leaky arm would fail that
test by construction, and suppressing it would defeat its purpose.

Example
-------
    CUDA_VISIBLE_DEVICES=2 python scripts/03_leakage_experiment.py \\
        --data-root data/DMR-IR \\
        --patients  results/dmr_patient_manifest.csv \\
        --images    results/dmr_image_manifest.csv \\
        --splits    results/dmr_patient_splits.csv \\
        --output-dir results/experiment_dmr_01_leakage \\
        --freeze-backbone --epochs 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dmr import (
    ALL_VIEWS,
    DYNAMIC,
    STATIC,
    Normalisation,
    aggregate_to_patient,
    cohort_statistics,
    select_frames,
)
from src.data.dmr_datasets import DmrFrameDataset
from src.data.splits import make_image_level_splits, measure_split_contamination
from src.evaluation.summary import select_threshold, summarize_predictions
from src.models.single_view import (
    DEFAULT_ENCODER,
    build_model_from_state,
    count_parameters,
    fetch_pretrained_state,
)
from src.preprocessing.augmentations import build_train_augmentation, describe
from src.training.trainer import TrainConfig, train_fold
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

PROTOCOLS = {"static": STATIC, "dynamic": DYNAMIC, "all": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/DMR-IR"))
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--images", type=Path, default=Path("results/dmr_image_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/dmr_patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/experiment_dmr_01_leakage"))
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="all",
                   help="which acquisition protocol's frames enter the pool")
    p.add_argument("--views", nargs="+", default=list(ALL_VIEWS), choices=list(ALL_VIEWS))
    p.add_argument("--arms", nargs="+", default=["patient_level", "image_level"],
                   choices=["patient_level", "image_level"])
    p.add_argument("--normalisation", choices=["per_image", "fixed"], default="per_image",
                   help="per_image matches the DMR-IR literature; fixed keeps absolute temperature")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="linear probe; the low-capacity arm, far cheaper on 6051 frames")
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--weights-cache", type=Path, default=Path("results/pretrained"))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-folds", type=int, default=0,
                   help="run only the first N folds of each repeat; smoke-test switch")
    return p.parse_args()


def build_pool(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frame pool both arms share, and the patient manifest restricted to it."""
    patients = read_results_csv(args.patients).dropna(subset=["patient_binary_label"])
    images = read_results_csv(args.images)
    frames = select_frames(images, protocol=PROTOCOLS[args.protocol], views=tuple(args.views))
    frames = frames[frames["patient_id"].isin(set(patients["patient_id"]))].copy()
    if frames.empty:
        raise SystemExit(f"No frames for protocol={args.protocol} views={args.views}")
    frames["image_id"] = frames["image_path"]
    patients = patients[patients["patient_id"].isin(set(frames["patient_id"]))]
    return frames.reset_index(drop=True), patients.reset_index(drop=True)


def fold_frames(frames: pd.DataFrame, fold_rows: pd.DataFrame, arm: str) -> dict[str, pd.DataFrame]:
    """Map a fold's assignment onto frames, by patient or by frame depending on the arm."""
    out = {}
    for split in ("train", "val", "test"):
        rows = fold_rows[fold_rows["split"] == split]
        if arm == "patient_level":
            out[split] = frames[frames["patient_id"].isin(set(rows["patient_id"]))]
        else:
            out[split] = frames[frames["image_id"].isin(set(rows["image_id"]))]
    return out


def run_arm(arm: str, frames: pd.DataFrame, splits: pd.DataFrame, args: argparse.Namespace,
            state: dict, device: torch.device) -> pd.DataFrame:
    """Train and score every fold of one arm, returning frame-level predictions."""
    augment = build_train_augmentation()
    rows = []
    for (repeat, fold), fold_rows in splits.groupby(["repeat", "fold"]):
        if args.limit_folds and fold > args.limit_folds:
            continue
        part = fold_frames(frames, fold_rows, arm)
        if any(p.empty for p in part.values()):
            print(f"  [{arm}] r{repeat} f{fold}: empty partition, skipped", flush=True)
            continue

        # 'fixed' statistics come from this fold's training frames only. Estimating them on
        # the pool would let the test partition's temperature distribution set the scale.
        if args.normalisation == "fixed":
            mean, std = cohort_statistics(part["train"]["image_path"], args.data_root, seed=args.seed)
            norm = Normalisation("fixed", mean, std)
        else:
            norm = Normalisation("per_image")

        def ds(split, aug=None):
            return DmrFrameDataset(part[split], args.data_root, args.image_size, norm,
                                   augment=aug, id_column="patient_id")

        model = build_model_from_state(state, args.encoder, freeze_backbone=args.freeze_backbone)
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                          patience=args.patience, mixed_precision=not args.no_amp,
                          num_workers=args.num_workers, freeze_backbone=args.freeze_backbone,
                          early_stopping=not args.no_early_stopping)
        t0 = time.time()
        res = train_fold(model, ds("train", augment), ds("val"), ds("test"), cfg, device,
                         log_prefix="")
        threshold = select_threshold(res["val_y"], res["val_prob"])
        print(f"  [{arm}] r{repeat} f{fold}: {len(part['train'])} train / {len(part['test'])} test "
              f"frames, val_auroc {res['best_val_auroc']:.3f}, {time.time() - t0:.0f}s", flush=True)

        test_ids = part["test"]["image_id"].tolist()
        for image_id, pid, yt, pp in zip(test_ids, res["test_ids"], res["test_y"], res["test_prob"]):
            rows.append({"image_id": image_id, "patient_id": pid, "repeat": int(repeat),
                         "fold": int(fold), "y_true": int(yt), "y_prob": float(pp),
                         "y_pred": int(pp >= threshold), "threshold": threshold,
                         "model": arm, "config": json.dumps({"arm": arm})})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames, patients = build_pool(args)
    patient_splits = read_results_csv(args.splits)
    patient_splits = patient_splits[patient_splits["patient_id"].isin(set(patients["patient_id"]))]
    n_repeats = patient_splits["repeat"].nunique()
    n_folds = patient_splits["fold"].nunique()
    print(f"Pool: {len(frames)} frames, {frames['patient_id'].nunique()} patients, "
          f"protocol={args.protocol}, views={args.views}")
    print(f"Device: {device}; folds {n_folds}x{n_repeats}")

    image_splits = make_image_level_splits(frames, n_splits=n_folds, n_repeats=n_repeats,
                                           seeds=sorted(patient_splits["seed"].unique()))
    contamination = pd.concat([
        measure_split_contamination(patient_splits.assign(image_id="")).assign(arm="patient_level"),
        measure_split_contamination(image_splits).assign(arm="image_level"),
    ], ignore_index=True)

    state = fetch_pretrained_state(args.encoder, args.weights_cache / f"{args.encoder}.pt")
    n_params = count_parameters(build_model_from_state(state, args.encoder,
                                                       freeze_backbone=args.freeze_backbone))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contamination.to_csv(args.output_dir / "contamination.csv", index=False)
    print("\nTest frames sharing a patient with training:")
    print(contamination.groupby("arm")["fraction_contaminated"].mean().round(4).to_string())

    summaries, per_frame_all = [], []
    for arm in args.arms:
        print(f"\n=== {arm} ===", flush=True)
        splits = patient_splits if arm == "patient_level" else image_splits
        preds = run_arm(arm, frames, splits, args, state, device)
        if preds.empty:
            continue
        per_frame_all.append(preds)

        # Persist the raw predictions *before* any aggregation or summarisation. Analysis is
        # cheap and rerunnable; these frames cost an hour of GPU per arm, and an exception
        # in the summary would otherwise discard them.
        patient_level = arm == "patient_level"
        name = "predictions.csv" if patient_level else "predictions_image_level_LEAKY.csv"
        (aggregate_to_patient(preds) if patient_level else preds).to_csv(
            args.output_dir / name, index=False)

        # Two units of analysis from the same predictions. Per-frame rows are not
        # independent, so their bootstrap interval is optimistic; it is reported because it
        # is what image-level work reports, and labelled accordingly.
        #
        # Under the image-level split a patient's frames land in several folds of one
        # repeat, so the per-patient view must pool across folds; keying it on the fold
        # would hand the bootstrap several rows for the same patient.
        by_patient = aggregate_to_patient(preds, by_fold=patient_level).assign(
            model=f"{arm}__per_patient")
        by_frame = preds.assign(model=f"{arm}__per_frame",
                                patient_id=preds["image_id"])  # unit = frame
        for unit in (by_patient, by_frame):
            _, ci, summary = summarize_predictions(unit, args.n_bootstrap, args.seed)
            summaries.append(summary.merge(ci[ci["metric"] == "roc_auc"][
                ["model", "estimate", "ci_lower", "ci_upper"]], on="model", how="left"))

    if not summaries:
        raise SystemExit("No arm produced predictions.")
    table = pd.concat(summaries, ignore_index=True)
    table.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(per_frame_all, ignore_index=True).to_csv(
        args.output_dir / "predictions_all_frames.csv", index=False)

    (args.output_dir / "config.json").write_text(json.dumps({
        "experiment": args.output_dir.name,
        "research_question": "how much of a reported DMR-IR score is produced by the split unit",
        "arms": args.arms,
        "protocol": args.protocol, "views": args.views,
        "encoder": args.encoder, "trainable_parameters": n_params,
        "freeze_backbone": args.freeze_backbone, "epochs": args.epochs,
        "normalisation": args.normalisation,
        "augmentation": describe(10.0, 0.05, (0.9, 1.1)),
        "leaky_arm_note": "image_level predictions are written to predictions_image_level_LEAKY.csv; "
                          "they violate patient-level holdout by design and must never be "
                          "reported as a clean estimate",
        "inputs": input_fingerprint(args.patients, args.splits),
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
    }, indent=2), encoding="utf-8")

    cols = ["model", "roc_auc_mean", "estimate", "ci_lower", "ci_upper",
            "balanced_accuracy_mean", "accuracy_at_selected_threshold_mean"]
    print("\n" + table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print(f"\nWrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
