#!/usr/bin/env python3
"""Does a thermography model trained on one cohort work on another?

This is the decisive test for DMR-IR, and the reason is measured, not assumed. Acquisition
epoch predicts malignancy there at AUROC 0.981; the epoch is legible from the pixels at
0.814 among patients sharing a diagnosis (`scripts/24_epoch_control.py`); and the image adds
nothing to the calendar (`scripts/16_incremental_value.py`, delta -0.003 [-0.015, +0.009]).
A patient-level AUROC of 0.916 on DMR-IR is therefore uninterpretable on its own: it is
consistent with a model reading pathology and with one reading a batch signature.

A second cohort separates them. The Cali cohort (Rodríguez-Guerrero et al., Data in Brief
2024; 119 patients, 84/35) was acquired in another country, on another camera, under another
protocol, in 2021-2022. DMR-IR's acquisition signature does not exist there. A model that
learned pathology should retain some discrimination; a model that learned when a DMR-IR
image was taken should collapse to chance. **Chance is the null being tested, and either
outcome is a result.**

Protocol. One source model per repeat: it trains on the `train` patients of
(repeat, `--source-fold`) of the frozen source splits and picks its decision threshold on
that fold's `val` patients. It then scores *every* target patient once. The target cohort is
never used for fitting, model selection or thresholding, so `target_splits.csv` marks every
target patient as held out in every repeat, and the leakage test checks that.

The same models also score their own source test fold, written to
`predictions_source_internal.csv`. That is the in-domain reference the transfer number must
be read against: the drop between them is the quantity of interest, not the transfer AUROC
alone.

Normalisation is the second thing this script tests, and it is not cosmetic:

  per_image   each frame standardised to its own mean and variance. Discards absolute
              temperature, and with it any fixed offset between the two cameras. The
              default, because it is the only mode under which the two cohorts are
              comparable without further assumption.
  fixed       the *source* mean and standard deviation applied to the target. Preserves
              absolute temperature and therefore also any calibration difference between
              cameras. Running both says whether a transfer failure is about pathology or
              about degrees Celsius.

Both cohorts must be in degrees Celsius. DMR-IR TIFFs already are; for Cali use the
radiometric tree and a manifest built over it, not the auto-ranged JPEG renders:

    python scripts/20_extract_radiometric.py \\
        --data-root data/Breast-Thermal-Image \\
        --out       data/Breast-Thermal-Image-Radiometric
    python scripts/01_create_manifest.py \\
        --data-root data/Breast-Thermal-Image-Radiometric --image-ext .tiff \\
        --output    results/patient_manifest_radiometric.csv \\
        --images-output results/image_manifest_radiometric.csv

Example
-------
    CUDA_VISIBLE_DEVICES=4 python scripts/27_transfer_eval.py \\
        --source dmr --target cali --freeze-backbone \\
        --output-dir results/experiment_dmr_06_transfer_dmr_to_cali
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
    STATIC,
    Normalisation,
    aggregate_to_patient,
    cohort_statistics,
    restrict_to_available,
    select_frames,
)
from src.data.dmr_datasets import DmrFrameDataset
from src.evaluation.summary import select_threshold, summarize_predictions
from src.models.single_view import (
    DEFAULT_ENCODER,
    build_model_from_state,
    count_parameters,
    fetch_pretrained_state,
)
from src.preprocessing.augmentations import build_train_augmentation
from src.training.trainer import TrainConfig, train_fold
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

COHORTS = ("dmr", "cali")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=COHORTS, default="dmr")
    p.add_argument("--target", choices=COHORTS, default="cali")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: results/experiment_transfer_<source>_to_<target>")

    p.add_argument("--dmr-root", type=Path, default=Path("data/DMR-IR"))
    p.add_argument("--dmr-patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--dmr-images", type=Path, default=Path("results/dmr_image_manifest.csv"))
    p.add_argument("--dmr-splits", type=Path, default=Path("results/dmr_patient_splits.csv"))
    p.add_argument("--dmr-protocol", default=STATIC)
    p.add_argument("--dmr-views", nargs="+", default=["frontal"], choices=list(ALL_VIEWS))

    p.add_argument("--cali-root", type=Path, default=Path("data/Breast-Thermal-Image-Radiometric"))
    p.add_argument("--cali-manifest", type=Path,
                   default=Path("results/patient_manifest_radiometric.csv"))
    p.add_argument("--cali-splits", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--cali-view", default="anterior")

    p.add_argument("--source-fold", type=int, default=1,
                   help="which fold of each repeat supplies the train/val partition")
    p.add_argument("--normalisation", choices=["per_image", "fixed"], default="per_image")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--weights-cache", type=Path, default=Path("results/pretrained"))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-repeats", type=int, default=0)
    return p.parse_args()


def load_cohort(name: str, args: argparse.Namespace) -> tuple[Path, pd.DataFrame, Path]:
    """Return (data_root, one-row-per-frame table, splits path) for a cohort.

    The two cohorts are stored differently -- DMR-IR long, one row per frame; Cali wide, one
    row per patient with a column per view -- so both are reduced to the same long form here
    and every later stage is cohort-agnostic.
    """
    if name == "dmr":
        patients = read_results_csv(args.dmr_patients).dropna(subset=["patient_binary_label"])
        images = read_results_csv(args.dmr_images)
        frames = select_frames(images, protocol=args.dmr_protocol, views=tuple(args.dmr_views))
        frames = frames[frames["patient_id"].isin(set(patients["patient_id"]))]
        keep = ["patient_id", "image_path", "patient_binary_label"]
        return args.dmr_root, frames[keep].reset_index(drop=True), args.dmr_splits

    manifest = read_results_csv(args.cali_manifest).dropna(subset=["patient_binary_label"])
    column = f"{args.cali_view}_path"
    if column not in manifest.columns:
        raise SystemExit(f"{args.cali_manifest} has no column {column!r}; available views: "
                         f"{sorted(c[:-5] for c in manifest.columns if c.endswith('_path'))}")
    frames = manifest[["patient_id", column, "patient_binary_label"]].rename(
        columns={column: "image_path"})
    frames = frames[frames["image_path"].notna() & frames["image_path"].astype(str).str.len().gt(0)]
    if frames["image_path"].astype(str).str.endswith(".jpg").any():
        raise SystemExit(
            f"{args.cali_manifest} points at .jpg renders, which are per-image auto-ranged and "
            "carry no comparable temperature scale. Build a manifest over the radiometric "
            "tree instead -- see this script's docstring.")
    return args.cali_root, frames.reset_index(drop=True), args.cali_splits


def main() -> int:
    args = parse_args()
    if args.source == args.target:
        raise SystemExit("--source and --target must differ; this measures transfer between "
                         "cohorts, and the two must never be pooled (see data/README.md)")
    out = args.output_dir or Path(f"results/experiment_transfer_{args.source}_to_{args.target}")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_root, src_frames, src_splits_path = load_cohort(args.source, args)
    tgt_root, tgt_frames, _ = load_cohort(args.target, args)
    src_splits = read_results_csv(src_splits_path)

    src_patients = src_frames[["patient_id", "patient_binary_label"]].drop_duplicates()
    src_patients, _ = restrict_to_available(src_patients, src_frames)
    src_splits = src_splits[src_splits["patient_id"].isin(set(src_patients["patient_id"]))]

    tgt_patients = tgt_frames[["patient_id", "patient_binary_label"]].drop_duplicates()
    counts = tgt_patients["patient_binary_label"].value_counts()
    print(f"source {args.source}: {len(src_patients)} patients, {len(src_frames)} frames")
    print(f"target {args.target}: {len(tgt_patients)} patients "
          f"({int(counts.get(0, 0))} benign / {int(counts.get(1, 0))} malignant), "
          f"{len(tgt_frames)} frames")
    print(f"device={device}  normalisation={args.normalisation}", flush=True)

    state = fetch_pretrained_state(args.encoder, args.weights_cache / f"{args.encoder}.pt")
    augment = build_train_augmentation()
    repeats = sorted(src_splits["repeat"].unique())
    if args.limit_repeats:
        repeats = repeats[: args.limit_repeats]

    transfer_rows, internal_rows = [], []
    for repeat in repeats:
        g = src_splits[(src_splits["repeat"] == repeat) & (src_splits["fold"] == args.source_fold)]
        if g.empty:
            raise SystemExit(f"repeat {repeat} has no fold {args.source_fold} in {src_splits_path}")
        ids = {s: set(g.loc[g["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
        part = {s: src_frames[src_frames["patient_id"].isin(ids[s])] for s in ids}

        # 'fixed' statistics come from the source training frames and are then applied
        # unchanged to the target -- that is the point of the arm, not an oversight.
        if args.normalisation == "fixed":
            mean, std = cohort_statistics(part["train"]["image_path"], src_root, seed=args.seed)
            norm = Normalisation("fixed", mean, std)
            print(f"  [r{repeat}] source temperature scale: mean {mean:.2f} sd {std:.2f}")
        else:
            norm = Normalisation("per_image")

        def ds(table, root, aug=None):
            return DmrFrameDataset(table, root, args.image_size, norm, augment=aug,
                                   id_column="patient_id")

        seed_everything(args.seed + int(repeat) * 100)
        model = build_model_from_state(state, args.encoder, freeze_backbone=args.freeze_backbone)
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, patience=args.patience,
                          mixed_precision=not args.no_amp, num_workers=args.num_workers,
                          freeze_backbone=args.freeze_backbone)
        t0 = time.time()
        res = train_fold(model, ds(part["train"], src_root, augment), ds(part["val"], src_root),
                         ds(tgt_frames, tgt_root), cfg, device)
        threshold = select_threshold(res["val_y"], res["val_prob"])
        print(f"  [r{repeat}] trained on {len(part['train'])} source frames, "
              f"val_auroc {res['best_val_auroc']:.3f}, {time.time() - t0:.0f}s", flush=True)

        for pid, yt, pp in zip(res["test_ids"], res["test_y"], res["test_prob"]):
            transfer_rows.append({"patient_id": pid, "repeat": int(repeat),
                                  "fold": int(args.source_fold), "y_true": int(yt),
                                  "y_prob": float(pp), "y_pred": int(pp >= threshold),
                                  "threshold": threshold, "model": "transfer",
                                  "config": json.dumps({"source": args.source})})

        # The same model on its own held-out source patients: the in-domain reference.
        from torch.utils.data import DataLoader

        from src.training.trainer import predict
        loader = DataLoader(ds(part["test"], src_root), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        p_int, y_int, id_int = predict(model, loader, device)
        for pid, yt, pp in zip(id_int, y_int, p_int):
            internal_rows.append({"patient_id": pid, "repeat": int(repeat),
                                  "fold": int(args.source_fold), "y_true": int(yt),
                                  "y_prob": float(pp), "y_pred": int(pp >= threshold),
                                  "threshold": threshold, "model": "source_internal",
                                  "config": json.dumps({"source": args.source})})

    if not transfer_rows:
        raise SystemExit("No repeat produced predictions.")
    out.mkdir(parents=True, exist_ok=True)

    transfer = aggregate_to_patient(pd.DataFrame(transfer_rows))
    internal = aggregate_to_patient(pd.DataFrame(internal_rows))
    transfer.to_csv(out / "predictions.csv", index=False)
    internal.to_csv(out / "predictions_source_internal.csv", index=False)

    # The target cohort is held out in its entirety, in every repeat. Recording that as a
    # split file is what lets the leakage test verify it rather than take it on trust.
    tgt_splits = pd.concat([
        tgt_patients.assign(repeat=int(r), fold=int(args.source_fold), seed=args.seed,
                            split="test").rename(columns={"patient_binary_label": "label"})
        for r in repeats], ignore_index=True)
    tgt_splits = tgt_splits[["repeat", "fold", "seed", "split", "patient_id", "label"]]
    tgt_splits.to_csv(out / "target_splits.csv", index=False)
    tgt_patients.to_csv(out / "target_manifest.csv", index=False)

    summaries = []
    for name, preds in (("transfer", transfer), ("source_internal", internal)):
        _, ci, summary = summarize_predictions(preds, args.n_bootstrap, args.seed)
        summaries.append(summary.merge(ci[ci["metric"] == "roc_auc"][
            ["model", "estimate", "ci_lower", "ci_upper"]], on="model", how="left"))
    table = pd.concat(summaries, ignore_index=True)
    table.to_csv(out / "metrics.csv", index=False)

    (out / "config.json").write_text(json.dumps({
        "experiment": out.name,
        "research_question": f"does a model trained on {args.source} discriminate on {args.target}",
        "null_hypothesis": "chance (AUROC 0.5) on the target; a collapse to chance is evidence "
                           "that the source model learned cohort-specific signal, and is as "
                           "much a result as a successful transfer",
        "source": args.source, "target": args.target,
        "source_fold": args.source_fold, "repeats": [int(r) for r in repeats],
        "n_source_patients": len(src_patients), "n_target_patients": len(tgt_patients),
        "normalisation": args.normalisation,
        "normalisation_note": "under 'fixed' the source mean and sd are applied to the target, "
                              "so a failure may be a camera calibration difference rather than "
                              "a pathology one; run both modes before concluding",
        "reference": "predictions_source_internal.csv holds the same models scored on their own "
                     "held-out source patients; the drop between the two is the quantity of "
                     "interest, not the transfer AUROC alone",
        "threshold_note": "the decision threshold is chosen on source validation patients, so "
                          "balanced accuracy on the target also carries any calibration shift; "
                          "AUROC does not",
        "encoder": args.encoder, "freeze_backbone": args.freeze_backbone,
        "trainable_parameters": count_parameters(
            build_model_from_state(state, args.encoder, freeze_backbone=args.freeze_backbone)),
        "inputs": input_fingerprint(out / "target_manifest.csv", out / "target_splits.csv"),
        "source_inputs": input_fingerprint(
            args.dmr_patients if args.source == "dmr" else args.cali_manifest, src_splits_path),
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
    }, indent=2), encoding="utf-8")

    cols = ["model", "estimate", "ci_lower", "ci_upper", "roc_auc_mean", "roc_auc_std",
            "balanced_accuracy_mean"]
    print("\n" + table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print("\nRead the transfer row against source_internal, not against 0.5 alone.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
