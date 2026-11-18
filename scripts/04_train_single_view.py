#!/usr/bin/env python3
"""Anterior-only patient-level baseline with a single pre-specified encoder.

Research question: does a pretrained convolutional encoder applied to the anterior
thermogram exceed the two reference baselines -- age alone (AUROC 0.758,
`scripts/09_metadata_baseline.py`) and bilateral asymmetry descriptors (0.631,
`scripts/09b_asymmetry_baseline.py`) -- on the same frozen patient folds?

The answer that matters is the *difference* from those baselines on paired patient-level
predictions, not the absolute AUROC. Age is strongly associated with the target in this
cohort and is partly legible in a thermogram, so an image model near 0.75 would not have
demonstrated that it uses thermal information.

Protocol, identical to the two baselines so the numbers are comparable:
  * folds come from results/patient_splits.csv, used unchanged;
  * fit on *train*; choose the stopping epoch and the decision threshold on *val*;
  * score *test* exactly once;
  * out-of-fold predictions pooled within a repeat, one prediction per patient per repeat.

One patient contributes one anterior image, so there is no view aggregation.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SingleViewDataset
from src.data.manifest import resolve_image_path
from src.evaluation.summary import select_threshold, summarize_predictions
from src.models.single_view import (
    DEFAULT_ENCODER,
    build_model_from_state,
    count_parameters,
    fetch_pretrained_state,
)
from src.preprocessing.augmentations import build_train_augmentation, describe
from src.training.trainer import TrainConfig, extract_features, train_fold
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/Breast-Thermal-Image"))
    p.add_argument("--manifest", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/experiment_02_single_view"))
    p.add_argument("--view", default="anterior")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="linear probe: train the classifier only")
    p.add_argument("--no-early-stopping", action="store_true",
                   help="run the full epoch budget and score the final model (no epoch selection)")
    p.add_argument("--backbone-lr-scale", type=float, default=1.0,
                   help="backbone learning rate as a fraction of the head's (e.g. 0.01)")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--run-name", default=None,
                   help="model name recorded in predictions.csv (default: derived from the arm)")
    p.add_argument("--max-collapsed-folds", type=int, default=5,
                   help="quality gate: maximum folds allowed to emit a near-constant prediction")
    p.add_argument("--collapse-std", type=float, default=0.05)
    p.add_argument("--weights-cache", type=Path, default=Path("results/pretrained"),
                   help="directory holding the downloaded encoder weights; reused across runs")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-folds", type=int, default=0,
                   help="smoke test: run only the first N folds and write nothing final")
    p.add_argument("--verbose-epochs", action="store_true")
    p.add_argument("--save-features", action="store_true",
                   help="after each fold, write the penultimate-layer representation of "
                        "every patient of that fold to <output-dir>/features/. Needed by "
                        "scripts/40_finetuned_era_probe.py, which asks whether a fine-tuned "
                        "representation still encodes the acquisition era; the frozen "
                        "decodes reported in the paper cannot answer that. Adds one forward "
                        "pass per fold and does not touch training.")
    a = p.parse_args()

    seed_everything(a.seed)
    manifest = read_results_csv(a.manifest).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)

    # The split file is built over the whole cohort, but a view-restricted manifest holds
    # fewer patients -- 231 of 236 hold a DMR-IR frontal view. Without this the fold loop
    # indexes the manifest with patients it does not contain and dies on a bare KeyError.
    # The count is printed and recorded because dropping patients changes the cohort a
    # result describes, and that must not be discovered later from a log.
    known = set(manifest["patient_id"])
    absent = sorted(set(splits["patient_id"]) - known)
    if absent:
        print(f"Restricting splits to the {len(known)} patients in {a.manifest}: "
              f"{len(absent)} dropped, e.g. {absent[:5]}", flush=True)
        splits = splits[splits["patient_id"].isin(known)]
        if splits.empty:
            raise SystemExit(f"No patient of {a.splits} appears in {a.manifest}")

    path_col = f"{a.view}_path"
    if path_col not in manifest.columns:
        raise SystemExit(f"Manifest has no column {path_col!r}")
    if manifest[path_col].astype(str).str.startswith("/").any():
        raise SystemExit(
            f"{a.manifest} stores absolute paths. Regenerate it:\n"
            f"    python scripts/00_audit_dataset.py --data-root {a.data_root}")
    probe = resolve_image_path(a.data_root, manifest[path_col].iloc[0])
    if not probe.exists():
        raise SystemExit(f"Dataset not found: {probe} does not exist. Check --data-root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: no CUDA device; this will be extremely slow.", flush=True)
    print(f"device={device} encoder={a.encoder} view={a.view}", flush=True)

    arm = ("linear_probe" if a.freeze_backbone else
           "finetune_discriminative_lr" if a.backbone_lr_scale != 1.0 else
           "finetune_fixed_budget" if a.no_early_stopping else "finetune_earlystop_auroc")
    run_name = a.run_name or f"{a.encoder}_{arm}"
    print(f"arm={arm} run_name={run_name}", flush=True)

    # Fetched once, before the loop: the fold loop must not depend on the network.
    pretrained_state = fetch_pretrained_state(a.encoder, a.weights_cache / f"{a.encoder}.pt")

    augment = build_train_augmentation()
    by_id = manifest.set_index("patient_id")
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate,
                      weight_decay=a.weight_decay, patience=a.patience,
                      mixed_precision=not a.no_amp, num_workers=a.num_workers,
                      freeze_backbone=a.freeze_backbone,
                      early_stopping=not a.no_early_stopping,
                      backbone_lr_scale=a.backbone_lr_scale,
                      warmup_epochs=a.warmup_epochs)

    fold_keys = sorted({(int(r), int(f)) for r, f in zip(splits["repeat"], splits["fold"])})
    if a.limit_folds:
        fold_keys = fold_keys[:a.limit_folds]
        print(f"SMOKE TEST: {len(fold_keys)} fold(s) only; results will not be written.", flush=True)

    rows, fold_log, started = [], [], time.time()
    for n, (repeat, fold) in enumerate(fold_keys, start=1):
        g = splits[(splits["repeat"] == repeat) & (splits["fold"] == fold)]
        ids = {s: g.loc[g["split"] == s, "patient_id"].tolist() for s in ("train", "val", "test")}
        for x, y in (("train", "test"), ("val", "test"), ("train", "val")):
            overlap = set(ids[x]) & set(ids[y])
            assert not overlap, f"{x}/{y} overlap at repeat={repeat} fold={fold}: {sorted(overlap)[:5]}"

        # A fresh model per fold: carrying weights across folds would leak test patients
        # of one fold into the initialisation used to score another.
        seed_everything(a.seed + repeat * 100 + fold)
        model = build_model_from_state(pretrained_state, a.encoder,
                                       freeze_backbone=a.freeze_backbone)

        datasets = {
            s: SingleViewDataset(by_id.loc[ids[s]].reset_index(), a.data_root, a.view,
                                 a.image_size, augment if s == "train" else None)
            for s in ("train", "val", "test")
        }
        out = train_fold(model, datasets["train"], datasets["val"], datasets["test"], cfg, device,
                         log_prefix=f"[r{repeat} f{fold}]" if a.verbose_epochs else "")
        threshold = select_threshold(out["val_y"], out["val_prob"])

        for pid, yt, pp in zip(out["test_ids"], out["test_y"], out["test_prob"]):
            rows.append({
                "patient_id": pid, "repeat": repeat, "fold": fold,
                "y_true": int(yt), "y_prob": float(pp), "y_pred": int(pp >= threshold),
                "threshold": float(threshold), "model": run_name,
                "config": json.dumps({"lr": a.learning_rate, "epochs": a.epochs,
                                      "best_epoch": out["best_epoch"]}, sort_keys=True),
                "n_train": len(ids["train"]), "n_val": len(ids["val"]), "n_test": len(ids["test"]),
            })
        # The representation that produced the predictions above, for every partition of this
        # fold. All three are needed: a probe of the era on this fold's representation has to
        # be fitted on its training patients and scored on its held-out ones, exactly like the
        # diagnostic model, or it would report an in-sample number.
        if a.save_features:
            feature_dir = a.output_dir / "features"
            feature_dir.mkdir(parents=True, exist_ok=True)
            blocks = {}
            for split in ("train", "val", "test"):
                x, y, pid = extract_features(model, datasets[split], device,
                                             a.batch_size, a.num_workers)
                blocks[f"{split}_x"] = x.astype(np.float32)
                blocks[f"{split}_y"] = y.astype(np.int8)
                blocks[f"{split}_ids"] = np.asarray(pid, dtype=object)
            np.savez_compressed(feature_dir / f"r{repeat}_f{fold}.npz",
                                encoder=a.encoder, run_name=run_name, **blocks)

        fold_log.append({"repeat": repeat, "fold": fold, "best_epoch": out["best_epoch"],
                         "best_val_auroc": out["best_val_auroc"], "epochs_run": out["epochs_run"]})
        elapsed = time.time() - started
        print(f"[{n}/{len(fold_keys)}] r{repeat} f{fold}: best epoch {out['best_epoch']} "
              f"val_auroc {out['best_val_auroc']:.3f} | {elapsed/n:.0f}s/fold "
              f"| eta {(len(fold_keys)-n)*elapsed/n/60:.0f} min", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preds = pd.DataFrame(rows)
    if a.limit_folds:
        print(f"\nSmoke test complete: {len(preds)} predictions over {len(fold_keys)} fold(s). "
              "Re-run without --limit-folds for the full experiment.")
        return 0

    # Quality gate, evaluated before any test metric is read. A fold whose predictions are
    # near-constant has not produced a classifier, and pooling such folds measures the
    # training failure rather than the modality. The criterion references only the spread of
    # the predictions, never their agreement with the labels, so applying it cannot select
    # for a favourable result.
    spread = preds.groupby(["repeat", "fold"])["y_prob"].std()
    collapsed = spread[spread < a.collapse_std]
    gate = {
        "criterion": f"at most {a.max_collapsed_folds} of {len(spread)} folds may have "
                     f"prediction std < {a.collapse_std}",
        "collapsed_folds": int(len(collapsed)), "total_folds": int(len(spread)),
        "threshold": a.collapse_std, "max_allowed": a.max_collapsed_folds,
        "passed": bool(len(collapsed) <= a.max_collapsed_folds),
        "collapsed_fold_ids": [f"r{r}f{f}" for r, f in collapsed.index],
    }
    if not gate["passed"]:
        quarantined = a.output_dir.with_name(a.output_dir.name + "_FAILED_QUALITY_GATE")
        # Features are written inside the fold loop, before the gate can be evaluated. They
        # have to follow the rest of the run into quarantine, or a later probe would read a
        # failed run's representation out of a directory whose name does not say so.
        if a.save_features and (a.output_dir / "features").exists():
            quarantined.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(quarantined / "features", ignore_errors=True)
            shutil.move(str(a.output_dir / "features"), str(quarantined / "features"))
        a.output_dir = quarantined
        print(f"\n{'='*78}\nQUALITY GATE FAILED: {gate['collapsed_folds']}/{gate['total_folds']} folds "
              f"emitted a near-constant prediction (std < {a.collapse_std}).\n"
              f"This run did not produce a working classifier on most folds and its metrics do "
              f"not measure the modality.\nArtifacts written to {a.output_dir} for diagnosis; "
              f"they must not be reported as a result.\n{'='*78}", flush=True)

    per_repeat, ci, summary = summarize_predictions(preds, a.n_bootstrap, a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(a.output_dir / "predictions.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    pd.DataFrame(fold_log).to_csv(a.output_dir / "fold_log.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": "experiment_02_single_view",
        "research_question": "does an anterior-view CNN exceed the E7/E7b reference baselines",
        "arm": arm, "model_name": run_name,
        "success_criterion": "pre-specified: the lower bound of the bootstrap 95% CI on the "
                             "AUROC difference against logistic_age_only must exceed 0 "
                             "(scripts/14_compare_models.py --reference logistic_age_only)",
        "quality_gate": gate,
        "encoder": a.encoder, "pretrained": True, "freeze_backbone": a.freeze_backbone,
        "early_stopping": not a.no_early_stopping,
        "encoder_selection": "fixed in advance; not chosen by comparison (see paper Section on single-view)",
        "trainable_parameters": count_parameters(
            build_model_from_state(pretrained_state, a.encoder, freeze_backbone=a.freeze_backbone)),
        "view": a.view, "image_size": a.image_size,
        "n_patients_scored": len(known),
        "n_patients_dropped_absent_from_manifest": len(absent),
        # The full list, not a sample: tests/test_no_prediction_leakage.py subtracts it from
        # the fold's held-out set to tell a legitimate cohort restriction apart from an
        # experiment that silently failed to score somebody.
        "patients_dropped": absent,
        "epochs_cap": a.epochs, "patience": a.patience, "batch_size": a.batch_size,
        "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
        "backbone_lr_scale": a.backbone_lr_scale, "warmup_epochs": a.warmup_epochs,
        # Recorded because it is the quantity that explains the instability: at 80 training
        # patients and batch 16 an epoch is five steps, so a short budget fine-tunes 28M
        # parameters in well under a hundred updates.
        "optimiser_steps_per_fold": int(a.epochs * np.ceil(preds["n_train"].iloc[0] / a.batch_size)),
        "optimiser": "AdamW",
        "scheduler": "LinearLR warmup then CosineAnnealingLR" if a.warmup_epochs else "CosineAnnealingLR",
        "loss": "BCEWithLogits with pos_weight from the training partition",
        "normalisation": "per-image z-score; exports are auto-ranged so ImageNet statistics do not apply",
        "augmentation": describe(10.0, 0.05, (0.9, 1.1)),
        "model_selection": ("epoch with the highest validation AUROC; weights restored before scoring test"
                            if not a.no_early_stopping else
                            "none: fixed epoch budget, the final model is scored"),
        "threshold_rule": "balanced accuracy maximized on validation patients of each fold",
        "aggregation": "one anterior image per patient; out-of-fold predictions pooled within a repeat",
        "features_saved": bool(a.save_features),
        "data_root": str(a.data_root), "n_bootstrap": a.n_bootstrap, "seed": a.seed,
        "inputs": input_fingerprint(a.manifest, a.splits),
        "device": str(device), "runtime_seconds": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(preds)} predictions to {a.output_dir}")
    print(f"Quality gate: {gate['collapsed_folds']}/{gate['total_folds']} collapsed folds "
          f"(max {a.max_collapsed_folds}) -> {'PASSED' if gate['passed'] else 'FAILED'}")
    show = ["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "balanced_accuracy_mean"]
    print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    roc = ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]]
    print("\nPatient-level bootstrap 95% CI (AUROC):")
    print(roc.round(3).to_string(index=False))
    print("\nReference baselines on identical folds: age only 0.758, asymmetry 0.631.")
    print("The pre-specified success criterion is not the AUROC above but the paired difference:")
    print("  python scripts/14_compare_models.py --reference logistic_age_only \\")
    print(f"    --predictions {a.output_dir}/predictions.csv \\")
    print("      results/experiment_07_metadata/predictions.csv \\")
    print("      results/experiment_07b_asymmetry/predictions.csv")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
