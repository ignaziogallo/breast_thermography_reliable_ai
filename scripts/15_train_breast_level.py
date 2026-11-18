#!/usr/bin/env python3
"""Breast-level classification from the oblique views.

Two weaknesses of the anterior-view experiment are addressed at once.

*Label dilution.* The anterior thermogram contains both breasts, so for a malignant patient
half the image is the healthy contralateral breast while the label says "malignant". Each
oblique view depicts a single anatomical side and carries that side's own label.

*Demographic confounding.* Age alone reaches AUROC 0.758 on this cohort and no image model
has matched it, so an image result near that level is not attributable to thermal content.
The two breasts of one patient share age, BMI, body temperature, camera and session
exactly. A comparison made **within** a patient therefore cannot be explained by any
patient-level covariate, which makes it the cleanest test of thermal signal available here.

Outputs three levels of analysis:
  predictions_breast.csv  one row per breast (238 per repeat)
  predictions.csv         one row per patient, max over the two breasts -- directly
                          comparable with the age baseline on the same folds
  within_patient.csv      the paired, confounder-immune test: in a patient with one
                          diseased and one normal breast, is the diseased one ranked higher?

Both breasts of a patient always share a fold; the patient splits are used unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import BreastViewDataset
from src.data.manifest import build_breast_table, resolve_image_path
from src.evaluation.statistical_tests import within_patient_test
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/Breast-Thermal-Image"))
    p.add_argument("--manifest", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/experiment_09_breast_level"))
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--weights-cache", type=Path, default=Path("results/pretrained"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--backbone-lr-scale", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-folds", type=int, default=0)
    p.add_argument("--max-collapsed-folds", type=int, default=5)
    p.add_argument("--collapse-std", type=float, default=0.05)
    p.add_argument("--verbose-epochs", action="store_true")
    a = p.parse_args()

    seed_everything(a.seed)
    manifest = read_results_csv(a.manifest).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)

    breasts = build_breast_table(manifest)
    print(f"Breast-level samples: {len(breasts)} from {breasts.patient_id.nunique()} patients")
    print(f"  class counts: {breasts.breast_class.value_counts().to_dict()}")
    print(f"  target (PM): {int(breasts.breast_label.sum())} positive, "
          f"{int((1 - breasts.breast_label).sum())} negative")

    if breasts["image_path"].astype(str).str.startswith("/").any():
        raise SystemExit(f"{a.manifest} stores absolute paths. Re-run scripts/00_audit_dataset.py.")
    probe = resolve_image_path(a.data_root, breasts["image_path"].iloc[0])
    if not probe.exists():
        raise SystemExit(f"Dataset not found: {probe} does not exist. Check --data-root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} encoder={a.encoder}", flush=True)
    pretrained_state = fetch_pretrained_state(a.encoder, a.weights_cache / f"{a.encoder}.pt")

    augment = build_train_augmentation()
    by_patient = {pid: g for pid, g in breasts.groupby("patient_id")}
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate,
                      weight_decay=a.weight_decay, mixed_precision=not a.no_amp,
                      num_workers=a.num_workers, early_stopping=False,
                      backbone_lr_scale=a.backbone_lr_scale, warmup_epochs=a.warmup_epochs)

    fold_keys = sorted({(int(r), int(f)) for r, f in zip(splits["repeat"], splits["fold"])})
    if a.limit_folds:
        fold_keys = fold_keys[:a.limit_folds]
        print(f"SMOKE TEST: {len(fold_keys)} fold(s); nothing final will be written.", flush=True)

    rows, started = [], time.time()
    for n, (repeat, fold) in enumerate(fold_keys, start=1):
        g = splits[(splits["repeat"] == repeat) & (splits["fold"] == fold)]
        ids = {s: g.loc[g["split"] == s, "patient_id"].tolist() for s in ("train", "val", "test")}
        for x, y in (("train", "test"), ("val", "test"), ("train", "val")):
            overlap = set(ids[x]) & set(ids[y])
            assert not overlap, f"{x}/{y} overlap at repeat={repeat} fold={fold}"

        tables = {s: pd.concat([by_patient[pid] for pid in ids[s]], ignore_index=True)
                  for s in ("train", "val", "test")}
        # Both breasts of a patient must land in the same partition.
        for s in ("train", "val", "test"):
            assert set(tables[s]["patient_id"]) == set(ids[s]), f"{s} breast/patient mismatch"

        seed_everything(a.seed + repeat * 100 + fold)
        model = build_model_from_state(pretrained_state, a.encoder)
        datasets = {s: BreastViewDataset(tables[s], a.data_root, a.image_size,
                                         augment if s == "train" else None)
                    for s in ("train", "val", "test")}
        out = train_fold(model, datasets["train"], datasets["val"], datasets["test"], cfg, device,
                         log_prefix=f"[r{repeat} f{fold}]" if a.verbose_epochs else "")
        threshold = select_threshold(out["val_y"], out["val_prob"])

        meta = tables["test"].set_index("sample_id")
        for sid, yt, pp in zip(out["test_ids"], out["test_y"], out["test_prob"]):
            m = meta.loc[sid]
            rows.append({
                "sample_id": sid, "patient_id": m["patient_id"], "side": m["side"],
                "breast_class": m["breast_class"], "repeat": repeat, "fold": fold,
                "y_true": int(yt), "y_prob": float(pp), "y_pred": int(pp >= threshold),
                "threshold": float(threshold), "model": f"{a.encoder}_breast_level",
            })
        elapsed = time.time() - started
        print(f"[{n}/{len(fold_keys)}] r{repeat} f{fold}: val_auroc {out['best_val_auroc']:.3f} "
              f"| {elapsed/n:.0f}s/fold | eta {(len(fold_keys)-n)*elapsed/n/60:.0f} min", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    breast_preds = pd.DataFrame(rows)
    if a.limit_folds:
        print(f"\nSmoke test complete: {len(breast_preds)} breast predictions.")
        return 0

    # Patient level: a patient is malignant if either breast is, so the patient score is the
    # larger of the two. Produces one row per patient per repeat, matching every other
    # experiment's format and therefore comparable on identical folds.
    patient_preds = (breast_preds.groupby(["model", "repeat", "fold", "patient_id"])
                     .agg(y_prob=("y_prob", "max"), threshold=("threshold", "first"))
                     .reset_index())
    truth = manifest.set_index("patient_id")["patient_binary_label"].astype(int)
    patient_preds["y_true"] = patient_preds["patient_id"].map(truth)
    patient_preds["y_pred"] = (patient_preds["y_prob"] >= patient_preds["threshold"]).astype(int)
    patient_preds["config"] = json.dumps({"aggregation": "max over the two breasts"})

    spread = patient_preds.groupby(["repeat", "fold"])["y_prob"].std()
    collapsed = spread[spread < a.collapse_std]
    gate = {"collapsed_folds": int(len(collapsed)), "total_folds": int(len(spread)),
            "threshold": a.collapse_std, "max_allowed": a.max_collapsed_folds,
            "passed": bool(len(collapsed) <= a.max_collapsed_folds)}
    if not gate["passed"]:
        a.output_dir = a.output_dir.with_name(a.output_dir.name + "_FAILED_QUALITY_GATE")
        print(f"\nQUALITY GATE FAILED: {gate['collapsed_folds']}/{gate['total_folds']} folds "
              f"near-constant. Artifacts in {a.output_dir}; not a reportable result.", flush=True)

    within = pd.concat(
        [within_patient_test(g).assign(repeat=r) for r, g in breast_preds.groupby("repeat")],
        ignore_index=True)
    per_repeat, ci, summary = summarize_predictions(patient_preds, a.n_bootstrap, a.seed)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    breast_preds.to_csv(a.output_dir / "predictions_breast.csv", index=False)
    patient_preds.to_csv(a.output_dir / "predictions.csv", index=False)
    within.to_csv(a.output_dir / "within_patient.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": "experiment_09_breast_level",
        "research_question": "does an oblique-view model discriminate the diseased breast "
                             "from its contralateral counterpart within the same patient",
        "encoder": a.encoder, "arm": "finetune_discriminative_lr",
        "trainable_parameters": count_parameters(build_model_from_state(pretrained_state, a.encoder)),
        "n_breast_samples": int(len(breasts)),
        "breast_class_counts": {k: int(v) for k, v in breasts.breast_class.value_counts().items()},
        "quality_gate": gate,
        "epochs": a.epochs, "batch_size": a.batch_size, "learning_rate": a.learning_rate,
        "backbone_lr_scale": a.backbone_lr_scale, "warmup_epochs": a.warmup_epochs,
        "augmentation": describe(10.0, 0.05, (0.9, 1.1)),
        "model_selection": "none: fixed epoch budget, the final model is scored",
        "threshold_rule": "balanced accuracy maximized on validation breasts of each fold",
        "patient_aggregation": "max over the two breasts",
        "within_patient_test": "sign test on the ordering of the two breasts of one patient; "
                               "immune to any patient-level covariate, including age",
        "inputs": input_fingerprint(a.manifest, a.splits),
        "device": str(device), "seed": a.seed,
        "runtime_seconds": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(breast_preds)} breast and {len(patient_preds)} patient predictions "
          f"to {a.output_dir}")
    print(f"Quality gate: {gate['collapsed_folds']}/{gate['total_folds']} -> "
          f"{'PASSED' if gate['passed'] else 'FAILED'}")
    print("\nPatient level (max over breasts), bootstrap 95% CI:")
    print(ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]]
          .round(3).to_string(index=False))
    print("\nWithin-patient paired test -- immune to age, BMI, temperature, camera, session:")
    agg = (within.groupby("contrast")
           .agg(n=("n_patients", "first"), proportion=("proportion", "mean"),
                p_median=("p_one_sided", "median")).round(4))
    print(agg.to_string())
    print("\nUnder the null the proportion is 0.5. This is the test that a patient-level "
          "confounder cannot produce.")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
