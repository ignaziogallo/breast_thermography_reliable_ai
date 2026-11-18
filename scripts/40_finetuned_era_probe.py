#!/usr/bin/env python3
"""Is the acquisition era still legible in a *fine-tuned* representation?

The paper's legibility column is honest about what it is: one frozen-probe decode per
backbone, which is why the era value repeats across the frozen and fine-tuned rows of the
encoder table. The exploitation claim, however, is about fine-tuned and strong arms. Between
those two facts sits a question nobody has answered here: when the encoder is adapted to
predict malignancy, does its representation keep the era, lose it, or sharpen it?

Any of the three is publishable and none of them moves the central claim:

  keeps / sharpens   the diagnostic objective does not discourage acquisition features, and
                     the frozen decode understates what a trained model has available;
  loses              era legibility is a property of generic ImageNet features rather than of
                     what this task learns -- which would make the score-level exploitation
                     result *more* interesting, not less, because the information would then
                     have to survive a representation that is discarding it.

Design. A fine-tuned model exists per fold, and its representation is its own -- 25 folds are
25 different feature spaces, so pooling them and probing once would average over incomparable
bases. Instead the probe is fitted inside each fold, on that fold's *training* patients, and
scored on that fold's held-out patients; the out-of-fold predictions are then pooled exactly
like a diagnostic model's. The contrast is benign 2012 vs 2013, diagnosis held constant, the
same one the frozen decoders answer, so the numbers sit in the same column.

Requires features written by `scripts/04_train_single_view.py --save-features`. They are not
in the repository: no checkpoints or activations are stored, which is why this needs a rerun
of the fine-tuned arms on the GPU host (`jobs/f3_e5_finetuned_features.txt`).

Example
-------
    python scripts/40_finetuned_era_probe.py \\
        --runs results/experiment_dmr_11_encoder_convnext_tiny_finetune \\
               results/experiment_dmr_11_encoder_swin_tiny_patch4_window7_224_finetune
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dmr import ERA_CONTRASTS
from src.evaluation.predictions import pool_repeats
from src.evaluation.resampling import bootstrap_ci, paired_auc_difference, permutation_p
from src.utils.io import ensure_parent, read_results_csv, write_json

# The clean contrast: benign patients of the two well-populated years.
BENIGN, (EARLY,), (LATE,) = ERA_CONTRASTS["within_benign"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True,
                   help="experiment directories holding a features/ subdirectory")
    p.add_argument("--reference", type=Path, default=None,
                   help="a run whose representation the others are compared against, "
                        "paired over the patients both scored. Without it the fine-tuned "
                        "numbers can only be read against the era decoders of the encoder "
                        "table, which differ from this probe in estimator *and* folds as "
                        "well as in representation -- two changes at once, which is not a "
                        "comparison. Point this at the frozen arm of the same encoder.")
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("results/confounders/finetuned_era_probe.csv"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-permutation", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def era_labels(patients: pd.DataFrame) -> pd.Series:
    """1 = the later era, restricted to benign patients of the two well-populated years."""
    df = patients.copy()
    df["year"] = pd.to_datetime(df["visit_date"], errors="coerce").dt.year
    df = df[(df["patient_binary_label"].astype(int) == BENIGN) & df["year"].isin((EARLY, LATE))]
    return (df.set_index("patient_id")["year"] == LATE).astype(int)


def summarise(run: Path, preds: pd.DataFrame, encoder: str | None,
              args) -> tuple[dict, pd.DataFrame]:
    """Pool the out-of-fold era predictions and put an interval and a p on them."""
    pooled = pool_repeats(preds)
    score, truth = pooled["y_prob"].to_numpy(), pooled["y_true"].to_numpy()
    auc = roc_auc_score(truth, score)
    lo, hi, valid = bootstrap_ci(score, truth, args.n_bootstrap,
                                 np.random.default_rng(args.seed))
    pval = permutation_p(score, truth, auc, args.n_permutation,
                         np.random.default_rng(args.seed))
    return {"run": run.name, "encoder": encoder, "n_patients": len(pooled),
            "n_folds": preds.groupby(["repeat", "fold"]).ngroups,
            "era_auroc_benign": auc, "era_auroc_benign_lo": lo, "era_auroc_benign_hi": hi,
            "era_auroc_benign_perm_p": pval, "n_bootstrap_valid": valid}, pooled


def probe_run(run: Path, era: pd.Series, args) -> tuple[dict, pd.DataFrame] | None:
    """Fit the era probe inside each fold of one run, or pool its stored predictions."""
    folds = sorted((run / "features").glob("r*_f*.npz"))
    if not folds:
        # The activations are deliberately untracked (20 MB per arm), but the probe's own
        # out-of-fold predictions are. Reading those back lets the paired comparison, the
        # intervals and the table be reproduced away from the host that trained the arm --
        # which is where the paper is written.
        stored = run / "era_probe_predictions.csv"
        if stored.exists():
            preds = read_results_csv(stored)
            print(f"  {run.name}: features absent; pooling the stored probe predictions "
                  f"({len(preds)} rows, not refitted)")
            return summarise(run, preds, None, args)
        print(f"  [skip] {run}: no features/ (rerun the arm with --save-features)")
        return None

    estimator = Pipeline([("scale", StandardScaler()),
                          ("model", LogisticRegression(max_iter=5000,
                                                       class_weight="balanced", C=1.0))])
    rows, encoder = [], None
    for path in folds:
        with np.load(path, allow_pickle=True) as z:
            encoder = str(z["encoder"])
            parts = {}
            for split in ("train", "val", "test"):
                ids = pd.Index([str(i) for i in z[f"{split}_ids"]])
                keep = ids.isin(era.index)
                parts[split] = (z[f"{split}_x"][keep], era.reindex(ids[keep]).to_numpy(),
                                ids[keep])
        # The probe may see both eras in training and only one in the held-out set; the
        # first is fatal to fitting, the second only to scoring that fold.
        if len(np.unique(parts["train"][1])) < 2 or len(parts["test"][2]) == 0:
            continue
        model = estimator.fit(parts["train"][0], parts["train"][1])
        prob = model.predict_proba(parts["test"][0])[:, 1]
        repeat, fold = path.stem[1:].split("_f")
        for pid, y, p in zip(parts["test"][2], parts["test"][1], prob):
            rows.append({"patient_id": pid, "repeat": int(repeat), "fold": int(fold),
                         "y_true": int(y), "y_prob": float(p)})

    if not rows:
        print(f"  [skip] {run}: no fold had both eras in its training partition")
        return None

    preds = pd.DataFrame(rows)
    preds.to_csv(run / "era_probe_predictions.csv", index=False)
    return summarise(run, preds, encoder, args)


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, rng) -> dict:
    """Era AUROC of `a` minus that of `b`, over the patients both runs scored.

    The two probes rank the same people, so the difference is paired and gets a proper
    interval -- which reading two independent rows of a table cannot give.
    """
    ids = a.index.intersection(b.index)
    era = a.loc[ids, "y_true"].to_numpy().astype(bool)
    d = paired_auc_difference(a.loc[ids, "y_prob"].to_numpy(), b.loc[ids, "y_prob"].to_numpy(),
                              era, n_boot, rng)
    return {"n_patients": len(ids), "delta": d["delta"],
            "delta_lo": d["delta_lo"], "delta_hi": d["delta_hi"],
            "bootstrap_p": min(1.0, d["bootstrap_p"])}


def main(argv=None) -> int:
    args = parse_args(argv)
    patients = read_results_csv(args.patients).dropna(subset=["patient_binary_label"])
    era = era_labels(patients)
    print(f"within-benign era cohort: {len(era)} patients, "
          f"{int((era == 0).sum())} in {EARLY} / {int((era == 1).sum())} in {LATE}\n")

    runs = list(args.runs) + ([args.reference] if args.reference else [])
    scored = {run: out for run in runs if (out := probe_run(run, era, args))}
    if not scored:
        raise SystemExit("No run had usable features. Run jobs/f3_e5_finetuned_features.txt "
                         "on the GPU host first.")
    df = pd.DataFrame([row for row, _ in scored.values()])
    ensure_parent(args.output)
    df.to_csv(args.output, index=False)
    write_json({
        "question": "does a fine-tuned diagnostic representation still encode the "
                    "acquisition era",
        "contrast": f"benign {EARLY} vs {LATE}",
        "probe": "standardised logistic regression, C=1, fitted per fold on that fold's "
                 "training patients and scored on its held-out patients",
        "comparison": "the frozen-representation decodes reported in the encoder table "
                      "(0.813 ConvNeXt-T, 0.821 Swin-T, 0.823 ResNet-50)",
    }, args.output.parent / "finetuned_era_probe_config.json")

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {args.output}")

    if args.reference and args.reference in scored:
        _, ref = scored[args.reference]
        rows = []
        for run, (row, pooled) in scored.items():
            if run == args.reference:
                continue
            rows.append({"run": row["run"], "reference": args.reference.name,
                         "era_auroc": row["era_auroc_benign"],
                         **paired_delta(pooled, ref, args.n_bootstrap,
                                        np.random.default_rng(args.seed))})
        deltas = pd.DataFrame(rows)
        # Named after --output, not fixed: two invocations writing one filename means the
        # second silently discards the first, which is what happened to the ConvNeXt pair.
        out = args.output.with_name(f"{args.output.stem}_paired.csv")
        deltas.to_csv(out, index=False)
        print("\nPaired against the reference representation, same probe, same folds, "
              "same patients:")
        print(deltas.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"Wrote {out}")
    else:
        print("No --reference given: these numbers are not comparable to the encoder "
              "table's era decodes, which use a different estimator and different folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
