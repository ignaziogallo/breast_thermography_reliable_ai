#!/usr/bin/env python3
"""The four diagnostics on which the epoch-selecting run was discarded.

The supplement reports a run that chose its stopping epoch by maximising AUROC on a
15-patient validation partition, and reports four numbers as the reason it was discarded:
the correlation between the selected validation AUROC and the test AUROC, the mean optimism
between them, the median selected epoch, and how many folds emitted a near-constant
probability. Those four were computed by hand when the run was quarantined and lived only
in prose, which makes them the only numbers in the paper that no script reproduces.

This script recomputes all four from the run's own stored files, so they are checkable like
everything else:

    fold_log.csv      best_epoch and best_val_auroc, written per fold by the trainer
    predictions.csv   the held-out patient scores, from which the test AUROC and the
                      prediction spread of each fold are recomputed here

Every criterion is computed without reference to the test labels *of the selection
decision*: the run was discarded because validation AUROC does not predict test AUROC and
because most folds collapsed, and both of those are properties of the run, not of a
threshold tuned after seeing the outcome. The test AUROC enters only to *measure* the
optimism that the selection produced.

Example
-------
    python scripts/43_selection_diagnostics.py
    python scripts/43_selection_diagnostics.py \\
        --run results/experiment_02_single_view_earlystop_auroc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

# The gate the rest of the pipeline uses. A fold whose predictions have a standard deviation
# below this is emitting one number for every patient, which an AUROC still scores.
COLLAPSE_SD = 0.05


def fold_table(run: Path) -> pd.DataFrame:
    """One row per (repeat, fold): selected epoch, validation AUROC, test AUROC, spread."""
    log = pd.read_csv(run / "fold_log.csv")
    preds = pd.read_csv(run / "predictions.csv")
    rows = []
    for (repeat, fold), group in preds.groupby(["repeat", "fold"]):
        rows.append({
            "repeat": int(repeat),
            "fold": int(fold),
            "n_test": len(group),
            "test_auroc": float(roc_auc_score(group["y_true"], group["y_prob"])),
            "prediction_sd": float(group["y_prob"].std(ddof=0)),
        })
    scored = pd.DataFrame(rows)
    return scored.merge(log, on=["repeat", "fold"], how="inner")


def summarise(table: pd.DataFrame, n_val: int) -> pd.DataFrame:
    """The four diagnostics, and the counts they are read against, as one long table."""
    collapsed = int((table["prediction_sd"] < COLLAPSE_SD).sum())
    mean_val = float(table["best_val_auroc"].mean())
    mean_test = float(table["test_auroc"].mean())
    return pd.DataFrame([
        {"statistic": "folds", "value": float(len(table))},
        {"statistic": "collapsed_folds", "value": float(collapsed)},
        {"statistic": "corr_selected_val_vs_test",
         "value": float(table["best_val_auroc"].corr(table["test_auroc"]))},
        {"statistic": "mean_selected_val_auroc", "value": mean_val},
        {"statistic": "mean_test_auroc", "value": mean_test},
        {"statistic": "mean_optimism", "value": mean_val - mean_test},
        {"statistic": "median_selected_epoch", "value": float(table["best_epoch"].median())},
        {"statistic": "n_validation_patients", "value": float(n_val)},
    ])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path,
                   default=Path("results/experiment_02_single_view_earlystop_auroc"),
                   help="the quarantined epoch-selecting run")
    p.add_argument("--output", type=Path,
                   default=Path("results/confounders/selection_diagnostics.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    if not (a.run / "fold_log.csv").exists():
        raise SystemExit(
            f"{a.run} has no fold_log.csv. This script reads a completed run of "
            "scripts/04_train_single_view.py; see the README for which stored runs ship "
            "with the reproducibility package.")

    table = fold_table(a.run)
    n_val = int(pd.read_csv(a.run / "predictions.csv", usecols=["n_val"])["n_val"].iloc[0])
    summary = summarise(table, n_val)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.output, index=False)
    table.to_csv(a.output.with_name(a.output.stem + "_per_fold.csv"), index=False)

    width = max(len(s) for s in summary["statistic"])
    for _, row in summary.iterrows():
        print(f"{row['statistic']:<{width}}  {row['value']:.4f}")
    print(f"\nwrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
