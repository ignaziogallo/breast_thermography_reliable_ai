"""From per-patient predictions to the tables every run directory stores.

Shared by the tabular and the image experiments alike, so that a `metrics.csv` or a
`bootstrap_ci.csv` means the same thing whichever script wrote it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.evaluation.bootstrap import patient_bootstrap
from src.evaluation.calibration import calibration_metrics
from src.evaluation.metrics import binary_metrics
from src.evaluation.predictions import pool_repeats

AUROC_INTERVAL_COLUMNS = ["model", "estimate", "ci_lower", "ci_upper"]


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on validation patients only."""
    best_t, best_score = 0.5, -np.inf
    for t in np.unique(np.concatenate([p_val, [0.5]])):
        score = balanced_accuracy_score(y_val, (p_val >= t).astype(int))
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t


def summarize_predictions(preds: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42):
    """Per-repeat metrics, patient-level bootstrap CIs, and a mean/sd summary.

    Returns `(per_repeat, ci, summary)`. The interval is bootstrapped on the
    repeat-averaged probability of each patient, so its resampling unit is the patient
    whatever the number of repeats.
    """
    per_repeat, ci_rows = [], []
    for name, mg in preds.groupby("model"):
        for repeat, rg in mg.groupby("repeat"):
            assert not rg["patient_id"].duplicated().any(), \
                f"{name} repeat {repeat}: a patient received more than one out-of-fold prediction"
            m = binary_metrics(rg["y_true"], rg["y_prob"])
            m.update(calibration_metrics(rg["y_true"], rg["y_prob"]))
            m["accuracy_at_selected_threshold"] = float((rg["y_pred"] == rg["y_true"]).mean())
            per_repeat.append({"model": name, "repeat": int(repeat), "n_patients": len(rg), **m})

        pooled = pool_repeats(mg).reset_index()
        ci, _ = patient_bootstrap(pooled["y_true"], pooled["y_prob"], pooled["patient_id"],
                                  n_bootstrap=n_bootstrap, seed=seed)
        ci.insert(0, "model", name)
        ci_rows.append(ci)

    per_repeat = pd.DataFrame(per_repeat)
    metric_cols = [c for c in per_repeat.columns if c not in ("model", "repeat", "n_patients")]
    summary = per_repeat.groupby("model")[metric_cols].agg(["mean", "std"]).round(4).reset_index()
    summary.columns = ["model"] + [f"{a}_{b}" for a, b in summary.columns[1:]]
    return per_repeat, pd.concat(ci_rows, ignore_index=True), summary


def auroc_intervals(ci: pd.DataFrame) -> pd.DataFrame:
    """The AUROC rows of a bootstrap table: one row per model with its estimate and CI."""
    return ci[ci["metric"] == "roc_auc"][AUROC_INTERVAL_COLUMNS]


def with_auroc_interval(summary: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    """A mean/sd summary with each model's bootstrap AUROC interval merged in.

    The layout of `metrics.csv` for the runs that report several units of analysis or
    several arms in one file.
    """
    return summary.merge(auroc_intervals(ci), on="model", how="left")


def stored_auroc_interval(path, model: str | None = None,
                          metric: str = "roc_auc") -> tuple[float, float, float] | None:
    """One arm's stored estimate and interval, whichever layout the file uses.

    `bootstrap_ci.csv` carries one row per metric and needs filtering; a `metrics.csv` built
    by `with_auroc_interval` already holds the AUROC interval merged into its summary and
    has no `metric` column. Returning None rather than raising lets a table or a figure skip
    an arm that has not been run instead of failing the whole script.
    """
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return None
    if not set(AUROC_INTERVAL_COLUMNS) <= set(df.columns):
        return None
    if "metric" in df.columns:
        df = df[df["metric"] == metric]
    if model is not None:
        df = df[df["model"] == model]
    if df.empty:
        return None
    r = df.iloc[0]
    return float(r["estimate"]), float(r["ci_lower"]), float(r["ci_upper"])
