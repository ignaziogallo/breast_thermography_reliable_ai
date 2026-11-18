"""Patient-level cross-validation for tabular models.

This is the single implementation of the fold loop for every tabular experiment: the
metadata and asymmetry baselines, the metadata attribution on DMR-IR, the incremental-value
models and the signature probes. The leakage-critical steps -- which patients a model may
see, which split may select a hyperparameter, which split may set a decision threshold --
live here and nowhere else, so that a mistake cannot be made in one experiment and not
another.

Protocol, per fold:
  * fit on *train* patients only;
  * choose the hyperparameter and the decision threshold on *val* patients;
  * score *test* patients exactly once.
Out-of-fold predictions are pooled within a repeat, giving one prediction per patient per
repeat.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.data.splits import SPLIT_NAMES, fold_patient_ids
from src.evaluation.summary import select_threshold


def grid_points(grid: dict) -> list[dict]:
    """Every combination of a hyperparameter grid, in a deterministic order."""
    keys = sorted(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def run_patient_cv(table: pd.DataFrame, feature_cols: list[str], splits: pd.DataFrame,
                   estimator, grid: dict, model_name: str) -> pd.DataFrame:
    """Run the frozen patient folds for one model and return per-patient predictions."""
    missing = [c for c in feature_cols if c not in table.columns]
    if missing:
        raise ValueError(f"{model_name}: features absent from the feature table: {missing}")
    if "patient_id" in table.columns:
        # A duplicated patient_id makes .loc[ids] return more rows than requested, which
        # silently misaligns labels against predictions. Refuse rather than mispair.
        dup = table.loc[table["patient_id"].duplicated(), "patient_id"].unique()
        if len(dup):
            raise ValueError(
                f"{model_name}: feature table has duplicate patient_id rows "
                f"({len(dup)}, e.g. {sorted(dup)[:5]}). Labels would be misaligned.")
    meta = table.set_index("patient_id") if "patient_id" in table.columns else table
    absent = set(splits["patient_id"]) - set(meta.index)
    if absent:
        raise ValueError(f"{model_name}: {len(absent)} split patients absent from the feature "
                         f"table, e.g. {sorted(absent)[:5]}")
    rows = []

    for (repeat, fold), g in splits.groupby(["repeat", "fold"]):
        # The premise of the whole study: the three partitions are disjoint, or it fails.
        ids = fold_patient_ids(g)

        X = {s: meta.loc[ids[s], feature_cols].to_numpy(dtype=float) for s in SPLIT_NAMES}
        y = {s: meta.loc[ids[s], "patient_binary_label"].to_numpy(dtype=int) for s in SPLIT_NAMES}

        best, best_auc = None, -np.inf
        for params in grid_points(grid) or [{}]:
            est = clone(estimator).set_params(**params)
            est.fit(X["train"], y["train"])
            p_val = est.predict_proba(X["val"])[:, 1]
            auc = roc_auc_score(y["val"], p_val) if len(np.unique(y["val"])) == 2 else np.nan
            if np.isfinite(auc) and auc > best_auc:
                best, best_auc = (est, params, p_val), auc
        if best is None:  # degenerate validation split
            est = clone(estimator)
            est.fit(X["train"], y["train"])
            best = (est, {}, est.predict_proba(X["val"])[:, 1])

        est, params, p_val = best
        threshold = select_threshold(y["val"], p_val)
        p_test = est.predict_proba(X["test"])[:, 1]

        for pid, yt, pp in zip(ids["test"], y["test"], p_test):
            rows.append({
                "patient_id": pid, "repeat": int(repeat), "fold": int(fold),
                "y_true": int(yt), "y_prob": float(pp),
                "y_pred": int(pp >= threshold), "threshold": threshold,
                "model": model_name, "config": json.dumps(params, sort_keys=True),
                "n_train": len(ids["train"]), "n_val": len(ids["val"]), "n_test": len(ids["test"]),
            })
    return pd.DataFrame(rows)
