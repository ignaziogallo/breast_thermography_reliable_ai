"""Patient-level bootstrap intervals for the metrics of `src.evaluation.metrics`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_metrics


def patient_bootstrap(y_true, y_prob, patient_ids, n_bootstrap: int = 1000,
                      seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates with percentile intervals, resampling patients with replacement.

    Returns `(summary, draws)`: one row per metric with its estimate and 95% interval, and
    the per-resample values behind it. A resample holding a single class is skipped, so
    `n_bootstrap_valid` can be smaller than `n_bootstrap`.
    """
    df = pd.DataFrame({"patient_id": patient_ids, "y_true": y_true, "y_prob": y_prob})
    # The resampling unit must be the patient. Frame-level rows would be treated as
    # independent observations and every interval would come out too narrow.
    if df["patient_id"].duplicated().any():
        raise ValueError("patient_bootstrap expects one independent prediction per patient. "
                         "Aggregate views first.")
    rng = np.random.default_rng(seed)
    n = len(df)
    draws = []
    for b in range(n_bootstrap):
        sample = df.iloc[rng.integers(0, n, n)]
        if sample["y_true"].nunique() < 2:
            continue
        draws.append({"bootstrap_iteration": b,
                      **binary_metrics(sample["y_true"], sample["y_prob"])})
    boot = pd.DataFrame(draws)

    rows = []
    for metric, value in binary_metrics(df["y_true"], df["y_prob"]).items():
        values = boot[metric].dropna() if metric in boot else pd.Series(dtype=float)
        rows.append({
            "metric": metric,
            "estimate": value,
            "ci_lower": values.quantile(.025) if len(values) else np.nan,
            "ci_upper": values.quantile(.975) if len(values) else np.nan,
            "n_bootstrap_valid": len(values),
        })
    return pd.DataFrame(rows), boot
