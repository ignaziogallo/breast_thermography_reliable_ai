"""The pre-specified gate that quarantines a run whose folds did not produce a classifier.

A fold whose predictions are near-constant has not learned anything, and pooling such folds
measures the training failure rather than the modality. The criterion references only the
spread of the predictions, never their agreement with the labels, so applying it cannot
select for a favourable result. A run that fails is written under a directory name carrying
`QUARANTINE_SUFFIX` and is reported as a failed configuration, never as a result.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

QUARANTINE_SUFFIX = "_FAILED_QUALITY_GATE"


def prediction_spread_gate(preds: pd.DataFrame, collapse_std: float,
                           max_collapsed_folds: int) -> dict:
    """Count the folds emitting a near-constant prediction and decide whether the run passes.

    `preds` holds one row per scored unit with `repeat`, `fold` and `y_prob`. The returned
    record is what a run stores in its `config.json` under `quality_gate`.
    """
    spread = preds.groupby(["repeat", "fold"])["y_prob"].std()
    collapsed = spread[spread < collapse_std]
    return {
        "criterion": f"at most {max_collapsed_folds} of {len(spread)} folds may have "
                     f"prediction std < {collapse_std}",
        "collapsed_folds": int(len(collapsed)), "total_folds": int(len(spread)),
        "threshold": collapse_std, "max_allowed": max_collapsed_folds,
        "passed": bool(len(collapsed) <= max_collapsed_folds),
        "collapsed_fold_ids": [f"r{r}f{f}" for r, f in collapsed.index],
    }


def quarantined(output_dir: Path) -> Path:
    """Where a run that failed the gate is written instead of `output_dir`."""
    return output_dir.with_name(output_dir.name + QUARANTINE_SUFFIX)
