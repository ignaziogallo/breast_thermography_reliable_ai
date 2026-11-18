"""Calibration of predicted probabilities: Brier score and expected calibration error."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width ECE: the bin-weighted gap between mean confidence and observed rate."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # The last bin is closed on the right so that a probability of exactly 1 is counted.
        in_bin = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return float(ece)


def calibration_metrics(y_true, y_prob) -> dict:
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
