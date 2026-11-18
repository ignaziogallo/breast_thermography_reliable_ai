"""Stored prediction files, reduced to the unit every interval is computed on: the patient.

A `predictions.csv` holds one row per patient per repeat. Averaging the repeats gives one
out-of-fold score per patient, which is what a patient-level bootstrap resamples and what a
paired comparison between two arms is made on.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import read_results_csv


def pool_repeats(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per patient, indexed by `patient_id`: the label and the mean probability."""
    return preds.groupby("patient_id").agg(y_true=("y_true", "first"),
                                           y_prob=("y_prob", "mean"))


def pooled_scores(predictions: Path, model: str | None = None) -> pd.DataFrame:
    """One score per patient from a stored run: the mean out-of-fold probability.

    `model` selects one arm of a file that holds several; a file without a `model` column
    is taken whole. The result is indexed by `patient_id` with columns `score` and `y_true`.
    """
    preds = read_results_csv(predictions)
    if model is not None and "model" in preds.columns:
        preds = preds[preds["model"] == model]
        if preds.empty:
            raise ValueError(f"{predictions}: no rows for model {model!r}")
    return preds.groupby("patient_id").agg(score=("y_prob", "mean"), y_true=("y_true", "first"))
