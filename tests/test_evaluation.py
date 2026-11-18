"""Tests for the evaluation helpers every script shares.

These routines carry the statistics the paper reports -- the shortcut-usage interval and
permutation test, the Holm correction, the quality gate -- so each is pinned to something it
can be checked against independently: scikit-learn's AUROC, a hand-computed adjustment, a
constructed set of folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data.manifest import resolve_image_path
from src.data.splits import fold_patient_ids
from src.evaluation.predictions import pool_repeats, pooled_scores
from src.evaluation.quality_gate import QUARANTINE_SUFFIX, prediction_spread_gate, quarantined
from src.evaluation.resampling import (
    auc_from_ranks,
    bootstrap_ci,
    paired_auc_difference,
    permutation_p,
)
from src.evaluation.statistical_tests import holm
from src.evaluation.summary import auroc_intervals, stored_auroc_interval, with_auroc_interval
from src.utils.io import posix


def scores(n=60, shift=1.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return rng.normal(labels * shift, 1.0), labels


def test_rank_auc_is_the_auroc_including_ties():
    from scipy.stats import rankdata

    score, labels = scores()
    score = np.round(score, 1)                       # force ties
    assert len(np.unique(score)) < len(score)
    got = auc_from_ranks(rankdata(score), labels.astype(bool))
    assert got == pytest.approx(roc_auc_score(labels, score), abs=1e-12)


def test_bootstrap_interval_is_reproducible_and_brackets_the_estimate():
    score, labels = scores(shift=1.5)
    first = bootstrap_ci(score, labels, 300, np.random.default_rng(42))
    again = bootstrap_ci(score, labels, 300, np.random.default_rng(42))
    assert first == again
    lower, upper, valid = first
    assert valid == 300
    assert lower <= roc_auc_score(labels, score) <= upper


def test_bootstrap_reports_nothing_when_one_group_is_absent():
    score, _ = scores()
    lower, upper, valid = bootstrap_ci(score, np.zeros(len(score), dtype=int), 50,
                                       np.random.default_rng(0))
    assert np.isnan(lower) and np.isnan(upper) and valid == 0


def test_permutation_p_separates_signal_from_noise():
    signal, labels = scores(shift=3.0)
    noise = np.random.default_rng(1).normal(size=len(labels))
    p_signal = permutation_p(signal, labels, roc_auc_score(labels, signal), 999,
                             np.random.default_rng(42))
    p_noise = permutation_p(noise, labels, roc_auc_score(labels, noise), 999,
                            np.random.default_rng(42))
    # No relabelling beats a near-perfect separation, so the estimate is its floor.
    assert p_signal == pytest.approx(1 / 1000)
    assert p_noise > 0.05


def test_paired_difference_is_zero_for_identical_scores():
    score, labels = scores()
    d = paired_auc_difference(score, score.copy(), labels.astype(bool), 200,
                              np.random.default_rng(42))
    assert d["delta"] == 0 and d["delta_lo"] == 0 and d["delta_hi"] == 0
    assert d["bootstrap_p"] == 2.0                    # both tails are 1; callers clip at 1


def test_paired_difference_detects_a_better_score():
    good, labels = scores(shift=2.5)
    poor = np.random.default_rng(3).normal(size=len(labels))
    d = paired_auc_difference(good, poor, labels.astype(bool), 300, np.random.default_rng(42))
    assert d["delta"] == pytest.approx(d["auc_a"] - d["auc_b"])
    assert d["delta_lo"] > 0


def test_holm_matches_a_hand_computed_adjustment():
    adjusted = holm(np.array([0.01, 0.04, 0.03]))
    # sorted 0.01, 0.03, 0.04 -> 3*0.01, max(0.03, 2*0.03), max(0.06, 1*0.04)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm(np.array([0.5, 0.9])).max() <= 1.0


def test_pooling_averages_repeats_within_a_patient(tmp_path):
    preds = pd.DataFrame({
        "patient_id": ["A", "A", "B", "B"], "repeat": [1, 2, 1, 2],
        "y_true": [0, 0, 1, 1], "y_prob": [0.2, 0.4, 0.9, 0.7], "model": ["m"] * 4,
    })
    pooled = pool_repeats(preds)
    assert pooled.loc["A", "y_prob"] == pytest.approx(0.3)
    assert pooled.loc["B", "y_true"] == 1

    path = tmp_path / "predictions.csv"
    preds.to_csv(path, index=False)
    stored = pooled_scores(path, "m")
    assert list(stored.columns) == ["score", "y_true"]
    assert stored.loc["B", "score"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="no rows for model"):
        pooled_scores(path, "another_model")


def test_quality_gate_counts_near_constant_folds():
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(1, 6):
        spread = 0.001 if fold <= 2 else 0.3          # two collapsed folds of five
        for p in 0.5 + spread * rng.standard_normal(20):
            rows.append({"repeat": 1, "fold": fold, "y_prob": p})
    preds = pd.DataFrame(rows)

    gate = prediction_spread_gate(preds, collapse_std=0.05, max_collapsed_folds=1)
    assert gate["collapsed_folds"] == 2 and gate["total_folds"] == 5
    assert gate["collapsed_fold_ids"] == ["r1f1", "r1f2"]
    assert gate["passed"] is False
    assert prediction_spread_gate(preds, 0.05, 2)["passed"] is True
    assert quarantined(Path("results/run")).name == "run" + QUARANTINE_SUFFIX


def test_fold_patient_ids_refuses_an_overlap():
    clean = pd.DataFrame({"repeat": 1, "fold": 1,
                          "split": ["train", "train", "val", "test"],
                          "patient_id": ["A", "B", "C", "D"]})
    assert fold_patient_ids(clean) == {"train": ["A", "B"], "val": ["C"], "test": ["D"]}
    leaky = clean.assign(patient_id=["A", "B", "C", "A"])
    with pytest.raises(AssertionError, match="train/test overlap"):
        fold_patient_ids(leaky)


def test_stored_interval_reads_both_layouts(tmp_path):
    ci = pd.DataFrame({"model": ["m", "m"], "metric": ["roc_auc", "pr_auc"],
                       "estimate": [0.8, 0.6], "ci_lower": [0.7, 0.5], "ci_upper": [0.9, 0.7]})
    summary = pd.DataFrame({"model": ["m"], "roc_auc_mean": [0.79]})
    assert list(auroc_intervals(ci)["estimate"]) == [0.8]

    ci.to_csv(tmp_path / "bootstrap_ci.csv", index=False)
    with_auroc_interval(summary, ci).to_csv(tmp_path / "metrics.csv", index=False)
    assert stored_auroc_interval(tmp_path / "bootstrap_ci.csv", "m") == (0.8, 0.7, 0.9)
    assert stored_auroc_interval(tmp_path / "metrics.csv", "m") == (0.8, 0.7, 0.9)
    assert stored_auroc_interval(tmp_path / "metrics.csv", "other") is None
    assert stored_auroc_interval(tmp_path / "missing.csv", "m") is None


def test_recorded_paths_use_forward_slashes_on_every_platform():
    assert posix(Path("results") / "dmr_patient_manifest.csv") == "results/dmr_patient_manifest.csv"


@pytest.mark.parametrize("entry", ["/abs/image.tiff", "C:/data/image.tiff", "C:\\data\\image.tiff"])
def test_an_absolute_manifest_entry_is_refused_whatever_its_flavour(entry):
    with pytest.raises(ValueError, match="absolute path"):
        resolve_image_path(Path("data/DMR-IR"), entry)
