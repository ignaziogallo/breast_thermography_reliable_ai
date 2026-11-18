"""Tests for the confounder-conditional discrimination estimator.

The estimator is the one the confounder-controlled analysis rests on, so what is checked
here is not that it runs but that it means what the manuscript will say it means: that an
unrestricted caliper reproduces the ordinary AUROC exactly, that a score which is purely the
confounder collapses to chance once the caliper is narrow, and that a genuine within-stratum
signal survives the same restriction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.confounding.conditional import (
    bootstrap_conditional,
    caliper_series,
    conditional_c_statistic,
    paired_conditional_difference,
)


def test_unrestricted_caliper_reproduces_auroc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = rng.normal(size=200) + y
    conf = rng.normal(size=200)
    got = conditional_c_statistic(y, score, conf, caliper=None)
    assert got["c_statistic"] == pytest.approx(roc_auc_score(y, score), abs=1e-9)
    assert got["n_pairs"] == int((y == 1).sum() * (y == 0).sum())


def test_ties_count_as_half_a_pair():
    y = [0, 1]
    got = conditional_c_statistic(y, [0.5, 0.5], [0.0, 0.0], caliper=None)
    assert got["c_statistic"] == pytest.approx(0.5)


def test_pure_confounder_collapses_to_chance_under_a_narrow_caliper():
    """A score that *is* the confounder must lose its discrimination when pairs are matched.

    This is the property the whole design depends on. Two acquisition eras, benign early and
    malignant late, reproduce DMR-IR's structure in miniature: unrestricted, the confounder
    separates the classes perfectly; restricted to pairs acquired within a short window, it
    cannot.
    """
    rng = np.random.default_rng(1)
    n = 60
    conf = np.concatenate([rng.uniform(0, 100, n), rng.uniform(50, 150, n)])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    unrestricted = conditional_c_statistic(y, conf, conf, caliper=None)["c_statistic"]
    narrow = conditional_c_statistic(y, conf, conf, caliper=5.0)
    assert unrestricted > 0.8
    assert narrow["n_pairs"] > 50, "the fixture must retain pairs, or it tests nothing"
    assert abs(narrow["c_statistic"] - 0.5) < 0.15


def test_within_stratum_signal_survives_the_caliper():
    """A score carrying real signal keeps it when the confounder is held constant."""
    rng = np.random.default_rng(2)
    n = 80
    conf = np.concatenate([rng.uniform(0, 100, n), rng.uniform(0, 100, n)])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    score = y * 2.0 + rng.normal(scale=0.5, size=2 * n)      # independent of the confounder
    narrow = conditional_c_statistic(y, score, conf, caliper=5.0)
    assert narrow["c_statistic"] > 0.9


def test_bootstrap_flags_chance_and_reports_pair_counts():
    rng = np.random.default_rng(3)
    y = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)
    conf = rng.uniform(0, 10, 80)
    signal = bootstrap_conditional(y, y + rng.normal(scale=0.3, size=80), conf, 2.0,
                                   n_bootstrap=200, seed=0)
    noise = bootstrap_conditional(y, rng.normal(size=80), conf, 2.0, n_bootstrap=200, seed=0)
    assert signal["exceeds_chance"] is True
    assert noise["exceeds_chance"] is False
    assert signal["n_pairs"] == noise["n_pairs"] > 0
    assert signal["ci_lower"] <= signal["c_statistic"] <= signal["ci_upper"]


def test_paired_difference_is_zero_for_identical_scores():
    rng = np.random.default_rng(4)
    y = np.concatenate([np.zeros(30), np.ones(30)]).astype(int)
    conf = rng.uniform(0, 10, 60)
    s = rng.normal(size=60)
    got = paired_conditional_difference(y, s, s, conf, 3.0, n_bootstrap=200, seed=0)
    assert got["delta"] == pytest.approx(0.0)
    assert got["improvement_demonstrated"] is False
    assert got["bootstrap_p"] == pytest.approx(1.0)


def test_paired_difference_detects_a_real_gap():
    rng = np.random.default_rng(5)
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(int)
    conf = rng.uniform(0, 10, 120)
    good = y + rng.normal(scale=0.3, size=120)
    bad = rng.normal(size=120)
    got = paired_conditional_difference(y, good, bad, conf, 3.0, n_bootstrap=300, seed=0)
    assert got["delta"] > 0.3
    assert got["improvement_demonstrated"] is True


def test_caliper_series_rejects_duplicated_patients():
    df = pd.DataFrame({"patient_id": ["a", "a"], "patient_binary_label": [0, 1],
                       "conf": [1.0, 2.0], "s": [0.1, 0.9]})
    with pytest.raises(ValueError, match="one row per patient"):
        caliper_series(df, {"m": "s"}, "conf", [None], n_bootstrap=10)


def test_caliper_series_shape_and_columns():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(60)],
        "patient_binary_label": np.concatenate([np.zeros(30), np.ones(30)]).astype(int),
        "conf": rng.uniform(0, 10, 60),
        "a": rng.normal(size=60), "b": rng.normal(size=60),
    })
    out = caliper_series(df, {"A": "a", "B": "b"}, "conf", [None, 2.0], n_bootstrap=50)
    assert len(out) == 4
    assert set(out["caliper"]) == {"none", 2.0}
    assert {"c_statistic", "ci_lower", "ci_upper", "n_pairs", "exceeds_chance"} <= set(out.columns)


def test_degenerate_arm_returns_nan_rather_than_raising():
    """A caliper narrow enough to admit no pair is an expected end of the series."""
    got = conditional_c_statistic([0, 1], [0.1, 0.9], [0.0, 1000.0], caliper=1.0)
    assert got["n_pairs"] == 0
    assert np.isnan(got["c_statistic"])


def sessions_fixture():
    """Label-pure sessions a few days apart, plus one mixed session.

    This is DMR-IR in miniature: a score that reads the acquisition session orders almost
    every date-comparable pair correctly, because the sessions either side of a short
    caliper carry different diagnoses. Only requiring the two patients to share a session
    closes that route.
    """
    rows = []
    # The score is a per-session constant that happens to align with that session's
    # diagnosis: the acquisition signature of a label-pure session, and nothing else.
    for s, (day, label) in enumerate([(0, 0), (5, 1), (10, 0), (15, 1), (20, 0), (25, 1)]):
        for _ in range(6):
            rows.append({"session": f"S{s}", "day": float(day), "y": label,
                         "session_score": 3.0 if label else 1.0})
    for i in range(6):                                   # one mixed session, no image signal
        rows.append({"session": "MIX", "day": 12.0, "y": i % 2, "session_score": 2.0})
    return pd.DataFrame(rows)


def test_session_signature_survives_a_caliper_but_not_the_stratum():
    df = sessions_fixture()
    caliper = conditional_c_statistic(df.y, df.session_score, df.day, caliper=7.0)
    stratum = conditional_c_statistic(df.y, df.session_score, df.day, caliper=None,
                                      stratum=df.session)
    assert caliper["n_pairs"] > 50
    assert caliper["c_statistic"] > 0.8, "the fixture must reproduce the shortcut"
    # Within a session the score is constant, so every admissible pair is a tie: 0.5 exactly.
    assert stratum["c_statistic"] == pytest.approx(0.5)
    assert stratum["n_pairs"] < caliper["n_pairs"]


def test_stratum_admits_only_pairs_sharing_the_value():
    y = [0, 1, 0, 1]
    got = conditional_c_statistic(y, [0.1, 0.9, 0.9, 0.1], [0.0, 0.0, 0.0, 0.0],
                                  caliper=None, stratum=["a", "a", "b", "b"])
    assert got["n_pairs"] == 2                       # (a0,a1) and (b0,b1), not the cross pairs
    assert got["c_statistic"] == pytest.approx(0.5)  # one ordered, one reversed


def test_stratum_with_no_shared_value_returns_nan():
    got = conditional_c_statistic([0, 1], [0.1, 0.9], [0.0, 0.0], caliper=None,
                                  stratum=["a", "b"])
    assert got["n_pairs"] == 0 and np.isnan(got["c_statistic"])


def test_caliper_series_adds_one_stratum_row_per_score():
    df = sessions_fixture()
    df["patient_id"] = [f"P{i:03d}" for i in range(len(df))]
    df["patient_binary_label"] = df["y"]
    out = caliper_series(df, {"session": "session_score"}, "day", [None, 7.0],
                         n_bootstrap=50, stratum_column="session")
    assert set(out["caliper"]) == {"none", 7.0, "same_session"}
    assert len(out) == 3
