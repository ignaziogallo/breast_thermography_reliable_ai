"""Tests for confounder-controlled cohort construction.

Two of these are validity tests rather than unit tests. A matched cohort that had been
selected with any knowledge of how a model scored its patients would produce an AUROC that
describes the selection, and no leakage test downstream could see it; the enforcement is
that the construction API cannot accept a score at all, which is asserted here. The second
checks the covariate rule: matching on a field recorded *because* a lesion was suspected
would balance the classes on the diagnosis.
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.matching import cohorts as M

POST_DIAGNOSTIC = {"biopsy", "mastectomy", "mammography", "radiotherapy"}


def toy(n_benign=20, n_malignant=20, benign_days=(0, 100), malignant_days=(200, 300)):
    rng = np.random.default_rng(0)
    days = np.concatenate([rng.uniform(*benign_days, n_benign),
                           rng.uniform(*malignant_days, n_malignant)])
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n_benign + n_malignant)],
        "patient_binary_label": np.concatenate([np.zeros(n_benign), np.ones(n_malignant)]).astype(int),
        "acquisition_days": days,
        "acquisition_year": (days // 100).astype(int),
        "age_at_visit": rng.normal(55, 10, n_benign + n_malignant),
    })


def test_no_construction_function_accepts_a_score():
    """Cohort construction must be blind to model output, by signature, not by convention."""
    forbidden = {"score", "scores", "prediction", "predictions", "y_prob", "probability"}
    for name, fn in vars(M).items():
        if name.startswith("_") or not inspect.isfunction(fn) or fn.__module__ != M.__name__:
            continue
        params = set(inspect.signature(fn).parameters)
        assert not (params & forbidden), f"{name} accepts a model score: {params & forbidden}"


def test_no_post_diagnostic_covariate_is_matched_on_by_default():
    """The configured covariates must precede any suspicion of disease."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_matched", Path(__file__).resolve().parents[1] / "scripts" / "29_build_matched_cohorts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, cfg in module.DATASETS.items():
        configured = set(cfg["covariates"])
        for group in cfg["propensity_covariates"].values():
            configured |= set(group)
        assert not (configured & POST_DIAGNOSTIC), \
            f"{name} matches on care-pathway fields: {configured & POST_DIAGNOSTIC}"


def test_acquisition_days_is_monotone_and_starts_at_zero():
    days = M.acquisition_days(pd.Series(["2013-01-05", "2012-01-01", "2014-01-01"]))
    assert days.min() == 0
    assert list(days.rank()) == [2.0, 1.0, 3.0]


def test_overlap_cohort_keeps_only_the_shared_range():
    df = toy(benign_days=(0, 150), malignant_days=(100, 300))
    out = M.overlap_cohort(df, "acquisition_days")
    assert out["acquisition_days"].between(100, 150).all()
    assert out["patient_binary_label"].nunique() == 2


def test_stratum_balanced_cohort_equalises_the_classes_within_each_stratum():
    df = toy(benign_days=(0, 300), malignant_days=(0, 300))
    out = M.stratum_balanced_cohort(df, "acquisition_year", seed=1)
    counts = out.groupby("acquisition_year")["patient_binary_label"].value_counts().unstack(fill_value=0)
    assert (counts[0] == counts[1]).all()


def test_stratum_balanced_cohort_drops_single_class_strata():
    """The label-pure sessions are exactly what makes the confounder predictive."""
    df = toy(benign_days=(0, 100), malignant_days=(200, 300))
    out = M.stratum_balanced_cohort(df, "acquisition_year", seed=1)
    assert out.empty


def test_matching_respects_the_caliper_and_uses_each_control_once():
    df = toy(benign_days=(0, 300), malignant_days=(0, 300))
    pairs = M.nearest_neighbour_pairs(df, "acquisition_days", caliper=5.0)
    assert (pairs["distance"] <= 5.0).all()
    assert pairs["benign_id"].is_unique and pairs["malignant_id"].is_unique


def test_matching_returns_nothing_when_the_classes_never_coexist():
    df = toy(benign_days=(0, 100), malignant_days=(1000, 1100))
    assert M.nearest_neighbour_pairs(df, "acquisition_days", caliper=5.0).empty


def test_pairs_to_cohort_carries_the_pair_identifier():
    df = toy(benign_days=(0, 300), malignant_days=(0, 300))
    pairs = M.nearest_neighbour_pairs(df, "acquisition_days", caliper=10.0)
    cohort = M.pairs_to_cohort(df, pairs)
    assert len(cohort) == 2 * len(pairs)
    assert cohort.groupby("pair_id")["patient_binary_label"].nunique().eq(2).all()


def test_smd_falls_after_matching():
    df = toy(benign_days=(0, 100), malignant_days=(50, 150))
    before = M.standardized_mean_differences(df, ["acquisition_days"])["smd"].abs().iloc[0]
    pairs = M.nearest_neighbour_pairs(df, "acquisition_days", caliper=5.0)
    after = M.standardized_mean_differences(M.pairs_to_cohort(df, pairs),
                                            ["acquisition_days"])["smd"].abs().iloc[0]
    assert after < before


def test_describe_cohort_counts_both_classes():
    df = toy()
    got = M.describe_cohort(df, "toy", "acquisition_days")
    assert (got["n_patients"], got["n_benign"], got["n_malignant"]) == (40, 20, 20)
    assert got["acquisition_days_malignant_min"] >= got["acquisition_days_benign_max"]


def test_emitted_ladder_is_consistent_with_its_split_files():
    """Every rung that emitted splits must have split exactly the patients it recorded."""
    root = Path("results/matching")
    if not root.exists():
        pytest.skip("no matched cohorts built yet")
    for ladder_path in root.glob("*/ladder.csv"):
        ladder = pd.read_csv(ladder_path)
        for row in ladder.itertuples():
            rung = ladder_path.parent / row.cohort
            if not (rung / "cohort.csv").exists():
                pytest.skip("per-patient cohort files are rebuilt from the datasets, not distributed")
            cohort = pd.read_csv(rung / "cohort.csv", dtype={"patient_id": str})
            assert len(cohort) == row.n_patients
            splits_path = rung / "splits.csv"
            if not row.evaluable:
                assert not splits_path.exists(), f"{rung} is not evaluable but emitted splits"
                continue
            splits = pd.read_csv(splits_path, dtype={"patient_id": str})
            assert set(splits["patient_id"]) == set(cohort["patient_id"])
            for (rep, fold), g in splits.groupby(["repeat", "fold"]):
                sets = {s: set(g.loc[g["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
                assert not sets["train"] & sets["test"]
                assert not sets["val"] & sets["test"]
                assert not sets["train"] & sets["val"]


def test_size_control_draws_the_requested_class_counts():
    df = toy(n_benign=40, n_malignant=30)
    out = M.size_control_cohort(df, 10, 8, seed=1)
    counts = out["patient_binary_label"].value_counts()
    assert (int(counts.get(0, 0)), int(counts.get(1, 0))) == (10, 8)
    assert out["patient_id"].is_unique


def test_size_control_differs_between_seeds_and_refuses_impossible_draws():
    df = toy(n_benign=40, n_malignant=30)
    a = set(M.size_control_cohort(df, 10, 8, seed=1)["patient_id"])
    b = set(M.size_control_cohort(df, 10, 8, seed=2)["patient_id"])
    assert a != b, "two seeds drew the same subset; the control would have no spread"
    with pytest.raises(ValueError):
        M.size_control_cohort(df, 100, 8, seed=1)


def test_size_control_leaves_the_confounder_alone():
    """The point of the control: same size, confounding intact.

    A matched rung removes the confounder and shrinks the cohort at once. The control must
    reproduce only the second, so its confounder imbalance has to stay close to the full
    cohort's rather than collapsing the way a matched rung's does.
    """
    df = toy(n_benign=40, n_malignant=40)          # classes fully separated in time
    full = abs(M.standardized_mean_differences(df, ["acquisition_days"])["smd"].iloc[0])
    ctrl = M.size_control_cohort(df, 10, 10, seed=3)
    got = abs(M.standardized_mean_differences(ctrl, ["acquisition_days"])["smd"].iloc[0])
    assert got > 0.5 * full
