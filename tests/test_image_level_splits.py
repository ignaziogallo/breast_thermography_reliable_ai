"""The image-level split is a leakage control; these tests check it still leaks.

`make_image_level_splits` exists to reproduce the protocol behind the near-perfect DMR-IR
accuracies in the literature, so that the size of the effect can be measured under an
otherwise identical setup. A control arm that quietly stopped being contaminated -- because
someone "fixed" it, or because a cohort has one frame per patient -- would make the
comparison in `scripts/03_leakage_experiment.py` report a gap of zero and look like good
news. These tests fail in that case.

They import only pandas and scikit-learn, so they run on a development workstation with no
deep-learning stack installed; `tests/test_dmr_data.py` covers the tensor path.
"""
import pandas as pd
import pytest

from src.data.splits import (
    make_image_level_splits,
    make_repeated_patient_splits,
    measure_split_contamination,
    validate_split_integrity,
)

FRAMES_PER_PATIENT = 11
N_PATIENTS = 15


@pytest.fixture
def frames() -> pd.DataFrame:
    """A cohort shaped like DMR-IR: many correlated frames per patient."""
    rows = []
    for i in range(N_PATIENTS):
        pid = f"DMR{i:04d}"
        for k in range(FRAMES_PER_PATIENT):
            rows.append({"patient_id": pid, "image_path": f"images/{i:04d}/frame_{k}.tiff",
                         "patient_binary_label": int(i % 3 == 0)})
    return pd.DataFrame(rows)


@pytest.fixture
def patients(frames: pd.DataFrame) -> pd.DataFrame:
    return frames.groupby("patient_id", as_index=False)["patient_binary_label"].first()


def test_image_level_split_covers_every_frame_exactly_once_per_fold(frames):
    splits = make_image_level_splits(frames, n_splits=3, n_repeats=2)
    assert splits["image_id"].nunique() == len(frames)
    for (rep, fold), g in splits.groupby(["repeat", "fold"]):
        assert len(g) == len(frames), f"r{rep} f{fold} does not partition the pool"
        assert not g["image_id"].duplicated().any()


def test_image_level_split_is_contaminated_by_construction(frames):
    splits = make_image_level_splits(frames, n_splits=3, n_repeats=2)
    contamination = measure_split_contamination(splits)
    assert contamination["fraction_contaminated"].mean() > 0.9, (
        "the leakage control is no longer leaking; scripts/03_leakage_experiment.py would "
        "report a spurious gap of zero"
    )
    assert (contamination["n_contaminated"] > 0).all()


def test_patient_level_split_is_clean_on_the_same_pool(frames, patients):
    splits = make_repeated_patient_splits(patients, n_splits=3, n_repeats=2)
    assert not validate_split_integrity(splits)
    contamination = measure_split_contamination(splits.assign(image_id=""))
    assert (contamination["fraction_contaminated"] == 0.0).all()


def test_the_two_protocols_differ_only_in_grouping(frames, patients):
    """Same fold count, same seeds: the gap they produce cannot be a budget difference."""
    seeds = [42, 43]
    image = make_image_level_splits(frames, n_splits=3, n_repeats=2, seeds=seeds)
    patient = make_repeated_patient_splits(patients, n_splits=3, n_repeats=2, seeds=seeds)
    assert set(image["repeat"]) == set(patient["repeat"])
    assert set(image["fold"]) == set(patient["fold"])
    assert set(image["split"]) == set(patient["split"]) == {"train", "val", "test"}


def test_image_level_split_is_deterministic(frames):
    a = make_image_level_splits(frames, n_splits=3, n_repeats=1, seeds=[7])
    b = make_image_level_splits(frames, n_splits=3, n_repeats=1, seeds=[7])
    pd.testing.assert_frame_equal(a, b)
    c = make_image_level_splits(frames, n_splits=3, n_repeats=1, seeds=[8])
    assert not a.equals(c)


def test_measure_contamination_reports_per_fold_exposure(frames):
    splits = make_image_level_splits(frames, n_splits=3, n_repeats=1)
    contamination = measure_split_contamination(splits)
    assert len(contamination) == 3
    assert set(contamination.columns) >= {"repeat", "fold", "n_test", "n_test_patients",
                                          "n_contaminated", "fraction_contaminated"}
    assert (contamination["n_contaminated"] <= contamination["n_test"]).all()


def test_seed_shortage_is_refused(frames):
    with pytest.raises(ValueError, match="n_repeats seeds"):
        make_image_level_splits(frames, n_splits=3, n_repeats=3, seeds=[1, 2])
