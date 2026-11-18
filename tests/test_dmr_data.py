"""Tests for the DMR-IR adapters and the image-level split used as a leakage control.

These run against a synthetic cohort written to a temporary directory, so they need neither
the 9 GB mirror nor a GPU. They do need torch, which is installed on the compute server and
not on a development workstation, so the module skips where it cannot run --
`tests/test_image_level_splits.py` covers the split logic with pandas alone and runs
everywhere.

What they pin down is behaviour that would otherwise surface only as a wrong number in a
results table: that a mask is never paired with a different frame's image, that a missing
view degrades to a flagged zero frame instead of a crash, that `fixed` normalisation refuses
to run without training statistics, and that frame predictions collapse to one row per
patient before anything bootstraps them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("torch", reason="torch is not installed; it is needed only to re-run training")

from src.data.dmr import (  # noqa: E402
    ALL_VIEWS,
    DYNAMIC,
    STATIC,
    Normalisation,
    aggregate_to_patient,
    build_view_table,
    cohort_statistics,
    restrict_to_available,
    select_frames,
    sequence_table,
)
from src.data.dmr_datasets import (  # noqa: E402
    DmrFrameDataset,
    DmrMultiViewDataset,
    DmrSequenceDataset,
)

N_PATIENTS = 12
N_DYNAMIC = 6


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """A miniature DMR-IR: float32 TIFFs in degrees Celsius plus the two manifests."""
    root = tmp_path_factory.mktemp("dmr")
    rng = np.random.default_rng(0)
    image_rows, patient_rows = [], []

    for i in range(N_PATIENTS):
        pid = f"DMR{i:04d}"
        label = int(i % 3 == 0)
        pdir = root / "images" / f"{i:04d}"
        mdir = root / "masks" / f"{i:04d}"
        pdir.mkdir(parents=True)
        mdir.mkdir(parents=True)

        def write(name: str, view: int, protocol: str) -> None:
            # Realistic formats, not just realistic shapes. Frames are float32 degrees
            # Celsius at 640x480; masks are 8-bit PNGs at 224x224, because PNG cannot store
            # mode 'F' at all -- writing the fixture in float32 is what the mirror could
            # never have shipped, and PIL refuses it.
            arr = (30.0 + rng.normal(0, 1.5, (480, 640))).astype(np.float32)
            Image.fromarray(arr, mode="F").save(pdir / f"{name}.tiff")
            mask = np.zeros((224, 224), dtype=np.uint8)
            mask[60:160, 60:160] = 255
            Image.fromarray(mask, mode="L").save(mdir / f"{name}.png")
            image_rows.append({
                "patient_id": pid,
                "image_path": f"images/{i:04d}/{name}.tiff",
                "mask_path": f"masks/{i:04d}/{name}.png",
                "view_code": view, "protocol": protocol,
                "patient_binary_label": label,
                "sha256": f"{i:04d}{name}",
            })

        for view in range(5):
            write(f"{1000 + i}_Static_{view:04d}_000", view, STATIC)
        for seq in range(N_DYNAMIC):
            write(f"{1000 + i}_Dinamic_0000_{seq:03d}", 0, DYNAMIC)

        patient_rows.append({"patient_id": pid, "patient_binary_label": label,
                             "age_at_visit": 40 + i, "visit_date": f"201{3 + i % 4}-0{1 + i % 8}-01"})

    images = pd.DataFrame(image_rows)
    patients = pd.DataFrame(patient_rows)
    images.to_csv(root / "image_manifest.csv", index=False)
    patients.to_csv(root / "patient_manifest.csv", index=False)
    return root, images, patients


def test_select_frames_filters_and_orders(cohort):
    _, images, _ = cohort
    static = select_frames(images, protocol=STATIC)
    assert len(static) == N_PATIENTS * 5
    assert set(static["view_name"]) == set(ALL_VIEWS)

    dyn = select_frames(images, protocol=DYNAMIC, views=("frontal",))
    assert len(dyn) == N_PATIENTS * N_DYNAMIC
    # Frame order drives the temporal model; an unsorted sequence would silently scramble it.
    for _, g in dyn.groupby("patient_id"):
        assert g["frame_index"].tolist() == sorted(g["frame_index"].tolist())


def test_select_frames_rejects_unknown_view(cohort):
    _, images, _ = cohort
    with pytest.raises(ValueError, match="unknown view"):
        select_frames(images, views=("frontal", "lateral_left"))


def test_build_view_table_is_one_row_per_patient(cohort):
    _, images, patients = cohort
    table = build_view_table(images, patients, protocol=STATIC)
    assert len(table) == N_PATIENTS
    assert not table["patient_id"].duplicated().any()
    for view in ALL_VIEWS:
        assert (table[f"{view}_path"].str.len() > 0).all()


def test_build_view_table_require_drops_incomplete_patients(cohort):
    _, images, patients = cohort
    thinned = images[~((images["patient_id"] == "DMR0000") & (images["view_code"] == 2))]
    table = build_view_table(thinned, patients, protocol=STATIC, require=("oblique90_a",))
    assert "DMR0000" not in set(table["patient_id"])
    assert len(table) == N_PATIENTS - 1

    # Without `require` the patient is kept and the missing view is an empty string, which
    # is what makes view-dropout robustness measurable instead of a crash.
    kept = build_view_table(thinned, patients, protocol=STATIC)
    assert kept.loc[kept["patient_id"] == "DMR0000", "oblique90_a_path"].item() == ""


def test_normalisation_modes_differ_and_fixed_needs_statistics():
    arr = np.array([[20.0, 30.0], [40.0, 50.0]], dtype=np.float32)
    per_image = Normalisation("per_image").apply(arr)
    assert per_image.mean() == pytest.approx(0.0, abs=1e-6)

    fixed = Normalisation("fixed", mean=35.0, std=10.0).apply(arr)
    assert fixed.mean() == pytest.approx(0.0, abs=1e-6)
    # A frame offset by 5 degrees keeps that offset under 'fixed' and loses it under
    # 'per_image'. That difference is the whole point of exposing the choice.
    shifted = arr + 5.0
    assert not np.allclose(Normalisation("fixed", 35.0, 10.0).apply(shifted), fixed)
    assert np.allclose(Normalisation("per_image").apply(shifted), per_image, atol=1e-6)

    with pytest.raises(ValueError, match="TRAINING frames only"):
        Normalisation("fixed")
    with pytest.raises(ValueError, match="mode must be one of"):
        Normalisation("minmax")


def test_cohort_statistics_recover_the_temperature_scale(cohort):
    root, images, _ = cohort
    mean, std = cohort_statistics(images["image_path"].head(8), root)
    assert 28.0 < mean < 32.0
    assert 1.0 < std < 2.0


def test_frame_dataset_shapes_and_masking(cohort):
    root, images, patients = cohort
    frames = select_frames(images, protocol=STATIC).head(4)
    ds = DmrFrameDataset(frames, root, image_size=32)
    x, y, pid = ds[0]
    assert x.shape == (3, 32, 32)
    assert y.item() in (0.0, 1.0)
    assert pid == frames["patient_id"].iloc[0]

    # Masks ship at 224x224 against 640x480 frames; applying one must not change the
    # model's input resolution.
    masked = DmrFrameDataset(frames, root, image_size=32, use_masks=True)
    assert masked[0][0].shape == (3, 32, 32)
    assert not np.allclose(masked[0][0].numpy(), x.numpy())


def test_frame_dataset_id_column_selects_the_unit_of_observation(cohort):
    root, images, _ = cohort
    frames = select_frames(images, protocol=STATIC).head(4)
    assert DmrFrameDataset(frames, root, image_size=32, id_column="image_path")[0][2].endswith(".tiff")
    with pytest.raises(ValueError, match="id_column"):
        DmrFrameDataset(frames, root, image_size=32, id_column="nope")


def test_multiview_dataset_stacks_views_and_flags_absences(cohort):
    root, images, patients = cohort
    thinned = images[~((images["patient_id"] == "DMR0000") & (images["view_code"] == 2))]
    table = build_view_table(thinned, patients, protocol=STATIC)
    ds = DmrMultiViewDataset(table, root, image_size=32)
    # (views, present, label, id): both tensors are model inputs and come first, so that
    # src.training.trainer can unpack any dataset as (*inputs, y, id).
    stack, present, y, pid = ds[list(table["patient_id"]).index("DMR0000")]
    assert stack.shape == (len(ALL_VIEWS), 3, 32, 32)
    assert present.tolist() == [True, True, False, True, True]
    assert stack[2].abs().sum().item() == 0.0


def test_sequence_dataset_pads_to_a_fixed_length(cohort):
    root, images, patients = cohort
    frames = sequence_table(images, patients, min_frames=N_DYNAMIC)
    assert frames["patient_id"].nunique() == N_PATIENTS

    ds = DmrSequenceDataset(frames, root, n_frames=N_DYNAMIC + 2, image_size=32)
    seq, y, pid = ds[0]
    assert seq.shape == (N_DYNAMIC + 2, 3, 32, 32)
    # Padding repeats the last real frame rather than inserting zeros.
    assert np.allclose(seq[-1].numpy(), seq[N_DYNAMIC - 1].numpy())

    truncated = DmrSequenceDataset(frames, root, n_frames=3, image_size=32)
    assert truncated[0][0].shape == (3, 3, 32, 32)


def test_sequence_augmentation_is_shared_across_timesteps(cohort):
    """One geometry per sequence: per-frame transforms would destroy the recovery signal."""
    from src.preprocessing.augmentations import build_train_augmentation

    root, images, patients = cohort
    frames = sequence_table(images, patients, min_frames=N_DYNAMIC)
    plain = DmrSequenceDataset(frames, root, n_frames=N_DYNAMIC + 2, image_size=32)
    augmented = DmrSequenceDataset(frames, root, n_frames=N_DYNAMIC + 2, image_size=32,
                                   augment=build_train_augmentation(degrees=25.0))

    seq = augmented[0][0]
    assert seq.shape == plain[0][0].shape
    assert not np.allclose(seq.numpy(), plain[0][0].numpy()), "augmentation had no effect"
    # The padded tail repeats one frame, so after a shared transform those frames stay
    # identical; under per-frame augmentation they would have diverged.
    assert np.allclose(seq[-1].numpy(), seq[N_DYNAMIC - 1].numpy())


def test_sequence_table_drops_short_sequences(cohort):
    _, images, patients = cohort
    assert sequence_table(images, patients, min_frames=N_DYNAMIC + 1).empty


def test_aggregate_to_patient_collapses_frames(cohort):
    preds = pd.DataFrame({
        "patient_id": ["DMR0001"] * 3 + ["DMR0002"] * 2,
        "repeat": 1, "fold": 1, "model": "arm",
        "y_true": [1, 1, 1, 0, 0],
        # DMR0002's frames straddle the threshold: 0.7 alone predicts positive, their mean
        # 0.4 does not. Values avoid a mean of exactly 0.5, which would test the tie-break
        # rule instead of the aggregation.
        "y_prob": [0.2, 0.4, 0.6, 0.1, 0.7],
        "threshold": 0.5,
    })
    out = aggregate_to_patient(preds)
    assert len(out) == 2
    assert out.loc[out["patient_id"] == "DMR0001", "y_prob"].item() == pytest.approx(0.4)
    assert out.loc[out["patient_id"] == "DMR0001", "n_frames"].item() == 3
    # y_pred must be recomputed from the aggregated probability, not inherited per frame.
    assert out.loc[out["patient_id"] == "DMR0002", "y_prob"].item() == pytest.approx(0.4)
    assert out.loc[out["patient_id"] == "DMR0002", "y_pred"].item() == 0

    assert aggregate_to_patient(preds, how="max").loc[
        lambda d: d["patient_id"] == "DMR0002", "y_prob"].item() == pytest.approx(0.7)
    with pytest.raises(ValueError, match="mean"):
        aggregate_to_patient(preds, how="median")


def test_aggregate_by_fold_false_pools_a_patient_across_folds():
    """An image-level split scatters one patient's frames over every fold of a repeat.

    Keying the per-patient view on the fold then emits several rows for the same patient,
    which summarize_predictions rejects because its bootstrap assumes one observation each.
    This is the crash that ended a completed 25-fold run of scripts/03_leakage_experiment.py
    after the GPU work was done.
    """
    from src.evaluation.summary import summarize_predictions

    preds = pd.DataFrame({
        "patient_id": ["A"] * 3 + ["B"] * 3,
        "image_id": [f"i{i}" for i in range(6)],
        "repeat": 1, "fold": [1, 2, 3, 1, 2, 3], "model": "image_level",
        "y_true": [1, 1, 1, 0, 0, 0], "y_prob": [0.2, 0.6, 0.8, 0.1, 0.3, 0.5],
        "threshold": 0.5,
    })
    keyed_on_fold = aggregate_to_patient(preds, by_fold=True)
    assert len(keyed_on_fold) == 6
    with pytest.raises(AssertionError, match="more than one out-of-fold prediction"):
        summarize_predictions(keyed_on_fold, n_bootstrap=20)

    pooled = aggregate_to_patient(preds, by_fold=False)
    assert len(pooled) == 2
    assert pooled.loc[pooled["patient_id"] == "A", "y_prob"].item() == pytest.approx(0.2 / 3 + 0.6 / 3 + 0.8 / 3)
    summarize_predictions(pooled, n_bootstrap=20)   # must not raise

    # Under a patient-level split each patient sits in one fold per repeat, so keeping the
    # fold changes nothing and preserves the column the leakage test needs.
    one_fold = preds[preds["fold"] == 1]
    assert len(aggregate_to_patient(one_fold, by_fold=True)) == 2
    assert "fold" in aggregate_to_patient(one_fold, by_fold=True).columns


def test_restrict_to_available_drops_patients_without_frames(cohort):
    """A patient with no frame of the selected view must leave the cohort before splitting."""
    root, images, patients = cohort
    frontal = select_frames(images, protocol=STATIC, views=("frontal",))
    thinned = frontal[frontal["patient_id"] != "DMR0000"]

    kept, dropped = restrict_to_available(patients, thinned)
    assert dropped == ["DMR0000"]
    assert "DMR0000" not in set(kept["patient_id"])
    assert len(kept) == len(patients) - 1

    unchanged, none_dropped = restrict_to_available(patients, frontal)
    assert none_dropped == []
    assert len(unchanged) == len(patients)
