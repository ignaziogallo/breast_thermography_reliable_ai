"""Torch datasets for the second cohort: one view per patient, or one oblique view per breast.

In the single-view dataset one patient contributes exactly one image, so the dataset index
*is* the patient index and no aggregation step exists in which views of the same patient
could be mixed across folds.

Normalisation deserves comment. The exports are per-image auto-ranged greyscale renderings
(351/357 saturate at grey 255), so a fixed photometric scale does not exist and ImageNet
channel statistics are not meaningful here. Every image is therefore standardised
individually to zero mean and unit variance, which is what the data permits; the
pretrained stem adapts to that input distribution during full fine-tuning. See
data/README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.manifest import resolve_image_path
from src.utils.io import read_thermal_image


def load_standardised(path: Path, size: int) -> torch.Tensor:
    """Load one thermogram as a 1xHxW tensor, standardised within the image."""
    arr = read_thermal_image(path).astype(np.float32)
    std = float(arr.std())
    arr = (arr - float(arr.mean())) / (std if std > 0 else 1.0)
    t = torch.from_numpy(arr).unsqueeze(0)
    if t.shape[-2:] != (size, size):
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return t


class BreastViewDataset(Dataset):
    """One oblique thermogram per breast, labelled at breast level.

    Both breasts of a patient must be assigned to the same fold; that is enforced by the
    caller, which selects rows by `patient_id` from the frozen patient splits.
    """

    def __init__(self, table: pd.DataFrame, data_root: Path, image_size: int = 224,
                 augment=None):
        dup = table.loc[table["sample_id"].duplicated(), "sample_id"].unique()
        if len(dup):
            raise ValueError(f"Duplicate breast samples: {sorted(dup)[:5]}")
        self.sample_ids = table["sample_id"].tolist()
        self.patient_ids = table["patient_id"].tolist()
        self.paths = [resolve_image_path(data_root, p) for p in table["image_path"]]
        self.labels = table["breast_label"].astype(int).tolist()
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, i: int):
        x = load_standardised(self.paths[i], self.image_size)
        if self.augment is not None:
            x = self.augment(x)
        x = x.expand(3, -1, -1).contiguous()
        return x, torch.tensor(float(self.labels[i])), self.sample_ids[i]


class SingleViewDataset(Dataset):
    """Anterior-view dataset over a set of patients.

    `table` must hold one row per patient with `patient_id`, `<view>_path` and
    `patient_binary_label`. Augmentation is passed in and must be supplied only for the
    training partition.
    """

    def __init__(self, table: pd.DataFrame, data_root: Path, view: str = "anterior",
                 image_size: int = 224, augment=None):
        dup = table.loc[table["patient_id"].duplicated(), "patient_id"].unique()
        if len(dup):
            raise ValueError("SingleViewDataset expects one row per patient; duplicated: "
                             f"{sorted(dup)[:5]}")

        column = f"{view}_path"
        if column not in table.columns:
            raise ValueError(
                f"SingleViewDataset: no column {column!r} in the manifest. Available views: "
                f"{sorted(c[:-5] for c in table.columns if c.endswith('_path'))}")
        # A patient lacking this view carries an empty string in the manifest, which
        # pd.read_csv reads back as NaN -- and NaN reaches Path() as a float, failing deep
        # inside pathlib with no mention of the patient or the view. Refuse here instead.
        missing = table.loc[table[column].isna() | table[column].astype(str).str.len().eq(0),
                            "patient_id"].tolist()
        if missing:
            raise ValueError(
                f"SingleViewDataset: {len(missing)} patient(s) have no {view!r} view, e.g. "
                f"{sorted(missing)[:5]}. Build a manifest restricted to patients holding it:\n"
                f"    python scripts/22_build_dmr_view_table.py --views {view} --require all \\\n"
                f"        --output results/dmr_view_table_{view}.csv")

        self.patient_ids = table["patient_id"].tolist()
        self.paths = [resolve_image_path(data_root, p) for p in table[column]]
        self.labels = table["patient_binary_label"].astype(int).tolist()
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, i: int):
        x = load_standardised(self.paths[i], self.image_size)
        if self.augment is not None:
            x = self.augment(x)
        # Replicate to three channels for the pretrained stem; the image is single-band.
        x = x.expand(3, -1, -1).contiguous()
        return x, torch.tensor(float(self.labels[i])), self.patient_ids[i]
