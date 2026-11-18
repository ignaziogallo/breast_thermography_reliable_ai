"""Torch datasets over the DMR-IR manifests: one frame, a stack of views, a dynamic sequence.

The tables these datasets read, and the normalisation they apply, are defined in
`src.data.dmr`, which does not need torch. Everything that touches a tensor is here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dmr import ALL_VIEWS, Normalisation
from src.data.manifest import resolve_image_path
from src.utils.io import read_thermal_image


def load_frame(path: Path, size: int, norm: Normalisation,
               mask_path: Path | None = None) -> torch.Tensor:
    """Load one thermogram as a 1xHxW tensor under the given normalisation.

    A mask, when supplied, is upsampled to the frame's resolution before being applied: the
    mirror stores masks at 224x224 while frames are 640x480, and resizing the *frame* down
    to the mask instead would silently change the input resolution of the experiment.
    Masking happens before normalisation under `per_image`, so the statistics describe the
    breast region rather than the background.
    """
    arr = read_thermal_image(path).astype(np.float32)
    if mask_path is not None:
        m = read_thermal_image(mask_path).astype(np.float32)
        if m.shape != arr.shape:
            m = torch.nn.functional.interpolate(
                torch.from_numpy(m)[None, None], size=arr.shape, mode="nearest"
            )[0, 0].numpy()
        arr = np.where(m > 0, arr, float(arr[m > 0].mean()) if (m > 0).any() else float(arr.mean()))
    t = torch.from_numpy(norm.apply(arr).astype(np.float32)).unsqueeze(0)
    if t.shape[-2:] != (size, size):
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return t


class DmrFrameDataset(Dataset):
    """One row per frame, from the long manifest.

    The returned identifier is configurable because the two protocols this repository
    compares differ precisely in what an observation *is*: `patient_id` for the
    patient-level protocol, `image_path` for the image-level protocol that treats correlated
    frames of one patient as independent examples.
    """

    def __init__(self, table: pd.DataFrame, data_root: Path, image_size: int = 224,
                 norm: Normalisation | None = None, augment=None,
                 id_column: str = "patient_id", use_masks: bool = False):
        if id_column not in table.columns:
            raise ValueError(f"id_column {id_column!r} absent from the frame table")
        self.ids = table[id_column].astype(str).tolist()
        self.patient_ids = table["patient_id"].astype(str).tolist()
        self.paths = [resolve_image_path(data_root, p) for p in table["image_path"]]
        self.masks = None
        if use_masks:
            if "mask_path" not in table.columns:
                raise ValueError("use_masks=True but the frame table has no mask_path column")
            self.masks = [resolve_image_path(data_root, p) if isinstance(p, str) and p else None
                          for p in table["mask_path"]]
        self.labels = table["patient_binary_label"].astype(int).tolist()
        self.image_size = image_size
        self.norm = norm or Normalisation()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        x = load_frame(self.paths[i], self.image_size, self.norm,
                       self.masks[i] if self.masks else None)
        if self.augment is not None:
            x = self.augment(x)
        # Replicate to three channels for the pretrained stem; the frame is single-band.
        return x.expand(3, -1, -1).contiguous(), torch.tensor(float(self.labels[i])), self.ids[i]


class DmrMultiViewDataset(Dataset):
    """One row per patient, returning a stack of V views as a VxCxHxW tensor.

    Views are emitted in the fixed order given by `views`, so a fusion module can rely on
    channel position. A patient missing a view yields a zero frame and a False in the mask,
    which is what makes view-dropout robustness measurable rather than a crash.

    The item is `(views, present, label, patient_id)`. The mask precedes the label because
    `src.training.trainer` unpacks a batch as `(*inputs, y, id)`: both tensors are model
    inputs, and putting them first lets the multi-view and single-view experiments share
    one fold loop.
    """

    def __init__(self, table: pd.DataFrame, data_root: Path, views: tuple[str, ...] = ALL_VIEWS,
                 image_size: int = 224, norm: Normalisation | None = None, augment=None):
        dup = table.loc[table["patient_id"].duplicated(), "patient_id"].unique()
        if len(dup):
            raise ValueError(f"expects one row per patient; duplicated: {sorted(dup)[:5]}")
        self.views = tuple(views)
        self.patient_ids = table["patient_id"].astype(str).tolist()
        self.paths = [[(resolve_image_path(data_root, r[f"{v}_path"])
                        if isinstance(r[f"{v}_path"], str) and r[f"{v}_path"] else None)
                       for v in self.views] for _, r in table.iterrows()]
        self.labels = table["patient_binary_label"].astype(int).tolist()
        self.image_size = image_size
        self.norm = norm or Normalisation()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, i: int):
        frames, present = [], []
        for path in self.paths[i]:
            if path is None:
                frames.append(torch.zeros(3, self.image_size, self.image_size))
                present.append(False)
                continue
            x = load_frame(path, self.image_size, self.norm)
            if self.augment is not None:
                x = self.augment(x)
            frames.append(x.expand(3, -1, -1).contiguous())
            present.append(True)
        return (torch.stack(frames), torch.tensor(present),
                torch.tensor(float(self.labels[i])), self.patient_ids[i])


class DmrSequenceDataset(Dataset):
    """One row per patient, returning the dynamic sequence as a TxCxHxW tensor.

    Frames are ordered by acquisition index. `n_frames` truncates or pads the sequence to a
    fixed length so that a batch collates; padding repeats the last frame, which is the
    neutral choice for a recovery curve that has flattened.

    Augmentation is applied to the *stacked* sequence, so one sampled rotation and shift
    covers every frame. Augmenting frame by frame would give each timestep its own geometry
    and destroy the pixel correspondence the model needs to read a temperature trajectory --
    turning the recovery signal into registration noise.
    """

    def __init__(self, frames: pd.DataFrame, data_root: Path, n_frames: int = 20,
                 image_size: int = 224, norm: Normalisation | None = None, augment=None):
        ordered = frames.sort_values(["patient_id", "frame_index", "image_path"])
        self.patient_ids, self.paths, self.labels = [], [], []
        for pid, g in ordered.groupby("patient_id", sort=True):
            self.patient_ids.append(str(pid))
            self.paths.append([resolve_image_path(data_root, p) for p in g["image_path"]])
            self.labels.append(int(g["patient_binary_label"].iloc[0]))
        self.n_frames = n_frames
        self.image_size = image_size
        self.norm = norm or Normalisation()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, i: int):
        paths = self.paths[i][: self.n_frames]
        frames = [load_frame(path, self.image_size, self.norm).expand(3, -1, -1).contiguous()
                  for path in paths]
        while len(frames) < self.n_frames:
            frames.append(frames[-1].clone())
        stack = torch.stack(frames)
        if self.augment is not None:
            # One call over the TxCxHxW stack: torchvision v2 samples its parameters once
            # per call, so every timestep receives the identical transform.
            stack = self.augment(stack)
        return stack, torch.tensor(float(self.labels[i])), self.patient_ids[i]
