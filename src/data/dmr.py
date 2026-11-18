"""DMR-IR cohort adapters: view tables, frame selection, and the normalisation decision.

`results/dmr_image_manifest.csv` is *long* -- one row per frame, 6051 rows for 236 patients
-- while `src.data.dataset.SingleViewDataset` and `scripts/04_train_single_view.py` expect a
*wide* table with one row per patient and a `<view>_path` column. `build_view_table` bridges
the two, so the DMR cohort flows through the existing patient-level machinery unchanged.

Two decisions are made here rather than left implicit at the call site.

**View naming.** DMR-IR encodes the view as 0-4. The 45/90 grouping is measured from the
protocol's frame counts, but the left/right assignment follows Figure 7 of Silva et al.
(2014) and is *inferred, not verified* (see `data/README.md`). Naming a column
`lateral45_left_path` would launder that inference into an assertion every downstream reader
would trust. The columns are therefore named `oblique45_a` / `oblique45_b`, neutral labels
that carry no anatomical claim; rename `VIEW_NAMES` in one place once the mirror-correlation
check has actually been run.

**Normalisation.** Unlike this study's own cohort, DMR-IR TIFFs are raw temperature
matrices, so a grey level is physically meaningful and comparable across files. Per-image
standardisation is therefore a *choice* that discards absolute temperature, not a
requirement imposed by auto-ranged renders. Both modes are offered:

  ``per_image``  z-score each frame independently. Discards absolute temperature. This is
                 what the DMR-IR literature does, and what makes results comparable to it.
  ``fixed``      subtract a single cohort mean and standard deviation, preserving absolute
                 temperature differences between patients.

`fixed` statistics must be estimated on training frames only; `cohort_statistics` exists to
make that the easy path, and passing test frames to it is the mistake it is designed to
prevent. A model trained under `fixed` on a cohort where acquisition epoch predicts the
label can key on a room-temperature drift, so the mode belongs in every reported config.

Nothing here imports torch: the manifests and the tables built from them are plain pandas,
so cohort construction and every analysis of stored predictions run on a machine with no
deep-learning stack. The datasets that feed a model live in `src.data.dmr_datasets`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.manifest import resolve_image_path
from src.utils.io import read_thermal_image

# view_code -> column stem. 'a'/'b' rather than right/left: the side is inferred (see above).
VIEW_NAMES = {0: "frontal", 1: "oblique45_a", 2: "oblique90_a", 3: "oblique45_b", 4: "oblique90_b"}
ALL_VIEWS = tuple(VIEW_NAMES[c] for c in sorted(VIEW_NAMES))
STATIC, DYNAMIC = "Static", "Dinamic"          # 'Dinamic' is the mirror's spelling
NORMALISATIONS = ("per_image", "fixed")

# Acquisition-era contrasts: arm -> (diagnosis held constant, early years, late years).
# `None` keeps both diagnoses. The within-benign contrast is the clean one: both eras are
# well populated and the disease is held constant. The within-malignant contrast is not
# attributable, because no benign patient was acquired after 2015, so a protocol change and
# a change of case mix cannot be separated there. Every script that asks whether a score or
# a representation tracks the calendar reads its eras from here.
ERA_CONTRASTS = {
    "within_benign": (0, (2012,), (2013,)),
    "within_malignant": (1, (2013, 2014, 2015), (2018, 2019, 2020)),
    "all": (None, (2012, 2013), (2018, 2019, 2020)),
}


@dataclass
class Normalisation:
    """How a frame's temperatures are mapped to model input."""

    mode: str = "per_image"
    mean: float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in NORMALISATIONS:
            raise ValueError(f"mode must be one of {NORMALISATIONS}, got {self.mode!r}")
        if self.mode == "fixed" and (self.mean is None or self.std is None):
            raise ValueError(
                "mode='fixed' needs mean and std. Estimate them on TRAINING frames only, "
                "with cohort_statistics(); statistics taken over the whole cohort leak the "
                "test partition's temperature distribution into training."
            )

    def apply(self, arr: np.ndarray) -> np.ndarray:
        if self.mode == "per_image":
            std = float(arr.std())
            return (arr - float(arr.mean())) / (std if std > 0 else 1.0)
        return (arr - float(self.mean)) / (float(self.std) or 1.0)

    def describe(self) -> dict:
        return {"normalisation": self.mode, "mean": self.mean, "std": self.std}


def cohort_statistics(paths, data_root: Path, max_frames: int = 400,
                      seed: int = 42) -> tuple[float, float]:
    """Mean and standard deviation of temperature over a sample of frames.

    Pass *training* frames only. Sampling keeps this to a few seconds on a cohort of
    thousands of frames; the estimate feeds a single global affine transform, so the
    sampling error is negligible next to the between-patient spread it normalises.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("cohort_statistics received no frames")
    rng = np.random.default_rng(seed)
    if len(paths) > max_frames:
        paths = [paths[i] for i in rng.choice(len(paths), max_frames, replace=False)]
    total, total_sq, n = 0.0, 0.0, 0
    for rel in paths:
        arr = read_thermal_image(resolve_image_path(data_root, rel)).astype(np.float64)
        total += arr.sum()
        total_sq += (arr ** 2).sum()
        n += arr.size
    mean = total / n
    var = max(total_sq / n - mean ** 2, 0.0)
    return float(mean), float(np.sqrt(var) or 1.0)


def _frame_index(images: pd.DataFrame) -> pd.Series:
    """Sequence number parsed from `<record>_<protocol>_<view>_<seq>.tiff`."""
    stem = images["image_path"].str.split("/").str[-1].str.rsplit(".", n=1).str[0]
    return pd.to_numeric(stem.str.split("_").str[-1], errors="coerce")


def select_frames(images: pd.DataFrame, protocol: str | None = None,
                  views: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Filter the long image manifest by protocol and canonical view name."""
    out = images.copy()
    out["view_code"] = pd.to_numeric(out["view_code"], errors="coerce").astype("Int64")
    out["view_name"] = out["view_code"].map(VIEW_NAMES)
    out["frame_index"] = _frame_index(out)
    if protocol is not None:
        out = out[out["protocol"] == protocol]
    if views is not None:
        unknown = set(views) - set(ALL_VIEWS)
        if unknown:
            raise ValueError(f"unknown view names {sorted(unknown)}; expected {ALL_VIEWS}")
        out = out[out["view_name"].isin(views)]
    order = ["patient_id", "view_name", "frame_index", "image_path"]
    return out.sort_values(order).reset_index(drop=True)


def build_view_table(images: pd.DataFrame, patients: pd.DataFrame, protocol: str = STATIC,
                     views: tuple[str, ...] = ALL_VIEWS,
                     require: tuple[str, ...] = ()) -> pd.DataFrame:
    """One row per patient with a `<view>_path` column each: the wide form the single-view
    trainer (`scripts/04_train_single_view.py`) expects.

    `require` names the views a patient must have to be kept; patients missing any of them
    are dropped, and the caller is expected to record how many. A missing view yields an
    empty string, matching `src.data.manifest.discover_patients`.

    A handful of patients hold two static frames for one view. The first by sorted path is
    taken, deterministically, so that a rerun selects the same frame.
    """
    frames = select_frames(images, protocol=protocol, views=views)
    # drop_duplicates, not groupby().first(): the latter takes the first *non-null* value
    # per column independently, so a row whose mask_path is empty would be paired with a
    # different frame's mask. Keeping whole rows keeps image and mask consistent.
    first = frames.drop_duplicates(subset=["patient_id", "view_name"], keep="first")
    wide = first.pivot(index="patient_id", columns="view_name", values="image_path")
    wide = wide.reindex(columns=list(views))
    wide.columns = [f"{c}_path" for c in wide.columns]
    wide = wide.fillna("").reset_index()

    table = patients.merge(wide, on="patient_id", how="inner")
    for view in require:
        col = f"{view}_path"
        if col not in table.columns:
            raise ValueError(f"required view {view!r} was not selected; views={views}")
        table = table[table[col].astype(str).str.len() > 0]
    return table.reset_index(drop=True)


def sequence_table(images: pd.DataFrame, patients: pd.DataFrame, view: str = "frontal",
                   min_frames: int = 20, max_frames: int | None = None) -> pd.DataFrame:
    """Long table of dynamic frames, ordered, for patients with a complete sequence.

    The dynamic protocol cools the breast and samples the thermal recovery, so the frames
    are a physiological time series, not 20 independent examples. Ordering is by the `seq`
    field of the filename, which the design audit confirms is contiguous 0..n-1.
    """
    frames = select_frames(images, protocol=DYNAMIC, views=(view,))
    counts = frames.groupby("patient_id").size()
    keep = set(counts[counts >= min_frames].index) & set(patients["patient_id"])
    frames = frames[frames["patient_id"].isin(keep)]
    if max_frames is not None:
        frames = frames.groupby("patient_id", group_keys=False).head(max_frames)
    return frames.reset_index(drop=True)


def restrict_to_available(cohort: pd.DataFrame,
                          frames: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop patients holding no frame in `frames`, and say which.

    A cohort is defined from the patient manifest but the frames are selected by protocol
    and view, and the two disagree: 5 of the 236 DMR-IR patients have no frontal view at
    all. Left in, such a patient is assigned to a fold, contributes nothing to any loader
    and is silently never scored -- which surfaces much later as a leakage test reporting a
    held-out patient that went missing. Restricting before the splits are built is what
    keeps the fold definition and the data in agreement.
    """
    available = set(frames["patient_id"])
    dropped = sorted(set(cohort["patient_id"]) - available)
    return cohort[cohort["patient_id"].isin(available)].reset_index(drop=True), dropped


def aggregate_to_patient(preds: pd.DataFrame, how: str = "mean",
                         by_fold: bool = True) -> pd.DataFrame:
    """Collapse frame-level predictions to one row per patient.

    The patient is the experimental unit, so a patient contributing 27 frames must not
    contribute 27 rows to a bootstrap that assumes independent observations.

    `by_fold` controls whether the fold is part of the key. Under a patient-level split a
    patient belongs to exactly one test fold per repeat, so keeping the fold is free and
    preserves the column the leakage test needs. Under an *image-level* split the patient's
    frames are scattered over every fold, so keying on the fold would emit several rows per
    patient per repeat -- which `summarize_predictions` rejects, correctly, since its
    bootstrap assumes one observation per patient. Pass `by_fold=False` there.
    """
    if how not in ("mean", "max"):
        raise ValueError(f"how must be 'mean' or 'max', got {how!r}")
    wanted = ("model", "repeat", "fold") if by_fold else ("model", "repeat")
    keys = [c for c in wanted if c in preds.columns]
    agg = {"y_true": ("y_true", "first"), "y_prob": ("y_prob", how),
           "n_frames": ("y_prob", "size")}
    for col in ("threshold", "config"):
        if col in preds.columns:
            agg[col] = (col, "first")
    out = preds.groupby(keys + ["patient_id"], as_index=False).agg(**agg)
    if "threshold" in out.columns:
        out["y_pred"] = (out["y_prob"] >= out["threshold"]).astype(int)
    return out
