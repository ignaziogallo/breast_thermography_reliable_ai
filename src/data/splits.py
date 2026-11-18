"""Cross-validation folds: the patient-level protocol, and the image-level leakage control.

A splits table has one row per (repeat, fold, split, patient) with the columns `repeat`,
`fold`, `seed`, `split`, `patient_id` and `label`. Folds are built once, stored, and read by
every experiment; no experiment re-derives them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

SPLIT_NAMES = ("train", "val", "test")


def make_repeated_patient_splits(manifest: pd.DataFrame, n_splits: int = 5, n_repeats: int = 5,
                                 seeds: list[int] | None = None,
                                 val_fraction: float = 0.15) -> pd.DataFrame:
    """Repeated stratified group k-fold over patients, with a validation subset per fold.

    The patient is the grouping variable, so every image of a patient stays in one
    partition. Repeat `r` uses seed `seeds[r]` (42, 43, ... by default); the validation
    patients are a stratified draw from that fold's training portion.
    """
    df = manifest.dropna(subset=["patient_binary_label"]).copy()
    df["patient_binary_label"] = df["patient_binary_label"].astype(int)
    seeds = seeds or list(range(42, 42 + n_repeats))
    if len(seeds) < n_repeats:
        raise ValueError("Need at least n_repeats seeds")
    rows = []
    X = np.zeros((len(df), 1))
    y = df["patient_binary_label"].to_numpy()
    groups = df["patient_id"].to_numpy()
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seeds[rep])
        for fold, (trainval_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            trainval = df.iloc[trainval_idx].reset_index(drop=True)
            test = df.iloc[test_idx]
            # Stratified patient-level validation split within the training portion.
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                         random_state=seeds[rep] + fold * 1000)
            tv_y = trainval["patient_binary_label"].to_numpy()
            try:
                tr_rel, va_rel = next(sss.split(np.zeros((len(trainval), 1)), tv_y))
            except ValueError:
                # Small-class fallback: deterministic class-aware allocation.
                rng = np.random.default_rng(seeds[rep] + fold * 1000)
                va_rel = []
                tr_rel = []
                for cls in sorted(trainval["patient_binary_label"].unique()):
                    inds = np.flatnonzero(tv_y == cls)
                    rng.shuffle(inds)
                    n_val = max(1, int(round(len(inds) * val_fraction)))
                    va_rel.extend(inds[:n_val])
                    tr_rel.extend(inds[n_val:])
                tr_rel, va_rel = np.array(tr_rel), np.array(va_rel)
            parts = {
                "train": trainval.iloc[tr_rel],
                "val": trainval.iloc[va_rel],
                "test": test,
            }
            for split, part in parts.items():
                for _, r in part.iterrows():
                    rows.append({"repeat": rep + 1, "fold": fold, "seed": seeds[rep],
                                 "split": split, "patient_id": r["patient_id"],
                                 "label": int(r["patient_binary_label"])})
    return pd.DataFrame(rows)


def make_image_level_splits(images: pd.DataFrame, n_splits: int = 5, n_repeats: int = 5,
                            seeds: list[int] | None = None,
                            val_fraction: float = 0.15) -> pd.DataFrame:
    """Randomly partition *frames*, ignoring which patient they came from.

    This produces leakage on purpose. DMR-IR holds up to 27 correlated frames per patient --
    five static views plus a 20-frame dynamic recovery sequence, some of them byte-identical
    to each other -- so a random frame split places near-copies of the same breast in train
    and test simultaneously. A model can then score a test frame by recognising the patient
    rather than the disease.

    It exists so the size of that effect can be *measured* against
    `make_repeated_patient_splits` under an otherwise identical protocol, which is the
    mechanism behind the near-perfect accuracies this literature reports. Never use it as
    the evaluation protocol for a reported result: predictions produced from these folds
    must be written under a filename that `tests/test_no_prediction_leakage.py` does not
    collect, because by construction they would fail it.

    The emitted schema matches `make_repeated_patient_splits` with an extra `image_id`
    column, so the same fold loop consumes either.
    """
    df = images.dropna(subset=["patient_binary_label"]).copy().reset_index(drop=True)
    df["patient_binary_label"] = df["patient_binary_label"].astype(int)
    if "image_id" not in df.columns:
        df["image_id"] = df["image_path"]
    seeds = seeds or list(range(42, 42 + n_repeats))
    if len(seeds) < n_repeats:
        raise ValueError("Need at least n_repeats seeds")

    rows = []
    y = df["patient_binary_label"].to_numpy()
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seeds[rep])
        folds = cv.split(np.zeros((len(df), 1)), y)
        for fold, (trainval_idx, test_idx) in enumerate(folds, start=1):
            trainval = df.iloc[trainval_idx]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                         random_state=seeds[rep] + fold * 1000)
            tr_rel, va_rel = next(sss.split(np.zeros((len(trainval), 1)),
                                            trainval["patient_binary_label"].to_numpy()))
            parts = {"train": trainval.iloc[tr_rel], "val": trainval.iloc[va_rel],
                     "test": df.iloc[test_idx]}
            for split, part in parts.items():
                for r in part.itertuples(index=False):
                    rows.append({"repeat": rep + 1, "fold": fold, "seed": seeds[rep],
                                 "split": split, "image_id": r.image_id,
                                 "patient_id": r.patient_id,
                                 "label": int(r.patient_binary_label)})
    return pd.DataFrame(rows)


def fold_patient_ids(fold_rows: pd.DataFrame) -> dict[str, list]:
    """The patients of one (repeat, fold), keyed by split, in the order the file lists them.

    Disjointness is the premise of the whole study, so it is asserted wherever a fold is
    consumed and not only where the folds are built: a split file can be regenerated,
    merged or edited between the two.
    """
    ids = {s: fold_rows.loc[fold_rows["split"] == s, "patient_id"].tolist() for s in SPLIT_NAMES}
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        overlap = set(ids[a]) & set(ids[b])
        assert not overlap, (
            f"{a}/{b} overlap at repeat={fold_rows['repeat'].iloc[0]} "
            f"fold={fold_rows['fold'].iloc[0]}: {sorted(overlap)[:5]}")
    return ids


def measure_split_contamination(splits: pd.DataFrame) -> pd.DataFrame:
    """Per fold, how much of the test partition shares a patient with training.

    Quantifies what an image-level split actually does, rather than asserting it is bad:
    the fraction of test frames whose patient also appears in train or val is the exposure
    the reported metric is contaminated by.
    """
    rows = []
    for (rep, fold), g in splits.groupby(["repeat", "fold"]):
        seen = set(g.loc[g["split"].isin(("train", "val")), "patient_id"])
        test = g[g["split"] == "test"]
        contaminated = test["patient_id"].isin(seen)
        rows.append({
            "repeat": int(rep), "fold": int(fold),
            "n_test": len(test), "n_test_patients": test["patient_id"].nunique(),
            "n_contaminated": int(contaminated.sum()),
            "fraction_contaminated": float(contaminated.mean()) if len(test) else 0.0,
        })
    return pd.DataFrame(rows)


def validate_split_integrity(splits: pd.DataFrame) -> list[str]:
    """Every patient overlap between two splits of one fold, as readable messages."""
    errors = []
    for (rep, fold), g in splits.groupby(["repeat", "fold"]):
        sets = {s: set(g.loc[g["split"] == s, "patient_id"]) for s in SPLIT_NAMES}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                errors.append(f"repeat={rep} fold={fold}: {a}/{b} overlap {sorted(overlap)}")
    return errors
