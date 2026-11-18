"""The second cohort's `Diagnostics.xlsx`: per-breast labels and patient covariates."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LABELS = {"N", "PB", "PM"}

# Canonical field -> the spellings of its column header seen in the distributed spreadsheet.
COLUMN_ALIASES = {
    "image": ["image", "patient", "id"],
    "left": ["left"],
    "right": ["right"],
    "weight": ["weight (kg)", "weight(kg)", "weight"],
    "height": ["height(cm)", "height (cm)", "height"],
    "temperature": ["temp(°c)", "temp (°c)", "temperature", "temp"],
    "age": ["age(years)", "age (years)", "age"],
}


def _norm_col(c: object) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def load_diagnostics(path: Path) -> pd.DataFrame:
    """Parse Diagnostics.xlsx robustly and return one row per patient."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The second cohort is not redistributed with this code: "
            "obtain it from its publication and place it so that "
            "<data-root>/{Benign,Malignant}/IIR####/IIR####_{anterior,oblleft,oblright}.jpg "
            "and <data-root>/Diagnostics.xlsx resolve, or point --data-root at it. See "
            "'Dataset preparation' in the README.")
    raw = pd.read_excel(path, engine="openpyxl")
    raw.columns = [_norm_col(c) for c in raw.columns]

    cols = {field: next((name for name in names if name in raw.columns), None)
            for field, names in COLUMN_ALIASES.items()}
    missing_required = [k for k in ("image", "left", "right") if cols[k] is None]
    if missing_required:
        raise ValueError(f"Diagnostics.xlsx missing required columns: {missing_required}; "
                         f"found {list(raw.columns)}")

    out = pd.DataFrame()
    out["patient_id"] = raw[cols["image"]].astype(str).str.strip()
    out = out[out["patient_id"].str.match(r"^IIR\d+$", na=False)].copy()
    for side in ("left", "right"):
        out[f"{side}_label"] = raw.loc[out.index, cols[side]].astype(str).str.strip().str.upper()
        out.loc[~out[f"{side}_label"].isin(LABELS), f"{side}_label"] = pd.NA

    for name in ("weight", "height", "temperature", "age"):
        col = cols[name]
        out[name] = pd.to_numeric(raw.loc[out.index, col], errors="coerce") if col else pd.NA

    h_m = out["height"] / 100.0
    out["bmi"] = out["weight"] / (h_m ** 2)
    out.loc[(h_m <= 0) | h_m.isna(), "bmi"] = pd.NA
    out = out.reset_index(drop=True)

    # A repeated patient row multiplies through the manifest merge and silently misaligns
    # every downstream label. Refuse to return an ambiguous diagnosis table.
    dup = out.loc[out["patient_id"].duplicated(keep=False)].sort_values("patient_id")
    if not dup.empty:
        ids = dup["patient_id"].unique()
        raise ValueError(
            f"{path} contains {len(ids)} patient identifiers on more than one row "
            f"(e.g. {list(ids[:5])}; {len(dup)} rows in total). Each patient must appear "
            "exactly once. Remove the repeated rows from the spreadsheet and re-run the audit."
        )
    return out


def derive_patient_label(left: object, right: object,
                         rule: str = "any_pm_malignant") -> int | None:
    """The patient-level binary label from the two breast-level labels.

    Under `any_pm_malignant` a patient is malignant if at least one breast is `PM` and
    non-malignant if a breast is `N` or `PB`; a patient with no recognised label gets None
    and is excluded from the splits.
    """
    labels = {str(left).upper(), str(right).upper()}
    if rule == "any_pm_malignant":
        if "PM" in labels:
            return 1
        if labels.intersection({"N", "PB"}):
            return 0
        return None
    raise ValueError(f"Unsupported patient label rule: {rule}")
