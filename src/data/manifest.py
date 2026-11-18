"""Patient and image manifests for the second cohort (one folder per patient, three views).

A manifest is the definition of a cohort: one row per patient with its label, covariates
and one `<view>_path` column per view. Paths are stored relative to the dataset root, so a
manifest built on one machine stays valid on another.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image

from src.data.metadata import derive_patient_label, load_diagnostics
from src.utils.io import file_sha256

VIEWS = ("anterior", "oblleft", "oblright")


def resolve_image_path(data_root: Path, relative_path: str) -> Path:
    """Rebuild an absolute path from a manifest entry.

    Manifests store paths relative to the dataset root so that a manifest produced on
    one machine stays valid on another. Consumers must therefore always pair a manifest
    with the --data-root it was built from.

    An absolute entry is rejected rather than joined: ``Path(root) / "/abs/path"`` silently
    discards ``root`` and yields ``/abs/path``, which then fails much later as a confusing
    FileNotFoundError pointing at another machine's layout. The entry is tested under both
    path flavours, because a manifest written on a POSIX host has to be refused on Windows
    as well, where a leading slash alone does not make a path absolute.
    """
    text = str(relative_path)
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ValueError(
            f"Manifest entry is an absolute path: {relative_path!r}. It was produced by an "
            "older version of scripts/00_audit_dataset.py. Re-run the audit to regenerate "
            "the manifests with paths relative to --data-root."
        )
    return Path(data_root) / Path(relative_path)


def build_breast_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Expand the patient manifest into one row per breast, using the oblique views.

    Each oblique view depicts one anatomical side (see data/README.md), so its breast-level
    label from Diagnostics.xlsx applies to the whole image. This differs from the anterior
    view, which contains both breasts: for a malignant patient, half of that image is the
    healthy contralateral breast, so the patient-level label is diluted by construction.

    Two consequences follow. The training set doubles, from 119 images to 238. And the two
    breasts of one patient share age, BMI, body temperature, camera, session and posture,
    so a comparison made *within* a patient is immune to the demographic confounding that
    dominates the patient-level task.
    """
    rows = []
    for r in manifest.itertuples(index=False):
        for side, path_col, label_col in (("left", "oblleft_path", "left_label"),
                                          ("right", "oblright_path", "right_label")):
            cls = getattr(r, label_col)
            path = getattr(r, path_col)
            if not isinstance(path, str) or not path or not isinstance(cls, str):
                continue
            rows.append({
                "sample_id": f"{r.patient_id}_{side}",
                "patient_id": r.patient_id,
                "side": side,
                "image_path": path,
                "breast_class": cls,                      # N / PB / PM
                "breast_label": int(cls == "PM"),         # target: carcinoma in this breast
                "patient_binary_label": int(r.patient_binary_label),
                "age": r.age,
            })
    return pd.DataFrame(rows)


def phash_file(path: Path) -> str:
    """Dependency-free 64-bit perceptual dHash used only to flag near duplicates."""
    with Image.open(path) as im:
        gray = im.convert("L").resize((9, 8))
        a = np.asarray(gray, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:016x}"


def discover_patients(data_root: Path, image_ext: str = ".jpg") -> pd.DataFrame:
    """Discover the cohort under `data_root`.

    `image_ext` selects the file type: ".jpg" for the delivered auto-ranged renders, ".tiff"
    for the float32 degrees-Celsius tree produced by scripts/20_extract_radiometric.py. The
    two trees are otherwise identical in layout, so a manifest can be built over either.
    """
    rows = []
    for folder_label in ("Benign", "Malignant"):
        class_dir = data_root / folder_label
        if not class_dir.exists():
            continue
        patient_dirs = sorted(x for x in class_dir.iterdir()
                              if x.is_dir() and x.name.startswith("IIR"))
        for pdir in patient_dirs:
            row = {"patient_id": pdir.name, "folder_label": folder_label.lower()}
            for view in VIEWS:
                path = pdir / f"{pdir.name}_{view}{image_ext}"
                # Relative to data_root: absolute paths would bake one machine's layout
                # into a committed artifact and break portability across dev/compute hosts.
                row[f"{view}_path"] = (path.relative_to(data_root).as_posix()
                                       if path.exists() else "")
            rows.append(row)
    df = pd.DataFrame(rows)
    dup = (df.loc[df["patient_id"].duplicated(keep=False), "patient_id"].unique()
           if not df.empty else [])
    if len(dup):
        raise ValueError(
            f"{data_root}: {len(dup)} patient directories occur under more than one class "
            f"folder (e.g. {sorted(dup)[:5]}). A patient must belong to exactly one class."
        )
    return df


def build_patient_manifest(data_root: Path, label_rule: str = "any_pm_malignant",
                           compute_hashes: bool = True,
                           image_ext: str = ".jpg") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The patient manifest and the long image manifest of the second cohort.

    Folders supply the images and the folder-level class, `Diagnostics.xlsx` the per-breast
    labels and covariates; the two are joined on the patient identifier and every
    disagreement between them is recorded in a column rather than resolved silently.
    """
    patients = discover_patients(data_root, image_ext)
    diag = load_diagnostics(data_root / "Diagnostics.xlsx")
    manifest = patients.merge(diag, on="patient_id", how="outer", indicator=True)
    manifest["diagnostics_match"] = manifest["_merge"].eq("both")
    manifest.drop(columns=["_merge"], inplace=True)
    manifest["patient_binary_label"] = [
        derive_patient_label(left, right, label_rule)
        for left, right in zip(manifest["left_label"], manifest["right_label"])]
    manifest["folder_binary_label"] = manifest["folder_label"].map({"benign": 0, "malignant": 1})
    manifest["folder_diagnostics_consistent"] = manifest["folder_binary_label"].eq(
        manifest["patient_binary_label"])
    manifest["available_views"] = manifest[[f"{v}_path" for v in VIEWS]].apply(
        lambda r: sum(bool(x) for x in r), axis=1)

    # Last line of defence: the manifest defines the cohort, and one row per patient is the
    # invariant every later stage assumes. A duplicated row does not fail loudly downstream,
    # it mispairs labels with predictions and quietly inflates measured performance.
    dup = manifest.loc[manifest["patient_id"].duplicated(keep=False), "patient_id"].unique()
    if len(dup):
        raise ValueError(
            f"Manifest has {len(manifest)} rows for {manifest['patient_id'].nunique()} patients; "
            f"{len(dup)} are duplicated (e.g. {sorted(dup)[:5]}). Refusing to emit it."
        )
    only_folders = manifest.loc[~manifest["diagnostics_match"], "patient_id"].tolist()
    only_diag = manifest.loc[manifest["folder_label"].isna(), "patient_id"].tolist()
    if only_folders or only_diag:
        print(f"Warning: {len(only_folders)} patients have image folders but no Diagnostics row "
              f"({only_folders[:5]}); {len(only_diag)} appear in Diagnostics but have no folder "
              f"({only_diag[:5]}). They carry no label and are excluded from splits.")

    images = []
    for _, r in manifest.iterrows():
        for view in VIEWS:
            ptxt = r.get(f"{view}_path", "")
            if not isinstance(ptxt, str) or not ptxt:
                continue
            p = resolve_image_path(data_root, ptxt)
            rec = {
                "patient_id": r["patient_id"], "folder_label": r.get("folder_label"),
                "patient_binary_label": r.get("patient_binary_label"), "view": view,
                "image_path": ptxt, "exists": p.exists(),
            }
            if p.exists() and compute_hashes:
                rec["sha256"] = file_sha256(p)
                try:
                    rec["phash"] = phash_file(p)
                except Exception:  # noqa: BLE001 - an unreadable image is an audit finding
                    rec["phash"] = ""
            images.append(rec)
    return manifest.sort_values("patient_id").reset_index(drop=True), pd.DataFrame(images)
