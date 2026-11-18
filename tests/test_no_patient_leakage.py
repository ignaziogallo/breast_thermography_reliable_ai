from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import validate_split_integrity


def test_no_patient_overlap_if_splits_exist():
    path = Path("results/patient_splits.csv")
    if not path.exists():
        pytest.skip("Run scripts/02_create_patient_splits.py first")
    splits = pd.read_csv(path)
    errors = validate_split_integrity(splits)
    assert not errors, "\n".join(errors)


def test_no_exact_hash_cross_split_if_manifests_exist():
    spath, ipath = Path("results/patient_splits.csv"), Path("results/image_manifest.csv")
    if not spath.exists() or not ipath.exists():
        pytest.skip("Run audit and split scripts first")
    splits, images = pd.read_csv(spath), pd.read_csv(ipath)
    if "sha256" not in images.columns:
        pytest.skip("Image hashes unavailable")
    for (rep, fold), sg in splits.groupby(["repeat", "fold"]):
        merged = images.merge(sg[["patient_id","split"]], on="patient_id", how="inner")
        dup = merged[merged.duplicated("sha256", keep=False) & merged["sha256"].fillna("").ne("")]
        for h, g in dup.groupby("sha256"):
            assert g["split"].nunique() == 1, f"Exact image hash {h} crosses splits at repeat={rep}, fold={fold}: {g[['patient_id','split','image_path']].to_dict('records')}"
