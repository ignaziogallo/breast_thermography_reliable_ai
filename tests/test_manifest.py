import re
from pathlib import Path

import pandas as pd
import pytest

from src.data.manifest import VIEWS, resolve_image_path
from src.data.metadata import derive_patient_label
from src.utils.io import read_results_csv

PATH_COLUMNS = {
    "results/patient_manifest.csv": [f"{v}_path" for v in VIEWS],
    "results/image_manifest.csv": ["image_path"],
    "results/near_duplicate_candidates.csv": ["path_a", "path_b"],
}


def test_patient_label_rule():
    assert derive_patient_label("PM", "N") == 1
    assert derive_patient_label("PB", "N") == 0
    assert derive_patient_label("N", "N") == 0


def test_patient_label_rule_is_order_independent():
    assert derive_patient_label("N", "PM") == derive_patient_label("PM", "N") == 1


@pytest.mark.parametrize("csv_name", sorted(PATH_COLUMNS))
def test_manifests_store_relative_paths(csv_name):
    """Absolute paths bake one machine's layout into a committed artifact.

    Manifests are produced on the compute server and read on other hosts, so a path such
    as /home/<user>/... silently invalidates the file everywhere else.
    """
    path = Path(csv_name)
    if not path.exists():
        pytest.skip(f"{csv_name} not generated yet")
    df = pd.read_csv(path)
    for col in PATH_COLUMNS[csv_name]:
        if col not in df.columns:
            continue
        absolute = [v for v in df[col].dropna().astype(str) if v.startswith("/") or ":\\" in v]
        assert not absolute, f"{csv_name}:{col} contains absolute paths, e.g. {absolute[:2]}"


def test_resolve_image_path_rebuilds_a_full_path():
    assert resolve_image_path(Path("/data/root"), "Benign/IIR0001/IIR0001_anterior.jpg") == \
        Path("/data/root/Benign/IIR0001/IIR0001_anterior.jpg")


def test_resolve_image_path_rejects_absolute_entries():
    """pathlib would discard the data root and yield the absolute path unchanged."""
    with pytest.raises(ValueError, match="absolute path"):
        resolve_image_path(Path("/data/root"), "/somewhere/else/IIR0001_anterior.jpg")


def test_manifest_has_exactly_one_row_per_patient():
    """The cohort definition. A duplicated row mispairs labels with predictions.

    This is a regression test for a real incident: a Diagnostics.xlsx holding every
    patient twice produced a 241-row manifest for 119 patients, which silently corrupted
    the labels of a completed experiment rather than failing.
    """
    path = Path("results/patient_manifest.csv")
    if not path.exists():
        pytest.skip("Run scripts/00_audit_dataset.py first")
    m = pd.read_csv(path)
    dup = m.loc[m["patient_id"].duplicated(keep=False), "patient_id"].unique()
    assert not len(dup), (
        f"patient_manifest.csv has {len(m)} rows for {m['patient_id'].nunique()} patients; "
        f"duplicated: {sorted(dup)[:5]}")


def test_manifest_patient_ids_are_wellformed():
    """Catches a conflicted CSV: git markers survive parsing as extra data rows."""
    path = Path("results/patient_manifest.csv")
    if not path.exists():
        pytest.skip("Run scripts/00_audit_dataset.py first")
    ids = pd.read_csv(path)["patient_id"].astype(str)
    bad = [i for i in ids if not re.fullmatch(r"IIR\d+", i)]
    assert not bad, f"patient_manifest.csv has malformed patient_id values: {bad[:5]}"


@pytest.mark.parametrize("csv_name", ["results/patient_manifest.csv", "results/patient_splits.csv",
                                      "results/image_manifest.csv"])
def test_results_csvs_have_no_conflict_markers(csv_name):
    path = Path(csv_name)
    if not path.exists():
        pytest.skip(f"{csv_name} not generated yet")
    read_results_csv(path)  # raises if markers are present


def test_split_labels_agree_with_manifest():
    """The split file caches a label; if it drifts from the manifest, folds and scoring disagree."""
    mpath, spath = Path("results/patient_manifest.csv"), Path("results/patient_splits.csv")
    if not mpath.exists() or not spath.exists():
        pytest.skip("Run the audit and split scripts first")
    truth = pd.read_csv(mpath).set_index("patient_id")["patient_binary_label"].to_dict()
    splits = pd.read_csv(spath)
    bad = {pid for pid, lab in zip(splits["patient_id"], splits["label"])
           if pid in truth and int(truth[pid]) != int(lab)}
    assert not bad, f"patient_splits.csv disagrees with the manifest for {sorted(bad)[:5]}"
