"""Leakage regression tests on experiment prediction files.

`test_no_patient_leakage.py` checks that the *split definition* is clean. These tests check
something stronger and more easily broken: that what an experiment actually *scored* was
the held-out patients of that fold, and nothing else. A model that silently predicted on
patients it had trained on would pass the split test and fail here.

Every experiment that writes results/<experiment>/predictions.csv is checked
automatically; experiments that have not been run yet are skipped.

An experiment is checked against *its own* inputs, not against a fixed pair of files. The
repository holds more than one cohort -- the Cali `IIR*` patients and the DMR-IR `DMR*`
patients -- with a manifest and a split file each, so a hard-coded path would compare an
experiment's predictions against a fold definition built for different people and report a
leakage failure that is really a bookkeeping mismatch. Each experiment records the manifest
and splits it consumed in its `config.json` (see `src.utils.io.input_fingerprint`); that is
the pairing used here, with the original cohort as the fallback for runs predating it.
"""
import json
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

RESULTS = Path("results")
REQUIRED_COLUMNS = {"patient_id", "repeat", "fold", "y_true", "y_prob", "model"}
DEFAULT_SPLITS = RESULTS / "patient_splits.csv"
DEFAULT_MANIFEST = RESULTS / "patient_manifest.csv"


def _prediction_files() -> list[Path]:
    return sorted(RESULTS.glob("experiment_*/predictions.csv"))


def _config(pred_path: Path) -> dict:
    """The experiment's own record of what it consumed and which patients it dropped.

    Every script writes `config.json` as its final act, so its absence means the run did not
    finish. Checking such a directory against the default cohort produces a confusing
    mismatch failure for what is really an experiment still in flight.
    """
    config = pred_path.parent / "config.json"
    if not config.exists():
        pytest.skip(f"{pred_path.parent} has no config.json: the run did not complete, so "
                    "there is no record of which cohort it was evaluated against")
    return json.loads(config.read_text(encoding="utf-8"))


def _declared_input(pred_path: Path, key: str, default: Path) -> Path:
    """The manifest or splits file this experiment recorded that it consumed."""
    declared = _config(pred_path).get("inputs", {}).get(key)
    return Path(declared) if declared else default


def _declared_dropped(pred_path: Path) -> set[str]:
    """Patients the experiment recorded that it deliberately excluded.

    A view-restricted manifest holds fewer patients than the cohort-wide split file -- 231
    of 236 hold a DMR-IR frontal view -- so those patients legitimately go unscored. They
    are subtracted from the expected held-out set rather than the equality being dropped,
    which would also stop catching an experiment that silently skipped somebody.
    """
    config = _config(pred_path)
    declared = set(config.get("patients_dropped", []))
    n_declared = config.get("n_patients_dropped_absent_from_manifest")
    if n_declared is not None and len(declared) != int(n_declared):
        pytest.fail(f"{pred_path.parent}/config.json says {n_declared} patients were dropped "
                    f"but lists {len(declared)}; the list is truncated and cannot be trusted")
    return declared


@cache
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_splits(pred_path: Path) -> tuple[pd.DataFrame, Path]:
    path = _declared_input(pred_path, "splits_path", DEFAULT_SPLITS)
    if not path.exists():
        pytest.skip(f"{pred_path} declares {path}, which is absent; regenerate it first")
    return _read(path), path


@pytest.mark.parametrize("pred_path", _prediction_files() or [None])
def test_predictions_never_include_a_training_patient(pred_path):
    """No scored patient may appear in the train or val split of the same fold."""
    if pred_path is None:
        pytest.skip("No experiment has produced predictions.csv yet")
    splits, splits_path = _load_splits(pred_path)
    dropped = _declared_dropped(pred_path)
    preds = pd.read_csv(pred_path)

    missing = REQUIRED_COLUMNS - set(preds.columns)
    assert not missing, f"{pred_path} is missing required columns: {sorted(missing)}"

    # A cohort mismatch produces the same symptom as leakage -- every scored patient looks
    # "unexpected" -- so it is diagnosed separately and named for what it is.
    scored_all, split_all = set(preds["patient_id"]), set(splits["patient_id"])
    assert scored_all & split_all, (
        f"{pred_path} shares no patient with {splits_path}: the experiment was evaluated "
        f"against a different cohort. Scored e.g. {sorted(scored_all)[:3]}, "
        f"{splits_path} holds e.g. {sorted(split_all)[:3]}."
    )

    fold_ids = {
        (int(rep), int(fold)): {
            s: set(g.loc[g["split"] == s, "patient_id"]) for s in ("train", "val", "test")
        }
        for (rep, fold), g in splits.groupby(["repeat", "fold"])
    }

    for (model, rep, fold), g in preds.groupby(["model", "repeat", "fold"]):
        key = (int(rep), int(fold))
        assert key in fold_ids, f"{pred_path}: {model} scored fold {key}, absent from {splits_path}"
        scored = set(g["patient_id"])

        train_leak = scored & fold_ids[key]["train"]
        assert not train_leak, (
            f"{pred_path}: {model} repeat={rep} fold={fold} scored {len(train_leak)} patient(s) "
            f"that were in its own TRAINING set: {sorted(train_leak)[:5]}"
        )
        val_leak = scored & fold_ids[key]["val"]
        assert not val_leak, (
            f"{pred_path}: {model} repeat={rep} fold={fold} scored {len(val_leak)} patient(s) "
            f"used for model selection: {sorted(val_leak)[:5]}"
        )
        expected = fold_ids[key]["test"] - dropped
        assert scored == expected, (
            f"{pred_path}: {model} repeat={rep} fold={fold} did not score exactly the held-out "
            f"patients (missing {sorted(expected - scored)[:5]}, "
            f"unexpected {sorted(scored - expected)[:5]}; "
            f"{len(dropped)} declared absent from the manifest)"
        )


@pytest.mark.parametrize("pred_path", _prediction_files() or [None])
def test_one_prediction_per_patient_per_repeat(pred_path):
    """Patient-level bootstrap assumes independent rows; duplicates would inflate n."""
    if pred_path is None:
        pytest.skip("No experiment has produced predictions.csv yet")
    preds = pd.read_csv(pred_path)
    for (model, rep), g in preds.groupby(["model", "repeat"]):
        dup = g.loc[g["patient_id"].duplicated(), "patient_id"].unique()
        assert len(dup) == 0, (
            f"{pred_path}: {model} repeat={rep} has multiple out-of-fold predictions for "
            f"{len(dup)} patient(s): {sorted(dup)[:5]}"
        )


@pytest.mark.parametrize("pred_path", _prediction_files() or [None])
def test_labels_agree_with_manifest(pred_path):
    """A silently remapped target would make every downstream metric meaningless."""
    if pred_path is None:
        pytest.skip("No experiment has produced predictions.csv yet")
    manifest_path = _declared_input(pred_path, "manifest_path", DEFAULT_MANIFEST)
    if not manifest_path.exists():
        pytest.skip(f"{pred_path} declares {manifest_path}, which is absent")
    truth = _read(manifest_path).set_index("patient_id")["patient_binary_label"].to_dict()
    preds = pd.read_csv(pred_path)

    # Against the wrong manifest no id matches and the comparison below is vacuous, so the
    # overlap is asserted first: this test must never pass by finding nothing to check.
    known = [pid for pid in preds["patient_id"] if pid in truth]
    assert known, (
        f"{pred_path}: not one scored patient appears in {manifest_path}; the labels were "
        "never actually checked."
    )
    mismatched = [
        pid for pid, y in zip(preds["patient_id"], preds["y_true"])
        if pid in truth and int(truth[pid]) != int(y)
    ]
    assert not mismatched, f"{pred_path}: y_true disagrees with {manifest_path} for {sorted(set(mismatched))[:5]}"
