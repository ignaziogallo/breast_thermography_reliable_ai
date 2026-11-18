"""File-level helpers shared by every script: hashing, thermogram loading, guarded reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFLICT_MARKERS = ("<<<<<<< ", "=======\n", ">>>>>>> ")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in blocks so a large file is never held in memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_thermal_image(path: Path) -> np.ndarray:
    """Load a thermogram as a 2-D float array, preserving whatever precision it holds.

    The cohort exists in two forms: the delivered 8-bit auto-ranged JPEG renders, and the
    float32 degrees-Celsius TIFFs from scripts/20_extract_radiometric.py. `Image.convert("L")`
    is right for the first and destroys the second -- on a mode-'F' TIFF spanning 21.3 to
    35.4 degC it clips to 8-bit integers and collapses 2729 distinct values into 15. Every
    loader must therefore branch on the mode rather than convert unconditionally.
    """
    from PIL import Image

    with Image.open(path) as im:
        # 'F' float32, 'I' int32 and 'I;16' 16-bit already carry one channel of real data.
        if im.mode in ("F", "I") or im.mode.startswith("I;"):
            return np.asarray(im, dtype=float)
        return np.asarray(im.convert("L"), dtype=float)


def read_results_csv(path: Path) -> pd.DataFrame:
    """Read a generated CSV, refusing one that still contains git conflict markers.

    A conflicted CSV still parses: git interleaves both audit runs and the markers become
    extra data rows. A 119-patient manifest silently becomes 241 rows for 122 "patients",
    every label is duplicated, and downstream `.loc` lookups mispair labels with
    predictions. This happened once and cost a full experiment run, so it is checked
    rather than trusted.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [m.strip() for m in CONFLICT_MARKERS if m in text]
    if found:
        raise ValueError(
            f"{path} contains unresolved git conflict markers ({', '.join(found)}). "
            "Do not edit it by hand: discard it and regenerate, e.g.\n"
            f"    git checkout --theirs {path} || git checkout HEAD -- {path}\n"
            "    python scripts/00_audit_dataset.py --data-root data/Breast-Thermal-Image"
        )
    return pd.read_csv(path)


def posix(path: Path) -> str:
    """A path as it is recorded in a `config.json`: with forward slashes on every platform.

    `str(path)` writes backslashes on Windows, and a run recorded there could then not be
    checked against its inputs on the host that trained it.
    """
    return Path(path).as_posix()


def input_fingerprint(manifest_path: Path, splits_path: Path) -> dict:
    """Identify exactly which manifest and split file an experiment consumed.

    Without this, a results directory cannot be checked against the inputs that produced
    it: if the manifest is regenerated after an experiment runs, the stored predictions
    silently refer to a cohort definition that no longer exists on disk, and the
    disagreement is only detectable as a mysterious label mismatch.
    """
    manifest = pd.read_csv(manifest_path)
    labels = manifest["patient_binary_label"].value_counts(dropna=False)
    return {
        "manifest_path": posix(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_rows": int(len(manifest)),
        "manifest_unique_patients": int(manifest["patient_id"].nunique()),
        "label_counts": {str(k): int(v) for k, v in labels.items()},
        "splits_path": posix(splits_path),
        "splits_sha256": file_sha256(splits_path),
    }


def ensure_parent(path: Path) -> None:
    """Create the directory a file is about to be written into."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(obj: dict, path: Path, **dumps_kwargs) -> None:
    """Write `obj` as indented JSON: the layout of every `config.json` in `results/`.

    Extra keyword arguments go to `json.dumps`. None is passed by default, so a value that
    is not JSON-serialisable raises here instead of being silently stringified into a run's
    record of what it did.
    """
    ensure_parent(path)
    Path(path).write_text(json.dumps(obj, indent=2, **dumps_kwargs), encoding="utf-8")
