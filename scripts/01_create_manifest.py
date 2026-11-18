#!/usr/bin/env python3
"""Build the second cohort's patient and image manifests, without running the audit.

The same manifests `scripts/00_audit_dataset.py` writes, for when only the cohort
definition is needed -- for instance over the radiometric TIFF tree, which holds the same
patients under another file extension (`--image-ext .tiff`).

Example
-------
    python scripts/01_create_manifest.py --data-root data/Breast-Thermal-Image
    python scripts/01_create_manifest.py --data-root data/Breast-Thermal-Image-Radiometric \\
        --image-ext .tiff --output results/patient_manifest_radiometric.csv \\
        --images-output results/image_manifest_radiometric.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import build_patient_manifest


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--images-output", type=Path, default=Path("results/image_manifest.csv"))
    p.add_argument("--no-hashes", action="store_true")
    p.add_argument("--image-ext", default=".jpg",
                   help="'.jpg' for the renders, '.tiff' for the radiometric tree")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest, images = build_patient_manifest(a.data_root, compute_hashes=not a.no_hashes,
                                              image_ext=a.image_ext)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(a.output, index=False)
    images.to_csv(a.images_output, index=False)
    print(f"Wrote {len(manifest)} patient rows to {a.output}")
    print(f"Wrote {len(images)} image rows to {a.images_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
