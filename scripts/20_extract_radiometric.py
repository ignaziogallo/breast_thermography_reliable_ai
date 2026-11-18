#!/usr/bin/env python3
"""Recover absolute temperature from the radiometric FLIR JPEGs and write float32 TIFFs in degC.

The visible greyscale in these files is auto-ranged per image -- 351/357 saturate at 255 --
which is why the pixels were previously treated as carrying no temperature. That is true of
the *render*, but the render is not the data: every one of the 357 files is a radiometric
FLIR JPEG whose APP1 segments still hold the raw 16-bit thermal array and the camera's
calibration constants. This script extracts them and applies the FLIR radiometric equation,
so that grey level stops being an arbitrary per-file scale and becomes degrees Celsius,
comparable across the whole cohort.

That matters for three reasons: per-image normalisation becomes a modelling choice rather
than something the data forces; bilateral-asymmetry features become differences in degrees
instead of differences between two independently rescaled images; and the cohort lands in
the same physical units as the DMR-IR mirror, which is a precondition for any cross-cohort
comparison and for comparing against work that trains on raw temperature.

Radiometry
----------
Signal to temperature, with the emissivity, reflected-radiation and atmospheric corrections
the camera itself would apply::

    raw_refl = R1/(R2*(exp(B/T_refl) - F)) - O          reflected-radiation signal
    raw_atm  = R1/(R2*(exp(B/T_atm ) - F)) - O          path-radiance signal
    raw_obj  = (S - (1-eps)*tau*raw_refl - (1-tau)*raw_atm) / (eps*tau)
    T        = B/ln(R1/(R2*(raw_obj + O)) + F) - 273.15

`tau` is the atmospheric transmission from the stored alpha/beta/X constants, the relative
humidity and the object distance. At the 1 m acquisition distance it is ~0.99, so the
dominant correction is emissivity (0.95) against a 20 degC reflected background. Pass
`--no-corrections` to skip it and get the uncorrected object-signal temperature instead;
the manifest records both so the difference is auditable rather than assumed.

Output mirrors the input tree, so the result is a drop-in dataset root::

    <out>/Benign/IIR0001/IIR0001_anterior.tiff
    <out>/Diagnostics.xlsx
    <out>/radiometric_manifest.csv

Example
-------
    python scripts/20_extract_radiometric.py \
        --data-root data/Breast-Thermal-Image \
        --out       data/Breast-Thermal-Image-Radiometric
"""

import argparse
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# FFF record types
REC_RAW_DATA = 1
REC_CAMERA_INFO = 32

# CameraInfo field offsets (little-endian floats unless noted), matching ExifTool's FLIR table.
CI = {
    "emissivity": 0x020, "object_distance": 0x024, "reflected_temp": 0x028,
    "atmospheric_temp": 0x02C, "ir_window_temp": 0x030, "ir_window_transmission": 0x034,
    "relative_humidity": 0x03C, "planck_r1": 0x058, "planck_b": 0x05C, "planck_f": 0x060,
    "atm_alpha1": 0x070, "atm_alpha2": 0x074, "atm_beta1": 0x078, "atm_beta2": 0x07C,
    "atm_x": 0x080,
}
CI_PLANCK_O = 0x308   # int32
CI_PLANCK_R2 = 0x30C  # float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/Breast-Thermal-Image"))
    p.add_argument("--out", type=Path, default=Path("data/Breast-Thermal-Image-Radiometric"))
    p.add_argument("--no-corrections", action="store_true",
                   help="skip emissivity/reflected/atmospheric correction")
    p.add_argument("--overwrite", action="store_true", help="rewrite existing TIFFs")
    p.add_argument("--limit", type=int, default=None, help="process only the first N images")
    return p.parse_args()


def flir_records(path: Path) -> dict[int, bytes]:
    """FFF records keyed by record type, reassembled from the APP1 chunks of a FLIR JPEG."""
    data = path.read_bytes()
    chunks: dict[int, bytes] = {}
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of scan: no metadata beyond this point
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + length]
        if payload.startswith(b"FLIR\x00"):
            chunks[payload[6]] = payload[8:]
        i += 2 + length

    if not chunks:
        return {}
    fff = b"".join(chunks[k] for k in sorted(chunks))
    if not fff.startswith(b"FFF\x00"):
        return {}

    dir_offset, dir_count = struct.unpack(">II", fff[0x18:0x20])
    records: dict[int, bytes] = {}
    for entry in range(dir_count):
        off = dir_offset + entry * 0x20
        if off + 0x20 > len(fff):
            break
        rec_type = struct.unpack(">H", fff[off:off + 2])[0]
        data_off, data_len = struct.unpack(">II", fff[off + 0x0C:off + 0x14])
        if data_len and data_off + data_len <= len(fff):
            records[rec_type] = fff[data_off:data_off + data_len]
    return records


def raw_thermal(record: bytes) -> np.ndarray:
    """Raw 16-bit thermal array. Byte order is declared by the first word of the record."""
    byte_order = "<" if record[0:2] == b"\x02\x00" else ">"
    width, height = struct.unpack(byte_order + "HH", record[2:6])
    expected = 32 + width * height * 2
    if len(record) < expected:
        raise ValueError(f"raw record is {len(record)} bytes, expected {expected} for {width}x{height}")
    return np.frombuffer(record[32:expected], dtype=byte_order + "u2").reshape(height, width).astype(np.float64)


def camera_info(record: bytes) -> dict[str, float]:
    info = {name: struct.unpack("<f", record[off:off + 4])[0] for name, off in CI.items()}
    info["planck_o"] = float(struct.unpack("<i", record[CI_PLANCK_O:CI_PLANCK_O + 4])[0])
    info["planck_r2"] = struct.unpack("<f", record[CI_PLANCK_R2:CI_PLANCK_R2 + 4])[0]
    return info


def atmospheric_transmission(c: dict[str, float]) -> float:
    """Transmission of the air column between camera and subject (~0.99 at 1 m)."""
    t_atm_c = c["atmospheric_temp"] - 273.15
    humidity = c["relative_humidity"]
    if humidity > 1.0:            # some cameras store percent rather than a fraction
        humidity /= 100.0
    water = humidity * np.exp(1.5587 + 6.939e-2 * t_atm_c
                              - 2.7816e-4 * t_atm_c ** 2 + 6.8455e-7 * t_atm_c ** 3)
    root_d = np.sqrt(max(c["object_distance"], 0.0))
    x = c["atm_x"]
    tau = (x * np.exp(-root_d * (c["atm_alpha1"] + c["atm_beta1"] * np.sqrt(water)))
           + (1 - x) * np.exp(-root_d * (c["atm_alpha2"] + c["atm_beta2"] * np.sqrt(water))))
    return float(np.clip(tau, 0.1, 1.0))


def signal_to_celsius(signal: np.ndarray, c: dict[str, float], corrections: bool) -> np.ndarray:
    r1, r2, b, f, o = c["planck_r1"], c["planck_r2"], c["planck_b"], c["planck_f"], c["planck_o"]

    if corrections:
        eps = c["emissivity"]
        tau = atmospheric_transmission(c)
        raw_refl = r1 / (r2 * (np.exp(b / c["reflected_temp"]) - f)) - o
        raw_atm = r1 / (r2 * (np.exp(b / c["atmospheric_temp"]) - f)) - o
        signal = (signal - (1 - eps) * tau * raw_refl - (1 - tau) * raw_atm) / (eps * tau)

    # Guard the log: the argument must stay positive for the inversion to be defined.
    denom = r2 * (signal + o)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = r1 / denom + f
        celsius = b / np.log(arg) - 273.15
    return np.where(np.isfinite(celsius), celsius, np.nan)


def main() -> int:
    args = parse_args()
    root: Path = args.data_root
    if not root.exists():
        print(f"ERROR: {root} does not exist.", file=sys.stderr)
        return 1

    sources = sorted(root.glob("*/*/*.jpg"))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        print(f"ERROR: no images found under {root}/<class>/<patient>/*.jpg", file=sys.stderr)
        return 1
    print(f"Found {len(sources)} JPEGs under {root}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    calibrations: Counter = Counter()
    failures: list[str] = []

    for n, src in enumerate(sources, start=1):
        rel = src.relative_to(root)
        try:
            records = flir_records(src)
            if REC_RAW_DATA not in records or REC_CAMERA_INFO not in records:
                raise ValueError("no FLIR radiometric payload")
            signal = raw_thermal(records[REC_RAW_DATA])
            cal = camera_info(records[REC_CAMERA_INFO])
            celsius = signal_to_celsius(signal, cal, corrections=not args.no_corrections)
            uncorrected = signal_to_celsius(signal, cal, corrections=False)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the cohort
            failures.append(f"{rel}: {exc}")
            continue

        calibrations[(round(cal["planck_r1"], 1), round(cal["planck_b"], 1), cal["planck_f"],
                      cal["planck_o"], round(cal["planck_r2"], 7), round(cal["emissivity"], 3))] += 1

        dest = (args.out / rel).with_suffix(".tiff")
        if args.overwrite or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(celsius.astype(np.float32), mode="F").save(dest)

        rows.append({
            "image_path": rel.with_suffix(".tiff").as_posix(),
            "source_path": rel.as_posix(),
            "height": celsius.shape[0], "width": celsius.shape[1],
            "t_min": float(np.nanmin(celsius)), "t_max": float(np.nanmax(celsius)),
            "t_mean": float(np.nanmean(celsius)), "t_std": float(np.nanstd(celsius)),
            "t_p99": float(np.nanpercentile(celsius, 99)),
            "t_mean_uncorrected": float(np.nanmean(uncorrected)),
            "n_nan": int(np.isnan(celsius).sum()),
            "emissivity": cal["emissivity"], "object_distance": cal["object_distance"],
            "reflected_temp_k": cal["reflected_temp"],
            "atmospheric_transmission": atmospheric_transmission(cal),
            "planck_r1": cal["planck_r1"], "planck_b": cal["planck_b"], "planck_f": cal["planck_f"],
            "planck_o": cal["planck_o"], "planck_r2": cal["planck_r2"],
        })
        if n % 50 == 0:
            print(f"  {n}/{len(sources)}")

    if not rows:
        print("ERROR: nothing could be extracted.", file=sys.stderr)
        return 1

    import pandas as pd

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out / "radiometric_manifest.csv", index=False)

    diagnostics = root / "Diagnostics.xlsx"
    if diagnostics.exists():
        shutil.copy2(diagnostics, args.out / "Diagnostics.xlsx")

    print(f"\nWrote {len(manifest)} TIFFs to {args.out}")
    print(f"Calibration tuples across the cohort: {len(calibrations)}")
    for tup, count in calibrations.most_common():
        print(f"  R1={tup[0]} B={tup[1]} F={tup[2]} O={tup[3]} R2={tup[4]} eps={tup[5]}  -> {count} files")
    if len(calibrations) > 1:
        print("  WARNING: more than one calibration in the cohort. Temperatures from different\n"
              "  calibrations are not directly comparable; check before pooling them.")

    print(f"\nTemperature (degC): min {manifest.t_min.min():.1f}..{manifest.t_min.max():.1f} | "
          f"max {manifest.t_max.min():.1f}..{manifest.t_max.max():.1f} | "
          f"mean {manifest.t_mean.mean():.1f} +- {manifest.t_mean.std():.1f}")
    shift = (manifest.t_mean - manifest.t_mean_uncorrected).mean()
    print(f"Mean shift from the emissivity/reflected/atmospheric correction: {shift:+.2f} degC")

    implausible = manifest[(manifest.t_max < 28) | (manifest.t_max > 42)]
    if len(implausible):
        print(f"\nWARNING: {len(implausible)} images have a peak outside 28-42 degC:")
        for _, r in implausible.head(10).iterrows():
            print(f"  {r.source_path}: max {r.t_max:.1f} degC")
    if manifest.n_nan.sum():
        print(f"\nWARNING: {int(manifest.n_nan.sum())} pixels were outside the invertible range "
              "and are NaN in the output.")
    if failures:
        print(f"\n{len(failures)} files failed:")
        for line in failures[:10]:
            print(f"  {line}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
