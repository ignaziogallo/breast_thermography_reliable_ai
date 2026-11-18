"""Bilateral thermal asymmetry features.

Contralateral comparison is the oldest and most clinically grounded reading strategy in
breast thermography: a healthy thorax is approximately symmetric about the sagittal
midline, and a unilateral lesion perturbs that symmetry. This module turns that reading
into a handful of interpretable scalars, so that the deep models can be asked a sharp
question -- do they beat a subtraction?

Two constraints shape every feature here.

*Scale invariance.* The exports are per-image auto-ranged renderings (351/357 saturate at
grey 255), so grey level is not temperature and is not comparable between files. Every
image is therefore z-scored before use, and every feature is computed **within** a single
image. See data/README.md.

*Side ignorance.* The patient-level target does not say which breast is diseased, and at
test time the side is unknown. Features are therefore magnitudes (absolute values, maxima)
and never signed side contrasts, which would leak the lateralisation the model is not
given.

No feature uses the label, so extracting them for the whole cohort before splitting is not
leakage; the fold-wise standardisation applied by the estimator pipeline is what must stay
inside the training partition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.io import read_thermal_image

FEATURE_NAMES = [
    "asym_mean_abs",
    "asym_p95_abs",
    "asym_area_frac",
    "asym_net_abs",
    "asym_focality",
    "breast_band_std",
    "hotspot_intensity",
    "obl_contrast_diff_abs",
    "obl_contrast_max",
    "obl_std_diff_abs",
]

# Vertical window of the body bounding box treated as the breast region. Deliberately
# coarse: a learned segmentation would be a separate experiment and must not be smuggled
# in here.
BREAST_BAND = (0.32, 0.78)


def _otsu_threshold(a: np.ndarray, bins: int = 64) -> float:
    hist, edges = np.histogram(a, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(a.mean())
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return float(a.mean())
    cum = np.cumsum(hist * centres)
    m0 = np.divide(cum, w0, out=np.zeros_like(cum, dtype=float), where=w0 > 0)
    m1 = np.divide(cum[-1] - cum, w1, out=np.zeros_like(cum, dtype=float), where=w1 > 0)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def _load_z(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (z-scored image, body mask).

    The mask separates the warm body from the cold background.
    """
    raw = read_thermal_image(path)
    mask = raw > _otsu_threshold(raw)
    std = raw.std()
    z = (raw - raw.mean()) / (std if std > 0 else 1.0)
    return z, mask


def _breast_band(mask: np.ndarray) -> slice:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        h = mask.shape[0]
        return slice(int(BREAST_BAND[0] * h), int(BREAST_BAND[1] * h))
    top, bottom = rows[0], rows[-1]
    height = bottom - top + 1
    return slice(int(top + BREAST_BAND[0] * height), int(top + BREAST_BAND[1] * height))


def _difference_map(z: np.ndarray, mask: np.ndarray,
                    band: slice) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the torso about the body midline and subtract; return (diff, valid mask)."""
    zb, mb = z[band], mask[band]
    cols = np.flatnonzero(mb.any(axis=0))
    midline = int(round(cols.mean())) if cols.size else zb.shape[1] // 2
    half = min(midline, zb.shape[1] - midline)
    if half < 4:  # degenerate: fall back to the geometric centre
        midline, half = zb.shape[1] // 2, zb.shape[1] // 2
    left = zb[:, midline - half:midline]
    right = zb[:, midline:midline + half]
    valid = mb[:, midline - half:midline] & np.fliplr(mb[:, midline:midline + half])
    return left - np.fliplr(right), valid


def _oblique_contrast(path: Path) -> tuple[float, float]:
    """Within-image breast-versus-upper-chest contrast, and breast-band heterogeneity."""
    z, mask = _load_z(path)
    band = _breast_band(mask)
    rows = np.flatnonzero(mask.any(axis=1))
    top = rows[0] if rows.size else 0
    height = (rows[-1] - rows[0] + 1) if rows.size else z.shape[0]
    upper = slice(int(top + 0.05 * height), int(top + 0.28 * height))
    breast_vals = z[band][mask[band]]
    upper_vals = z[upper][mask[upper]]
    if breast_vals.size == 0 or upper_vals.size == 0:
        return 0.0, 0.0
    return float(breast_vals.mean() - upper_vals.mean()), float(breast_vals.std())


def extract_features(anterior: Path, oblleft: Path, oblright: Path) -> dict[str, float]:
    """The asymmetry descriptors of one patient, from the three views of one session."""
    z, mask = _load_z(anterior)
    band = _breast_band(mask)
    diff, valid = _difference_map(z, mask, band)

    d = np.abs(diff[valid]) if valid.any() else np.abs(diff).ravel()
    signed = diff[valid] if valid.any() else diff.ravel()
    body_band = z[band][mask[band]]
    if d.size == 0 or body_band.size == 0:
        return {name: float("nan") for name in FEATURE_NAMES}

    mean_abs = float(d.mean())
    left_c, left_s = _oblique_contrast(oblleft)
    right_c, right_s = _oblique_contrast(oblright)

    return {
        "asym_mean_abs": mean_abs,
        "asym_p95_abs": float(np.percentile(d, 95)),
        # extent of the asymmetric territory: |diff| above one within-image sd
        "asym_area_frac": float((d > 1.0).mean()),
        # net side imbalance, magnitude only so that the diseased side is not revealed
        "asym_net_abs": float(abs(signed.mean())),
        # focal hot spot versus diffuse asymmetry
        "asym_focality": float(np.percentile(d, 99) / (mean_abs + 1e-6)),
        "breast_band_std": float(body_band.std()),
        "hotspot_intensity": float(np.percentile(body_band, 99) - np.percentile(body_band, 50)),
        "obl_contrast_diff_abs": float(abs(left_c - right_c)),
        "obl_contrast_max": float(max(left_c, right_c)),
        "obl_std_diff_abs": float(abs(left_s - right_s)),
    }
