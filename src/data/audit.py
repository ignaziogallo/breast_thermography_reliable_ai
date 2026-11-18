"""Dataset audit of the second cohort: counts, duplicates, covariate screening, figures.

Run before any model is fitted. It writes the machine-readable audit files under
`results/`, the covariate distributions as figures, and a short Markdown report; it changes
nothing about the cohort, it only describes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import pointbiserialr

COVARIATES = ("age", "weight", "height", "bmi", "temperature")

AXIS_LABELS = {
    "age": "Age (years)",
    "weight": "Weight (kg)",
    "height": "Height (cm)",
    "bmi": r"BMI (kg/m$^2$)",
    "temperature": "Body temperature (°C)",
}

# Okabe--Ito colourblind-safe pair. Hatching is applied as a secondary encoding so the
# two groups remain distinguishable in greyscale print and under colour-vision deficiency.
CLASS_STYLE = {
    0: {"label": "Non-malignant", "color": "#0072B2", "hatch": "//"},
    1: {"label": "Malignant", "color": "#E69F00", "hatch": "\\\\"},
}

LABEL_RULE_NOTE = (
    "The default patient-level binary label is **malignant if at least one breast is PM; "
    "otherwise non-malignant (N/PB)**. This rule is explicit and should be reconsidered "
    "after inspecting the breast-level label counts.")
DATA_QUALITY_NOTE = (
    "Exact duplicates are listed in `results/exact_duplicates.csv`. Cross-patient "
    "perceptual-hash candidates are listed in `results/near_duplicate_candidates.csv`. "
    "These are audit flags and require visual confirmation before exclusion.")


def _axis_label(col: str) -> str:
    return AXIS_LABELS.get(col, col.replace("_", " ").title())


def _finish(fig, ax, out: Path) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts of patients are integers
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_hist(df: pd.DataFrame, col: str, out: Path) -> None:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins="auto", color="#4D4D4D")
    ax.set_xlabel(_axis_label(col))
    ax.set_ylabel("Patients")
    _finish(fig, ax, out)


def _save_hist_by_class(df: pd.DataFrame, col: str, out: Path,
                        label_col: str = "patient_binary_label", n_bins: int = 14) -> None:
    """Class-stratified distribution of a covariate.

    Counts rather than densities are plotted: with 35 positives, a density estimate
    would imply more resolution than the sample supports. Group sizes are therefore
    stated in the legend so the unequal class sizes cannot be misread as shape.
    """
    x = pd.to_numeric(df[col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() == 0 or y[valid].nunique() < 2:
        return
    x, y = x[valid], y[valid].astype(int)

    bins = np.histogram_bin_edges(x, bins=n_bins)  # shared bins across both groups
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for cls in (0, 1):
        style = CLASS_STYLE[cls]
        g = x[y == cls]
        ax.hist(g, bins=bins, alpha=0.55, color=style["color"], hatch=style["hatch"],
                edgecolor=style["color"], linewidth=1.1,
                label=f"{style['label']} (n={len(g)})")
        ax.axvline(g.median(), color=style["color"], linestyle="--", linewidth=1.3)

    ax.set_xlabel(_axis_label(col))
    ax.set_ylabel("Patients")
    ax.legend(frameon=False, loc="best")
    _finish(fig, ax, out)


def _near_duplicate_pairs(images: pd.DataFrame, threshold: int) -> list[dict]:
    """Cross-patient image pairs whose perceptual hashes differ by at most `threshold` bits."""
    pairs = []
    if "phash" not in images.columns:
        return pairs
    vals = images.loc[images["phash"].fillna("").ne("")].reset_index(drop=True)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if vals.loc[i, "patient_id"] == vals.loc[j, "patient_id"]:
                continue
            try:
                a, b = int(vals.loc[i, "phash"], 16), int(vals.loc[j, "phash"], 16)
                distance = (a ^ b).bit_count()
            except Exception:  # noqa: BLE001 - a malformed hash is skipped, not fatal
                continue
            if distance <= threshold:
                pairs.append({"patient_a": vals.loc[i, "patient_id"],
                              "path_a": vals.loc[i, "image_path"],
                              "patient_b": vals.loc[j, "patient_id"],
                              "path_b": vals.loc[j, "image_path"],
                              "phash_distance": distance})
    return pairs


def _covariate_statistics(manifest: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    """Per-covariate summary and point-biserial screening against the label, with figures."""
    rows = []
    for col in COVARIATES:
        s = pd.to_numeric(manifest[col], errors="coerce")
        row = {"variable": col, "n": int(s.notna().sum()), "missing": int(s.isna().sum()),
               "mean": s.mean(), "std": s.std(), "median": s.median(),
               "min": s.min(), "max": s.max()}
        for label in (0, 1):
            x = s[manifest["patient_binary_label"].eq(label)]
            row[f"mean_label_{label}"] = x.mean()
            row[f"std_label_{label}"] = x.std()
        valid = s.notna() & manifest["patient_binary_label"].notna()
        if valid.sum() >= 3 and manifest.loc[valid, "patient_binary_label"].nunique() == 2:
            r, p = pointbiserialr(manifest.loc[valid, "patient_binary_label"].astype(int), s[valid])
            row["point_biserial_r"] = r
            row["point_biserial_p"] = p
        rows.append(row)
        _save_hist(manifest, col, figures_dir / f"metadata_{col}")
        _save_hist_by_class(manifest, col, figures_dir / f"metadata_{col}_by_class")
    return pd.DataFrame(rows)


def run_audit(manifest: pd.DataFrame, images: pd.DataFrame, reports_dir: Path,
              results_dir: Path, figures_dir: Path | None = None,
              phash_threshold: int = 4) -> dict:
    """Audit the cohort and write the report, the audit tables and the covariate figures."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = figures_dir or (reports_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    inconsistent = manifest["folder_diagnostics_consistent"] == False  # noqa: E712 - NaN-aware
    summary = {
        "n_manifest_rows": int(len(manifest)),
        "n_patients_in_folders": int(manifest["folder_label"].notna().sum()),
        "n_patients_in_diagnostics": int(manifest["diagnostics_match"].sum()),
        "n_images_found": int(images["exists"].sum()) if not images.empty else 0,
        "patients_with_all_3_views": int((manifest["available_views"] == 3).sum()),
        "patients_missing_views": int((manifest["available_views"] < 3).sum()),
        "folder_diagnostics_inconsistencies": int(inconsistent.sum()),
        "patient_label_counts":
            manifest["patient_binary_label"].value_counts(dropna=False).to_dict(),
        "left_label_counts": manifest["left_label"].value_counts(dropna=False).to_dict(),
        "right_label_counts": manifest["right_label"].value_counts(dropna=False).to_dict(),
    }

    if "sha256" in images.columns and not images.empty:
        exact_dups = images[images.duplicated("sha256", keep=False)
                            & images["sha256"].ne("")].sort_values("sha256")
        summary["exact_duplicate_image_rows"] = int(len(exact_dups))
        summary["exact_duplicate_hash_groups"] = (int(exact_dups["sha256"].nunique())
                                                  if not exact_dups.empty else 0)
        exact_dups.to_csv(results_dir / "exact_duplicates.csv", index=False)

    near_pairs = _near_duplicate_pairs(images, phash_threshold)
    pd.DataFrame(near_pairs).to_csv(results_dir / "near_duplicate_candidates.csv", index=False)
    summary["near_duplicate_candidate_pairs"] = len(near_pairs)

    stats = _covariate_statistics(manifest, figures_dir)
    stats.to_csv(results_dir / "metadata_statistics.csv", index=False)

    # Machine-readable one-row audit summary.
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary.items()}
    pd.DataFrame([flat]).to_csv(results_dir / "data_audit.csv", index=False)

    lines = ["# Dataset Audit", "", "## Core counts", ""]
    lines += [f"- **{k.replace('_', ' ')}:** {v}" for k, v in summary.items()]
    lines += ["", "## Label rule", "", LABEL_RULE_NOTE, "",
              "## Metadata statistics", "", "```\n" + stats.to_string(index=False) + "\n```", "",
              "## Data-quality checks", "", DATA_QUALITY_NOTE, ""]
    (reports_dir / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
