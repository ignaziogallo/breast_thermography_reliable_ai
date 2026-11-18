#!/usr/bin/env python3
"""Recompute every reported statistic, table and figure from the stored predictions.

This is the analysis-only path (Section 2 of the README): no GPU and no images. Each step
reads patient-level prediction files and manifests and rewrites the statistics file the
manuscript quotes. Those per-patient files are not distributed: build the manifests from
your copy of the datasets (README Section 5) and regenerate the predictions on a GPU
(README Section 8) first. The last step is the verification gate:
`36_consistency_sweep.py --strict` re-checks every number printed in the paper against the
files the earlier steps just wrote, and exits non-zero if one drifts.

Steps run in place, overwriting the statistics under `results/` with freshly computed ones.
That is the point: the shipped tree and the recomputed tree should be identical, and the
sweep is what says so. Copy `results/` first if you want a byte-level before/after diff.

    python scripts/reproduce_paper.py            # everything, then verify
    python scripts/reproduce_paper.py --list     # print the plan and exit
    python scripts/reproduce_paper.py --only 34 39   # selected steps, then verify
    python scripts/reproduce_paper.py --no-verify

Expect roughly 15-30 minutes on a laptop CPU; the bootstrap and permutation loops dominate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DMR_ARMS = [
    "results/experiment_dmr_01_leakage_combined/predictions.csv",
    "results/experiment_dmr_11_allframes_swin_tiny_patch4_window7_224/predictions.csv",
    "results/experiment_dmr_11_encoder_convnext_tiny_finetune/predictions.csv",
    "results/experiment_dmr_11_encoder_swin_tiny_patch4_window7_224_finetune/predictions.csv",
    "results/experiment_dmr_11_encoder_swin_tiny_patch4_window7_224_probe/predictions.csv",
]

COHORT2_ARMS = [
    "results/experiment_ver_11_encoder_convnext_tiny_finetune/predictions.csv",
    "results/experiment_ver_11_encoder_swin_tiny_patch4_window7_224_finetune/predictions.csv",
    "results/experiment_02_single_view_linear_probe/predictions.csv",
]

# Built from the datasets by README Section 5; every analysis step needs at least one.
REQUIRED_MANIFESTS = [
    "results/dmr_patient_manifest.csv",
    "results/patient_manifest.csv",
]

# (step id, what it produces for the manuscript, argv after the interpreter)
STEPS: list[tuple[str, str, list[str]]] = [
    ("34", "shortcut exploitation, all arms (Tab. 2; App. C, Tab. 5-6)",
     ["scripts/34_exploitation_interval.py"]),
    ("30a", "conditional discrimination, DMR-IR, with the same-date-group column "
            "(Fig. 4; App. H, Tab. 11)",
     ["scripts/30_conditional_evaluation.py", "--predictions", *DMR_ARMS,
      "--manifest", "results/dmr_patient_manifest.csv", "--confounder", "visit_date",
      "--stratum", "visit_date",
      "--output-dir", "results/confounders/dmr_session_conditional"]),
    ("30b", "the same grid without the date-group stratum, which the multiplicity "
            "correction consumes",
     ["scripts/30_conditional_evaluation.py", "--predictions", *DMR_ARMS,
      "--manifest", "results/dmr_patient_manifest.csv", "--confounder", "visit_date",
      "--output-dir", "results/confounders/dmr_encoders_conditional"]),
    ("30c", "conditional discrimination, second cohort, age calipers (Fig. 4b)",
     ["scripts/30_conditional_evaluation.py", "--predictions", *COHORT2_ARMS,
      "--manifest", "results/patient_manifest.csv", "--confounder", "age",
      "--calipers", "10", "5", "2",
      "--output-dir", "results/confounders/verification_conditional"]),
    ("35", "Holm correction over the 25-cell caliper family (Tab. 11 daggers)",
     ["scripts/35_caliper_multiplicity.py"]),
    ("32", "matched ladder against its size controls, both cohorts and both all-frames "
           "arms (Fig. 3; App. F-G, Tab. 9-10)",
     ["scripts/32_matched_ladder_report.py"]),
    ("38", "matching diagnostics, standardised mean differences (App. O, Tab. 15-16)",
     ["scripts/38_matching_balance_report.py"]),
    ("41", "cross-label acquisition support against a permutation null "
           "(Fig. 5; App. I, Tab. 12)",
     ["scripts/41_cross_label_support.py"]),
    ("42", "age scored directly on both cohorts (App. M)",
     ["scripts/42_age_direction.py"]),
    ("43", "the four diagnostics that quarantined the epoch-selecting run (App. L)",
     ["scripts/43_selection_diagnostics.py"]),
    ("37", "shortcut exploitation on the second cohort (App. N, Tab. 14)",
     ["scripts/37_verification_confounder.py"]),
    ("39", "where the era signature lives, from cached frame descriptors (App. Q, Tab. 17)",
     ["scripts/39_signature_source.py"]),
    ("40a", "era legibility under fine-tuning, ConvNeXt-T (App. R, Tab. 18)",
     ["scripts/40_finetuned_era_probe.py",
      "--runs", "results/experiment_dmr_15_features_convnext_finetune",
      "--reference", "results/experiment_dmr_15_features_convnext_frozen",
      "--output", "results/confounders/finetuned_era_probe_convnext.csv"]),
    ("40b", "era legibility under fine-tuning, Swin-T (App. R, Tab. 18)",
     ["scripts/40_finetuned_era_probe.py",
      "--runs", "results/experiment_dmr_15_features_swin_finetune",
      "--reference", "results/experiment_dmr_15_features_swin_frozen",
      "--output", "results/confounders/finetuned_era_probe_swin.csv"]),
    ("31", "the confounder-control figures (Fig. 3 and Fig. 4)",
     ["scripts/31_figure_confounder_control.py"]),
    ("33", "the remaining manuscript figures (Fig. 1, Fig. 2)",
     ["scripts/33_paper_figures.py"]),
    ("28", "prints the manuscript's LaTeX tables from the files above",
     ["scripts/28_generate_dmr_tables.py", "--all"]),
]

VERIFY = ("36", "checks every number printed in the paper against results/",
          ["scripts/36_consistency_sweep.py", "--strict"])


def run(step: tuple[str, str, list[str]], log_dir: Path) -> tuple[str, bool, float]:
    step_id, what, argv = step
    log = log_dir / f"{step_id}.log"
    print(f"[{step_id}] {what}")
    start = time.time()
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call([sys.executable, *argv], cwd=ROOT,
                               stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    print(f"      {'ok' if code == 0 else f'FAILED (exit {code})'}"
          f"  {elapsed:.0f}s  log: {log.relative_to(ROOT)}")
    return step_id, code == 0, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="run only these step ids (see --list)")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the final consistency sweep")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "reproduce_logs")
    args = parser.parse_args()

    plan = STEPS if not args.only else [s for s in STEPS if s[0] in set(args.only)]
    if args.only and len(plan) != len(set(args.only)):
        unknown = set(args.only) - {s[0] for s in STEPS}
        raise SystemExit(f"unknown step id(s): {sorted(unknown)}")

    if args.list:
        for step_id, what, argv in [*plan, VERIFY]:
            print(f"{step_id:>4}  {what}\n      {' '.join(argv)}")
        return 0

    if plan:
        missing = [path for path in REQUIRED_MANIFESTS if not (ROOT / path).exists()]
        if missing:
            print("The per-patient manifests are not distributed, and these are missing:",
                  *(f"  {path}" for path in missing),
                  "Build them from your copy of the datasets (README Section 5) and",
                  "regenerate the predictions on a GPU (README Section 8), then rerun.",
                  sep="\n", file=sys.stderr)
            return 2
        absent = [path for path in DMR_ARMS + COHORT2_ARMS if not (ROOT / path).exists()]
        if absent:
            print(f"warning: {len(absent)} of {len(DMR_ARMS) + len(COHORT2_ARMS)} main-arm "
                  "prediction files are absent; the steps reading them will fail "
                  "(README Section 8 regenerates them).", file=sys.stderr)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    results = [run(step, args.log_dir) for step in plan]
    if not args.no_verify:
        results.append(run(VERIFY, args.log_dir))

    failed = [step_id for step_id, ok, _ in results if not ok]
    total = sum(elapsed for _, _, elapsed in results)
    print(f"\n{len(results) - len(failed)}/{len(results)} steps ok in {total / 60:.1f} min")
    if failed:
        print(f"failed: {', '.join(failed)}  (read the logs in {args.log_dir})")
        return 1
    print("Every number in the manuscript still matches the files these steps wrote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
