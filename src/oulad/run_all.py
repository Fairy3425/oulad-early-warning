"""Reproducible end-to-end pipeline entrypoint for OULAD Early Warning (P1-B6).

Safe-by-default design (AUD-B6-02, AUD-B6-03, AUD-B6-05 & AUD-B6-06):
- By default (`--skip-download` or `--skip-download --skip-locked-test`):
  * Stages 1-2 ingest all raw OULAD CSVs (including 2014J) into DuckDB/Parquets for schema
    checks and row-wise feature export (top-5 VLE activities are strictly Train-only 2013B+2013J).
  * Stages 3-4b train/select models, run 5-fold OOF calibration, and lock
    `artifacts/champion_locked.joblib` strictly on 2013 + 2014B.
  * `run_p1_b5_locked_evaluation_suite` is unreachable and 2014J is NEVER used for training,
    feature selection, calibration, threshold selection, or evaluation.
- `--include-locked-test` runs a POST-HOC HISTORICAL REPRODUCTION of the already-unlocked 2014J
  evaluation. It is NOT a new blind test and must NOT be used to select or tune models while
  claiming 2014J is an independent holdout.

Usage:
    # Safe fresh-clone workflow (rebuilds DB, Parquets, Val models & champion_locked.joblib):
    python -m oulad.run_all --skip-download --skip-locked-test

    # Dry-run input & stage validation:
    python -m oulad.run_all --skip-download --dry-run

    # Explicit opt-in post-hoc historical reproduction (NOT a new blind test):
    python -m oulad.run_all --skip-download --include-locked-test
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oulad.baselines import run_p1_b3_validation_suite
from oulad.config import (
    CUTOFFS,
    DB_PATH,
    FIGURES_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_TABLES,
    REPORTS_DIR,
    SEED,
)
from oulad.db import (
    compute_student_overlap,
    get_connection,
    get_registration_and_missing_stats,
    load_raw_tables,
    run_quality_checks,
)
from oulad.evaluate import (
    fit_and_lock_phase_a_champion,
    run_p1_b5_locked_evaluation_suite,
)
from oulad.features import export_all_feature_parquets
from oulad.train_torch import ARTIFACTS_DIR, run_p1_b4_torch_suite


def verify_raw_csvs_present(raw_dir: Path) -> list[str]:
    """Return list of missing OULAD CSV filenames inside raw_dir."""
    return [f"{name}.csv" for name in RAW_TABLES if not (raw_dir / f"{name}.csv").is_file()]


def run_stage_db_and_features(
    *,
    raw_dir: Path,
    db_path: Path,
    processed_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Execute P1-B1 (DuckDB load + 9 quality checks) and P1-B2 (6 feature Parquets)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    con = get_connection(db_path)
    try:
        row_counts = load_raw_tables(raw_dir, con)
        quality_checks = run_quality_checks(con)
        reg_stats = get_registration_and_missing_stats(con, cutoffs=CUTOFFS)
        overlap = compute_student_overlap(con)

        dq_report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "OULAD (UCI ID 349, CC BY 4.0)",
            "row_counts": row_counts,
            "quality_checks": quality_checks,
            "registration_and_missing_stats": reg_stats,
            "returning_vs_new_students": overlap,
        }
        (reports_dir / "data_quality.json").write_text(
            json.dumps(dq_report, indent=2), encoding="utf-8"
        )

        cohort_summary = export_all_feature_parquets(con, out_dir=processed_dir, cutoffs=CUTOFFS)
        (reports_dir / "cohort_summary.json").write_text(
            json.dumps(cohort_summary, indent=2), encoding="utf-8"
        )
    finally:
        con.close()

    return {"data_quality": dq_report, "cohort_summary": cohort_summary}


def run_pipeline(
    *,
    skip_download: bool = False,
    skip_locked_test: bool = True,
    include_locked_test: bool = False,
    dry_run: bool = False,
    disable_mlflow: bool = False,
    prefer_cuda: bool = False,
    seed: int = SEED,
    raw_dir: Path = RAW_DIR,
    db_path: Path = DB_PATH,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Orchestrate the reproducible OULAD Early Warning pipeline (safe-by-default)."""
    run_locked_2014j = bool(include_locked_test and not skip_locked_test)

    missing_csvs = verify_raw_csvs_present(raw_dir)
    if missing_csvs:
        hint = (
            "Download OULAD CSV files (UCI ID 349, CC BY 4.0) into "
            f"{raw_dir} and re-run with --skip-download --skip-locked-test."
        )
        raise FileNotFoundError(
            f"Missing raw OULAD CSV file(s): {', '.join(missing_csvs)}. {hint}"
        )

    stages = [
        "1_db_load_and_quality_checks",
        "2_feature_engineering_operational_and_common_14_21_28",
        "3_baselines_logreg_lgbm_shap_and_sensitivity_val_select",
        "4_pytorch_tabular_mlp_val_select",
        "4b_phase_a_oof_calibration_and_champion_lock",
    ]
    if run_locked_2014j:
        stages.append("5_locked_test_2014j_evaluation")

    plan = {
        "dry_run": dry_run,
        "skip_download": skip_download,
        "skip_locked_test": not run_locked_2014j,
        "include_locked_test": run_locked_2014j,
        "seed": seed,
        "raw_dir": str(raw_dir),
        "db_path": str(db_path),
        "processed_dir": str(processed_dir),
        "reports_dir": str(reports_dir),
        "figures_dir": str(figures_dir),
        "artifacts_dir": str(artifacts_dir),
        "stages": stages,
    }
    if dry_run:
        return plan

    stage_log: list[dict[str, Any]] = []

    print("[Stage 1-2] Loading DuckDB, 9 SQL quality checks & 6 feature Parquets...", flush=True)
    stage_12 = run_stage_db_and_features(
        raw_dir=raw_dir,
        db_path=db_path,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )
    stage_log.append(
        {
            "stage": "1_2_db_and_features",
            "status": "PASS",
            "n_parquets": len(stage_12["cohort_summary"]),
            "quality_violations": sum(stage_12["data_quality"]["quality_checks"].values()),
        }
    )

    print("[Stage 3] Training LogReg, LightGBM, 12 cutoffs, SHAP & sensitivity...", flush=True)
    metrics_val, _ = run_p1_b3_validation_suite(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        cutoffs=CUTOFFS,
        seed=seed,
    )
    stage_log.append(
        {
            "stage": "3_baselines_val_select",
            "status": "PASS",
            "lightgbm_val_pr_auc": metrics_val["models_val_select_t28_operational"]["lightgbm"][
                "pr_auc"
            ],
        }
    )

    print("[Stage 4] Training PyTorch TabularMLP on val_select (2014J excluded)...", flush=True)
    metrics_val_full = run_p1_b4_torch_suite(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        artifacts_dir=artifacts_dir,
        seed=seed,
        prefer_cuda=prefer_cuda,
    )
    stage_log.append(
        {
            "stage": "4_pytorch_tabular_mlp_val_select",
            "status": "PASS",
            "tabular_mlp_val_pr_auc": metrics_val_full["models_val_select_t28_operational"][
                "tabular_mlp"
            ]["pr_auc"],
            "device_used": metrics_val_full["tabular_mlp_training_info"]["device_used"],
        }
    )

    locked_report: dict[str, Any] | None = None
    if run_locked_2014j:
        print(
            "[Stage 4b-5] Running Phase A lock + POST-HOC historical 2014J reproduction...",
            flush=True,
        )
        locked_report = run_p1_b5_locked_evaluation_suite(
            processed_dir=processed_dir,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            artifacts_dir=artifacts_dir,
            cutoffs=CUTOFFS,
            seed=seed,
            enable_mlflow=not disable_mlflow,
        )
        champ_name = locked_report["evaluation_protocol"]["champion_selected_on_val_select"]
        champ_tau = locked_report["models_test_locked_2014j_t28"][champ_name]["locked_threshold"]
        bundle_path = artifacts_dir / "champion_locked.joblib"
        stage_log.append({"stage": "5_locked_test_2014j_evaluation", "status": "PASS"})
    else:
        print(
            "[Stage 4b] Phase A OOF calibration & champion lock strictly on 2013+2014B "
            "(2014J evaluation SKIPPED)...",
            flush=True,
        )
        phase_a = fit_and_lock_phase_a_champion(
            processed_dir=processed_dir,
            artifacts_dir=artifacts_dir,
            seed=seed,
        )
        champ_name = str(phase_a["champion_name"])
        champ_tau = float(phase_a["champion_tau"])
        bundle_path = Path(phase_a["champion_bundle_path"])
        stage_log.append(
            {
                "stage": "4b_phase_a_oof_calibration_and_champion_lock",
                "status": "PASS",
                "champion_name": champ_name,
                "calibration_method": phase_a["champion_cal"].chosen_method,
                "locked_threshold": champ_tau,
                "champion_bundle_exists": bundle_path.is_file(),
                "locked_test_2014j_evaluated": False,
            }
        )

    return {
        "status": "completed",
        "seed": seed,
        "stages_executed": stages,
        "stage_log": stage_log,
        "champion_name": champ_name,
        "locked_threshold": champ_tau,
        "champion_bundle_path": str(bundle_path),
        "champion_bundle_exists": bundle_path.is_file(),
        "locked_test_executed": locked_report is not None,
        "locked_test_2014j_evaluated": locked_report is not None,
        "data_boundary_note": (
            "Stages 1-2 ingest raw OULAD CSVs (incl. 2014J) into DuckDB/Parquets for schema "
            "checks & row-wise feature export (top-5 VLE activities are Train-only 2013B+2013J). "
            "Stages 3-4b strictly exclude 2014J from training, feature selection, calibration, "
            "threshold selection, and evaluation."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for python -m oulad.run_all."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible OULAD Early Warning pipeline. Safe by default: skips locked "
            "2014J evaluation unless --include-locked-test is explicitly passed."
        )
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing OULAD CSVs in --raw-dir without attempting network download.",
    )
    parser.add_argument(
        "--skip-locked-test",
        action="store_true",
        help=(
            "Skip locked 2014J evaluation (safe default). Still runs Phase A OOF calibration "
            "on 2014B val_calib to generate artifacts/champion_locked.joblib for predict --sample."
        ),
    )
    parser.add_argument(
        "--include-locked-test",
        action="store_true",
        help=(
            "Explicitly opt in to a post-hoc historical reproduction of the already-unlocked "
            "2014J evaluation (Stage 5). This is NOT a new blind test; its output must NOT be "
            "used to select or tune models while claiming 2014J is an independent holdout."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned stages without executing or modifying files.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Skip optional local MLflow SQLite logging.",
    )
    parser.add_argument(
        "--prefer-cuda",
        action="store_true",
        help="Allow CUDA for PyTorch TabularMLP if available (defaults to deterministic CPU).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Global random seed (default: 42).")
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help="Directory with 7 OULAD CSVs."
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Output DuckDB file path.")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR, help="Output Parquet directory."
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR, help="Output JSON/CSV reports directory."
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR, help="Output figures directory."
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=ARTIFACTS_DIR, help="Output model artifacts dir."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    if args.skip_locked_test and args.include_locked_test:
        raise ValueError(
            "Conflicting flags: cannot pass both --skip-locked-test and --include-locked-test."
        )

    summary = run_pipeline(
        skip_download=args.skip_download,
        skip_locked_test=not args.include_locked_test,
        include_locked_test=args.include_locked_test,
        dry_run=args.dry_run,
        disable_mlflow=args.disable_mlflow,
        prefer_cuda=args.prefer_cuda,
        seed=args.seed,
        raw_dir=args.raw_dir,
        db_path=args.db_path,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(json.dumps(summary, indent=2), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
