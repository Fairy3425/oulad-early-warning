"""Reproducible end-to-end pipeline entrypoint for OULAD Early Warning (P1-B6).

Usage:
    python -m oulad.run_all --skip-download
    python -m oulad.run_all --skip-download --dry-run
    python -m oulad.run_all --skip-download --skip-locked-test

Stages:
    1. DuckDB load (nullstr='?') + 9 SQL quality checks + Returning/New audit -> data_quality.json
    2. Start-of-Day Operational & Safe Common feature Parquets (14/21/28) -> cohort_summary.json
    3. Classical baselines (LogReg, LightGBM), 12 cutoff models, SHAP & sensitivity -> metrics_val
    4. PyTorch TabularMLP (Unknown=0, unweighted BCE) -> tabular_mlp_best.pt & learning_curve.png
    5. Explicit 5-fold OOF calibration, threshold lock & single 2014J unlock -> metrics_test_locked
"""

import argparse
import json
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
from oulad.evaluate import run_p1_b5_locked_evaluation_suite
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
    skip_locked_test: bool = False,
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
    """Orchestrate the full reproducible OULAD Early Warning pipeline."""
    missing_csvs = verify_raw_csvs_present(raw_dir)
    if missing_csvs:
        hint = (
            "Download OULAD CSV files (UCI ID 349, CC BY 4.0) into "
            f"{raw_dir} and re-run with --skip-download."
        )
        raise FileNotFoundError(
            f"Missing raw OULAD CSV file(s): {', '.join(missing_csvs)}. {hint}"
        )

    plan = {
        "dry_run": dry_run,
        "skip_download": skip_download,
        "skip_locked_test": skip_locked_test,
        "seed": seed,
        "raw_dir": str(raw_dir),
        "db_path": str(db_path),
        "processed_dir": str(processed_dir),
        "reports_dir": str(reports_dir),
        "artifacts_dir": str(artifacts_dir),
        "stages": [
            "1_db_load_and_quality_checks",
            "2_feature_engineering_operational_and_common_14_21_28",
            "3_baselines_logreg_lgbm_shap_and_sensitivity_val_select",
            "4_pytorch_tabular_mlp_val_select",
            *([] if skip_locked_test else ["5_oof_calibration_lock_and_2014j_eval"]),
        ],
    }
    if dry_run:
        return plan

    stage_12 = run_stage_db_and_features(
        raw_dir=raw_dir,
        db_path=db_path,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )
    metrics_val, _ = run_p1_b3_validation_suite(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        cutoffs=CUTOFFS,
        seed=seed,
    )
    metrics_val_full = run_p1_b4_torch_suite(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        artifacts_dir=artifacts_dir,
        seed=seed,
        prefer_cuda=prefer_cuda,
    )

    locked_report: dict[str, Any] | None = None
    if not skip_locked_test:
        locked_report = run_p1_b5_locked_evaluation_suite(
            processed_dir=processed_dir,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            artifacts_dir=artifacts_dir,
            cutoffs=CUTOFFS,
            seed=seed,
            enable_mlflow=not disable_mlflow,
        )

    return {
        "status": "completed",
        "seed": seed,
        "stages_executed": plan["stages"],
        "n_parquets": len(stage_12["cohort_summary"]),
        "val_select_models": list(metrics_val_full.get("models_val_select_t28_operational", {})),
        "val_select_lightgbm_pr_auc": metrics_val["models_val_select_t28_operational"]["lightgbm"][
            "pr_auc"
        ],
        "locked_test_executed": locked_report is not None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for python -m oulad.run_all."""
    parser = argparse.ArgumentParser(
        description="Run the reproducible OULAD Early Warning pipeline (P1-B1..B5)."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing OULAD CSVs in --raw-dir without attempting network download.",
    )
    parser.add_argument(
        "--skip-locked-test",
        action="store_true",
        help="Stop after validation modeling (P1-B1..B4) without unlocking 2014J.",
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
    summary = run_pipeline(
        skip_download=args.skip_download,
        skip_locked_test=args.skip_locked_test,
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
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
