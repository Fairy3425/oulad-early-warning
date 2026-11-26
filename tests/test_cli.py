"""Contract tests for P1-B6 CLI entrypoints (run_all.py & predict.py) without touching 2014J."""

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import pytest

from oulad.baselines import train_lightgbm
from oulad.config import RAW_TABLES, SEED
from oulad.evaluate import LGBMSklearnWrapper
from oulad.features import split_feature_target
from oulad.predict import build_illustrative_sample_profiles, predict_from_bundle
from oulad.predict import main as predict_main
from oulad.run_all import build_arg_parser, run_stage_db_and_features
from oulad.run_all import main as run_all_main
from oulad.split import presentation_forward_split


def _export_synthetic_csvs(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Export all 7 OULAD tables from a synthetic DuckDB connection into CSV files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for table in RAW_TABLES:
        df = con.execute(f"SELECT * FROM {table}").df()
        df.to_csv(raw_dir / f"{table}.csv", index=False, na_rep="?")


def test_predict_sample_bundle_and_cli_with_top5_shap(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """predict --sample must return bounded score, consistent alert, and 5 non-causal SHAP rows."""
    split = presentation_forward_split(synthetic_feature_table, seed=SEED)
    X_tr, y_tr = split_feature_target(split.train)
    X_vs, y_vs = split_feature_target(split.val_select)

    lgbm, cats = train_lightgbm(X_tr, y_tr, X_vs, y_vs, seed=SEED)
    bundle_path = tmp_path / "champion_locked.joblib"
    joblib.dump(
        {
            "champion_name": "lightgbm",
            "base_model": LGBMSklearnWrapper(lgbm, cats),
            "calibrator": None,
            "calibration_method": "uncalibrated",
            "threshold": 0.4279,
            "feature_columns": list(X_tr.columns),
            "cutoff": 28,
            "cohort_mode": "operational",
        },
        bundle_path,
    )

    X_demo = build_illustrative_sample_profiles(list(X_tr.columns))
    res = predict_from_bundle(X_demo, bundle_path=bundle_path, top_k_shap=5)
    assert res["champion_model"] == "lightgbm"
    assert res["calibration_method"] == "uncalibrated"
    assert "not causal" in res["shap_disclaimer"].lower()
    assert len(res["predictions"]) == 2
    for pred in res["predictions"]:
        assert 0.0 <= pred["risk_score"] <= 1.0
        expected = "AT_RISK_ALERT" if pred["risk_score"] >= 0.4279 else "NO_ALERT"
        assert pred["decision"] == expected
        assert len(pred["top_shap_drivers"]) == 5

    exit_code = predict_main(["--sample", "--bundle", str(bundle_path), "--json"])
    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["predictions"]) == 2


def test_run_all_dry_run_and_synthetic_stage12_isolated(
    synthetic_db: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run_all --dry-run validates inputs cleanly; Stage 1-2 runs end-to-end in tmp_path."""
    raw_dir = tmp_path / "raw"
    _export_synthetic_csvs(synthetic_db, raw_dir)

    exit_code = run_all_main(
        [
            "--skip-download",
            "--dry-run",
            "--raw-dir",
            str(raw_dir),
            "--db-path",
            str(tmp_path / "test.duckdb"),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["skip_locked_test"] is True
    assert plan["include_locked_test"] is False
    assert "4b_phase_a_oof_calibration_and_champion_lock" in plan["stages"]
    assert "5_locked_test_2014j_evaluation" not in plan["stages"]
    assert not (tmp_path / "test.duckdb").exists()

    # Execute isolated Stage 1-2 on synthetic CSVs in tmp_path (never touches real data/ or 2014J)
    out = run_stage_db_and_features(
        raw_dir=raw_dir,
        db_path=tmp_path / "test.duckdb",
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )
    assert (tmp_path / "reports" / "data_quality.json").is_file()
    assert (tmp_path / "reports" / "cohort_summary.json").is_file()
    assert len(out["cohort_summary"]) == 6
    assert all(v == 0 for v in out["data_quality"]["quality_checks"].values())


def test_predict_missing_bundle_in_clean_workspace_gives_safe_instructions(
    tmp_path: Path,
) -> None:
    """AUD-B6-03: Missing gitignored artifact must raise clear error pointing to safe command."""
    missing_bundle = tmp_path / "artifacts" / "champion_locked.joblib"
    with pytest.raises(FileNotFoundError) as exc_info:
        predict_main(["--sample", "--bundle", str(missing_bundle)])
    msg = str(exc_info.value)
    assert "--skip-locked-test" in msg
    assert "without re-running locked test 2014J" in msg


def test_run_all_rejects_conflicting_locked_test_flags(tmp_path: Path) -> None:
    """AUD-B6-02: Conflicting --skip-locked-test and --include-locked-test must fail loudly."""
    with pytest.raises(ValueError, match="Conflicting flags"):
        run_all_main(
            [
                "--skip-download",
                "--skip-locked-test",
                "--include-locked-test",
                "--raw-dir",
                str(tmp_path),
            ]
        )


def test_run_all_help_documents_post_hoc_historical_reproduction_warning() -> None:
    """AUD-B6-06: CLI help must clearly warn --include-locked-test is not a new blind test."""
    help_text = " ".join(build_arg_parser().format_help().lower().split())
    assert "post-hoc historical reproduction" in help_text
    assert "not a new blind test" in help_text
    assert "independent holdout" in help_text
