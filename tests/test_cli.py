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
from oulad.predict import (
    build_illustrative_sample_profiles,
    compute_bundle_artifact_fingerprint,
    predict_from_bundle,
)
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


def _dump_synthetic_bundle(
    synthetic_feature_table: pd.DataFrame,
    bundle_path: Path,
    *,
    evaluation_metadata: dict[str, object] | None = None,
    seed: int = SEED,
    threshold: float = 0.4279,
    feature_columns_override: list[str] | None = None,
    attach_matching_ece: float | None = None,
) -> tuple[list[str], dict[str, object]]:
    """Train a tiny synthetic LightGBM bundle for predict.py contract tests."""
    split = presentation_forward_split(synthetic_feature_table, seed=seed)
    X_tr, y_tr = split_feature_target(split.train)
    X_vs, y_vs = split_feature_target(split.val_select)
    if feature_columns_override is not None:
        X_tr = X_tr.loc[:, feature_columns_override]
        X_vs = X_vs.loc[:, feature_columns_override]
    lgbm, cats = train_lightgbm(X_tr, y_tr, X_vs, y_vs, seed=seed)
    cols = list(X_tr.columns)
    payload: dict[str, object] = {
        "champion_name": "lightgbm",
        "base_model": LGBMSklearnWrapper(lgbm, cats),
        "calibrator": None,
        "calibration_method": "uncalibrated",
        "threshold": threshold,
        "feature_columns": cols,
        "cutoff": 28,
        "cohort_mode": "operational",
    }
    if attach_matching_ece is not None:
        fp = compute_bundle_artifact_fingerprint(payload)
        payload["evaluation_metadata"] = {
            "artifact_fingerprint": fp,
            "model_name": "lightgbm",
            "calibration_method": "uncalibrated",
            "cutoff": 28,
            "cohort_mode": "operational",
            "threshold": threshold,
            "feature_columns": cols,
            "eval_dataset": "2014J locked test",
            "ece_10bins": attach_matching_ece,
        }
    elif evaluation_metadata is not None:
        payload["evaluation_metadata"] = evaluation_metadata
    joblib.dump(payload, bundle_path)
    return cols, payload


def test_predict_rejects_missing_required_feature_columns(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """F-04: Missing required columns must raise a clear ValueError (never silently NaN-fill)."""
    bundle_path = tmp_path / "bundle.joblib"
    cols, _ = _dump_synthetic_bundle(synthetic_feature_table, bundle_path)
    X_demo = build_illustrative_sample_profiles(cols).drop(columns=["clicks_total", "mean_score"])

    with pytest.raises(ValueError, match="Missing required feature column") as exc_info:
        predict_from_bundle(X_demo, bundle_path=bundle_path)
    assert "clicks_total" in str(exc_info.value)
    assert "mean_score" in str(exc_info.value)


def test_predict_ignores_and_reports_extra_columns_deterministically(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """F-04: Extra columns are explicitly ignored, reported, and do not alter predictions."""
    bundle_path = tmp_path / "bundle.joblib"
    cols, _ = _dump_synthetic_bundle(synthetic_feature_table, bundle_path)
    X_base = build_illustrative_sample_profiles(cols)
    X_extra = X_base.copy()
    X_extra["unexpected_col_a"] = 123
    X_extra["unexpected_col_b"] = "ignored"

    res_base = predict_from_bundle(X_base, bundle_path=bundle_path)
    res_extra = predict_from_bundle(X_extra, bundle_path=bundle_path)

    assert res_base["ignored_extra_columns"] == []
    assert res_extra["ignored_extra_columns"] == ["unexpected_col_a", "unexpected_col_b"]
    assert res_base["predictions"] == res_extra["predictions"]


def test_predict_custom_bundle_without_ece_metadata_never_fabricates_ece(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """F-04: Custom bundle lacking evaluation_metadata must NOT hardcode ECE=0.0542."""
    bundle_path = tmp_path / "custom_bundle.joblib"
    cols, _ = _dump_synthetic_bundle(synthetic_feature_table, bundle_path, evaluation_metadata=None)
    X_demo = build_illustrative_sample_profiles(cols)

    res = predict_from_bundle(X_demo, bundle_path=bundle_path)
    assert res["reported_ece_10bins"] is None
    assert "0.0542" not in res["calibration_note"]
    assert "no artifact-verified evaluation ece metadata" in res["calibration_note"].lower()


def test_predict_accepts_ece_metadata_for_exact_matching_artifact(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """ECE metadata bound by matching SHA-256 artifact fingerprint & fields is accepted."""
    bundle_path = tmp_path / "exact_artifact.joblib"
    cols, payload = _dump_synthetic_bundle(
        synthetic_feature_table,
        bundle_path,
        attach_matching_ece=0.0542,
    )
    X_demo = build_illustrative_sample_profiles(cols)
    res = predict_from_bundle(X_demo, bundle_path=bundle_path)

    expected_fp = compute_bundle_artifact_fingerprint(payload)
    assert res["artifact_fingerprint"] == expected_fp
    assert res["reported_ece_10bins"] == 0.0542
    assert "0.0542" in res["calibration_note"]
    assert expected_fp[:12] in res["calibration_note"]


def test_predict_rejects_ece_metadata_copied_between_distinct_models(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Metadata copied from Model A into independently trained Model B must be rejected.

    Both models share identical champion_name, cutoff, threshold, cohort_mode,
    calibration_method, and feature_columns — only the trained model state differs.
    """
    path_a = tmp_path / "model_a.joblib"
    cols_a, payload_a = _dump_synthetic_bundle(
        synthetic_feature_table,
        path_a,
        seed=SEED,
        attach_matching_ece=0.0542,
    )
    copied_meta = dict(payload_a["evaluation_metadata"])  # type: ignore[arg-type]

    # Train an independent Model B (different seed -> distinct booster state) with identical config
    path_b = tmp_path / "model_b.joblib"
    cols_b, payload_b = _dump_synthetic_bundle(
        synthetic_feature_table,
        path_b,
        seed=SEED + 17,
        evaluation_metadata=copied_meta,
    )
    assert cols_a == cols_b
    assert compute_bundle_artifact_fingerprint(payload_a) != compute_bundle_artifact_fingerprint(
        payload_b
    )

    X_demo = build_illustrative_sample_profiles(cols_b)
    res_b = predict_from_bundle(X_demo, bundle_path=path_b)
    assert res_b["reported_ece_10bins"] is None
    assert "0.0542" not in res_b["calibration_note"]


def test_predict_rejects_ece_metadata_on_feature_order_or_threshold_change(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Changing feature column order or operating threshold invalidates copied ECE metadata."""
    base_path = tmp_path / "base_model.joblib"
    cols, payload = _dump_synthetic_bundle(
        synthetic_feature_table,
        base_path,
        seed=SEED,
        threshold=0.4279,
        attach_matching_ece=0.0542,
    )
    copied_meta = dict(payload["evaluation_metadata"])  # type: ignore[arg-type]

    # Case 1: threshold changed while keeping exact same base_model and copied metadata
    thresh_path = tmp_path / "changed_threshold.joblib"
    thresh_payload = dict(payload)
    thresh_payload["threshold"] = 0.5000
    thresh_payload["evaluation_metadata"] = copied_meta
    joblib.dump(thresh_payload, thresh_path)

    X_demo = build_illustrative_sample_profiles(cols)
    res_thresh = predict_from_bundle(X_demo, bundle_path=thresh_path)
    assert res_thresh["reported_ece_10bins"] is None
    assert "0.0542" not in res_thresh["calibration_note"]

    # Case 2: feature_columns order changed (swap first two columns) with copied metadata
    swapped_cols = [cols[1], cols[0], *cols[2:]]
    order_path = tmp_path / "changed_order.joblib"
    _dump_synthetic_bundle(
        synthetic_feature_table,
        order_path,
        seed=SEED,
        threshold=0.4279,
        feature_columns_override=swapped_cols,
        evaluation_metadata=copied_meta,
    )
    X_swapped = build_illustrative_sample_profiles(swapped_cols)
    res_order = predict_from_bundle(X_swapped, bundle_path=order_path)
    assert res_order["reported_ece_10bins"] is None
    assert "0.0542" not in res_order["calibration_note"]


@pytest.mark.parametrize(
    "bad_fp",
    [
        "0" * 64,
        "not-a-valid-sha256",
        "",
        12345,
        None,
    ],
)
def test_predict_rejects_incorrect_or_malformed_artifact_fingerprint(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
    bad_fp: object,
) -> None:
    """Incorrect or malformed artifact_fingerprint must suppress ECE display."""
    bundle_path = tmp_path / "bad_fp_bundle.joblib"
    cols, payload = _dump_synthetic_bundle(
        synthetic_feature_table,
        bundle_path,
        attach_matching_ece=0.0542,
    )
    meta = dict(payload["evaluation_metadata"])  # type: ignore[arg-type]
    meta["artifact_fingerprint"] = bad_fp
    payload["evaluation_metadata"] = meta
    joblib.dump(payload, bundle_path)

    X_demo = build_illustrative_sample_profiles(cols)
    res = predict_from_bundle(X_demo, bundle_path=bundle_path)
    assert res["reported_ece_10bins"] is None
    assert "0.0542" not in res["calibration_note"]
