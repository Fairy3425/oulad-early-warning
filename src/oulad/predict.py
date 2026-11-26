"""CLI inference & non-causal SHAP explanation demo for locked champion model (P1-B6).

Usage:
    python -m oulad.predict --sample
    python -m oulad.predict --sample --json

Loads the locked champion bundle (artifacts/champion_locked.joblib, locked before 2014J unlock)
and scores self-contained illustrative Day-28 student profiles (never touches 2014J test data).
"""

import argparse
import hashlib
import hmac
import json
import math
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from oulad.baselines import infer_feature_types, prepare_lgbm_frame
from oulad.evaluate import LGBMSklearnWrapper, apply_calibrator
from oulad.train_torch import ARTIFACTS_DIR

DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "champion_locked.joblib"

SHAP_DISCLAIMER = (
    "SHAP values show local predictive attribution (log-odds direction for this model), "
    "NOT causal factors of student withdrawal or failure."
)


def build_illustrative_sample_profiles(feature_columns: list[str]) -> pd.DataFrame:
    """Build two self-contained illustrative Day-28 student profiles matching feature_columns.

    These are synthetic demo records (never read from locked 2014J test data):
      - demo_high_risk: missed early due assessment, low VLE clicks, long inactivity gap.
      - demo_low_risk: submitted due assessment on time with strong score and steady engagement.
    """
    high_risk: dict[str, Any] = {
        "code_module": "DDD",
        "gender": "M",
        "region": "London Region",
        "highest_education": "Lower Than A Level",
        "imd_band": "10-20",
        "age_band": "0-35",
        "num_of_prev_attempts": 1,
        "studied_credits": 120,
        "disability": "N",
        "date_registration": -25,
        "has_any_vle_click": 1,
        "clicks_pre_course": 4.0,
        "clicks_total": 28.0,
        "active_days": 3,
        "n_unique_sites": 5,
        "std_daily_clicks": 2.1,
        "first_click_day": -5,
        "last_click_day": 9,
        "days_since_last_click": 19,
        "clicks_w0": 16.0,
        "clicks_w1": 8.0,
        "clicks_w2": 0.0,
        "clicks_w3": 0.0,
        "clicks_act_forumng": 4.0,
        "clicks_act_homepage": 12.0,
        "clicks_act_oucontent": 8.0,
        "clicks_act_subpage": 4.0,
        "clicks_act_quiz": 0.0,
        "n_due_before_cutoff": 1,
        "n_submitted_due": 0,
        "submission_rate_due": 0.0,
        "n_submitted_early": 0,
        "n_banked": 0,
        "mean_score": np.nan,
        "has_score": 0,
        "mean_days_late": np.nan,
        "click_trend_ratio": 1.0 / 25.0,
    }
    low_risk: dict[str, Any] = {
        "code_module": "BBB",
        "gender": "F",
        "region": "South Region",
        "highest_education": "HE Qualification",
        "imd_band": "70-80%",
        "age_band": "35-55",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
        "date_registration": -60,
        "has_any_vle_click": 1,
        "clicks_pre_course": 45.0,
        "clicks_total": 420.0,
        "active_days": 22,
        "n_unique_sites": 34,
        "std_daily_clicks": 9.5,
        "first_click_day": -12,
        "last_click_day": 27,
        "days_since_last_click": 1,
        "clicks_w0": 90.0,
        "clicks_w1": 95.0,
        "clicks_w2": 90.0,
        "clicks_w3": 100.0,
        "clicks_act_forumng": 110.0,
        "clicks_act_homepage": 85.0,
        "clicks_act_oucontent": 140.0,
        "clicks_act_subpage": 55.0,
        "clicks_act_quiz": 30.0,
        "n_due_before_cutoff": 1,
        "n_submitted_due": 1,
        "submission_rate_due": 1.0,
        "n_submitted_early": 1,
        "n_banked": 0,
        "mean_score": 84.0,
        "has_score": 1,
        "mean_days_late": -2.0,
        "click_trend_ratio": 191.0 / 186.0,
    }

    rows = []
    for profile in (high_risk, low_risk):
        aligned = {col: profile.get(col, np.nan) for col in feature_columns}
        rows.append(aligned)
    df = pd.DataFrame(rows, columns=feature_columns)
    df.index = pd.Index(["demo_high_risk", "demo_low_risk"], name="sample_id")
    return df


def _extract_top_shap_drivers(
    base_model: Any,
    X_samples: pd.DataFrame,
    top_k: int = 5,
) -> list[list[dict[str, Any]]]:
    """Compute top-k non-causal SHAP drivers per row when base_model wraps LightGBM."""
    if not isinstance(base_model, LGBMSklearnWrapper):
        return [[] for _ in range(len(X_samples))]

    _, cat_cols = infer_feature_types(X_samples)
    X_prep, _ = prepare_lgbm_frame(
        X_samples, cat_cols, cat_categories=base_model.cat_categories
    )
    explainer = shap.TreeExplainer(base_model.model)
    raw_sv = explainer.shap_values(X_prep)
    if isinstance(raw_sv, list):
        sv = np.asarray(raw_sv[1], dtype=np.float64)
    elif np.asarray(raw_sv).ndim == 3:
        sv = np.asarray(raw_sv, dtype=np.float64)[:, :, 1]
    else:
        sv = np.asarray(raw_sv, dtype=np.float64)

    cols = list(X_prep.columns)
    all_drivers: list[list[dict[str, Any]]] = []
    for row_idx in range(len(X_samples)):
        row_sv = sv[row_idx]
        order = np.argsort(np.abs(row_sv))[::-1][:top_k]
        drivers: list[dict[str, Any]] = []
        for col_idx in order:
            col_name = cols[int(col_idx)]
            val = X_samples.iloc[row_idx][col_name]
            clean_val = None if pd.isna(val) else (
                round(float(val), 4) if isinstance(val, (float, np.floating)) else val
            )
            shap_val = float(row_sv[int(col_idx)])
            drivers.append(
                {
                    "feature": col_name,
                    "feature_value": clean_val,
                    "shap_log_odds": round(shap_val, 4),
                    "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                }
            )
        all_drivers.append(drivers)
    return all_drivers


def _canonical_component_bytes(obj: Any) -> bytes:
    """Serialize a model or calibrator component deterministically for artifact fingerprinting."""
    if obj is None:
        return b"NONE"
    if isinstance(obj, LGBMSklearnWrapper):
        payload = {
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "booster_model": obj.model.booster_.model_to_string(),
            "cat_categories": {
                str(k): list(v) for k, v in sorted(obj.cat_categories.items())
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hasattr(obj, "booster_") and hasattr(obj.booster_, "model_to_string"):
        payload = {
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "booster_model": obj.booster_.model_to_string(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hasattr(obj, "model") and hasattr(obj, "preprocessor") and hasattr(obj.model, "state_dict"):
        hasher = hashlib.sha256()
        hasher.update(f"{type(obj).__module__}.{type(obj).__qualname__}".encode("utf-8"))
        for key, tensor in sorted(obj.model.state_dict().items()):
            arr = tensor.detach().cpu().numpy()
            hasher.update(key.encode("utf-8"))
            hasher.update(str(arr.shape).encode("utf-8"))
            hasher.update(str(arr.dtype).encode("utf-8"))
            hasher.update(arr.tobytes())
        if hasattr(obj.preprocessor, "to_dict"):
            hasher.update(
                json.dumps(
                    obj.preprocessor.to_dict(), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        return hasher.digest()
    return pickle.dumps(obj, protocol=4)


def compute_bundle_artifact_fingerprint(bundle: dict[str, Any]) -> str:
    """Compute a stable SHA-256 fingerprint binding metadata to the exact inference artifact.

    Covers base_model state, calibrator state, ordered feature_columns, champion_name,
    calibration_method, cutoff, cohort_mode, and threshold. Deliberately excludes
    evaluation_metadata to avoid any circular dependency.
    """
    hasher = hashlib.sha256()
    config_payload = {
        "champion_name": str(bundle["champion_name"]),
        "calibration_method": str(bundle["calibration_method"]),
        "cutoff": int(bundle["cutoff"]),
        "cohort_mode": str(bundle["cohort_mode"]),
        "threshold": repr(float(bundle["threshold"])),
        "feature_columns": [str(c) for c in bundle["feature_columns"]],
    }
    hasher.update(b"CONFIG:")
    hasher.update(json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(b"|BASE_MODEL:")
    hasher.update(_canonical_component_bytes(bundle["base_model"]))
    hasher.update(b"|CALIBRATOR:")
    hasher.update(_canonical_component_bytes(bundle["calibrator"]))
    return hasher.hexdigest()


def _is_valid_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def _resolve_verified_ece_metadata(
    bundle: dict[str, Any],
    *,
    expected_fingerprint: str,
    champion_name: str,
    cal_method: str,
    cutoff: int,
    cohort_mode: str,
    threshold: float,
    feature_columns: list[str],
) -> tuple[float | None, str]:
    """Return ECE metadata ONLY when bound by an exact SHA-256 artifact fingerprint."""
    meta = bundle.get("evaluation_metadata")
    if isinstance(meta, dict):
        meta_fp = meta.get("artifact_fingerprint")
        meta_threshold = meta.get("threshold")
        threshold_ok = (
            isinstance(meta_threshold, (int, float))
            and not isinstance(meta_threshold, bool)
            and np.isfinite(meta_threshold)
            and math.isclose(float(meta_threshold), float(threshold), rel_tol=0.0, abs_tol=1e-12)
        )
        raw_ece = meta.get("ece_10bins")
        ece_ok = (
            isinstance(raw_ece, (int, float))
            and not isinstance(raw_ece, bool)
            and np.isfinite(raw_ece)
            and 0.0 <= float(raw_ece) <= 1.0
        )
        meta_cols = meta.get("feature_columns")
        cols_ok = isinstance(meta_cols, list) and [str(c) for c in meta_cols] == feature_columns

        matches_fields = (
            meta.get("model_name") == champion_name
            and meta.get("calibration_method") == cal_method
            and meta.get("cutoff") == cutoff
            and meta.get("cohort_mode") == cohort_mode
            and threshold_ok
            and cols_ok
        )
        fingerprint_ok = _is_valid_sha256_hex(meta_fp) and hmac.compare_digest(
            meta_fp.lower(), expected_fingerprint.lower()
        )
        if fingerprint_ok and matches_fields and ece_ok:
            ece_val = round(float(raw_ece), 4)
            dataset_label = str(meta.get("eval_dataset", "evaluation split"))
            return (
                ece_val,
                f"Artifact-verified bundle metadata ({dataset_label}, "
                f"sha256={expected_fingerprint[:12]}): calibration_method={cal_method}, "
                f"ECE_10bins={ece_val:.4f} (not claimed perfectly calibrated).",
            )

    if cal_method == "uncalibrated":
        return (
            None,
            "No artifact-verified evaluation ECE metadata attached to this bundle "
            "(scores are raw uncalibrated model outputs; calibration error is not estimated "
            "from inference samples).",
        )
    return (
        None,
        f"Post-hoc {cal_method} calibrator attached; no artifact-verified evaluation ECE "
        "metadata embedded in this bundle.",
    )


def predict_from_bundle(
    X_samples: pd.DataFrame,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    *,
    top_k_shap: int = 5,
) -> dict[str, Any]:
    """Score feature rows using a locked champion bundle with strict schema validation.

    Schema policy (F-04):
    - Missing required feature columns raise ValueError immediately (never silently filled).
    - Extra columns not in bundle['feature_columns'] are explicitly ignored and reported under
      'ignored_extra_columns'.
    """
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Locked champion bundle not found at {bundle_path}. "
            "Run `python -m oulad.run_all --skip-download --skip-locked-test` first to build it "
            "safely on 2013 + 2014B without re-running locked test 2014J."
        )

    bundle: dict[str, Any] = joblib.load(bundle_path)
    feature_columns: list[str] = list(bundle["feature_columns"])
    required_set = set(feature_columns)

    missing_cols = [c for c in feature_columns if c not in X_samples.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature column(s) ({len(missing_cols)}): "
            f"{', '.join(missing_cols)}"
        )
    extra_cols = [str(c) for c in X_samples.columns if c not in required_set]
    X_aligned = X_samples.loc[:, feature_columns].copy()

    base_model = bundle["base_model"]
    calibrator = bundle["calibrator"]
    threshold = float(bundle["threshold"])
    cal_method = str(bundle["calibration_method"])
    champion_name = str(bundle["champion_name"])
    cutoff = int(bundle["cutoff"])
    cohort_mode = str(bundle["cohort_mode"])

    artifact_fingerprint = compute_bundle_artifact_fingerprint(bundle)
    reported_ece, calibration_note = _resolve_verified_ece_metadata(
        bundle,
        expected_fingerprint=artifact_fingerprint,
        champion_name=champion_name,
        cal_method=cal_method,
        cutoff=cutoff,
        cohort_mode=cohort_mode,
        threshold=threshold,
        feature_columns=feature_columns,
    )

    probs = apply_calibrator(base_model, calibrator, X_aligned)
    shap_drivers = _extract_top_shap_drivers(base_model, X_aligned, top_k=top_k_shap)

    predictions: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(X_aligned.index.tolist()):
        prob = float(probs[idx])
        alert = bool(prob >= threshold)
        predictions.append(
            {
                "sample_id": str(sample_id),
                "risk_score": round(prob, 4),
                "score_type": (
                    "raw_uncalibrated_model_score"
                    if cal_method == "uncalibrated"
                    else f"{cal_method}_calibrated_probability"
                ),
                "locked_threshold": round(threshold, 4),
                "decision": "AT_RISK_ALERT" if alert else "NO_ALERT",
                "top_shap_drivers": shap_drivers[idx],
            }
        )

    return {
        "champion_model": champion_name,
        "cutoff_day": cutoff,
        "cohort_mode": cohort_mode,
        "calibration_method": cal_method,
        "locked_threshold": round(threshold, 4),
        "artifact_fingerprint": artifact_fingerprint,
        "reported_ece_10bins": reported_ece,
        "calibration_note": calibration_note,
        "ignored_extra_columns": extra_cols,
        "shap_disclaimer": SHAP_DISCLAIMER,
        "predictions": predictions,
    }


def _format_human_readable(result: dict[str, Any]) -> str:
    lines = [
        "=== OULAD Early Warning — Locked Champion Sample Prediction Demo ===",
        f"Champion Model    : {result['champion_model']} "
        f"(T={result['cutoff_day']} {result['cohort_mode']})",
        f"Calibration Method: {result['calibration_method']} | "
        f"Locked Threshold tau*: {result['locked_threshold']:.4f}",
        f"Calibration Note  : {result['calibration_note']}",
        f"SHAP Disclaimer   : {result['shap_disclaimer']}",
    ]
    if result.get("ignored_extra_columns"):
        lines.append(
            f"Ignored Extra Cols: {', '.join(result['ignored_extra_columns'])}"
        )
    lines.append("")
    for pred in result["predictions"]:
        lines.append(
            f"[{pred['sample_id']}] risk_score={pred['risk_score']:.4f} "
            f"({pred['score_type']}) vs tau*={pred['locked_threshold']:.4f} "
            f"-> {pred['decision']}"
        )
        lines.append("  Top 5 SHAP predictive drivers (non-causal):")
        for rank, drv in enumerate(pred["top_shap_drivers"], start=1):
            sign = "+" if drv["shap_log_odds"] >= 0 else ""
            lines.append(
                f"    {rank}. {drv['feature']} = {drv['feature_value']} "
                f"({sign}{drv['shap_log_odds']:.4f} log-odds, {drv['direction']})"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for python -m oulad.predict."""
    parser = argparse.ArgumentParser(
        description="Run locked champion early-warning inference with non-causal SHAP drivers."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Score two self-contained illustrative Day-28 student profiles (never touches 2014J).",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE_PATH,
        help="Path to locked champion joblib bundle.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    if not args.sample:
        raise SystemExit("Pass --sample to run the illustrative prediction demo.")

    if not args.bundle.is_file():
        raise FileNotFoundError(
            f"Locked champion bundle not found at {args.bundle}. "
            "Run `python -m oulad.run_all --skip-download --skip-locked-test` first to build it "
            "safely on 2013 + 2014B without re-running locked test 2014J."
        )
    bundle_meta: dict[str, Any] = joblib.load(args.bundle)
    X_sample = build_illustrative_sample_profiles(list(bundle_meta["feature_columns"]))
    result = predict_from_bundle(X_sample, bundle_path=args.bundle, top_k_shap=5)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_human_readable(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
