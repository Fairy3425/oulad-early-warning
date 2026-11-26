"""Contract unit tests for OOF calibration guards, threshold fallback, FP/FN & fairness (P1-B5)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from oulad.baselines import train_logistic_regression
from oulad.config import SEED, TEST_PRESENTATION
from oulad.evaluate import (
    build_illustrative_cost_benefit_table,
    compute_ece,
    compute_subgroup_fairness,
    extract_post_lock_error_samples,
    select_best_calibrator_oof,
    select_threshold_with_fallback,
)
from oulad.features import split_feature_target
from oulad.split import presentation_forward_split


def _fit_simple_binary_model(n_rows: int, seed: int = SEED) -> tuple[
    LogisticRegression, pd.DataFrame, pd.Series
]:
    """Fit a simple deterministic binary LogisticRegression on synthetic Gaussian features."""
    rng = np.random.default_rng(seed)
    X_tr = pd.DataFrame(rng.normal(size=(400, 3)), columns=["f1", "f2", "f3"])
    y_tr = pd.Series(
        (1.4 * X_tr["f1"] - 0.8 * X_tr["f2"] + rng.normal(0, 0.7, 400) > 0).astype(int)
    )

    model = LogisticRegression(random_state=seed)
    model.fit(X_tr, y_tr)

    X_cal = pd.DataFrame(rng.normal(size=(n_rows, 3)), columns=["f1", "f2", "f3"])
    y_cal = pd.Series(
        (1.4 * X_cal["f1"] - 0.8 * X_cal["f2"] + rng.normal(0, 0.7, n_rows) > 0).astype(int)
    )
    return model, X_cal, y_cal


def test_oof_calibration_blocks_isotonic_when_n_calib_below_1000() -> None:
    """AUD-08 / FA-02: n_calib < 1000 must strictly block isotonic and run explicit 5-fold OOF."""
    model, X_cal, y_cal = _fit_simple_binary_model(n_rows=200, seed=SEED)
    res = select_best_calibrator_oof(model, X_cal, y_cal, n_splits=5, seed=SEED)

    assert res.n_calib == 200
    assert res.isotonic_blocked is True
    assert "isotonic" not in res.candidate_metrics
    assert set(res.candidate_metrics.keys()) == {"uncalibrated", "sigmoid"}
    assert res.chosen_method in {"uncalibrated", "sigmoid"}
    assert len(res.oof_probs_winner) == 200
    assert np.all((res.oof_probs_winner >= 0.0) & (res.oof_probs_winner <= 1.0))


def test_oof_calibration_enables_isotonic_when_n_calib_ge_1000() -> None:
    """When n_calib >= 1000 & min_fold_train >= 800, all 3 methods run via FrozenEstimator."""
    model, X_cal, y_cal = _fit_simple_binary_model(n_rows=1050, seed=SEED)
    res = select_best_calibrator_oof(model, X_cal, y_cal, n_splits=5, seed=SEED)

    assert res.n_calib == 1050
    assert res.min_fold_train >= 800
    assert res.isotonic_blocked is False
    assert set(res.candidate_metrics.keys()) == {"uncalibrated", "sigmoid", "isotonic"}
    assert res.chosen_method in {"uncalibrated", "sigmoid", "isotonic"}
    if res.final_calibrator is not None:
        assert len(res.final_calibrator.calibrated_classifiers_) == 1


def test_threshold_selection_transparent_fallback_when_precision_unattainable() -> None:
    """AUD-08: Unattainable Precision>=0.70 must log target_precision_met=False & max_f1."""
    rng = np.random.default_rng(SEED)
    y_true = np.zeros(200, dtype=np.int64)
    y_true[:50] = 1  # 25% positive rate, pure random scores -> precision hovers near ~0.25-0.45
    y_prob = rng.uniform(0.1, 0.9, size=200)

    res = select_threshold_with_fallback(
        y_true, y_prob, target_precision=0.70, min_predicted_positives=30
    )
    assert res.target_precision_met is False
    assert res.fallback_used == "max_f1"
    assert res.max_achievable_precision < 0.70
    assert 0.0 < res.threshold < 1.0


def test_threshold_selection_meets_target_precision_when_attainable() -> None:
    """When Precision >= 0.70 is attainable (>= 30 pred pos), pick highest-Recall threshold."""
    y_true = np.array([1] * 60 + [0] * 140, dtype=np.int64)
    y_prob = np.concatenate([np.linspace(0.65, 0.99, 60), np.linspace(0.01, 0.55, 140)])

    res = select_threshold_with_fallback(
        y_true, y_prob, target_precision=0.70, min_predicted_positives=30
    )
    assert res.target_precision_met is True
    assert res.fallback_used is None
    assert res.oof_precision >= 0.70
    assert res.oof_n_predicted_positives >= 30


def test_post_lock_error_samples_use_min_20_len_without_fabrication() -> None:
    """AUD-08: Must return exact count when FP/FN < 20 and cap strictly at 20 when > 20."""
    # Case 1: only 3 FPs and 2 FNs -> must report exact 3 and 2 (never fabricate 20)
    df_small = pd.DataFrame(
        {
            "code_module": ["AAA"] * 10,
            "code_presentation": ["2014J"] * 10,
            "id_student": list(range(1, 11)),
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    probs_small = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.85])
    audit_small = extract_post_lock_error_samples(df_small, probs_small, threshold=0.5, max_k=20)
    assert audit_small["total_fp"] == 3 and audit_small["sampled_fp_count"] == 3
    assert len(audit_small["fp_samples"]) == 3
    assert audit_small["total_fn"] == 2 and audit_small["sampled_fn_count"] == 2
    assert len(audit_small["fn_samples"]) == 2

    # Case 2: 35 FPs -> must cap at 20
    df_large = pd.DataFrame(
        {
            "code_module": ["AAA"] * 35,
            "code_presentation": ["2014J"] * 35,
            "id_student": list(range(1, 36)),
            "target": [0] * 35,
        }
    )
    probs_large = np.linspace(0.6, 0.99, 35)
    audit_large = extract_post_lock_error_samples(df_large, probs_large, threshold=0.5, max_k=20)
    assert audit_large["total_fp"] == 35 and audit_large["sampled_fp_count"] == 20
    assert len(audit_large["fp_samples"]) == 20


def test_subgroup_fairness_columns_low_sample_warning_and_missing() -> None:
    """Fairness table must include n, n_pos, low_sample_warning (n < 30), and MISSING handling."""
    df = pd.DataFrame(
        {
            "gender": ["M"] * 35 + ["F"] * 10,  # F has n=10 < 30 -> warning True; M -> False
            "disability": ["N"] * 40 + ["Y"] * 5,
            "imd_band": ["20-30%"] * 35 + [None] * 10,
            "age_band": ["0-35"] * 45,
            "target": [1, 0] * 22 + [1],
        }
    )
    probs = np.where(df["target"] == 1, 0.8, 0.2)
    fair = compute_subgroup_fairness(df, probs, threshold=0.5, min_group_n=30)

    required_cols = {
        "group_col",
        "group_value",
        "n",
        "n_pos",
        "positive_rate",
        "recall",
        "precision",
        "fpr",
        "low_sample_warning",
    }
    assert required_cols.issubset(set(fair.columns))

    row_m = fair[(fair["group_col"] == "gender") & (fair["group_value"] == "M")].iloc[0]
    row_f = fair[(fair["group_col"] == "gender") & (fair["group_value"] == "F")].iloc[0]
    assert int(row_m["n"]) == 35 and bool(row_m["low_sample_warning"]) is False
    assert int(row_f["n"]) == 10 and bool(row_f["low_sample_warning"]) is True

    missing_imd = fair[(fair["group_col"] == "imd_band") & (fair["group_value"] == "MISSING")]
    assert len(missing_imd) == 1 and int(missing_imd.iloc[0]["n"]) == 10


def test_compute_ece_boundary_cases() -> None:
    """Perfect constant calibration yields ECE=0; completely inverted predictions yield ECE=1."""
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    assert compute_ece(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0
    assert compute_ece(y, np.array([1.0, 0.0, 1.0, 0.0])) == 1.0


def test_oof_calibration_5fold_disjoint_single_prediction_and_final_refit() -> None:
    """Audit: 5 disjoint OOF folds, each row predicted once, final calibrator fit on full calib."""
    model, X_cal, y_cal = _fit_simple_binary_model(n_rows=200, seed=SEED)
    # Give X_cal unique index values to verify strict fit/eval disjointness & full coverage
    X_cal = X_cal.copy()
    X_cal.index = pd.RangeIndex(1000, 1200)
    y_cal = y_cal.copy()
    y_cal.index = X_cal.index

    fit_calls: list[set[int]] = []
    eval_calls: list[set[int]] = []
    orig_fit = CalibratedClassifierCV.fit
    orig_pred = CalibratedClassifierCV.predict_proba

    def spy_fit(self: CalibratedClassifierCV, X: pd.DataFrame, y: np.ndarray, **kw: object):
        fit_calls.append(set(X.index.tolist()))
        return orig_fit(self, X, y, **kw)

    def spy_pred(self: CalibratedClassifierCV, X: pd.DataFrame):
        eval_calls.append(set(X.index.tolist()))
        return orig_pred(self, X)

    with (
        patch.object(CalibratedClassifierCV, "fit", spy_fit),
        patch.object(CalibratedClassifierCV, "predict_proba", spy_pred),
    ):
        res = select_best_calibrator_oof(model, X_cal, y_cal, n_splits=5, seed=SEED)

    # First 5 fits are the 5 outer folds of sigmoid (isotonic blocked because n_calib=200 < 1000)
    assert len(eval_calls) == 5
    seen_eval: list[int] = []
    for fold_i in range(5):
        fit_idx = fit_calls[fold_i]
        eval_idx = eval_calls[fold_i]
        assert len(fit_idx) == 160 and len(eval_idx) == 40
        assert fit_idx.isdisjoint(eval_idx)
        seen_eval.extend(eval_idx)

    # Every val_calib row received exactly one OOF prediction
    assert len(seen_eval) == 200
    assert set(seen_eval) == set(X_cal.index.tolist())

    # If a calibrator won, final fit must happen strictly AFTER OOF folds on all 200 val_calib rows
    if res.chosen_method != "uncalibrated":
        assert len(fit_calls) == 6
        assert fit_calls[-1] == set(X_cal.index.tolist())
    else:
        assert len(fit_calls) == 5


def _build_balanced_phase_a_frame(seed: int = SEED) -> pd.DataFrame:
    """Build balanced synthetic presentations so 5-fold StratifiedKFold on val_calib is clean."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    # 80 train (2013B+2013J), 100 balanced 2014B (-> 40 val_calib: 20 pos / 20 neg), 40 2014J
    specs = [("2013B", 40, 1), ("2013J", 40, 101), ("2014B", 100, 201), ("2014J", 40, 1001)]
    for pres, n_rows, start_id in specs:
        for i in range(n_rows):
            target = i % 2  # Exact 50/50 class balance per presentation
            clicks = float(rng.integers(5, 40) if target == 1 else rng.integers(50, 140))
            score = float(rng.uniform(25.0, 55.0) if target == 1 else rng.uniform(65.0, 95.0))
            rows.append(
                {
                    "code_module": "AAA" if i % 2 == 0 else "BBB",
                    "code_presentation": pres,
                    "id_student": start_id + i,
                    "gender": "M" if i % 2 == 0 else "F",
                    "clicks_total": clicks,
                    "mean_score": score,
                    "submission_rate_due": 0.2 if target == 1 else 0.9,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def test_phase_a_calibration_and_threshold_isolated_from_2014j_mutation() -> None:
    """Audit: Mutating 2014J features & labels must have ZERO impact on Phase A calibration/tau*."""
    df_orig = _build_balanced_phase_a_frame(seed=SEED)
    df_mut = df_orig.copy()
    mask_2014j = df_mut["code_presentation"] == TEST_PRESENTATION
    df_mut.loc[mask_2014j, "clicks_total"] = 999999.0
    df_mut.loc[mask_2014j, "mean_score"] = 0.0
    df_mut.loc[mask_2014j, "target"] = 1 - df_mut.loc[mask_2014j, "target"]

    def _run_phase_a(df: pd.DataFrame):
        sp = presentation_forward_split(df, seed=SEED)
        sp.assert_invariants()
        assert TEST_PRESENTATION not in set(sp.train["code_presentation"])
        assert TEST_PRESENTATION not in set(sp.val_calib["code_presentation"])
        # Base model training rows and calibration rows must be strictly disjoint presentations
        assert set(sp.train["code_presentation"]).isdisjoint(set(sp.val_calib["code_presentation"]))

        X_tr, y_tr = split_feature_target(sp.train)
        X_vc, y_vc = split_feature_target(sp.val_calib)
        model = train_logistic_regression(X_tr, y_tr, seed=SEED)
        cal = select_best_calibrator_oof(model, X_vc, y_vc, n_splits=5, seed=SEED)
        th = select_threshold_with_fallback(
            y_vc, cal.oof_probs_winner, target_precision=0.70, min_predicted_positives=5
        )
        return cal, th

    cal_a, th_a = _run_phase_a(df_orig)
    cal_b, th_b = _run_phase_a(df_mut)

    assert cal_a.chosen_method == cal_b.chosen_method
    assert cal_a.candidate_metrics == cal_b.candidate_metrics
    np.testing.assert_allclose(cal_a.oof_probs_winner, cal_b.oof_probs_winner, rtol=0, atol=0)
    assert th_a.threshold == th_b.threshold
    assert th_a.fallback_used == th_b.fallback_used


def _make_fake_skf_cls(bad_folds: list[tuple[np.ndarray, np.ndarray]]):
    """Create a stub StratifiedKFold class yielding bad_folds for outer OOF guard tests."""

    class _FakeSKF:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def split(self, X: object, y: object):
            return iter(bad_folds)

    return _FakeSKF


def test_oof_runtime_error_on_wrong_fold_count() -> None:
    """AUD-B6-01: Must raise RuntimeError (survives python -O) if fold count != n_splits."""
    model, X_cal, y_cal = _fit_simple_binary_model(n_rows=100, seed=SEED)
    bad_folds = [
        (np.arange(0, 75), np.arange(75, 100)),
        (np.arange(25, 100), np.arange(0, 25)),
    ]
    with patch("oulad.evaluate.StratifiedKFold", _make_fake_skf_cls(bad_folds)):
        with pytest.raises(RuntimeError, match="Expected 5 calibration folds"):
            select_best_calibrator_oof(model, X_cal, y_cal, n_splits=5, seed=SEED)


def test_oof_runtime_error_on_overlapping_fit_eval_folds() -> None:
    """AUD-B6-01: Must raise RuntimeError (survives python -O) if fit & eval indices overlap."""
    model, X_cal, y_cal = _fit_simple_binary_model(n_rows=100, seed=SEED)
    bad_folds = [
        (np.arange(0, 85), np.arange(80, 100)),  # 80..84 overlap!
        (np.arange(20, 100), np.arange(0, 20)),
        (np.concatenate([np.arange(0, 20), np.arange(40, 100)]), np.arange(20, 40)),
        (np.concatenate([np.arange(0, 40), np.arange(60, 100)]), np.arange(40, 60)),
        (np.concatenate([np.arange(0, 60), np.arange(80, 100)]), np.arange(60, 80)),
    ]
    with patch("oulad.evaluate.StratifiedKFold", _make_fake_skf_cls(bad_folds)):
        with pytest.raises(RuntimeError, match="overlap"):
            select_best_calibrator_oof(model, X_cal, y_cal, n_splits=5, seed=SEED)


def test_oof_runtime_error_on_missing_or_duplicate_oof_prediction() -> None:
    """AUD-B6-01: Must raise RuntimeError (survives python -O) if any row != 1 OOF prediction."""
    model, X_cal, y_cal = _fit_simple_binary_model(n_rows=100, seed=SEED)
    # Disjoint per fold, but index 0..19 predicted twice and 80..99 never predicted
    bad_folds = [
        (np.arange(20, 100), np.arange(0, 20)),
        (np.arange(20, 100), np.arange(0, 20)),
        (np.concatenate([np.arange(0, 20), np.arange(40, 100)]), np.arange(20, 40)),
        (np.concatenate([np.arange(0, 40), np.arange(60, 100)]), np.arange(40, 60)),
        (np.concatenate([np.arange(0, 60), np.arange(80, 100)]), np.arange(60, 80)),
    ]
    with patch("oulad.evaluate.StratifiedKFold", _make_fake_skf_cls(bad_folds)):
        with pytest.raises(RuntimeError, match="exactly one OOF probability"):
            select_best_calibrator_oof(model, X_cal, y_cal, n_splits=5, seed=SEED)


def test_illustrative_cost_benefit_uses_cost_ratio_tau_and_non_bayes_disclaimer() -> None:
    """AUD-B5-01: Must use cost_ratio_tau label and avoid claiming Bayes optimality for raw."""
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    p = np.array([0.1, 0.3, 0.2, 0.8], dtype=np.float64)
    rows = build_illustrative_cost_benefit_table(y, p, locked_threshold=0.4279)

    policies = {r["policy"] for r in rows}
    assert policies == {"locked_tau", "cost_ratio_tau"}
    assert "bayes_cost_tau" not in policies
    for r in rows:
        assert "not bayes-optimal for uncalibrated scores" in r["disclaimer"].lower()
        assert "illustrative cost assumption only" in r["disclaimer"].lower()
