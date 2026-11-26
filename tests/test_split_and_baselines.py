"""Unit tests for Presentation-Based Forward Split, 2014J lock, and P1-B3 baselines."""

import numpy as np
import pandas as pd
import pytest

from oulad.baselines import (
    SCORE_COLS,
    SENSITIVE_DEMOGRAPHIC_COLS,
    evaluate_returning_vs_new,
    predict_lgbm_proba,
    train_lightgbm,
    train_logistic_regression,
)
from oulad.config import SEED, TEST_PRESENTATION
from oulad.features import split_feature_target
from oulad.split import presentation_forward_split


def _make_synthetic_feature_table(seed: int = SEED) -> pd.DataFrame:
    """Create a synthetic feature table spanning 2013B, 2013J, 2014B, and 2014J."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    presentations = [
        ("2013B", 40, 1),
        ("2013J", 40, 41),
        ("2014B", 40, 61),  # ids 61..80 returning (50%), 81..100 new (50%)
        ("2014J", 30, 1001),
    ]
    for pres, count, start_id in presentations:
        for i in range(count):
            student_id = start_id + i
            clicks = int(rng.integers(0, 120))
            sub_rate = float(rng.uniform(0.0, 1.0))
            mean_score = float(rng.uniform(25.0, 95.0)) if clicks > 10 else np.nan
            risk_logit = 1.5 - 0.03 * clicks - 2.0 * sub_rate
            target = int(risk_logit + rng.normal(0, 0.5) > 0)
            rows.append(
                {
                    "code_module": "AAA" if i % 2 == 0 else "BBB",
                    "code_presentation": pres,
                    "id_student": student_id,
                    "gender": "M" if i % 2 == 0 else "F",
                    "region": "London Region" if i % 3 == 0 else "Scotland",
                    "highest_education": "HE Qualification" if i % 2 == 0 else "A Level",
                    "imd_band": "20-30%" if i % 2 == 0 else "70-80%",
                    "age_band": "0-35",
                    "num_of_prev_attempts": int(i % 3 == 0),
                    "studied_credits": 60,
                    "disability": "Y" if i % 5 == 0 else "N",
                    "date_registration": -15,
                    "has_any_vle_click": int(clicks > 0),
                    "clicks_pre_course": 5,
                    "clicks_total": clicks,
                    "active_days": min(clicks, 14),
                    "n_unique_sites": 3,
                    "std_daily_clicks": 2.5,
                    "first_click_day": -2 if clicks > 0 else np.nan,
                    "last_click_day": 25 if clicks > 0 else np.nan,
                    "days_since_last_click": 2 if clicks > 0 else np.nan,
                    "clicks_w0": clicks // 4,
                    "clicks_w1": clicks // 4,
                    "clicks_w2": clicks // 4,
                    "clicks_w3": clicks - 3 * (clicks // 4),
                    "n_due_before_cutoff": 2,
                    "n_submitted_due": int(round(sub_rate * 2)),
                    "submission_rate_due": sub_rate,
                    "n_submitted_early": 0,
                    "n_banked": 0,
                    "mean_score": mean_score,
                    "has_score": int(not np.isnan(mean_score)),
                    "mean_days_late": -1.0,
                    "click_trend_ratio": 1.0,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def test_presentation_forward_split_locks_2014j_and_disjoint_val() -> None:
    """2014J must be strictly isolated, and val_select/val_calib must be disjoint 60/40."""
    df = _make_synthetic_feature_table()
    split = presentation_forward_split(df, val_calib_ratio=0.40, seed=SEED)

    assert set(split.train["code_presentation"]) == {"2013B", "2013J"}
    assert set(split.val_select["code_presentation"]) == {"2014B"}
    assert set(split.val_calib["code_presentation"]) == {"2014B"}
    assert set(split.test_locked["code_presentation"]) == {TEST_PRESENTATION}

    assert len(split.val_select) == 24
    assert len(split.val_calib) == 16
    assert set(split.val_select["id_student"]).isdisjoint(set(split.val_calib["id_student"]))


def test_lgbm_enforces_subsample_freq_one_and_reproducibility() -> None:
    """LightGBM must enable row bagging (subsample_freq=1) and reproduce exactly with seed=42."""
    df = _make_synthetic_feature_table()
    split = presentation_forward_split(df, seed=SEED)
    X_tr, y_tr = split_feature_target(split.train)
    X_va, y_va = split_feature_target(split.val_select)

    m1, c1 = train_lightgbm(X_tr, y_tr, X_va, y_va, seed=SEED, stopping_rounds=20)
    m2, c2 = train_lightgbm(X_tr, y_tr, X_va, y_va, seed=SEED, stopping_rounds=20)

    params = m1.get_params()
    assert params["subsample"] == 0.8
    assert params["subsample_freq"] == 1
    assert c1 == c2

    p1 = predict_lgbm_proba(m1, X_va, c1)
    p2 = predict_lgbm_proba(m2, X_va, c2)
    np.testing.assert_allclose(p1, p2, rtol=0.0, atol=0.0)


def test_unseen_category_handled_safely_in_logreg_and_lgbm() -> None:
    """Both LogReg and LightGBM must handle unseen validation categories without crashing."""
    df = _make_synthetic_feature_table()
    split = presentation_forward_split(df, seed=SEED)
    X_tr, y_tr = split_feature_target(split.train)
    X_va, y_va = split_feature_target(split.val_select)

    X_va_unseen = X_va.copy()
    X_va_unseen.loc[0, "region"] = "Unseen Mars Region"
    X_va_unseen.loc[1, "imd_band"] = np.nan

    logreg = train_logistic_regression(X_tr, y_tr, seed=SEED)
    p_lr = logreg.predict_proba(X_va_unseen)[:, 1]
    assert np.all((p_lr >= 0.0) & (p_lr <= 1.0))

    lgbm, cats = train_lightgbm(X_tr, y_tr, X_va, y_va, seed=SEED, stopping_rounds=20)
    p_lgb = predict_lgbm_proba(lgbm, X_va_unseen, cats)
    assert np.all((p_lgb >= 0.0) & (p_lgb <= 1.0))


def test_2014j_mutation_does_not_affect_training_or_val_predictions() -> None:
    """Corrupting 2014J rows must produce bit-identical models and val_select predictions."""
    df_orig = _make_synthetic_feature_table()
    df_mut = df_orig.copy()
    mask_test = df_mut["code_presentation"] == TEST_PRESENTATION
    df_mut.loc[mask_test, "clicks_total"] = 999999
    df_mut.loc[mask_test, "target"] = 1 - df_mut.loc[mask_test, "target"]

    s1 = presentation_forward_split(df_orig, seed=SEED)
    s2 = presentation_forward_split(df_mut, seed=SEED)

    X_tr1, y_tr1 = split_feature_target(s1.train)
    X_va1, y_va1 = split_feature_target(s1.val_select)
    X_tr2, y_tr2 = split_feature_target(s2.train)
    X_va2, y_va2 = split_feature_target(s2.val_select)

    lr1 = train_logistic_regression(X_tr1, y_tr1, seed=SEED)
    lr2 = train_logistic_regression(X_tr2, y_tr2, seed=SEED)
    np.testing.assert_allclose(
        lr1.predict_proba(X_va1)[:, 1], lr2.predict_proba(X_va2)[:, 1], rtol=0.0, atol=0.0
    )

    gb1, c1 = train_lightgbm(X_tr1, y_tr1, X_va1, y_va1, seed=SEED, stopping_rounds=20)
    gb2, c2 = train_lightgbm(X_tr2, y_tr2, X_va2, y_va2, seed=SEED, stopping_rounds=20)
    np.testing.assert_allclose(
        predict_lgbm_proba(gb1, X_va1, c1),
        predict_lgbm_proba(gb2, X_va2, c2),
        rtol=0.0,
        atol=0.0,
    )


def test_returning_vs_new_and_sensitivity_column_sets() -> None:
    """Returning vs New counts must sum to val_select size and sensitivity tuples must be valid."""
    df = _make_synthetic_feature_table()
    split = presentation_forward_split(df, seed=SEED)
    X_tr, y_tr = split_feature_target(split.train)
    X_va, _ = split_feature_target(split.val_select)

    lr = train_logistic_regression(X_tr, y_tr, seed=SEED)
    probs = lr.predict_proba(X_va)[:, 1]
    seg = evaluate_returning_vs_new(split.val_select, split.train, probs)

    assert seg["returning"]["n"] + seg["new"]["n"] == len(split.val_select)
    assert seg["returning"]["n"] > 0 and seg["new"]["n"] > 0
    assert set(SENSITIVE_DEMOGRAPHIC_COLS) == {"gender", "disability", "imd_band"}
    assert set(SCORE_COLS) == {"mean_score", "has_score"}

    # Ensure dropping sensitive or score columns still trains cleanly
    for drop_cols in (SENSITIVE_DEMOGRAPHIC_COLS, SCORE_COLS):
        X_tr_sub = X_tr.drop(columns=list(drop_cols))
        X_va_sub = X_va.drop(columns=list(drop_cols))
        m_sub = train_logistic_regression(X_tr_sub, y_tr, seed=SEED)
        p_sub = m_sub.predict_proba(X_va_sub)[:, 1]
        assert len(p_sub) == len(X_va_sub)
        assert not any(c in X_tr_sub.columns for c in drop_cols)


def test_invalid_val_calib_ratio_raises() -> None:
    """Invalid split ratios outside (0, 1) must fail fast."""
    df = _make_synthetic_feature_table()
    with pytest.raises(ValueError, match="val_calib_ratio"):
        presentation_forward_split(df, val_calib_ratio=1.2)
