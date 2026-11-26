"""Parameterized anti-leakage tests across cutoffs [14, 21, 28] and both cohort modes (P1-B2)."""

from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from oulad.features import (
    FORBIDDEN_FEATURE_COLS,
    build_features,
    get_top_activities,
    split_feature_target,
)


@pytest.mark.parametrize("cutoff", [14, 21, 28])
@pytest.mark.parametrize("mode", ["operational", "common"])
def test_future_event_invariance(
    leakage_db: duckdb.DuckDBPyConnection,
    cutoff: int,
    mode: str,
) -> None:
    """Invariant 1: deleting all VLE & assessment events >= cutoff must not change X."""
    top_before = get_top_activities(leakage_db, feature_cutoff=cutoff)
    df_orig = build_features(
        leakage_db, feature_cutoff=cutoff, mode=mode, top_activities=top_before
    )
    X_orig, _ = split_feature_target(df_orig)

    # Confirm test is non-vacuous: boundary/future events actually exist before deletion
    future_vle = leakage_db.execute(
        "SELECT COUNT(*) FROM studentVle WHERE date >= ?", [cutoff]
    ).fetchone()[0]
    future_assess = leakage_db.execute(
        "SELECT COUNT(*) FROM studentAssessment WHERE date_submitted >= ?", [cutoff]
    ).fetchone()[0]
    assert future_vle > 0 and future_assess > 0

    leakage_db.execute("DELETE FROM studentVle WHERE date >= ?", [cutoff])
    leakage_db.execute("DELETE FROM studentAssessment WHERE date_submitted >= ?", [cutoff])

    top_after = get_top_activities(leakage_db, feature_cutoff=cutoff)
    assert top_before == top_after

    df_trunc = build_features(
        leakage_db, feature_cutoff=cutoff, mode=mode, top_activities=top_after
    )
    X_trunc, _ = split_feature_target(df_trunc)

    pd.testing.assert_frame_equal(X_orig, X_trunc)


@pytest.mark.parametrize("cutoff", [14, 21, 28])
@pytest.mark.parametrize("mode", ["operational", "common"])
def test_post_outcome_invariance(
    leakage_db: duckdb.DuckDBPyConnection,
    cutoff: int,
    mode: str,
) -> None:
    """Invariant 2: flipping final_result and mutating date_unregistration >= 28 leaves X fixed."""
    df_orig = build_features(leakage_db, feature_cutoff=cutoff, mode=mode)
    X_orig, y_orig = split_feature_target(df_orig)

    # Mutate post-cutoff outcomes (keeping eligibility >= 28 intact)
    leakage_db.execute("""
        UPDATE studentInfo
        SET final_result = CASE
            WHEN final_result IN ('Pass', 'Distinction') THEN 'Withdrawn'
            ELSE 'Pass'
        END
        WHERE id_student = 1 AND code_presentation = '2014B'
    """)
    leakage_db.execute("""
        UPDATE studentRegistration
        SET date_unregistration = 90
        WHERE id_student = 1 AND code_presentation = '2014B'
    """)

    df_mut = build_features(leakage_db, feature_cutoff=cutoff, mode=mode)
    X_mut, y_mut = split_feature_target(df_mut)

    pd.testing.assert_frame_equal(X_orig, X_mut)
    assert not y_orig.equals(y_mut)


@pytest.mark.parametrize("cutoff", [14, 21, 28])
@pytest.mark.parametrize("mode", ["operational", "common"])
def test_forbidden_columns_registration_bound_and_week_schema(
    leakage_db: duckdb.DuckDBPyConnection,
    cutoff: int,
    mode: str,
) -> None:
    """Invariant 3: 0 forbidden columns, 100% date_registration < cutoff, valid week columns."""
    df = build_features(leakage_db, feature_cutoff=cutoff, mode=mode)
    for col in FORBIDDEN_FEATURE_COLS:
        assert col not in df.columns

    assert df["date_registration"].notna().all()
    assert (df["date_registration"] < cutoff).all()

    if cutoff == 14:
        assert "clicks_w2" not in df.columns and "clicks_w3" not in df.columns
    elif cutoff == 21:
        assert "clicks_w2" in df.columns and "clicks_w3" not in df.columns
    else:
        assert "clicks_w2" in df.columns and "clicks_w3" in df.columns


@pytest.mark.parametrize("cutoff", [14, 21, 28])
@pytest.mark.parametrize("mode", ["operational", "common"])
def test_submission_rate_due_strictly_in_unit_interval(
    leakage_db: duckdb.DuckDBPyConnection,
    cutoff: int,
    mode: str,
) -> None:
    """Invariant 4 (AUD-02): submission_rate_due in [0.0, 1.0] with early/banked separated."""
    df = build_features(leakage_db, feature_cutoff=cutoff, mode=mode)
    valid_rates = df["submission_rate_due"].dropna()
    assert len(valid_rates) > 0
    assert ((valid_rates >= 0.0) & (valid_rates <= 1.0)).all()

    # Student 1 in 2014B submitted due #301 (date=10) on day 8, early #303 (due=35) on day 12,
    # banked #304 on day -1, and #302 (due=22) on day 14 (future at T=14, early at T=21, due at 28)
    s1 = df[(df["code_presentation"] == "2014B") & (df["id_student"] == 1)].iloc[0]
    assert 0.0 <= s1["submission_rate_due"] <= 1.0
    expected_early = 2 if cutoff == 21 else 1
    assert s1["n_submitted_early"] == expected_early
    assert s1["n_banked"] == 1


def test_cohort_boundary_membership_operational_vs_common(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """Verify exact eligibility of late registrant (10), early dropout (11), NULL reg (12)."""
    op14 = build_features(leakage_db, feature_cutoff=14, mode="operational")
    op21 = build_features(leakage_db, feature_cutoff=21, mode="operational")
    op28 = build_features(leakage_db, feature_cutoff=28, mode="operational")
    com14 = build_features(leakage_db, feature_cutoff=14, mode="common")
    com28 = build_features(leakage_db, feature_cutoff=28, mode="common")

    ids_op14 = set(op14["id_student"])
    ids_op21 = set(op21["id_student"])
    ids_op28 = set(op28["id_student"])
    ids_com14 = set(com14["id_student"])
    ids_com28 = set(com28["id_student"])

    # Late registrant (reg=16): excluded at Op T=14 & Common, included at Op T=21 & T=28
    assert 10 not in ids_op14 and 10 not in ids_com14
    assert 10 in ids_op21 and 10 in ids_op28

    # Early dropout (reg=-5, unreg=20): included at Op T=14, excluded at Op T=21/28 & Common
    assert 11 in ids_op14
    assert 11 not in ids_op21 and 11 not in ids_op28 and 11 not in ids_com14

    # NULL date_registration (12): always excluded
    for id_set in (ids_op14, ids_op21, ids_op28, ids_com14, ids_com28):
        assert 12 not in id_set

    # Safe Common Cohort keeps exact same student population across T=14 and T=28
    assert ids_com14 == ids_com28

    # Zero-click student (13): has_any_vle_click == 0 and days_since_last_click is NaN
    s13 = op28[op28["id_student"] == 13].iloc[0]
    assert s13["has_any_vle_click"] == 0
    assert pd.isna(s13["days_since_last_click"])


def test_top_activities_strictly_train_only(leakage_db: duckdb.DuckDBPyConnection) -> None:
    """Mutating 2014J VLE clicks must not change Train-only top_activities selection."""
    top_before = get_top_activities(leakage_db, feature_cutoff=28)
    leakage_db.execute("INSERT INTO studentVle VALUES ('BBB', '2014J', 2, 7, 5, 999999)")
    top_after = get_top_activities(leakage_db, feature_cutoff=28)
    assert top_before == top_after


def test_runtime_guard_catches_registration_leakage_under_optimization_mode(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """F-06: build_features must raise RuntimeError (survives python -O) on reg >= cutoff."""
    # Bypass SQL WHERE clause to verify post-query RuntimeError guard catches late registrant (16)
    with patch(
        "oulad.features._cohort_where_clause",
        return_value="sr.date_registration IS NOT NULL",
    ):
        with pytest.raises(RuntimeError, match="Temporal leakage"):
            build_features(leakage_db, feature_cutoff=14, mode="operational")


@pytest.mark.parametrize("cutoff", [21, 28])
def test_runtime_guard_catches_common_cohort_registration_ge_14_under_optimization_mode(
    leakage_db: duckdb.DuckDBPyConnection,
    cutoff: int,
) -> None:
    """Common mode must enforce date_registration < 14 post-query even at cutoffs 21 and 28."""
    # Inject exact boundary date_registration = 14 (valid in op T=21/28, illegal in common)
    leakage_db.execute(
        "INSERT INTO studentInfo VALUES ('AAA', '2014B', 999, 'F', 'London Region', "
        "'HE Qualification', '20-30%', '0-35', 0, 60, 'N', 'Pass')"
    )
    leakage_db.execute(
        "INSERT INTO studentRegistration VALUES ('AAA', '2014B', 999, 14, NULL)"
    )
    # Bypass SQL cohort WHERE clause so the post-query RuntimeError guard is exercised directly
    with patch(
        "oulad.features._cohort_where_clause",
        return_value="sr.date_registration IS NOT NULL",
    ):
        with pytest.raises(RuntimeError, match=r"date_registration >= 14.*mode=common"):
            build_features(leakage_db, feature_cutoff=cutoff, mode="common")
