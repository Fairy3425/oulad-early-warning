"""Leakage-free cohort definition and feature engineering for OULAD (P1-B2).

Enforces:
- AUD-01: Operational Cohort (primary) vs Safe Common Cohort (date_registration < 14).
- AUD-02: submission_rate_due strictly bounded in [0.0, 1.0], separating
  n_submitted_early and n_banked.
- Train-only top-5 VLE activity_type selection strictly from 2013B+2013J with date < T.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

from oulad.config import (
    CUTOFFS,
    PROCESSED_DIR,
    TRAIN_PRESENTATIONS,
)

CohortMode = Literal["operational", "common"]

KEY_COLS: tuple[str, ...] = ("code_module", "code_presentation", "id_student")
TARGET_COL = "target"
FORBIDDEN_FEATURE_COLS: tuple[str, ...] = ("final_result", "date_unregistration")


def get_top_activities(
    con: duckdb.DuckDBPyConnection,
    feature_cutoff: int,
    train_presentations: Sequence[str] = TRAIN_PRESENTATIONS,
    top_k: int = 5,
) -> list[str]:
    """Select top-k VLE activity_types strictly from Train presentations with date < T."""
    if feature_cutoff <= 0:
        raise ValueError(f"feature_cutoff must be positive, got {feature_cutoff}")
    train_sql = ", ".join(f"'{p}'" for p in train_presentations)
    rows = con.execute(
        f"""
        SELECT v.activity_type, SUM(sv.sum_click) AS total_clicks
        FROM studentVle sv
        JOIN vle v
          ON sv.code_module = v.code_module
         AND sv.code_presentation = v.code_presentation
         AND sv.id_site = v.id_site
        WHERE sv.code_presentation IN ({train_sql})
          AND sv.date < ?
          AND v.activity_type IS NOT NULL
        GROUP BY v.activity_type
        ORDER BY total_clicks DESC, v.activity_type ASC
        LIMIT ?
        """,
        [feature_cutoff, top_k],
    ).fetchall()
    return [str(r[0]) for r in rows]


def _cohort_where_clause(feature_cutoff: int, mode: CohortMode) -> str:
    """Return SQL WHERE clause implementing Operational or Safe Common Cohort (AUD-01)."""
    if mode == "operational":
        return (
            f"sr.date_registration IS NOT NULL "
            f"AND sr.date_registration < {feature_cutoff} "
            f"AND (sr.date_unregistration IS NULL OR sr.date_unregistration >= {feature_cutoff})"
        )
    if mode == "common":
        if feature_cutoff < 14:
            raise ValueError(
                f"Safe Common Cohort requires feature_cutoff >= 14, got {feature_cutoff}"
            )
        return (
            "sr.date_registration IS NOT NULL "
            "AND sr.date_registration < 14 "
            "AND (sr.date_unregistration IS NULL OR sr.date_unregistration >= 28)"
        )
    raise ValueError(f"Invalid mode '{mode}'. Expected 'operational' or 'common'.")


def build_features(
    con: duckdb.DuckDBPyConnection,
    feature_cutoff: int,
    mode: CohortMode = "operational",
    top_activities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a leakage-free feature table for Start-of-Day cutoff T.

    Returns a DataFrame containing KEY_COLS, engineered features (VLE, Assessment,
    Demographics & Registration), and binary TARGET_COL (1 = Withdrawn/Fail,
    0 = Pass/Distinction). Forbidden columns (final_result, date_unregistration)
    are strictly excluded.
    """
    if feature_cutoff not in CUTOFFS:
        raise ValueError(f"Unsupported feature_cutoff={feature_cutoff}; expected one of {CUTOFFS}")

    cohort_where = _cohort_where_clause(feature_cutoff, mode)
    activities = (
        list(top_activities)
        if top_activities is not None
        else get_top_activities(con, feature_cutoff=feature_cutoff)
    )

    # Dynamic complete-week VLE aggregations strictly < feature_cutoff
    week_exprs = [
        "COALESCE(SUM(CASE WHEN sv.date >= 0 AND sv.date < 7 THEN sv.sum_click ELSE 0 END), 0) "
        "AS clicks_w0",
        "COALESCE(SUM(CASE WHEN sv.date >= 7 AND sv.date < 14 THEN sv.sum_click ELSE 0 END), 0) "
        "AS clicks_w1",
    ]
    if feature_cutoff >= 21:
        week_exprs.append(
            "COALESCE(SUM(CASE WHEN sv.date >= 14 AND sv.date < 21 "
            "THEN sv.sum_click ELSE 0 END), 0) AS clicks_w2"
        )
    if feature_cutoff >= 28:
        week_exprs.append(
            "COALESCE(SUM(CASE WHEN sv.date >= 21 AND sv.date < 28 "
            "THEN sv.sum_click ELSE 0 END), 0) AS clicks_w3"
        )
    week_sql = ",\n            ".join(week_exprs)

    # Dynamic Train-only top-k activity_type click columns
    act_select_cols = []
    act_agg_exprs = []
    for act in activities:
        safe_col = f"clicks_act_{act.lower().replace(' ', '_')}"
        act_select_cols.append(f"COALESCE(va.{safe_col}, 0) AS {safe_col}")
        act_agg_exprs.append(
            f"COALESCE(SUM(CASE WHEN v.activity_type = '{act}' THEN sv.sum_click ELSE 0 END), 0) "
            f"AS {safe_col}"
        )
    act_select_sql = (
        ",\n        " + ",\n        ".join(act_select_cols) if act_select_cols else ""
    )
    act_agg_sql = (
        ",\n            " + ",\n            ".join(act_agg_exprs)
        if act_agg_exprs
        else ",\n            0 AS _dummy_act"
    )

    query = f"""
    WITH cohort AS (
        SELECT
            si.code_module,
            si.code_presentation,
            si.id_student,
            si.gender,
            si.region,
            si.highest_education,
            si.imd_band,
            si.age_band,
            si.num_of_prev_attempts,
            si.studied_credits,
            si.disability,
            sr.date_registration,
            CASE
                WHEN si.final_result IN ('Withdrawn', 'Fail') THEN 1
                ELSE 0
            END AS {TARGET_COL}
        FROM studentInfo si
        JOIN studentRegistration sr
          ON si.code_module = sr.code_module
         AND si.code_presentation = sr.code_presentation
         AND si.id_student = sr.id_student
        WHERE {cohort_where}
    ),
    vle_daily AS (
        SELECT
            sv.code_module,
            sv.code_presentation,
            sv.id_student,
            sv.date,
            SUM(sv.sum_click) AS daily_clicks
        FROM studentVle sv
        WHERE sv.date < {feature_cutoff}
        GROUP BY sv.code_module, sv.code_presentation, sv.id_student, sv.date
    ),
    vle_daily_stats AS (
        SELECT
            code_module,
            code_presentation,
            id_student,
            COALESCE(STDDEV_POP(daily_clicks), 0.0) AS std_daily_clicks
        FROM vle_daily
        GROUP BY code_module, code_presentation, id_student
    ),
    vle_agg AS (
        SELECT
            sv.code_module,
            sv.code_presentation,
            sv.id_student,
            COALESCE(SUM(CASE WHEN sv.date < 0 THEN sv.sum_click ELSE 0 END), 0)
                AS clicks_pre_course,
            COALESCE(SUM(sv.sum_click), 0) AS clicks_total,
            COUNT(DISTINCT sv.date) AS active_days,
            COUNT(DISTINCT sv.id_site) AS n_unique_sites,
            MIN(sv.date) AS first_click_day,
            MAX(sv.date) AS last_click_day,
            ({feature_cutoff - 1} - MAX(sv.date)) AS days_since_last_click,
            {week_sql}
            {act_agg_sql}
        FROM studentVle sv
        LEFT JOIN vle v
          ON sv.code_module = v.code_module
         AND sv.code_presentation = v.code_presentation
         AND sv.id_site = v.id_site
        WHERE sv.date < {feature_cutoff}
        GROUP BY sv.code_module, sv.code_presentation, sv.id_student
    ),
    due_counts AS (
        SELECT
            code_module,
            code_presentation,
            COUNT(DISTINCT id_assessment) AS n_due_before_cutoff
        FROM assessments
        WHERE date IS NOT NULL
          AND date < {feature_cutoff}
        GROUP BY code_module, code_presentation
    ),
    assess_agg AS (
        SELECT
            a.code_module,
            a.code_presentation,
            sa.id_student,
            COUNT(DISTINCT CASE
                WHEN sa.is_banked = 0 AND a.date IS NOT NULL AND a.date < {feature_cutoff}
                THEN sa.id_assessment
            END) AS n_submitted_due,
            COUNT(DISTINCT CASE
                WHEN sa.is_banked = 0 AND (a.date IS NULL OR a.date >= {feature_cutoff})
                THEN sa.id_assessment
            END) AS n_submitted_early,
            COUNT(DISTINCT CASE
                WHEN sa.is_banked = 1
                THEN sa.id_assessment
            END) AS n_banked,
            AVG(CASE WHEN sa.score IS NOT NULL THEN sa.score END) AS mean_score,
            AVG(CASE
                WHEN sa.is_banked = 0 AND a.date IS NOT NULL
                THEN (sa.date_submitted - a.date)
            END) AS mean_days_late
        FROM studentAssessment sa
        JOIN assessments a
          ON sa.id_assessment = a.id_assessment
        WHERE sa.date_submitted < {feature_cutoff}
        GROUP BY a.code_module, a.code_presentation, sa.id_student
    )
    SELECT
        c.code_module,
        c.code_presentation,
        c.id_student,
        -- Group 3: Demographics & Registration
        c.gender,
        c.region,
        c.highest_education,
        c.imd_band,
        c.age_band,
        c.num_of_prev_attempts,
        c.studied_credits,
        c.disability,
        c.date_registration,
        -- Group 1: VLE Behavior (< T)
        CASE WHEN COALESCE(va.clicks_total, 0) > 0 THEN 1 ELSE 0 END AS has_any_vle_click,
        COALESCE(va.clicks_pre_course, 0) AS clicks_pre_course,
        COALESCE(va.clicks_total, 0) AS clicks_total,
        COALESCE(va.active_days, 0) AS active_days,
        COALESCE(va.n_unique_sites, 0) AS n_unique_sites,
        COALESCE(vds.std_daily_clicks, 0.0) AS std_daily_clicks,
        va.first_click_day,
        va.last_click_day,
        va.days_since_last_click,
        COALESCE(va.clicks_w0, 0) AS clicks_w0,
        COALESCE(va.clicks_w1, 0) AS clicks_w1
        {", COALESCE(va.clicks_w2, 0) AS clicks_w2" if feature_cutoff >= 21 else ""}
        {", COALESCE(va.clicks_w3, 0) AS clicks_w3" if feature_cutoff >= 28 else ""}
        {act_select_sql},
        -- Group 2: Assessment Behavior (< T, AUD-02)
        COALESCE(dc.n_due_before_cutoff, 0) AS n_due_before_cutoff,
        COALESCE(aa.n_submitted_due, 0) AS n_submitted_due,
        CASE
            WHEN COALESCE(dc.n_due_before_cutoff, 0) > 0
            THEN CAST(COALESCE(aa.n_submitted_due, 0) AS DOUBLE)
                 / CAST(dc.n_due_before_cutoff AS DOUBLE)
            ELSE NULL
        END AS submission_rate_due,
        COALESCE(aa.n_submitted_early, 0) AS n_submitted_early,
        COALESCE(aa.n_banked, 0) AS n_banked,
        aa.mean_score,
        CASE WHEN aa.mean_score IS NOT NULL THEN 1 ELSE 0 END AS has_score,
        aa.mean_days_late,
        -- Target
        c.{TARGET_COL}
    FROM cohort c
    LEFT JOIN vle_agg va
      ON c.code_module = va.code_module
     AND c.code_presentation = va.code_presentation
     AND c.id_student = va.id_student
    LEFT JOIN vle_daily_stats vds
      ON c.code_module = vds.code_module
     AND c.code_presentation = vds.code_presentation
     AND c.id_student = vds.id_student
    LEFT JOIN due_counts dc
      ON c.code_module = dc.code_module
     AND c.code_presentation = dc.code_presentation
    LEFT JOIN assess_agg aa
      ON c.code_module = aa.code_module
     AND c.code_presentation = aa.code_presentation
     AND c.id_student = aa.id_student
    ORDER BY c.code_module, c.code_presentation, c.id_student
    """

    df = con.execute(query).df()

    # Add smoothed click_trend_ratio based on completed weeks strictly < feature_cutoff
    if feature_cutoff == 14:
        df["click_trend_ratio"] = (df["clicks_w1"] + 1.0) / (df["clicks_w0"] + 1.0)
    elif feature_cutoff == 21:
        df["click_trend_ratio"] = (df["clicks_w2"] + 1.0) / (
            df["clicks_w0"] + df["clicks_w1"] + 1.0
        )
    else:  # feature_cutoff == 28
        df["click_trend_ratio"] = (df["clicks_w2"] + df["clicks_w3"] + 1.0) / (
            df["clicks_w0"] + df["clicks_w1"] + 1.0
        )

    # Hard runtime invariant guards (AUD-01 & AUD-02; explicit RuntimeError survives python -O)
    if mode == "operational":
        max_allowed_registration = feature_cutoff
    elif mode == "common":
        max_allowed_registration = 14
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'operational' or 'common'.")

    if len(df) > 0:
        if not df["date_registration"].notna().all():
            raise RuntimeError("Cohort boundary violation: NULL date_registration detected.")
        if not (df["date_registration"] < max_allowed_registration).all():
            raise RuntimeError(
                f"Temporal leakage: date_registration >= {max_allowed_registration} "
                f"in mode={mode} (cutoff={feature_cutoff})"
            )
        valid_rates = df["submission_rate_due"].dropna()
        if len(valid_rates) > 0 and not ((valid_rates >= 0.0) & (valid_rates <= 1.0)).all():
            raise RuntimeError("AUD-02 violation: submission_rate_due outside [0.0, 1.0]")

    return df


def split_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model feature matrix X from keys and target y."""
    drop_cols = [c for c in ("code_presentation", "id_student", TARGET_COL) if c in df.columns]
    # Note: code_module is intentionally kept as a demographic/course feature per spec
    X = df.drop(columns=drop_cols).copy()
    y = df[TARGET_COL].astype(np.int64).copy()
    return X, y


def export_all_feature_parquets(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path = PROCESSED_DIR,
    cutoffs: Sequence[int] = CUTOFFS,
) -> list[dict[str, object]]:
    """Build and export all 6 feature Parquet files (operational + common x 14/21/28)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for cutoff in cutoffs:
        top_acts = get_top_activities(con, feature_cutoff=cutoff)
        for mode, prefix in (("operational", "op"), ("common", "common")):
            df = build_features(con, feature_cutoff=cutoff, mode=mode, top_activities=top_acts)
            out_path = out_dir / f"features_{prefix}_c{cutoff}.parquet"
            df.to_parquet(out_path, index=False)
            summary.append(
                {
                    "mode": mode,
                    "cutoff": cutoff,
                    "n_rows": int(len(df)),
                    "n_features": int(len(split_feature_target(df)[0].columns)),
                    "positive_rate": round(float(df[TARGET_COL].mean()), 4) if len(df) else 0.0,
                    "top_activities": top_acts,
                    "path": str(out_path.name),
                }
            )
    return summary
