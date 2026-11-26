"""Shared synthetic in-memory DuckDB fixtures for fast, offline CI tests."""

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from oulad.config import SEED


def populate_clean_synthetic_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate all 7 OULAD tables with clean, deterministic test rows."""
    con.execute("""
        CREATE TABLE courses (
            code_module VARCHAR,
            code_presentation VARCHAR,
            module_presentation_length INTEGER
        );
        INSERT INTO courses VALUES
            ('AAA', '2013B', 240),
            ('AAA', '2013J', 268),
            ('AAA', '2014B', 240),
            ('AAA', '2014J', 269),
            ('BBB', '2014J', 262);

        CREATE TABLE assessments (
            code_module VARCHAR,
            code_presentation VARCHAR,
            id_assessment INTEGER,
            assessment_type VARCHAR,
            date INTEGER,
            weight DOUBLE
        );
        INSERT INTO assessments VALUES
            ('AAA', '2013B', 101, 'TMA', 10, 10.0),
            ('AAA', '2013B', 102, 'TMA', 25, 20.0),
            ('AAA', '2013J', 201, 'TMA', 12, 10.0),
            ('AAA', '2013J', 202, 'TMA', 35, 20.0),
            ('AAA', '2014B', 301, 'TMA', 10, 10.0),
            ('AAA', '2014B', 302, 'TMA', 22, 20.0),
            ('AAA', '2014J', 401, 'TMA', 10, 10.0),
            ('AAA', '2014J', 402, 'Exam', NULL, 100.0);

        CREATE TABLE vle (
            id_site INTEGER,
            code_module VARCHAR,
            code_presentation VARCHAR,
            activity_type VARCHAR,
            week_from INTEGER,
            week_to INTEGER
        );
        INSERT INTO vle VALUES
            (1, 'AAA', '2013B', 'oucontent', 0, 2),
            (2, 'AAA', '2013B', 'forumng', 0, 4),
            (3, 'AAA', '2013J', 'oucontent', 0, 2),
            (4, 'AAA', '2013J', 'resource', 1, 3),
            (5, 'AAA', '2014B', 'oucontent', 0, 2),
            (6, 'AAA', '2014J', 'oucontent', 0, 2),
            (7, 'BBB', '2014J', 'quiz', 0, 2);

        CREATE TABLE studentInfo (
            code_module VARCHAR,
            code_presentation VARCHAR,
            id_student INTEGER,
            gender VARCHAR,
            region VARCHAR,
            highest_education VARCHAR,
            imd_band VARCHAR,
            age_band VARCHAR,
            num_of_prev_attempts INTEGER,
            studied_credits INTEGER,
            disability VARCHAR,
            final_result VARCHAR
        );
        INSERT INTO studentInfo VALUES
            ('AAA', '2013B', 1, 'M', 'London Region', 'HE Qualification',
             '50-60%', '0-35', 0, 60, 'N', 'Pass'),
            ('AAA', '2013J', 2, 'F', 'Scotland', 'A Level or Equivalent',
             '20-30%', '35-55', 0, 60, 'N', 'Fail'),
            ('AAA', '2013J', 3, 'M', 'Wales', 'Lower Than A Level',
             '10-20%', '0-35', 0, 120, 'Y', 'Withdrawn'),
            ('AAA', '2014B', 1, 'M', 'London Region', 'HE Qualification',
             '50-60%', '0-35', 1, 60, 'N', 'Distinction'),
            ('AAA', '2014B', 4, 'F', 'North Region', 'HE Qualification',
             '80-90%', '0-35', 0, 60, 'N', 'Fail'),
            ('AAA', '2014J', 2, 'F', 'Scotland', 'A Level or Equivalent',
             '20-30%', '35-55', 1, 60, 'N', 'Pass'),
            ('BBB', '2014J', 2, 'F', 'Scotland', 'A Level or Equivalent',
             '20-30%', '35-55', 0, 60, 'N', 'Withdrawn'),
            ('AAA', '2014J', 5, 'M', 'South Region', 'A Level or Equivalent',
             '70-80%', '0-35', 0, 60, 'N', 'Fail');

        CREATE TABLE studentRegistration (
            code_module VARCHAR,
            code_presentation VARCHAR,
            id_student INTEGER,
            date_registration INTEGER,
            date_unregistration INTEGER
        );
        INSERT INTO studentRegistration VALUES
            ('AAA', '2013B', 1, -30, NULL),
            ('AAA', '2013J', 2, -20, NULL),
            ('AAA', '2013J', 3, -15, 45),
            ('AAA', '2014B', 1, -25, NULL),
            ('AAA', '2014B', 4, -10, NULL),
            ('AAA', '2014J', 2, -18, NULL),
            ('BBB', '2014J', 2, -12, 60),
            ('AAA', '2014J', 5, -5, NULL);

        CREATE TABLE studentAssessment (
            id_assessment INTEGER,
            id_student INTEGER,
            date_submitted INTEGER,
            is_banked INTEGER,
            score DOUBLE
        );
        INSERT INTO studentAssessment VALUES
            (101, 1, 9, 0, 82.0),
            (102, 1, 24, 0, 78.0),
            (201, 2, 11, 0, 45.0),
            (301, 1, 8, 0, 91.0),
            (302, 4, 21, 0, NULL),
            (401, 2, 9, 0, 70.0),
            (401, 5, 12, 0, 35.0);

        CREATE TABLE studentVle (
            code_module VARCHAR,
            code_presentation VARCHAR,
            id_student INTEGER,
            id_site INTEGER,
            date INTEGER,
            sum_click INTEGER
        );
        INSERT INTO studentVle VALUES
            ('AAA', '2013B', 1, 1, -2, 5),
            ('AAA', '2013B', 1, 2, 5, 12),
            ('AAA', '2013J', 2, 3, 3, 4),
            ('AAA', '2013J', 3, 4, 10, 2),
            ('AAA', '2014B', 1, 5, 4, 15),
            ('AAA', '2014B', 4, 5, 12, 3),
            ('AAA', '2014J', 2, 6, 6, 8),
            ('BBB', '2014J', 2, 7, 15, 2),
            ('AAA', '2014J', 5, 6, 20, 1);
    """)


@pytest.fixture
def synthetic_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide an isolated, clean in-memory OULAD DuckDB connection per test."""
    con = duckdb.connect(":memory:")
    populate_clean_synthetic_db(con)
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def leakage_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide a synthetic DB enriched with exact cutoff boundary & early/banked edge cases."""
    con = duckdb.connect(":memory:")
    populate_clean_synthetic_db(con)
    con.execute("""
        -- Add edge-case students in 2014B/2014J:
        -- 10: late registrant (reg=16, unreg=NULL) -> excluded from Op T=14 & Common, in Op T=21/28
        -- 11: early dropout (reg=-5, unreg=20) -> in Op T=14, excluded from Op T=21/28 & Common
        -- 12: NULL date_registration -> always excluded
        -- 13: zero-click student (reg=-10, unreg=NULL) -> tests has_any_vle_click=0 & NULL days
        INSERT INTO studentInfo VALUES
            ('AAA', '2014B', 10, 'M', 'London Region', 'HE Qualification',
             '50-60%', '0-35', 0, 60, 'N', 'Pass'),
            ('AAA', '2014B', 11, 'F', 'Wales', 'A Level or Equivalent',
             '20-30%', '0-35', 0, 60, 'N', 'Withdrawn'),
            ('AAA', '2014B', 12, 'M', 'Scotland', 'A Level or Equivalent',
             '30-40%', '35-55', 0, 60, 'N', 'Fail'),
            ('AAA', '2014J', 13, 'F', 'North Region', 'HE Qualification',
             '70-80%', '0-35', 0, 60, 'N', 'Fail');

        INSERT INTO studentRegistration VALUES
            ('AAA', '2014B', 10, 16, NULL),
            ('AAA', '2014B', 11, -5, 20),
            ('AAA', '2014B', 12, NULL, NULL),
            ('AAA', '2014J', 13, -10, NULL);

        -- Exact boundary & future VLE events at date = 14, 21, 28, 35
        INSERT INTO studentVle VALUES
            ('AAA', '2013B', 1, 1, 14, 99),
            ('AAA', '2013B', 1, 2, 21, 88),
            ('AAA', '2013B', 1, 1, 28, 77),
            ('AAA', '2013B', 1, 1, 35, 66),
            ('AAA', '2014B', 1, 5, 14, 50),
            ('AAA', '2014B', 1, 5, 21, 40),
            ('AAA', '2014B', 1, 5, 28, 30),
            ('AAA', '2014B', 11, 5, 10, 6);

        -- Early submission (due=35, submitted=12) + banked (is_banked=1, submitted=-1)
        -- + exact boundary future submissions at 14, 21, 28
        INSERT INTO assessments VALUES
            ('AAA', '2014B', 303, 'TMA', 35, 30.0),
            ('AAA', '2014B', 304, 'TMA', 12, 10.0);

        INSERT INTO studentAssessment VALUES
            (303, 1, 12, 0, 88.0),
            (304, 1, -1, 1, 75.0),
            (302, 1, 14, 0, 60.0),
            (302, 1, 21, 0, 65.0),
            (302, 1, 28, 0, 70.0);
    """)
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def synthetic_feature_table() -> pd.DataFrame:
    """Provide a deterministic synthetic feature table across 2013B, 2013J, 2014B, and 2014J."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    presentations = [
        ("2013B", 40, 1),
        ("2013J", 40, 41),
        ("2014B", 40, 61),
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
