"""Two-way contract tests for OULAD DuckDB ingestion, 8 SQL checks, and overlap audit (P1-B1)."""

from pathlib import Path

import duckdb
import pytest

from oulad.db import (
    compute_student_overlap,
    get_registration_and_missing_stats,
    load_raw_tables,
    run_quality_checks,
)


def test_quality_checks_all_zero_on_clean_db(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Positive check: clean synthetic DB must produce 0 violations across all 9 checks."""
    results = run_quality_checks(synthetic_db)
    assert len(results) == 9
    assert all(v == 0 for v in results.values()), f"Unexpected violations: {results}"


def test_detects_duplicate_student_info_key(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #1: duplicate 3-column composite key must be caught."""
    synthetic_db.execute("""
        INSERT INTO studentInfo VALUES
            ('AAA', '2013B', 1, 'M', 'London Region', 'HE Qualification',
             '50-60%', '0-35', 0, 60, 'N', 'Pass')
    """)
    assert run_quality_checks(synthetic_db)["duplicate_student_info_keys"] == 1


def test_detects_invalid_final_result(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #2: unexpected final_result label must be caught."""
    synthetic_db.execute("""
        INSERT INTO studentInfo VALUES
            ('AAA', '2014J', 999, 'M', 'London Region', 'HE Qualification',
             '50-60%', '0-35', 0, 60, 'N', 'Dropped')
    """)
    assert run_quality_checks(synthetic_db)["invalid_final_result"] == 1


def test_detects_orphan_student_vle_site(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #3: studentVle row pointing to non-existent (module, pres, id_site)."""
    synthetic_db.execute(
        "INSERT INTO studentVle VALUES ('AAA', '2014J', 5, 99999, 10, 3)"
    )
    assert run_quality_checks(synthetic_db)["orphan_student_vle_sites"] == 1


def test_detects_orphan_student_assessment(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #4: studentAssessment row pointing to non-existent id_assessment."""
    synthetic_db.execute("INSERT INTO studentAssessment VALUES (99999, 5, 10, 0, 50.0)")
    assert run_quality_checks(synthetic_db)["orphan_student_assessments"] == 1


def test_detects_invalid_assessment_scores(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #5: score outside [0, 100] must be flagged while NULL remains valid."""
    synthetic_db.execute(
        "INSERT INTO studentAssessment VALUES (401, 5, 10, 0, 105.0), (401, 2, 10, 0, -5.0)"
    )
    assert run_quality_checks(synthetic_db)["invalid_assessment_scores"] == 2


def test_detects_non_positive_sum_click(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #6: sum_click <= 0 must be flagged."""
    synthetic_db.execute("INSERT INTO studentVle VALUES ('AAA', '2014J', 5, 6, 12, 0)")
    assert run_quality_checks(synthetic_db)["non_positive_sum_click"] == 1


def test_detects_vle_date_after_course_end(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #7: VLE date beyond module_presentation_length must be caught."""
    synthetic_db.execute("INSERT INTO studentVle VALUES ('AAA', '2013J', 2, 3, 300, 1)")
    assert run_quality_checks(synthetic_db)["vle_date_after_course_end"] == 1


def test_detects_imd_band_missing_ge_5pct(synthetic_db: duckdb.DuckDBPyConnection) -> None:
    """Negative check #8: missing imd_band >= 5% triggers violation flag (1/8 = 12.5%)."""
    synthetic_db.execute("UPDATE studentInfo SET imd_band = NULL WHERE id_student = 5")
    assert run_quality_checks(synthetic_db)["imd_band_missing_ge_5pct"] == 1


def test_detects_unregistration_before_registration(
    synthetic_db: duckdb.DuckDBPyConnection,
) -> None:
    """Negative check #9: date_unregistration < date_registration must be flagged."""
    synthetic_db.execute(
        "INSERT INTO studentRegistration VALUES ('AAA', '2014J', 998, 10, 5)"
    )
    assert run_quality_checks(synthetic_db)["unregistration_before_registration"] == 1


def test_load_raw_tables_parses_question_mark_as_null(tmp_path: Path) -> None:
    """Ensure read_csv_auto(..., nullstr='?') keeps numeric columns typed and parses '?' as NULL."""
    csv_samples = {
        "courses": "code_module,code_presentation,module_presentation_length\nAAA,2013J,268\n",
        "assessments": (
            "code_module,code_presentation,id_assessment,assessment_type,date,weight\n"
            "AAA,2013J,1,TMA,10,10\n"
            "AAA,2013J,2,Exam,?,100\n"
        ),
        "vle": (
            "id_site,code_module,code_presentation,activity_type,week_from,week_to\n"
            "10,AAA,2013J,oucontent,?,?\n"
        ),
        "studentInfo": (
            "code_module,code_presentation,id_student,gender,region,highest_education,"
            "imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
            "AAA,2013J,1,M,London,HE,?,0-35,0,60,N,Pass\n"
        ),
        "studentRegistration": (
            "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
            "AAA,2013J,1,-20,?\n"
            "AAA,2013J,2,?,?\n"
        ),
        "studentAssessment": (
            "id_assessment,id_student,date_submitted,is_banked,score\n"
            "1,1,9,0,85\n"
            "2,1,20,0,?\n"
        ),
        "studentVle": (
            "code_module,code_presentation,id_student,id_site,date,sum_click\n"
            "AAA,2013J,1,10,5,3\n"
        ),
    }
    for name, content in csv_samples.items():
        (tmp_path / f"{name}.csv").write_text(content, encoding="utf-8")

    con = duckdb.connect(":memory:")
    try:
        counts = load_raw_tables(tmp_path, con)
        assert counts["studentAssessment"] == 2

        score_type = con.execute(
            "SELECT typeof(score) FROM studentAssessment WHERE score IS NOT NULL LIMIT 1"
        ).fetchone()
        assert score_type is not None and score_type[0] != "VARCHAR"

        null_scores = con.execute(
            "SELECT COUNT(*) FROM studentAssessment WHERE score IS NULL"
        ).fetchone()
        assert null_scores == (1,)

        stats = get_registration_and_missing_stats(con)
        assert stats["date_registration_null_count"] == 1
        assert stats["imd_band_missing_count"] == 1
    finally:
        con.close()


def test_load_raw_tables_raises_on_missing_csv(tmp_path: Path) -> None:
    """Loading from an empty directory must raise FileNotFoundError listing missing files."""
    con = duckdb.connect(":memory:")
    try:
        with pytest.raises(FileNotFoundError, match="courses.csv"):
            load_raw_tables(tmp_path, con)
    finally:
        con.close()


def test_compute_student_overlap_returning_vs_new(
    synthetic_db: duckdb.DuckDBPyConnection,
) -> None:
    """Verify exact Returning vs New counts at both unique-student and enrollment levels."""
    overlap = compute_student_overlap(synthetic_db)

    # 2014B has students {1 (in 2013B -> returning), 4 (new)}
    assert overlap["2014B"]["unique_students_total"] == 2
    assert overlap["2014B"]["unique_students_returning"] == 1
    assert overlap["2014B"]["unique_students_new"] == 1
    assert overlap["2014B"]["unique_returning_pct"] == 50.0

    # 2014J has unique students {2 (in 2013J -> returning, 2 enrollments), 5 (new, 1 enrollment)}
    assert overlap["2014J"]["unique_students_total"] == 2
    assert overlap["2014J"]["unique_students_returning"] == 1
    assert overlap["2014J"]["unique_students_new"] == 1
    assert overlap["2014J"]["enrollments_total"] == 3
    assert overlap["2014J"]["enrollments_returning"] == 2
    assert overlap["2014J"]["enrollments_new"] == 1
    assert overlap["2014J"]["enrollments_returning_pct"] == 66.67
