"""DuckDB ingestion, 8 SQL data quality checks, and returning-student overlap audit (P1-B1)."""

from pathlib import Path
from typing import Any

import duckdb

from oulad.config import (
    CUTOFFS,
    RAW_TABLES,
    TRAIN_PRESENTATIONS,
    VALID_FINAL_RESULTS,
)


def get_connection(
    db_path: Path | str = ":memory:",
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection (in-memory for unit tests, file-backed for real data)."""
    if isinstance(db_path, Path):
        if not read_only:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path), read_only=read_only)
    return duckdb.connect(db_path, read_only=read_only)


def load_raw_tables(
    raw_dir: Path,
    con: duckdb.DuckDBPyConnection,
    tables: tuple[str, ...] = RAW_TABLES,
) -> dict[str, int]:
    """Load the 7 OULAD CSV files into DuckDB with nullstr='?' and return row counts.

    Raises:
        FileNotFoundError: If any expected OULAD CSV file is missing from ``raw_dir``.
    """
    missing = [name for name in tables if not (raw_dir / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing OULAD CSV file(s) in {raw_dir}: {', '.join(f'{m}.csv' for m in missing)}"
        )

    row_counts: dict[str, int] = {}
    for table in tables:
        csv_path = str((raw_dir / f"{table}.csv").resolve())
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            "SELECT * FROM read_csv_auto(?, nullstr='?', header=true)",
            [csv_path],
        )
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        row_counts[table] = int(count[0]) if count else 0
    return row_counts


def run_quality_checks(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Run the 8 mandatory SQL Data Quality Checks and return {check_name: n_violations}.

    Notes:
        - Never remove checks that report real-world anomalies (e.g. check #7);
          log them transparently in ``reports/data_quality.json``.
        - Check #8 returns 1 if ``imd_band`` missing rate >= 5%, else 0.
    """
    valid_results_sql = ", ".join(f"'{val}'" for val in VALID_FINAL_RESULTS)

    queries: dict[str, str] = {
        # Check #1: Composite primary key uniqueness in studentInfo
        "duplicate_student_info_keys": """
            SELECT COUNT(*) FROM (
                SELECT code_module, code_presentation, id_student
                FROM studentInfo
                GROUP BY code_module, code_presentation, id_student
                HAVING COUNT(*) > 1
            )
        """,
        # Check #2: Domain check on final_result
        "invalid_final_result": f"""
            SELECT COUNT(*) FROM studentInfo
            WHERE final_result IS NULL
               OR final_result NOT IN ({valid_results_sql})
        """,
        # Check #3: Referential integrity studentVle -> vle (3-column Left Anti-Join)
        "orphan_student_vle_sites": """
            SELECT COUNT(*) FROM studentVle sv
            LEFT JOIN vle v
              ON sv.code_module = v.code_module
             AND sv.code_presentation = v.code_presentation
             AND sv.id_site = v.id_site
            WHERE v.id_site IS NULL
        """,
        # Check #4: Referential integrity studentAssessment -> assessments (Left Anti-Join)
        "orphan_student_assessments": """
            SELECT COUNT(*) FROM studentAssessment sa
            LEFT JOIN assessments a
              ON sa.id_assessment = a.id_assessment
            WHERE a.id_assessment IS NULL
        """,
        # Check #5: Valid score range [0, 100] or NULL
        "invalid_assessment_scores": """
            SELECT COUNT(*) FROM studentAssessment
            WHERE score IS NOT NULL AND (score < 0 OR score > 100)
        """,
        # Check #6: Positive click counts
        "non_positive_sum_click": """
            SELECT COUNT(*) FROM studentVle
            WHERE sum_click IS NULL OR sum_click <= 0
        """,
        # Check #7: VLE interaction date within official module_presentation_length
        "vle_date_after_course_end": """
            SELECT COUNT(*) FROM studentVle sv
            JOIN courses c
              ON sv.code_module = c.code_module
             AND sv.code_presentation = c.code_presentation
            WHERE sv.date > c.module_presentation_length
        """,
        # Check #8: Missing rate of imd_band strictly below 5%
        "imd_band_missing_ge_5pct": """
            SELECT CASE
                WHEN COUNT(*) = 0 THEN 0
                WHEN (SUM(CASE WHEN imd_band IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) >= 0.05
                THEN 1
                ELSE 0
            END
            FROM studentInfo
        """,
        # Check #9: Unregistration must not precede registration when both dates exist
        "unregistration_before_registration": """
            SELECT COUNT(*) FROM studentRegistration
            WHERE date_registration IS NOT NULL
              AND date_unregistration IS NOT NULL
              AND date_unregistration < date_registration
        """,
    }

    results: dict[str, int] = {}
    for check_name, sql in queries.items():
        row = con.execute(sql).fetchone()
        results[check_name] = int(row[0]) if row else 0
    return results


def get_registration_and_missing_stats(
    con: duckdb.DuckDBPyConnection,
    cutoffs: tuple[int, ...] = CUTOFFS,
) -> dict[str, Any]:
    """Profile imd_band missing rate and date_registration NULL / >= cutoff counts."""
    imd_row = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN imd_band IS NULL THEN 1 ELSE 0 END) AS missing_imd
        FROM studentInfo
        """
    ).fetchone()
    total_info = int(imd_row[0]) if imd_row else 0
    missing_imd = int(imd_row[1]) if imd_row and imd_row[1] is not None else 0
    imd_missing_pct = round((missing_imd / total_info) * 100.0, 4) if total_info > 0 else 0.0

    reg_null_row = con.execute(
        "SELECT COUNT(*) FROM studentRegistration WHERE date_registration IS NULL"
    ).fetchone()
    reg_null_count = int(reg_null_row[0]) if reg_null_row else 0

    reg_ge_cutoffs: dict[str, int] = {}
    for cutoff in cutoffs:
        row = con.execute(
            "SELECT COUNT(*) FROM studentRegistration WHERE date_registration >= ?",
            [cutoff],
        ).fetchone()
        reg_ge_cutoffs[f"date_registration_ge_{cutoff}"] = int(row[0]) if row else 0

    return {
        "student_info_total_rows": total_info,
        "imd_band_missing_count": missing_imd,
        "imd_band_missing_pct": imd_missing_pct,
        "date_registration_null_count": reg_null_count,
        **reg_ge_cutoffs,
    }


def compute_student_overlap(
    con: duckdb.DuckDBPyConnection,
    train_presentations: tuple[str, ...] = TRAIN_PRESENTATIONS,
    target_presentations: tuple[str, ...] = ("2014B", "2014J"),
) -> dict[str, dict[str, int | float]]:
    """Compute Returning vs New student counts and shares on target presentations.

    A student in a target presentation is classified as ``returning`` if their
    ``id_student`` appeared in any presentation within ``train_presentations``.
    Both unique-student-level and enrollment-row-level counts are reported.
    """
    train_sql = ", ".join(f"'{p}'" for p in train_presentations)
    overlap_report: dict[str, dict[str, int | float]] = {}

    for pres in target_presentations:
        # Unique student-level overlap using INTERSECT
        unique_total_row = con.execute(
            "SELECT COUNT(DISTINCT id_student) FROM studentInfo WHERE code_presentation = ?",
            [pres],
        ).fetchone()
        unique_returning_row = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT id_student FROM studentInfo WHERE code_presentation = ?
                INTERSECT
                SELECT DISTINCT id_student FROM studentInfo WHERE code_presentation IN ({train_sql})
            )
            """,
            [pres],
        ).fetchone()

        # Enrollment-row-level overlap (matches modeling unit: module x presentation x student)
        enroll_row = con.execute(
            f"""
            WITH train_students AS (
                SELECT DISTINCT id_student
                FROM studentInfo
                WHERE code_presentation IN ({train_sql})
            )
            SELECT
                COUNT(*) AS total_enrollments,
                SUM(CASE WHEN ts.id_student IS NOT NULL THEN 1 ELSE 0 END) AS returning_enrollments
            FROM studentInfo si
            LEFT JOIN train_students ts
              ON si.id_student = ts.id_student
            WHERE si.code_presentation = ?
            """,
            [pres],
        ).fetchone()

        u_total = int(unique_total_row[0]) if unique_total_row else 0
        u_ret = int(unique_returning_row[0]) if unique_returning_row else 0
        u_new = u_total - u_ret

        e_total = int(enroll_row[0]) if enroll_row else 0
        e_ret = int(enroll_row[1]) if enroll_row and enroll_row[1] is not None else 0
        e_new = e_total - e_ret

        overlap_report[pres] = {
            "unique_students_total": u_total,
            "unique_students_returning": u_ret,
            "unique_students_new": u_new,
            "unique_returning_pct": round((u_ret / u_total) * 100.0, 2) if u_total else 0.0,
            "enrollments_total": e_total,
            "enrollments_returning": e_ret,
            "enrollments_new": e_new,
            "enrollments_returning_pct": round((e_ret / e_total) * 100.0, 2) if e_total else 0.0,
        }

    return overlap_report
