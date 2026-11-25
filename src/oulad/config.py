"""Central configuration and constants for the OULAD Early Warning project."""

from pathlib import Path

# Project directory layout (resolved relative to this file: src/oulad/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "oulad.duckdb"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
SEED = 42

# OULAD dataset contract (UCI ID 349, CC BY 4.0)
RAW_TABLES: tuple[str, ...] = (
    "courses",
    "assessments",
    "vle",
    "studentInfo",
    "studentRegistration",
    "studentAssessment",
    "studentVle",
)

VALID_FINAL_RESULTS: tuple[str, ...] = (
    "Pass",
    "Fail",
    "Withdrawn",
    "Distinction",
)

# Presentation-Based Forward Split (AUD-07: discloses 2013J-2014B calendar overlap)
TRAIN_PRESENTATIONS: tuple[str, ...] = ("2013B", "2013J")
VAL_PRESENTATION = "2014B"
TEST_PRESENTATION = "2014J"

# Start-of-Day cutoffs (T=28 is primary; T=14, 21 for operational comparison)
CUTOFFS: tuple[int, ...] = (14, 21, 28)
PRIMARY_CUTOFF = 28
