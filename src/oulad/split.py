"""Presentation-Based Forward Split and Returning/New segmentation for OULAD (P1-B3).

Enforces AUD-07:
- Train: 2013B + 2013J (fit all preprocessing, top-5 activities, and models).
- Validation: 2014B stratified-split 60/40 into val_select (selection & early stopping)
  and val_calib (OOF calibration & threshold selection in P1-B5).
- Locked Test: 2014J strictly isolated until the single final evaluation in P1-B5.

Calendar disclosure (AUD-07): 2013J (Oct 2013-Jun 2014) overlaps ~4 calendar months
with 2014B (Feb 2014-Oct 2014), so this is a Presentation-Based Forward Split rather
than strictly chronological. Train 2013 -> Test 2014J (Oct 2014) has zero overlap.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from oulad.config import (
    PROCESSED_DIR,
    SEED,
    TEST_PRESENTATION,
    TRAIN_PRESENTATIONS,
    VAL_PRESENTATION,
)
from oulad.features import TARGET_COL, CohortMode


@dataclass(frozen=True)
class SplitData:
    """Container for Presentation-Based Forward Split with strict 2014J isolation."""

    train: pd.DataFrame
    val_select: pd.DataFrame
    val_calib: pd.DataFrame
    test_locked: pd.DataFrame

    def assert_invariants(self) -> None:
        """Verify 2014J isolation and disjointness between val_select and val_calib.

        Raises explicit RuntimeError so all data-boundary guards survive `python -O`.
        """
        for name, frame in (
            ("train", self.train),
            ("val_select", self.val_select),
            ("val_calib", self.val_calib),
        ):
            if TEST_PRESENTATION in set(frame["code_presentation"]):
                raise RuntimeError(
                    f"Locked test presentation {TEST_PRESENTATION} leaked into {name}!"
                )

        if not set(self.train["code_presentation"]).issubset(set(TRAIN_PRESENTATIONS)):
            raise RuntimeError("train contains non-training presentations")

        for name, frame in (("val_select", self.val_select), ("val_calib", self.val_calib)):
            if len(frame) > 0 and set(frame["code_presentation"]) != {VAL_PRESENTATION}:
                raise RuntimeError(f"{name} must contain only {VAL_PRESENTATION}")

        if len(self.test_locked) > 0 and set(self.test_locked["code_presentation"]) != {
            TEST_PRESENTATION
        }:
            raise RuntimeError(f"test_locked must contain only {TEST_PRESENTATION}")

        sel_keys = set(
            zip(
                self.val_select["code_module"],
                self.val_select["code_presentation"],
                self.val_select["id_student"],
            )
        )
        cal_keys = set(
            zip(
                self.val_calib["code_module"],
                self.val_calib["code_presentation"],
                self.val_calib["id_student"],
            )
        )
        overlap = sel_keys & cal_keys
        if overlap:
            raise RuntimeError(f"val_select and val_calib overlap on {len(overlap)} rows")


def presentation_forward_split(
    df: pd.DataFrame,
    val_calib_ratio: float = 0.40,
    seed: int = SEED,
) -> SplitData:
    """Split a feature DataFrame into train, val_select (60%), val_calib (40%), test_locked."""
    if not 0.0 < val_calib_ratio < 1.0:
        raise ValueError(f"val_calib_ratio must be in (0, 1), got {val_calib_ratio}")

    train_df = df[df["code_presentation"].isin(TRAIN_PRESENTATIONS)].reset_index(drop=True)
    val_df = df[df["code_presentation"] == VAL_PRESENTATION].reset_index(drop=True)
    test_df = df[df["code_presentation"] == TEST_PRESENTATION].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("Training split (2013B+2013J) is empty.")
    if len(val_df) < 2:
        raise ValueError("Validation split (2014B) must have at least 2 rows to split 60/40.")

    class_counts = val_df[TARGET_COL].value_counts()
    stratify_col = (
        val_df[TARGET_COL] if len(class_counts) >= 2 and class_counts.min() >= 2 else None
    )

    val_select_df, val_calib_df = train_test_split(
        val_df,
        test_size=val_calib_ratio,
        stratify=stratify_col,
        random_state=seed,
    )

    split = SplitData(
        train=train_df,
        val_select=val_select_df.reset_index(drop=True),
        val_calib=val_calib_df.reset_index(drop=True),
        test_locked=test_df,
    )
    split.assert_invariants()
    return split


def get_returning_mask(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.Series:
    """Return boolean mask identifying Returning students (id_student seen in train_df)."""
    train_ids = set(train_df["id_student"].unique())
    return eval_df["id_student"].isin(train_ids)


def load_feature_parquet(
    cutoff: int,
    mode: CohortMode = "operational",
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Load an exported P1-B2 feature Parquet file for a given cutoff and cohort mode."""
    prefix = "op" if mode == "operational" else "common"
    path = processed_dir / f"features_{prefix}_c{cutoff}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {path}")
    return pd.read_parquet(path)
