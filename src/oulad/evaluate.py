"""OOF Calibration, Threshold Selection, Single Locked-Test (2014J) Evaluation & Fairness (P1-B5).

Enforces AUD-08, FA-02, FA-07, and AUD-16:
- Explicit StratifiedKFold(n_splits=5) loop with CalibratedClassifierCV(FrozenEstimator(...))
  on val_calib (compatible with scikit-learn >= 1.6; never passes cv=5 with FrozenEstimator).
- Hard code guard blocking isotonic calibration whenever n_calib < 1000 or min_fold_train < 800.
- Threshold tau* chosen strictly on OOF calibrated probabilities of val_calib, with deterministic
  fallback (max F1) if target Precision >= 0.70 (with n_predicted_positives >= 30) is unattainable.
- All models, calibrators, and thresholds are LOCKED before the single final unlock of 2014J,
  which evaluates in one pass: 3 primary models, Returning vs New, 12 cutoff models (14/21/28),
  Score-Free & Sensitive-Demographic Ablation, Illustrative Cost-Benefit, min(20, len) FP/FN
  error samples, and Subgroup Fairness (n, n_pos, low_sample_warning).
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from oulad.baselines import (
    SCORE_COLS,
    SENSITIVE_DEMOGRAPHIC_COLS,
    compute_binary_metrics,
    evaluate_returning_vs_new,
    predict_lgbm_proba,
    train_lightgbm,
    train_logistic_regression,
)
from oulad.config import (
    CUTOFFS,
    FIGURES_DIR,
    PRIMARY_CUTOFF,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
    TEST_PRESENTATION,
)
from oulad.features import CohortMode, split_feature_target
from oulad.split import SplitData, load_feature_parquet, presentation_forward_split
from oulad.torch_models import TabularMLP, TabularPreprocessor
from oulad.train_torch import (
    ARTIFACTS_DIR,
    load_torch_checkpoint,
    predict_torch_proba,
    save_torch_checkpoint,
    train_tabular_mlp,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAIRNESS_GROUP_COLS: tuple[str, ...] = ("gender", "disability", "imd_band", "age_band")


class LGBMSklearnWrapper(ClassifierMixin, BaseEstimator):
    """Frozen sklearn-compatible wrapper applying Train-only categorical schema for LightGBM."""

    def __init__(
        self,
        model: lgb.LGBMClassifier,
        cat_categories: dict[str, list[str]],
    ) -> None:
        self.model = model
        self.cat_categories = cat_categories
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LGBMSklearnWrapper":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = predict_lgbm_proba(self.model, X, self.cat_categories)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


class TorchSklearnWrapper(ClassifierMixin, BaseEstimator):
    """Frozen sklearn-compatible wrapper around (TabularMLP, TabularPreprocessor)."""

    def __init__(
        self,
        model: TabularMLP,
        preprocessor: TabularPreprocessor,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TorchSklearnWrapper":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = predict_torch_proba(self.model, self.preprocessor, X)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def compute_ece(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute equal-width Expected Calibration Error (ECE) over [0, 1]."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = len(y_arr)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (p_arr >= lo) & (p_arr <= hi)
        else:
            mask = (p_arr >= lo) & (p_arr < hi)
        count = int(mask.sum())
        if count > 0:
            acc = float(y_arr[mask].mean())
            conf = float(p_arr[mask].mean())
            ece += (count / n) * abs(acc - conf)
    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """Compute ranking, calibration (Brier/ECE), and thresholded classification metrics."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    y_pred = (p_arr >= threshold).astype(np.int64)

    n = int(len(y_arr))
    n_pos = int(y_arr.sum())
    has_both = len(np.unique(y_arr)) == 2
    tn, fp, fn, tp = (
        confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel().tolist()
        if n > 0
        else [0, 0, 0, 0]
    )

    return {
        "n": n,
        "n_pos": n_pos,
        "positive_rate": round(float(n_pos / n), 4) if n else 0.0,
        "pr_auc": round(float(average_precision_score(y_arr, p_arr)), 4) if has_both else None,
        "roc_auc": round(float(roc_auc_score(y_arr, p_arr)), 4) if has_both else None,
        "brier_score": round(float(brier_score_loss(y_arr, p_arr)), 4) if n else None,
        "ece_10bins": round(compute_ece(y_arr, p_arr, n_bins=10), 4) if n else None,
        "threshold": round(float(threshold), 4),
        "n_predicted_positives": int(y_pred.sum()),
        "precision": round(float(precision_score(y_arr, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_arr, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_arr, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@dataclass
class OOFCalibrationResult:
    """Result of explicit StratifiedKFold(5) OOF calibration selection on val_calib."""

    n_calib: int
    min_fold_train: int
    isotonic_blocked: bool
    candidate_metrics: dict[str, dict[str, float]]
    chosen_method: str
    oof_probs_raw: np.ndarray
    oof_probs_winner: np.ndarray
    final_calibrator: CalibratedClassifierCV | None


def select_best_calibrator_oof(
    base_model: Any,
    X_val_calib: pd.DataFrame,
    y_val_calib: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = SEED,
) -> OOFCalibrationResult:
    """Compare Uncalibrated vs sigmoid (vs isotonic if n_calib>=1000) via explicit 5-fold OOF.

    Uses scikit-learn >= 1.6 pattern: explicit StratifiedKFold loop with
    CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method).
    """
    y_arr = np.asarray(y_val_calib, dtype=np.int64)
    n_calib = int(len(y_arr))
    if n_calib < n_splits * 2:
        raise ValueError(f"val_calib too small ({n_calib}) for {n_splits}-fold OOF calibration.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X_val_calib, y_arr))
    min_fold_train = int(min(len(fit_idx) for fit_idx, _ in folds))

    # Hard AUD-08 / FA-02 guard against isotonic overfitting on small calibration sets
    isotonic_blocked = (n_calib < 1000) or (min_fold_train < 800)
    methods_to_test: tuple[str, ...] = (
        ("uncalibrated", "sigmoid")
        if isotonic_blocked
        else ("uncalibrated", "sigmoid", "isotonic")
    )

    raw_probs = np.asarray(base_model.predict_proba(X_val_calib)[:, 1], dtype=np.float64)
    oof_by_method: dict[str, np.ndarray] = {"uncalibrated": raw_probs}
    candidate_metrics: dict[str, dict[str, float]] = {
        "uncalibrated": {
            "brier_score": round(float(brier_score_loss(y_arr, raw_probs)), 5),
            "ece_10bins": round(compute_ece(y_arr, raw_probs, n_bins=10), 5),
        }
    }

    for method in methods_to_test:
        if method == "uncalibrated":
            continue
        oof_probs = np.zeros(n_calib, dtype=np.float64)
        for fit_idx, eval_idx in folds:
            cal = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
            cal.fit(X_val_calib.iloc[fit_idx], y_arr[fit_idx])
            oof_probs[eval_idx] = cal.predict_proba(X_val_calib.iloc[eval_idx])[:, 1]
        oof_by_method[method] = oof_probs
        candidate_metrics[method] = {
            "brier_score": round(float(brier_score_loss(y_arr, oof_probs)), 5),
            "ece_10bins": round(compute_ece(y_arr, oof_probs, n_bins=10), 5),
        }

    simplicity_rank = {"uncalibrated": 0, "sigmoid": 1, "isotonic": 2}
    chosen_method = min(
        methods_to_test,
        key=lambda m: (
            candidate_metrics[m]["brier_score"],
            candidate_metrics[m]["ece_10bins"],
            simplicity_rank[m],
        ),
    )

    final_calibrator: CalibratedClassifierCV | None = None
    if chosen_method != "uncalibrated":
        final_calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model), method=chosen_method
        )
        final_calibrator.fit(X_val_calib, y_arr)

    return OOFCalibrationResult(
        n_calib=n_calib,
        min_fold_train=min_fold_train,
        isotonic_blocked=isotonic_blocked,
        candidate_metrics=candidate_metrics,
        chosen_method=chosen_method,
        oof_probs_raw=raw_probs,
        oof_probs_winner=oof_by_method[chosen_method],
        final_calibrator=final_calibrator,
    )


def apply_calibrator(
    base_model: Any,
    calibrator: CalibratedClassifierCV | None,
    X: pd.DataFrame,
) -> np.ndarray:
    """Apply frozen calibrator if chosen, otherwise return raw base-model probabilities."""
    if calibrator is None:
        return np.asarray(base_model.predict_proba(X)[:, 1], dtype=np.float64)
    return np.asarray(calibrator.predict_proba(X)[:, 1], dtype=np.float64)


@dataclass
class ThresholdSelectionResult:
    """Selected operating threshold tau* from val_calib OOF probabilities with fallback log."""

    threshold: float
    target_precision: float
    min_predicted_positives: int
    target_precision_met: bool
    fallback_used: str | None
    max_achievable_precision: float
    oof_precision: float
    oof_recall: float
    oof_f1: float
    oof_n_predicted_positives: int


def select_threshold_with_fallback(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    target_precision: float = 0.70,
    min_predicted_positives: int = 30,
) -> ThresholdSelectionResult:
    """Select tau* maximizing Recall subject to Precision >= target_precision & n_pred >= min_pos.

    Deterministic transparent fallback (AUD-08): if no threshold satisfies the precision floor,
    records target_precision_met=False, logs honest max_achievable_precision, and selects the
    threshold maximizing F1.
    """
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)

    grid = np.unique(np.concatenate([np.round(np.linspace(0.05, 0.95, 91), 4), np.unique(p_arr)]))
    grid = grid[(grid > 0.0) & (grid < 1.0)]
    if len(grid) == 0:
        grid = np.array([0.5])

    best_meeting: tuple[float, float, float, float, int] | None = None
    best_f1: tuple[float, float, float, float, int] | None = None
    max_prec_with_min_pos = 0.0
    max_prec_any = 0.0

    for tau in grid:
        y_pred = (p_arr >= tau).astype(np.int64)
        n_pred = int(y_pred.sum())
        if n_pred == 0:
            continue
        prec = float(precision_score(y_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_arr, y_pred, zero_division=0))

        if prec > max_prec_any:
            max_prec_any = prec
        if n_pred >= min_predicted_positives and prec > max_prec_with_min_pos:
            max_prec_with_min_pos = prec

        if best_f1 is None or (f1, prec, float(tau)) > (best_f1[0], best_f1[1], best_f1[3]):
            best_f1 = (f1, prec, rec, float(tau), n_pred)

        if prec >= target_precision and n_pred >= min_predicted_positives:
            cand = (rec, f1, prec, float(tau), n_pred)
            if best_meeting is None or (cand[0], cand[1], cand[3]) > (
                best_meeting[0],
                best_meeting[1],
                best_meeting[3],
            ):
                best_meeting = cand

    honest_max_prec = max_prec_with_min_pos if max_prec_with_min_pos > 0.0 else max_prec_any

    if best_meeting is not None:
        rec, f1, prec, tau, n_pred = best_meeting
        return ThresholdSelectionResult(
            threshold=round(tau, 4),
            target_precision=target_precision,
            min_predicted_positives=min_predicted_positives,
            target_precision_met=True,
            fallback_used=None,
            max_achievable_precision=round(honest_max_prec, 4),
            oof_precision=round(prec, 4),
            oof_recall=round(rec, 4),
            oof_f1=round(f1, 4),
            oof_n_predicted_positives=n_pred,
        )

    if best_f1 is None:
        return ThresholdSelectionResult(
            threshold=0.5,
            target_precision=target_precision,
            min_predicted_positives=min_predicted_positives,
            target_precision_met=False,
            fallback_used="default_0.5",
            max_achievable_precision=round(honest_max_prec, 4),
            oof_precision=0.0,
            oof_recall=0.0,
            oof_f1=0.0,
            oof_n_predicted_positives=0,
        )
    f1, prec, rec, tau, n_pred = best_f1
    return ThresholdSelectionResult(
        threshold=round(tau, 4),
        target_precision=target_precision,
        min_predicted_positives=min_predicted_positives,
        target_precision_met=False,
        fallback_used="max_f1",
        max_achievable_precision=round(honest_max_prec, 4),
        oof_precision=round(prec, 4),
        oof_recall=round(rec, 4),
        oof_f1=round(f1, 4),
        oof_n_predicted_positives=n_pred,
    )


def build_illustrative_cost_benefit_table(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    locked_threshold: float,
    cost_ratios: Sequence[tuple[int, int]] = ((1, 5), (1, 10)),
) -> list[dict[str, Any]]:
    """Compute illustrative expected costs (FP:FN = 1:5, 1:10) with explicit disclaimer."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = max(len(y_arr), 1)
    rows: list[dict[str, Any]] = []

    for c_fp, c_fn in cost_ratios:
        # Bayes-optimal threshold under calibrated probabilities: C_FP / (C_FP + C_FN)
        tau_bayes = c_fp / (c_fp + c_fn)
        for label, tau in (("locked_tau", locked_threshold), ("bayes_cost_tau", tau_bayes)):
            y_pred = (p_arr >= tau).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel().tolist()
            total_cost = int(c_fp * fp + c_fn * fn)
            rows.append(
                {
                    "cost_ratio_fp_to_fn": f"{c_fp}:{c_fn}",
                    "policy": label,
                    "threshold": round(float(tau), 4),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "tn": int(tn),
                    "total_illustrative_cost": total_cost,
                    "cost_per_student": round(total_cost / n, 4),
                    "disclaimer": (
                        "Illustrative cost assumption only (not real institutional cost)."
                    ),
                }
            )
    return rows


def extract_post_lock_error_samples(
    df_test: pd.DataFrame,
    y_prob: np.ndarray | pd.Series,
    threshold: float,
    *,
    max_k: int = 20,
) -> dict[str, Any]:
    """Extract min(max_k, len) highest-confidence FP and FN samples without fabrication."""
    work = df_test.copy()
    work["pred_prob"] = np.asarray(y_prob, dtype=np.float64)
    work["pred_label"] = (work["pred_prob"] >= threshold).astype(np.int64)

    keep_cols = [
        c
        for c in (
            "code_module",
            "code_presentation",
            "id_student",
            "target",
            "pred_prob",
            "clicks_total",
            "active_days",
            "submission_rate_due",
            "mean_score",
            "days_since_last_click",
        )
        if c in work.columns
    ]

    fp_df = work[(work["target"] == 0) & (work["pred_label"] == 1)].sort_values(
        "pred_prob", ascending=False
    )
    fn_df = work[(work["target"] == 1) & (work["pred_label"] == 0)].sort_values(
        "pred_prob", ascending=True
    )

    k_fp = min(max_k, len(fp_df))
    k_fn = min(max_k, len(fn_df))

    def _clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for row in frame[keep_cols].to_dict(orient="records"):
            cleaned = {
                str(k): (
                    None
                    if pd.isna(v)
                    else (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                )
                for k, v in row.items()
            }
            recs.append(cleaned)
        return recs

    return {
        "policy": "Post-lock observational error audit ONLY (no model edits after 2014J unlock).",
        "max_k_requested": max_k,
        "total_fp": int(len(fp_df)),
        "sampled_fp_count": int(k_fp),
        "fp_samples": _clean_records(fp_df.iloc[:k_fp]),
        "total_fn": int(len(fn_df)),
        "sampled_fn_count": int(k_fn),
        "fn_samples": _clean_records(fn_df.iloc[:k_fn]),
    }


def compute_subgroup_fairness(
    df_test: pd.DataFrame,
    y_prob: np.ndarray | pd.Series,
    threshold: float,
    *,
    group_cols: Sequence[str] = FAIRNESS_GROUP_COLS,
    min_group_n: int = 30,
) -> pd.DataFrame:
    """Compute per-subgroup Fairness table with n, n_pos, Positive Rate, Recall, Precision, FPR."""
    y_all = df_test["target"].to_numpy(dtype=np.int64)
    p_all = np.asarray(y_prob, dtype=np.float64)
    pred_all = (p_all >= threshold).astype(np.int64)

    rows: list[dict[str, Any]] = []
    for col in group_cols:
        if col not in df_test.columns:
            continue
        series = df_test[col].fillna("MISSING").astype(str)
        for val in sorted(series.unique().tolist()):
            mask = (series == val).to_numpy()
            n_grp = int(mask.sum())
            y_g = y_all[mask]
            pred_g = pred_all[mask]
            n_pos = int(y_g.sum())
            tn, fp, fn, tp = confusion_matrix(y_g, pred_g, labels=[0, 1]).ravel().tolist()
            n_neg = tn + fp
            fpr = float(fp / n_neg) if n_neg > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rows.append(
                {
                    "group_col": col,
                    "group_value": val,
                    "n": n_grp,
                    "n_pos": n_pos,
                    "positive_rate": round(float(n_pos / n_grp), 4) if n_grp else 0.0,
                    "recall": round(rec, 4),
                    "precision": round(prec, 4),
                    "fpr": round(fpr, 4),
                    "low_sample_warning": bool(n_grp < min_group_n),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration_curve(
    y_calib: np.ndarray,
    raw_calib: np.ndarray,
    cal_calib: np.ndarray,
    y_test: np.ndarray,
    raw_test: np.ndarray,
    cal_test: np.ndarray,
    method_name: str,
    model_name: str,
    out_path: Path = FIGURES_DIR / "calibration_curve.png",
) -> None:
    """Save 2-panel reliability diagram (val_calib OOF vs 2014J Locked Test)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    panels = [
        (axes[0], y_calib, raw_calib, cal_calib, f"val_calib 5-Fold OOF (n={len(y_calib)})"),
        (axes[1], y_test, raw_test, cal_test, f"Locked Test 2014J (n={len(y_test)})"),
    ]
    for ax, y_t, p_raw, p_cal, title in panels:
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfect Calibration")
        frac_r, mean_r = calibration_curve(y_t, p_raw, n_bins=10, strategy="uniform")
        b_raw = brier_score_loss(y_t, p_raw)
        e_raw = compute_ece(y_t, p_raw, n_bins=10)
        ax.plot(
            mean_r,
            frac_r,
            marker="o",
            label=f"Raw (Brier={b_raw:.4f}, ECE={e_raw:.4f})",
        )
        if method_name != "uncalibrated":
            frac_c, mean_c = calibration_curve(y_t, p_cal, n_bins=10, strategy="uniform")
            b_cal = brier_score_loss(y_t, p_cal)
            e_cal = compute_ece(y_t, p_cal, n_bins=10)
            ax.plot(
                mean_c,
                frac_c,
                marker="s",
                label=f"{method_name.capitalize()} (Brier={b_cal:.4f}, ECE={e_cal:.4f})",
            )
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Positive Fraction")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Calibration Reliability Diagram — {model_name} (T=28 Operational, Method={method_name})",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_mlflow_core_runs(
    metrics_locked: dict[str, Any],
    db_path: Path = PROJECT_ROOT / "mlflow.db",
) -> dict[str, Any]:
    """Log 4 concise core runs to local SQLite MLflow tracking DB (Priority B, gitignored)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        return {"logged": False, "reason": "mlflow not installed"}

    uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("oulad-early-warning-p1-b5")

    run_ids: dict[str, str] = {}
    primary_models = metrics_locked["models_test_locked_2014j_t28"]
    for model_name in ("logistic_regression", "lightgbm", "tabular_mlp"):
        m = primary_models[model_name]["calibrated_or_final"]
        with mlflow.start_run(run_name=f"t28_{model_name}_locked_2014j") as run:
            mlflow.log_params(
                {
                    "cutoff": PRIMARY_CUTOFF,
                    "cohort": "operational",
                    "model": model_name,
                    "calibration_method": primary_models[model_name]["calibration_method"],
                    "threshold": m["threshold"],
                }
            )
            mlflow.log_metrics(
                {
                    "test_pr_auc": float(m["pr_auc"]),
                    "test_roc_auc": float(m["roc_auc"]),
                    "test_brier": float(m["brier_score"]),
                    "test_ece": float(m["ece_10bins"]),
                    "test_precision": float(m["precision"]),
                    "test_recall": float(m["recall"]),
                    "test_f1": float(m["f1"]),
                }
            )
            run_ids[model_name] = run.info.run_id

    sf = metrics_locked["sensitivity_checks_test_locked_2014j"]["score_free_date_graded_check"][
        "lightgbm"
    ]
    with mlflow.start_run(run_name="t28_lightgbm_score_free_2014j") as run:
        mlflow.log_params({"cutoff": PRIMARY_CUTOFF, "variant": "score_free"})
        mlflow.log_metrics(
            {
                "test_pr_auc": float(sf["pr_auc"]),
                "test_roc_auc": float(sf["roc_auc"]),
                "test_brier": float(sf["brier_score"]),
            }
        )
        run_ids["lightgbm_score_free"] = run.info.run_id

    return {
        "logged": True,
        "tracking_uri": "sqlite:///mlflow.db (gitignored)",
        "experiment": "oulad-early-warning-p1-b5",
        "run_ids": run_ids,
    }


def run_p1_b5_locked_evaluation_suite(
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    cutoffs: Sequence[int] = CUTOFFS,
    seed: int = SEED,
    enable_mlflow: bool = True,
) -> dict[str, Any]:
    """Execute Phase A (OOF calibration & lock on val_calib) then SINGLE unlock of 2014J.

    Strictly forbids any model selection, calibration fitting, or threshold tuning on 2014J.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PHASE A — TRAIN, SELECT, CALIBRATE & LOCK STRICTLY ON 2013 + 2014B ONLY
    # =========================================================================
    df_t28 = load_feature_parquet(PRIMARY_CUTOFF, mode="operational", processed_dir=processed_dir)
    split_t28: SplitData = presentation_forward_split(df_t28, seed=seed)
    split_t28.assert_invariants()
    assert TEST_PRESENTATION not in set(split_t28.train["code_presentation"])
    assert TEST_PRESENTATION not in set(split_t28.val_select["code_presentation"])
    assert TEST_PRESENTATION not in set(split_t28.val_calib["code_presentation"])

    X_tr, y_tr = split_feature_target(split_t28.train)
    X_vs, y_vs = split_feature_target(split_t28.val_select)
    X_vc, y_vc = split_feature_target(split_t28.val_calib)

    # 1. Fit/load the 3 primary models strictly on Train (+ early stopping on val_select)
    logreg_model = train_logistic_regression(X_tr, y_tr, seed=seed)
    lgbm_raw, lgbm_cats = train_lightgbm(X_tr, y_tr, X_vs, y_vs, seed=seed)
    lgbm_wrapper = LGBMSklearnWrapper(lgbm_raw, lgbm_cats)

    mlp_ckpt = artifacts_dir / "tabular_mlp_best.pt"
    if mlp_ckpt.exists():
        mlp_model, mlp_prep = load_torch_checkpoint(mlp_ckpt)
    else:
        mlp_res = train_tabular_mlp(X_tr, y_tr, X_vs, y_vs, seed=seed, prefer_cuda=False)
        save_torch_checkpoint(mlp_res, mlp_ckpt)
        mlp_model, mlp_prep = mlp_res.model, mlp_res.preprocessor
    mlp_wrapper = TorchSklearnWrapper(mlp_model, mlp_prep)

    primary_estimators: dict[str, Any] = {
        "logistic_regression": logreg_model,
        "lightgbm": lgbm_wrapper,
        "tabular_mlp": mlp_wrapper,
    }

    # Select champion strictly on val_select PR-AUC (never val_calib or 2014J)
    val_select_pr: dict[str, float] = {}
    for name, est in primary_estimators.items():
        p_vs = np.asarray(est.predict_proba(X_vs)[:, 1], dtype=np.float64)
        val_select_pr[name] = round(float(average_precision_score(y_vs, p_vs)), 4)
    champion_name = max(val_select_pr, key=lambda k: val_select_pr[k])

    # 2. Explicit StratifiedKFold(5) OOF calibration + threshold selection strictly on val_calib
    oof_cal_results: dict[str, OOFCalibrationResult] = {}
    thresh_results: dict[str, ThresholdSelectionResult] = {}
    for name, est in primary_estimators.items():
        cal_res = select_best_calibrator_oof(est, X_vc, y_vc, n_splits=5, seed=seed)
        th_res = select_threshold_with_fallback(
            y_vc, cal_res.oof_probs_winner, target_precision=0.70, min_predicted_positives=30
        )
        oof_cal_results[name] = cal_res
        thresh_results[name] = th_res

    champion_cal = oof_cal_results[champion_name]
    champion_tau = thresh_results[champion_name].threshold

    # 3. Train & lock all 12 cutoff models (operational & common x 14/21/28) BEFORE unlocking 2014J
    locked_cutoff_bundles: list[dict[str, Any]] = []
    for mode in ("operational", "common"):
        mode_typed: CohortMode = mode  # type: ignore[assignment]
        for cutoff in cutoffs:
            df_c = load_feature_parquet(cutoff, mode=mode_typed, processed_dir=processed_dir)
            sp_c = presentation_forward_split(df_c, seed=seed)
            sp_c.assert_invariants()
            X_tr_c, y_tr_c = split_feature_target(sp_c.train)
            X_vs_c, y_vs_c = split_feature_target(sp_c.val_select)
            lr_c = train_logistic_regression(X_tr_c, y_tr_c, seed=seed)
            gb_c, cats_c = train_lightgbm(X_tr_c, y_tr_c, X_vs_c, y_vs_c, seed=seed)
            locked_cutoff_bundles.append(
                {
                    "mode": mode,
                    "cutoff": cutoff,
                    "split": sp_c,
                    "logreg": lr_c,
                    "lgbm": LGBMSklearnWrapper(gb_c, cats_c),
                }
            )

    # 4. Train & lock sensitivity models (Ablation & Score-Free) BEFORE unlocking 2014J
    def _fit_dropped_lgbm(drop_cols: Sequence[str]) -> LGBMSklearnWrapper:
        cols = [c for c in drop_cols if c in X_tr.columns]
        gb_d, cats_d = train_lightgbm(
            X_tr.drop(columns=cols), y_tr, X_vs.drop(columns=cols), y_vs, seed=seed
        )
        return LGBMSklearnWrapper(gb_d, cats_d)

    locked_ablation_lgbm = _fit_dropped_lgbm(SENSITIVE_DEMOGRAPHIC_COLS)
    locked_score_free_lgbm = _fit_dropped_lgbm(SCORE_COLS)

    # Persist locked champion bundle (gitignored) for P1-B6 CLI demo (predict --sample)
    champion_bundle_path = artifacts_dir / "champion_locked.joblib"
    joblib.dump(
        {
            "champion_name": champion_name,
            "base_model": primary_estimators[champion_name],
            "calibrator": champion_cal.final_calibrator,
            "calibration_method": champion_cal.chosen_method,
            "threshold": champion_tau,
            "feature_columns": list(X_tr.columns),
            "cutoff": PRIMARY_CUTOFF,
            "cohort_mode": "operational",
        },
        champion_bundle_path,
    )

    # =========================================================================
    # PHASE B — SINGLE FINAL UNLOCK OF LOCKED TEST 2014J (READ-ONLY EVALUATION)
    # =========================================================================
    X_test, y_test = split_feature_target(split_t28.test_locked)
    y_test_np = y_test.to_numpy(dtype=np.int64)

    test_models_report: dict[str, Any] = {}
    returning_vs_new_test: dict[str, Any] = {}
    champion_raw_test: np.ndarray | None = None
    champion_cal_test: np.ndarray | None = None

    for name, est in primary_estimators.items():
        cal_res = oof_cal_results[name]
        tau_m = thresh_results[name].threshold
        p_raw = np.asarray(est.predict_proba(X_test)[:, 1], dtype=np.float64)
        p_cal = apply_calibrator(est, cal_res.final_calibrator, X_test)

        raw_m = compute_classification_metrics(y_test_np, p_raw, threshold=tau_m)
        cal_m = compute_classification_metrics(y_test_np, p_cal, threshold=tau_m)
        test_models_report[name] = {
            "calibration_method": cal_res.chosen_method,
            "locked_threshold": tau_m,
            "uncalibrated_raw": raw_m,
            "calibrated_or_final": cal_m,
            "brier_delta_post_minus_pre": round(
                float(cal_m["brier_score"]) - float(raw_m["brier_score"]), 4
            ),
            "ece_delta_post_minus_pre": round(
                float(cal_m["ece_10bins"]) - float(raw_m["ece_10bins"]), 4
            ),
        }
        returning_vs_new_test[name] = evaluate_returning_vs_new(
            split_t28.test_locked, split_t28.train, p_cal
        )
        if name == champion_name:
            champion_raw_test = p_raw
            champion_cal_test = p_cal

    assert champion_raw_test is not None and champion_cal_test is not None

    # 12 locked cutoff models evaluated once on 2014J -> cutoff_comparison_test.csv
    cutoff_test_rows: list[dict[str, Any]] = []
    for bundle in locked_cutoff_bundles:
        sp_c: SplitData = bundle["split"]
        X_te_c, y_te_c = split_feature_target(sp_c.test_locked)
        for m_name, m_est in (
            ("logistic_regression", bundle["logreg"]),
            ("lightgbm", bundle["lgbm"]),
        ):
            p_c = np.asarray(m_est.predict_proba(X_te_c)[:, 1], dtype=np.float64)
            m_dict = compute_binary_metrics(y_te_c, p_c)
            cutoff_test_rows.append(
                {
                    "mode": bundle["mode"],
                    "cutoff": bundle["cutoff"],
                    "model": m_name,
                    "n_test_2014j": m_dict["n"],
                    "n_features": int(X_te_c.shape[1]),
                    "positive_rate": m_dict["positive_rate"],
                    "pr_auc": m_dict["pr_auc"],
                    "roc_auc": m_dict["roc_auc"],
                    "brier_score": m_dict["brier_score"],
                }
            )
    cutoff_test_df = pd.DataFrame(cutoff_test_rows)
    cutoff_test_df.to_csv(reports_dir / "cutoff_comparison_test.csv", index=False)

    # Sensitivity checks evaluated once on 2014J
    full_lgbm_test_pr = float(
        test_models_report["lightgbm"]["uncalibrated_raw"]["pr_auc"]
    )
    full_lgbm_test_roc = float(
        test_models_report["lightgbm"]["uncalibrated_raw"]["roc_auc"]
    )

    X_test_abl = X_test.drop(columns=[c for c in SENSITIVE_DEMOGRAPHIC_COLS if c in X_test.columns])
    p_abl_test = np.asarray(locked_ablation_lgbm.predict_proba(X_test_abl)[:, 1], dtype=np.float64)
    abl_test_m = compute_binary_metrics(y_test_np, p_abl_test)

    X_test_sf = X_test.drop(columns=[c for c in SCORE_COLS if c in X_test.columns])
    p_sf_test = np.asarray(locked_score_free_lgbm.predict_proba(X_test_sf)[:, 1], dtype=np.float64)
    sf_test_m = compute_binary_metrics(y_test_np, p_sf_test)

    # Illustrative Cost-Benefit, Post-Lock Error Analysis (min(20, len)), & Subgroup Fairness
    cost_benefit = build_illustrative_cost_benefit_table(
        y_test_np, champion_cal_test, locked_threshold=champion_tau
    )
    error_audit = extract_post_lock_error_samples(
        split_t28.test_locked, champion_cal_test, threshold=champion_tau, max_k=20
    )
    fairness_df = compute_subgroup_fairness(
        split_t28.test_locked, champion_cal_test, threshold=champion_tau, min_group_n=30
    )
    fairness_df.to_csv(reports_dir / "fairness_test.csv", index=False)

    # Reliability diagram (val_calib OOF vs 2014J Locked Test)
    plot_calibration_curve(
        y_vc.to_numpy(dtype=np.int64),
        champion_cal.oof_probs_raw,
        champion_cal.oof_probs_winner,
        y_test_np,
        champion_raw_test,
        champion_cal_test,
        method_name=champion_cal.chosen_method,
        model_name=champion_name,
        out_path=figures_dir / "calibration_curve.png",
    )

    metrics_locked: dict[str, Any] = {
        "evaluation_protocol": {
            "split_name": "Presentation-Based Forward Split (AUD-07)",
            "locked_test_presentation": TEST_PRESENTATION,
            "single_unlock_discipline": True,
            "locked_before_test_unlock": True,
            "post_unlock_model_edits": 0,
            "champion_selected_on_val_select": champion_name,
            "val_select_pr_auc_ranking": val_select_pr,
            "n_test_2014j": int(len(y_test_np)),
            "test_positive_rate": round(float(y_test_np.mean()), 4),
            "val_calib_positive_rate": round(float(y_vc.mean()), 4),
            "seed": seed,
        },
        "oof_calibration_val_calib": {
            name: {
                "n_calib": res.n_calib,
                "min_fold_train": res.min_fold_train,
                "isotonic_blocked": res.isotonic_blocked,
                "candidate_oof_metrics": res.candidate_metrics,
                "chosen_method": res.chosen_method,
                "threshold_selection_oof": {
                    "threshold": thresh_results[name].threshold,
                    "target_precision": thresh_results[name].target_precision,
                    "min_predicted_positives": thresh_results[name].min_predicted_positives,
                    "target_precision_met": thresh_results[name].target_precision_met,
                    "fallback_used": thresh_results[name].fallback_used,
                    "max_achievable_precision": thresh_results[name].max_achievable_precision,
                    "oof_precision": thresh_results[name].oof_precision,
                    "oof_recall": thresh_results[name].oof_recall,
                    "oof_f1": thresh_results[name].oof_f1,
                    "oof_n_predicted_positives": thresh_results[name].oof_n_predicted_positives,
                },
            }
            for name, res in oof_cal_results.items()
        },
        "models_test_locked_2014j_t28": test_models_report,
        "returning_vs_new_test_locked_2014j_t28": returning_vs_new_test,
        "sensitivity_checks_test_locked_2014j": {
            "sensitive_demographic_ablation": {
                "dropped_columns": list(SENSITIVE_DEMOGRAPHIC_COLS),
                "lightgbm": abl_test_m,
                "delta_pr_auc_vs_full_raw": round(
                    float(abl_test_m["pr_auc"]) - full_lgbm_test_pr, 4
                ),
                "delta_roc_auc_vs_full_raw": round(
                    float(abl_test_m["roc_auc"]) - full_lgbm_test_roc, 4
                ),
            },
            "score_free_date_graded_check": {
                "dropped_columns": list(SCORE_COLS),
                "lightgbm": sf_test_m,
                "delta_pr_auc_vs_full_raw": round(
                    float(sf_test_m["pr_auc"]) - full_lgbm_test_pr, 4
                ),
                "delta_roc_auc_vs_full_raw": round(
                    float(sf_test_m["roc_auc"]) - full_lgbm_test_roc, 4
                ),
            },
        },
        "illustrative_cost_benefit_2014j": cost_benefit,
        "post_lock_error_analysis_2014j": error_audit,
        "fairness_summary_2014j": {
            "csv_path": "fairness_test.csv",
            "n_subgroups_evaluated": int(len(fairness_df)),
            "n_low_sample_warnings": int(fairness_df["low_sample_warning"].sum()),
        },
        "artifacts": {
            "calibration_curve": "calibration_curve.png",
            "cutoff_comparison_test_csv": "cutoff_comparison_test.csv",
            "fairness_test_csv": "fairness_test.csv",
            "champion_bundle": "artifacts/champion_locked.joblib (gitignored)",
        },
    }

    if enable_mlflow:
        metrics_locked["mlflow_tracking"] = log_mlflow_core_runs(metrics_locked)

    (reports_dir / "metrics_test_locked.json").write_text(
        json.dumps(metrics_locked, indent=2), encoding="utf-8"
    )
    return metrics_locked


if __name__ == "__main__":
    report = run_p1_b5_locked_evaluation_suite()
    print("=== P1-B5 Locked Test (2014J) Evaluation Complete ===")
    print("Champion:", report["evaluation_protocol"]["champion_selected_on_val_select"])
    print(json.dumps(report["models_test_locked_2014j_t28"], indent=2))
