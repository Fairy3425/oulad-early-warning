"""Baseline models (Logistic Regression, LightGBM), SHAP & sensitivity checks (P1-B3).

Enforces:
- Train-only preprocessing fit (Imputer, StandardScaler, OneHotEncoder, LGBM categories).
- LightGBM mandatory subsample_freq=1 (enabling row bagging with subsample=0.8).
- Separate model trained per cutoff T in {14, 21, 28} and cohort mode (operational vs common).
- SHAP TreeExplainer strictly as predictive attribution (NOT causation) on val_select (<= 2000).
- 2014J locked test is NEVER touched anywhere in P1-B3.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oulad.config import (
    CUTOFFS,
    FIGURES_DIR,
    PRIMARY_CUTOFF,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    TEST_PRESENTATION,
)
from oulad.features import CohortMode, split_feature_target
from oulad.split import (
    SplitData,
    get_returning_mask,
    load_feature_parquet,
    presentation_forward_split,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORICAL_COLS: tuple[str, ...] = (
    "code_module",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
)

# Priority B sensitivity checks
SENSITIVE_DEMOGRAPHIC_COLS: tuple[str, ...] = ("gender", "disability", "imd_band")
SCORE_COLS: tuple[str, ...] = ("mean_score", "has_score")


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into (numeric_cols, categorical_cols) preserving order."""
    cat_set = set(CATEGORICAL_COLS)
    cat_cols = [c for c in X.columns if c in cat_set]
    num_cols = [c for c in X.columns if c not in cat_set]
    return num_cols, cat_cols


def compute_binary_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float | int | None]:
    """Compute core probabilistic ranking and calibration metrics for binary classification."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    n_rows = int(len(y_arr))
    n_pos = int(y_arr.sum())
    pos_rate = float(n_pos / n_rows) if n_rows > 0 else 0.0

    has_both_classes = len(np.unique(y_arr)) == 2
    pr_auc = float(average_precision_score(y_arr, p_arr)) if has_both_classes else None
    roc_auc = float(roc_auc_score(y_arr, p_arr)) if has_both_classes else None
    brier = float(brier_score_loss(y_arr, p_arr)) if n_rows > 0 else None

    return {
        "n": n_rows,
        "n_pos": n_pos,
        "positive_rate": round(pos_rate, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
    }


def evaluate_returning_vs_new(
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Evaluate metrics separately for Returning vs New students (without using id_student)."""
    mask_ret = get_returning_mask(eval_df, train_df).to_numpy()
    y_true = eval_df["target"].to_numpy()
    n_total = int(len(eval_df))
    n_ret = int(mask_ret.sum())

    return {
        "returning_pct": round(100.0 * n_ret / n_total, 2) if n_total > 0 else 0.0,
        "returning": compute_binary_metrics(y_true[mask_ret], y_prob[mask_ret]),
        "new": compute_binary_metrics(y_true[~mask_ret], y_prob[~mask_ret]),
    }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = SEED,
) -> Pipeline:
    """Fit Logistic Regression with strictly train-only imputation, scaling, and OHE."""
    num_cols, cat_cols = infer_feature_types(X_train)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def prepare_lgbm_frame(
    X: pd.DataFrame,
    cat_cols: Sequence[str],
    cat_categories: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Cast categorical columns using strictly train-learned vocabularies for LightGBM.

    Unseen categories in validation/test map to NaN, which LightGBM routes cleanly.
    """
    out = X.copy()
    learned_cats: dict[str, list[str]] = {} if cat_categories is None else dict(cat_categories)

    for col in cat_cols:
        if col not in out.columns:
            continue
        series = out[col].fillna("MISSING").astype(str)
        if cat_categories is None:
            cats = sorted(series.unique().tolist())
            learned_cats[col] = cats
        else:
            cats = learned_cats[col]
            series = series.where(series.isin(cats), np.nan)
        out[col] = pd.Categorical(series, categories=cats)

    return out, learned_cats


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = SEED,
    stopping_rounds: int = 100,
) -> tuple[lgb.LGBMClassifier, dict[str, list[str]]]:
    """Fit LightGBM (subsample_freq=1) with early stopping strictly on val_select."""
    _, cat_cols = infer_feature_types(X_train)
    X_tr_prep, cat_categories = prepare_lgbm_frame(X_train, cat_cols)
    X_va_prep, _ = prepare_lgbm_frame(X_val, cat_cols, cat_categories=cat_categories)

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,  # Mandatory per LightGBM docs to actually enable row bagging
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )
    model.fit(
        X_tr_prep,
        y_train,
        eval_X=X_va_prep,
        eval_y=y_val,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
    )
    return model, cat_categories


def predict_lgbm_proba(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    cat_categories: dict[str, list[str]],
) -> np.ndarray:
    """Predict positive-class probabilities using frozen train-only categorical schema."""
    _, cat_cols = infer_feature_types(X)
    X_prep, _ = prepare_lgbm_frame(X, cat_cols, cat_categories=cat_categories)
    return np.asarray(model.predict_proba(X_prep)[:, 1], dtype=np.float64)


def _extract_positive_shap_values(raw_shap: Any) -> np.ndarray:
    """Normalize TreeExplainer SHAP output to a 2D (n_samples, n_features) array for class 1."""
    if isinstance(raw_shap, list):
        arr = np.asarray(raw_shap[1] if len(raw_shap) == 2 else raw_shap[0], dtype=np.float64)
    else:
        arr = np.asarray(raw_shap, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr


def generate_shap_summary(
    lgbm_model: lgb.LGBMClassifier,
    X_val_prep: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "shap_summary.png",
    max_samples: int = 2000,
    top_n: int = 10,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute TreeExplainer SHAP values on val_select (<= 2000) and save beeswarm + bar plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_sample = min(max_samples, len(X_val_prep))
    X_sample = (
        X_val_prep.sample(n=n_sample, random_state=seed)
        if len(X_val_prep) > n_sample
        else X_val_prep.copy()
    )

    explainer = shap.TreeExplainer(lgbm_model)
    shap_vals = _extract_positive_shap_values(explainer.shap_values(X_sample))
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top_features = [
        {"feature": str(X_sample.columns[i]), "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order[:top_n]
    ]

    # Plot 2-panel figure: Left = Beeswarm top 10, Right = Mean |SHAP| bar top 10
    X_plot = X_sample.copy()
    for col in X_plot.columns:
        if isinstance(X_plot[col].dtype, pd.CategoricalDtype):
            X_plot[col] = X_plot[col].cat.codes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    plt.sca(axes[0])
    shap.summary_plot(
        shap_vals,
        X_plot,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    axes[0].set_title(f"Top {top_n} SHAP Beeswarm (val_select, n={n_sample})", fontsize=11)

    top_names = [item["feature"] for item in top_features][::-1]
    top_vals = [item["mean_abs_shap"] for item in top_features][::-1]
    axes[1].barh(top_names, top_vals, color="#2563eb")
    axes[1].set_xlabel("Mean |SHAP value| (log-odds magnitude)")
    axes[1].set_title(f"Top {top_n} Global Predictive Attribution", fontsize=11)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        "LightGBM (T=28 Operational) — Predictive Attribution ONLY (SHAP != Causation)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_samples": n_sample,
        "figure_path": str(out_path.name),
        "disclaimer": (
            "SHAP values represent predictive attribution (what the model relied on), "
            "NOT causal drivers of dropout."
        ),
        "top_10_features": top_features,
    }


def _train_and_eval_pair(
    split: SplitData,
    drop_feature_cols: Sequence[str] = (),
    seed: int = SEED,
) -> dict[str, Any]:
    """Train LogReg & LightGBM strictly on split.train and evaluate strictly on split.val_select."""
    # Hard safety guard: never allow 2014J inside train or val_select
    split.assert_invariants()
    assert TEST_PRESENTATION not in set(split.train["code_presentation"])
    assert TEST_PRESENTATION not in set(split.val_select["code_presentation"])

    X_train, y_train = split_feature_target(split.train)
    X_val, y_val = split_feature_target(split.val_select)

    if drop_feature_cols:
        cols_to_drop = [c for c in drop_feature_cols if c in X_train.columns]
        X_train = X_train.drop(columns=cols_to_drop)
        X_val = X_val.drop(columns=cols_to_drop)

    logreg = train_logistic_regression(X_train, y_train, seed=seed)
    logreg_prob = logreg.predict_proba(X_val)[:, 1]
    logreg_metrics = compute_binary_metrics(y_val, logreg_prob)

    lgbm, cat_cats = train_lightgbm(X_train, y_train, X_val, y_val, seed=seed)
    lgbm_prob = predict_lgbm_proba(lgbm, X_val, cat_cats)
    lgbm_metrics = compute_binary_metrics(y_val, lgbm_prob)
    lgbm_metrics["best_iteration"] = int(lgbm.best_iteration_ or lgbm.n_estimators)

    return {
        "n_features": int(X_train.shape[1]),
        "logreg": logreg,
        "logreg_prob": logreg_prob,
        "logreg_metrics": logreg_metrics,
        "lgbm": lgbm,
        "lgbm_cats": cat_cats,
        "lgbm_prob": lgbm_prob,
        "lgbm_metrics": lgbm_metrics,
        "X_val": X_val,
    }


def run_p1_b3_validation_suite(
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    cutoffs: Sequence[int] = CUTOFFS,
    seed: int = SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run full P1-B3 modeling, cutoff comparison, SHAP, Returning/New & sensitivity checks.

    Strictly operates on Train (2013B+2013J) and val_select (60% of 2014B).
    val_calib (40% of 2014B) and test_locked (2014J) remain completely untouched.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cutoff_rows: list[dict[str, Any]] = []
    primary_split: SplitData | None = None
    primary_result: dict[str, Any] | None = None

    for mode in ("operational", "common"):
        mode_typed: CohortMode = mode  # type: ignore[assignment]
        for cutoff in cutoffs:
            df = load_feature_parquet(cutoff, mode=mode_typed, processed_dir=processed_dir)
            split = presentation_forward_split(df, seed=seed)
            res = _train_and_eval_pair(split, seed=seed)

            for model_name, m_dict in (
                ("logistic_regression", res["logreg_metrics"]),
                ("lightgbm", res["lgbm_metrics"]),
            ):
                cutoff_rows.append(
                    {
                        "mode": mode,
                        "cutoff": cutoff,
                        "model": model_name,
                        "n_train": int(len(split.train)),
                        "n_val_select": int(m_dict["n"]),
                        "n_features": res["n_features"],
                        "positive_rate": m_dict["positive_rate"],
                        "pr_auc": m_dict["pr_auc"],
                        "roc_auc": m_dict["roc_auc"],
                        "brier_score": m_dict["brier_score"],
                        "best_iteration": m_dict.get("best_iteration"),
                    }
                )

            if mode == "operational" and cutoff == PRIMARY_CUTOFF:
                primary_split = split
                primary_result = res

    assert primary_split is not None and primary_result is not None

    # Returning vs New breakdown on val_select (T=28 Operational)
    returning_vs_new = {
        "logistic_regression": evaluate_returning_vs_new(
            primary_split.val_select, primary_split.train, primary_result["logreg_prob"]
        ),
        "lightgbm": evaluate_returning_vs_new(
            primary_split.val_select, primary_split.train, primary_result["lgbm_prob"]
        ),
    }

    # SHAP TreeExplainer on val_select (<= 2000 rows, non-causal)
    _, cat_cols = infer_feature_types(primary_result["X_val"])
    X_val_lgbm_prep, _ = prepare_lgbm_frame(
        primary_result["X_val"], cat_cols, cat_categories=primary_result["lgbm_cats"]
    )
    shap_info = generate_shap_summary(
        primary_result["lgbm"],
        X_val_lgbm_prep,
        out_path=figures_dir / "shap_summary.png",
        max_samples=2000,
        seed=seed,
    )

    # Priority B Sensitivity 1: Sensitive-Demographic Ablation (drop gender, disability, imd_band)
    ablation_res = _train_and_eval_pair(
        primary_split, drop_feature_cols=SENSITIVE_DEMOGRAPHIC_COLS, seed=seed
    )
    full_lgbm_pr = float(primary_result["lgbm_metrics"]["pr_auc"])
    full_lgbm_roc = float(primary_result["lgbm_metrics"]["roc_auc"])
    ablation_lgbm_pr = float(ablation_res["lgbm_metrics"]["pr_auc"])
    ablation_lgbm_roc = float(ablation_res["lgbm_metrics"]["roc_auc"])

    # Priority B Sensitivity 2: Score-Free Sensitivity Check (AUD-02 missing date_graded)
    score_free_res = _train_and_eval_pair(
        primary_split, drop_feature_cols=SCORE_COLS, seed=seed
    )
    sf_lgbm_pr = float(score_free_res["lgbm_metrics"]["pr_auc"])
    sf_lgbm_roc = float(score_free_res["lgbm_metrics"]["roc_auc"])

    metrics_val: dict[str, Any] = {
        "split_protocol": {
            "name": "Presentation-Based Forward Split (AUD-07)",
            "calendar_overlap_disclosure": (
                "2013J (Oct 2013-Jun 2014) overlaps ~4 months with 2014B (Feb 2014-Oct 2014). "
                "Train 2013 -> Locked Test 2014J (Oct 2014) has zero calendar overlap."
            ),
            "train_presentations": ["2013B", "2013J"],
            "val_presentation": "2014B (60% val_select / 40% val_calib)",
            "test_presentation_locked": TEST_PRESENTATION,
            "test_2014j_untouched": True,
            "val_calib_untouched_in_p1_b3": True,
            "seed": seed,
            "primary_cutoff": PRIMARY_CUTOFF,
            "primary_cohort_mode": "operational",
            "split_sizes_t28_operational": {
                "train": int(len(primary_split.train)),
                "val_select": int(len(primary_split.val_select)),
                "val_calib": int(len(primary_split.val_calib)),
                "test_locked": int(len(primary_split.test_locked)),
            },
        },
        "models_val_select_t28_operational": {
            "logistic_regression": primary_result["logreg_metrics"],
            "lightgbm": primary_result["lgbm_metrics"],
        },
        "returning_vs_new_val_select_t28": returning_vs_new,
        "shap_val_select_t28_lightgbm": shap_info,
        "sensitivity_checks_val_select_t28": {
            "sensitive_demographic_ablation": {
                "dropped_columns": list(SENSITIVE_DEMOGRAPHIC_COLS),
                "n_features": ablation_res["n_features"],
                "lightgbm": ablation_res["lgbm_metrics"],
                "delta_pr_auc_vs_full": round(ablation_lgbm_pr - full_lgbm_pr, 4),
                "delta_roc_auc_vs_full": round(ablation_lgbm_roc - full_lgbm_roc, 4),
                "note": (
                    "Removing sensitive columns does not guarantee fairness due to proxy "
                    "variables; subgroup fairness is evaluated directly in P1-B5."
                ),
            },
            "score_free_date_graded_check": {
                "dropped_columns": list(SCORE_COLS),
                "n_features": score_free_res["n_features"],
                "lightgbm": score_free_res["lgbm_metrics"],
                "delta_pr_auc_vs_full": round(sf_lgbm_pr - full_lgbm_pr, 4),
                "delta_roc_auc_vs_full": round(sf_lgbm_roc - full_lgbm_roc, 4),
                "note": (
                    "OULAD lacks date_graded; Score-Free provides a conservative lower bound "
                    "when tutor-marked grading latency exceeds cutoff T=28."
                ),
            },
        },
    }

    cutoff_df = pd.DataFrame(cutoff_rows)
    cutoff_df.to_csv(reports_dir / "cutoff_comparison_val.csv", index=False)
    with (reports_dir / "metrics_val.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_val, fh, indent=2)

    return metrics_val, cutoff_df


if __name__ == "__main__":
    summary, table = run_p1_b3_validation_suite()
    print("Saved reports/metrics_val.json, reports/cutoff_comparison_val.csv, shap_summary.png")
    print(table.to_string(index=False))
