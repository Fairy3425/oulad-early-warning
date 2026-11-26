# Methodology & Evaluation Reference

This document provides the full technical details behind the cohort definitions, anti-leakage controls, validation/test split protocol, modeling benchmarks, out-of-fold calibration, sensitivity analyses, and illustrative cost–benefit calculations for `oulad-early-warning`.

For a high-level summary, see [README.md](../README.md). For intended/prohibited use, ethical risks, and the full 18-subgroup performance audit, see [MODEL_CARD.md](../MODEL_CARD.md).

---

## 1. Data Ingestion & Automated SQL Quality Audit

Raw OULAD tables (7 CSV files; Kuzilek et al., 2017, CC BY 4.0) are loaded into DuckDB (`data/oulad.duckdb`) using `read_csv_auto(..., nullstr='?', header=true)` so missing `?` entries are parsed as SQL `NULL` rather than coercing numeric columns to strings.

- **Ingested Row Counts** ([reports/data_quality.json](../reports/data_quality.json)):
  - `studentInfo`: `32,593` · `studentRegistration`: `32,593` · `studentVle`: `10,655,280` · `studentAssessment`: `173,912` · `assessments`: `206` · `vle`: `6,364` · `courses`: `22`.
- **9 Automated SQL Quality Checks (`0` violations across all 9):**
  1. `duplicate_student_info_keys`: composite key uniqueness on `(code_module, code_presentation, id_student)`.
  2. `invalid_final_result`: domain check `final_result IN {'Pass', 'Fail', 'Withdrawn', 'Distinction'}`.
  3. `orphan_student_vle_sites`: 3-column referential integrity `(code_module, code_presentation, id_site)` from `studentVle` to `vle`.
  4. `orphan_student_assessments`: referential integrity on `id_assessment` from `studentAssessment` to `assessments`.
  5. `invalid_assessment_scores`: score range `[0, 100]` or `NULL`.
  6. `non_positive_sum_click`: strictly positive click counts (`sum_click > 0`).
  7. `vle_date_after_course_end`: `studentVle.date <= courses.module_presentation_length`.
  8. `imd_band_missing_ge_5pct`: `imd_band` missing rate strictly `< 5%` (observed: `1,111 / 32,593 = 3.41%`).
  9. `unregistration_before_registration`: no record with `date_unregistration < date_registration`.
- **Registration Profile & Returning Students:**
  - `date_registration IS NULL`: `45` rows; `date_registration >= 14`: `59` rows; `>= 21`: `26` rows; `>= 28`: `18` rows (excluded from cohorts where registration falls at or after cutoff).
  - Students appearing in `2013B + 2013J` who re-enroll in later presentations are tracked explicitly as a valid `Returning` segment rather than dropped: `2014B` has `12.34%` returning enrollments (`963 / 7,804`); `2014J` has `8.87%` returning enrollments (`999 / 11,260`).

---

## 2. Start-of-Day Cohort Rules & Anti-Leakage Controls

Predictions are generated at the **Start of Day $T$** (using only events strictly `< T`, equivalent to End-of-Day $T - 1$) for $T \in \{14, 21, 28\}$, with **$T = 28$** (end of Week 4) as the primary horizon.

### 2.1 Operational Cohort vs. Safe Common Cohort

| Cohort Mode | Inclusion Criteria | Role in Evaluation | Total Enrollments ([reports/cohort_summary.json](../reports/cohort_summary.json)) |
| :--- | :--- | :--- | :---: |
| **Operational Cohort** *(Primary)* | `date_registration < T` AND `(date_unregistration IS NULL OR date_unregistration >= T)` | Represents real-world deployment at Day $T$; separate models are trained per cutoff $T$. | `T=14`: `28,062` · `T=21`: `27,880` · `T=28`: `27,528` |
| **Safe Common Cohort** *(Retrospective Sensitivity)* | `date_registration < 14` AND `(date_unregistration IS NULL OR date_unregistration >= 28)` | Isolates the incremental value of Week 3–4 signals on a fixed population while preventing post-Day-14 registration leakage. | `27,491` across `T=14, 21, 28` |

> **Methodological Caveat:** Because the Safe Common Cohort retrospectively conditions on students remaining enrolled through Day 28, it does **not** represent operational Day-14 deployment performance. All primary operational claims use the **Operational Cohort**.

### 2.2 Leakage-Free Feature Engineering

All behavioral and assessment queries join on the 3-column enrollment key `(code_module, code_presentation, id_student)` and strictly enforce `< T`:

- **VLE Engagement (`studentVle.date < T`):** `has_any_vle_click`, `clicks_pre_course` (`date < 0`), `clicks_total`, `active_days`, `n_unique_sites`, `std_daily_clicks`, `first_click_day`, `last_click_day`, `days_since_last_click` (`(T - 1) - MAX(date)`), complete-week click totals (`clicks_w0`, `clicks_w1`, plus `clicks_w2` for $T \ge 21$ and `clicks_w3` for $T \ge 28$), smoothed `click_trend_ratio`, and **top-5 `activity_type` columns fitted strictly on Train (`2013B + 2013J`, `date < T`)**: `forumng`, `homepage`, `oucontent`, `subpage`, `quiz`.
- **Assessment Progress (`studentAssessment.date_submitted < T`):**
  - `n_due_before_cutoff`: assessments in the same `(code_module, code_presentation)` with `assessments.date IS NOT NULL AND assessments.date < T`.
  - `n_submitted_due`: non-banked (`is_banked = 0`) submissions with both `date_submitted < T` **and** `assessments.date < T`.
  - `submission_rate_due = n_submitted_due / n_due_before_cutoff` (when denominator `> 0`, else `NaN`), strictly bounded in `[0.0, 1.0]`.
  - Separated counts: `n_submitted_early` (non-banked submitted `< T` for due dates `>= T` or `NULL`) and `n_banked` (`is_banked = 1`).
  - `mean_score`, `has_score`, and `mean_days_late` (`date_submitted - assessments.date` for non-banked assessments with due dates).
- **Demographics & Registration:** `code_module`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `num_of_prev_attempts`, `studied_credits`, `disability`, `date_registration`.
- **Forbidden Post-Outcome Columns:** `final_result` and `date_unregistration` are excluded from all feature tables and verified absent by automated tests. Runtime cohort/rate guards raise explicit `RuntimeError` so they remain active even under `python -O`.

---

## 3. Presentation-Based Forward Split & Calendar Overlap

To simulate forward deployment across academic terms (`seed = 42`, `T = 28 Operational`):

| Split | Presentations | Rows (`n`) | Positive Rate | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Train** | `2013B + 2013J` | `11,806` | — | Model fitting, imputation, scaling, encoding, top-5 VLE activity selection |
| **Validation Select (`val_select`)** | `60%` stratified of `2014B` | `3,905` | `48.43%` | Model selection, early stopping (LightGBM & PyTorch MLP), SHAP attribution |
| **Validation Calibration (`val_calib`)** | `40%` stratified of `2014B` | `2,604` | `48.43%` | Explicit 5-fold out-of-fold (OOF) probability calibration & operating threshold `tau*` |
| **Frozen Historical Test (`2014J`)** | `2014J` | `9,213` | `40.79%` | Held out from training, selection, calibration, and thresholding |

- **Calendar Overlap Disclosure:** `2013J` (Oct 2013–Jun 2014) overlaps ~4 calendar months with `2014B` (Feb 2014–Oct 2014), so this split is accurately termed a **Presentation-Based Forward Split** rather than strictly chronological. Between `Train (2013B + 2013J)` and `Test (2014J, Oct 2014)` there is zero calendar overlap.
- **Historical `2014J` Provenance:** The committed `2014J` evaluation artifacts ([reports/metrics_test_locked.json](../reports/metrics_test_locked.json), [reports/cutoff_comparison_test.csv](../reports/cutoff_comparison_test.csv), [reports/fairness_test.csv](../reports/fairness_test.csv)) were created in commit `2c8bab8` and have remained unchanged across subsequent Git history. Repository records alone do not independently establish the exact number of local evaluation executions prior to `2c8bab8`.

---

## 4. Model Architectures & Calibration Protocol

### 4.1 Three Benchmarked Models (`T = 28 Operational`, `seed = 42`)

1. **Logistic Regression (`src/oulad/baselines.py`):** `ColumnTransformer` (`SimpleImputer(median)` + `StandardScaler` for numerics; `MISSING` + `OneHotEncoder(handle_unknown="ignore")` for categoricals) + `LogisticRegression(max_iter=2000)`.
2. **LightGBM (`src/oulad/baselines.py`):** `LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, deterministic=True)` with `early_stopping(100)` on `val_select` (best iteration: `218`) and strictly Train-only categorical vocabularies.
3. **PyTorch `TabularMLP` (`src/oulad/torch_models.py`, `src/oulad/train_torch.py`):** Train-only `TabularPreprocessor` (`Unknown/Missing = 0`, known categories `1..K`, `nn.Embedding(K + 1, d)`), two hidden blocks (`Linear → LayerNorm → ReLU → Dropout(0.2)` with dimensions `128 → 64`), raw-logit output trained with unweighted `nn.BCEWithLogitsLoss()` and `AdamW(lr=1e-3, weight_decay=1e-4)` on CPU (`device_used = "cpu"`, early stopping `patience = 5`, best epoch: `2`). Unweighted BCE avoids intentional class-reweighting prior-odds distortion; post-hoc calibration and thresholding are handled separately.

### 4.2 Explicit 5-Fold Out-of-Fold (OOF) Calibration & Thresholding

To avoid in-sample optimism when calibrating and thresholding on `val_calib` (`n = 2,604`):

1. Run an explicit `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` loop over `val_calib`, fitting `CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)` on each training fold and scoring the held-out fold (compatible with scikit-learn ≥ 1.6).
2. Enforce a hard guard blocking `isotonic` regression whenever `n_calib < 1000` or `min_fold_train < 800` (on real `val_calib`, `n_calib = 2,604`, `min_fold_train = 2,083`, so `uncalibrated`, `sigmoid`, and `isotonic` were all eligible).
3. Select the calibration method minimizing 5-fold OOF Brier Score (tie-breaking by ECE, then model simplicity):
   - **LightGBM:** `uncalibrated` (`0.17912` raw vs `0.18011` sigmoid vs `0.18064` isotonic).
   - **Logistic Regression:** `sigmoid` (`0.18006` vs `0.18093` raw vs `0.18059` isotonic).
   - **TabularMLP:** `sigmoid` (`0.18424` vs `0.18535` raw vs `0.18429` isotonic).
4. Select operating threshold `tau*` on the winner's **OOF** probabilities maximizing Recall subject to `Precision >= 0.70` and `n_predicted_positives >= 30`, with a deterministic `max_f1` fallback if `0.70` is unattainable. All three models met the `0.70` floor on `val_calib` OOF (`LightGBM tau* = 0.4279`, OOF Precision `0.7011`, Recall `0.7328`).
5. Fit the winning calibrator (when not `uncalibrated`) on full `val_calib` and lock the champion bundle (`artifacts/champion_locked.joblib`).

---

## 5. Detailed Empirical Results

### 5.1 Cutoff Horizon Comparison: Validation vs. Frozen Historical `2014J`

**Validation (`val_select`, `2014B` 60%)** — Source: [reports/cutoff_comparison_val.csv](../reports/cutoff_comparison_val.csv)

| Cohort Mode | Cutoff (`T`) | `n_val_select` | LogReg PR-AUC / ROC-AUC | LightGBM PR-AUC / ROC-AUC / Brier |
| :--- | :---: | :---: | :---: | :---: |
| Operational | Day 14 | `3,993` | `0.6874` / `0.7038` | `0.7026` / `0.7130` / `0.2190` |
| Operational | Day 21 | `3,948` | `0.7601` / `0.7601` | `0.7527` / `0.7575` / `0.2000` |
| Operational *(Primary)* | Day 28 | `3,905` | `0.7815` / `0.7791` | **`0.7871` / `0.7842` / `0.1880`** |
| Safe Common | Day 14 | `3,901` | `0.6780` / `0.7017` | `0.6856` / `0.7070` / `0.2202` |
| Safe Common | Day 21 | `3,901` | `0.7539` / `0.7603` | `0.7394` / `0.7542` / `0.2014` |
| Safe Common | Day 28 | `3,901` | `0.7873` / `0.7878` | `0.7925` / `0.7937` / `0.1844` |

**Frozen Historical `2014J` Evaluation** *(committed in `2c8bab8`, unchanged thereafter; not re-executed)* — Source: [reports/cutoff_comparison_test.csv](../reports/cutoff_comparison_test.csv)

| Cohort Mode | Cutoff (`T`) | `n_test_2014j` | LogReg PR-AUC / ROC-AUC | LightGBM PR-AUC / ROC-AUC / Brier |
| :--- | :---: | :---: | :---: | :---: |
| Operational | Day 14 | `9,332` | `0.6137` / `0.7049` | `0.6312` / `0.7053` / `0.2127` |
| Operational | Day 21 | `9,291` | `0.6110` / `0.7120` | `0.6526` / `0.7205` / `0.2148` |
| Operational *(Primary)* | Day 28 | `9,213` | `0.6184` / `0.7239` | **`0.6813` / `0.7467` / `0.2040`** |
| Safe Common | Day 14 | `9,192` | `0.6070` / `0.7050` | `0.6239` / `0.7058` / `0.2109` |
| Safe Common | Day 21 | `9,192` | `0.6009` / `0.7103` | `0.6501` / `0.7227` / `0.2125` |
| Safe Common | Day 28 | `9,192` | `0.6190` / `0.7239` | `0.6798` / `0.7491` / `0.2028` |

### 5.2 Returning vs. New Students (`T = 28 Operational`, LightGBM)

| Evaluation Split | Segment | `n` (`Share`) | Positive Rate | PR-AUC | ROC-AUC | Brier | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `val_select` (`2014B` 60%) | Returning | `396` (`10.14%`) | `61.36%` | `0.8514` | `0.7976` | `0.1748` | [metrics_val.json](../reports/metrics_val.json) |
| `val_select` (`2014B` 60%) | New | `3,509` (`89.86%`) | `46.96%` | `0.7748` | `0.7778` | `0.1895` | [metrics_val.json](../reports/metrics_val.json) |
| Frozen Historical `2014J` | Returning | `782` (`8.49%`) | `57.93%` | `0.7987` | `0.7442` | `0.2037` | [metrics_test_locked.json](../reports/metrics_test_locked.json) |
| Frozen Historical `2014J` | New | `8,431` (`91.51%`) | `39.20%` | `0.6640` | `0.7428` | `0.2040` | [metrics_test_locked.json](../reports/metrics_test_locked.json) |

### 5.3 Sensitivity Analyses (`T = 28 Operational`, LightGBM)

1. **Sensitive-Demographic Ablation** *(drop `gender`, `disability`, `imd_band`)*:
   - `val_select`: `PR-AUC = 0.7829` (`-0.0042` vs full), `ROC-AUC = 0.7827` (`-0.0015`).
   - Frozen Historical `2014J`: `PR-AUC = 0.6799` (`-0.0014`), `ROC-AUC = 0.7466` (`-0.0001`).
   - *Interpretation:* Small aggregate deltas do **not** certify fairness because correlated proxy features (course module, region, prior education, VLE activity) remain present.
2. **Score-Free Sensitivity Check** *(drop `mean_score`, `has_score` to test the missing `date_graded` assumption)*:
   - `val_select`: `PR-AUC = 0.7570` (`-0.0301`), `ROC-AUC = 0.7553` (`-0.0289`), `Brier = 0.2001`.
   - Frozen Historical `2014J`: `PR-AUC = 0.7008` (`+0.0195`), `ROC-AUC = 0.7525` (`+0.0058`), `Brier = 0.1948`.
   - *Holdout Discipline:* Because the champion was locked prior to `2014J` evaluation, Full LightGBM remains the locked champion. The `2014J` score-free gain is recorded strictly as an exploratory post-unlock finding to be validated on a fresh, unused cohort.

### 5.4 Illustrative Cost–Benefit Analysis (Frozen Historical `2014J`, LightGBM)

Source: [reports/metrics_test_locked.json](../reports/metrics_test_locked.json) (frozen artifact retains legacy JSON policy label `"bayes_cost_tau"`; current `src/oulad/evaluate.py` uses `"cost_ratio_tau"`).

| Assumed Cost Ratio (`FP:FN`) | Threshold Policy | Threshold | FP | FN | Total Illustrative Cost | Cost / Student |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `1:5` | `locked_tau` (from `val_calib` OOF) | `0.4279` | `1,700` | `1,183` | `7,615` | `0.8265` |
| `1:5` | Fixed cost-ratio cutoff `C_FP/(C_FP+C_FN)` | `0.1667` | `4,216` | `180` | `5,116` | `0.5553` |
| `1:10` | `locked_tau` (from `val_calib` OOF) | `0.4279` | `1,700` | `1,183` | `13,530` | `1.4686` |
| `1:10` | Fixed cost-ratio cutoff `C_FP/(C_FP+C_FN)` | `0.0909` | `5,247` | `12` | `5,367` | `0.5825` |

> **Important Caveat:** Ratios `1:5` and `1:10` are **illustrative assumptions only**, not measured institutional costs. Because the locked LightGBM champion outputs **uncalibrated** scores (`ECE = 0.0542` on `2014J`), fixed cutoffs `0.1667` and `0.0909` are **not** claimed to be Bayes-optimal.

---

## 6. Global Explainability (Non-Causal SHAP)

`shap.TreeExplainer` was computed on `2,000` sampled rows of `val_select` (`T = 28 Operational`, LightGBM; [reports/metrics_val.json](../reports/metrics_val.json) & [reports/figures/shap_summary.png](../reports/figures/shap_summary.png)). Top 5 features by mean `|SHAP|` (log-odds magnitude):

1. `mean_score` (`0.4685`)
2. `submission_rate_due` (`0.3488`)
3. `code_module` (`0.2673`)
4. `highest_education` (`0.2603`)
5. `mean_days_late` (`0.1905`)

All SHAP outputs measure **predictive reliance** of the fitted model and must never be interpreted as causal drivers of student withdrawal or failure.
