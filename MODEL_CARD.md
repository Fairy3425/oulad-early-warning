# Model Card — OULAD Early Warning (`oulad-early-warning`)

## 1. Model Details

- **Model Name:** OULAD Day-28 Student Risk Early Warning — Locked Champion (`LightGBM`)
- **Version:** `v1.0-rc` *(verified on Linux `x86_64` Python `3.11` CPU GitHub Actions CI; the historical `2014J` evaluation report was committed in the `P1-B5` commit `2c8bab8` and has remained unchanged in the subsequent Git history. The available repository evidence does not independently establish the exact number of local evaluation executions before that commit.)*
- **Model Type:** Binary Gradient Boosted Decision Trees (`lightgbm.LGBMClassifier`, `subsample_freq=1`, deterministic `seed=42`), benchmarked against `LogisticRegression` (median impute + standard scale + one-hot) and a PyTorch `TabularMLP` (entity embeddings with `Unknown/Missing = 0`, LayerNorm, Dropout `0.2`, unweighted `BCEWithLogitsLoss`, trained on CPU).
- **Prediction Horizon & Cohort:** Day `T = 28` (Start-of-Day, strictly `< 28` features), Operational Cohort (`date_registration < 28` and still enrolled at Day 28).
- **Output:** Risk score in `[0, 1]` (`calibration_method = "uncalibrated"` per pre-specified OOF Brier selection rule on `val_calib`) and binary alert decision at locked operating threshold `tau* = 0.4279`.
- **Dataset & License:** Open University Learning Analytics Dataset (OULAD, UCI ID 349, **CC BY 4.0**; Kuzilek et al., 2017).

---

## 2. Intended Use & Prohibited Use

### Intended Use
- **Supportive Academic Advising Triage:** Prioritizing voluntary, non-punitive early outreach (tutor check-ins, study-skills workshops, technical onboarding assistance) around Week 4 of an online course.
- **Educational ML Engineering Portfolio Demonstration:** Illustrating leakage-free SQL feature engineering, temporal forward splits, out-of-fold calibration auditing, and subgroup performance transparency.

### Out-of-Scope / Prohibited Use
- **Automated Punitive Decisions:** Must **NOT** be used to deny enrollment, revoke scholarships/financial aid, grade students, or impose disciplinary sanctions.
- **Causal Intervention Claims:** Must **NOT** be used to claim that modifying a single feature (e.g., forcing extra VLE clicks) will causally prevent dropout; SHAP outputs are strictly predictive attributions.
- **Unmonitored Cross-Institution Deployment:** Must **NOT** be deployed to a new university, curriculum, or seasonal term without fresh validation, calibration checks, and human oversight.

---

## 3. Training, Validation & Locked Test Data

- **Split Protocol:** Presentation-Based Forward Split (`AUD-07`), `seed = 42`:
  - **Train (`2013B + 2013J`):** `n = 11,806` (`T=28 Operational`).
  - **Validation Select (`60%` stratified `2014B`):** `n = 3,905` (`pos_rate = 48.43%`) — model comparison, LightGBM/MLP early stopping, SHAP summary.
  - **Validation Calibration (`40%` stratified `2014B`):** `n = 2,604` (`pos_rate = 48.43%`) — explicit 5-fold OOF calibration (`StratifiedKFold(5)` + `FrozenEstimator`) & threshold `tau*` selection.
  - **Locked Test (`2014J`):** `n = 9,213` (`n_pos = 3,758`, `pos_rate = 40.79%`) — held out from training, model selection, calibration, and threshold selection. The historical `2014J` evaluation report was committed in the `P1-B5` commit `2c8bab8` and has remained unchanged in the subsequent Git history. The available repository evidence does not independently establish the exact number of local evaluation executions before that commit.
- **Calendar Overlap Disclosure (`AUD-07`):** `2013J` (Oct 2013–Jun 2014) overlaps ~4 months with `2014B` (Feb–Oct 2014). The final evaluation from `2013` Train to `2014J` Locked Test (Oct 2014) has zero calendar overlap.
- **Data Quality Verification (`reports/data_quality.json`):** `9/9` automated DuckDB checks returned `0` violations across `32,593` enrollments and `10,655,280` VLE interaction rows (`imd_band` missing rate `3.41% < 5%`, preserved explicitly as `MISSING` in subgroup audits).

---

## 4. Quantitative Evaluation Summary

### 4.1 Primary Models — Validation (`val_select`), OOF Calibration (`val_calib`) & Locked Test (`2014J`)

| Model | `val_select` PR-AUC / ROC-AUC | `val_calib` 5-Fold OOF Brier (Uncal / Sig / Iso) | Chosen Method (`tau*`) | `2014J` PR-AUC / ROC-AUC | `2014J` Raw → Final Brier (`Δ`) | `2014J` Raw → Final ECE (`Δ`) | `2014J` Prec / Rec / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM *(Champion)*** | `0.7871` / `0.7842` | **`0.17912`** / `0.18011` / `0.18064` | `uncalibrated` (`0.4279`) | **`0.6813` / `0.7467`** | `0.2040 → 0.2040` (`0.0000`) | `0.0542 → 0.0542` (`0.0000`) | `0.6023` / `0.6852` / `0.6411` |
| **TabularMLP (CPU)** | `0.7695` / `0.7689` | `0.18535` / **`0.18424`** / `0.18429` | `sigmoid` (`0.4156`) | `0.6376` / `0.7221` | `0.2128 → 0.2186` (`+0.0058`) | `0.0532 → 0.0713` (`+0.0181`) | `0.6022` / `0.6408` / `0.6209` |
| **LogisticRegression** | `0.7815` / `0.7791` | `0.18093` / **`0.18006`** / `0.18059` | `sigmoid` (`0.4398`) | `0.6184` / `0.7239` | `0.2180 → 0.2187` (`+0.0007`) | `0.0814 → 0.0812` (`-0.0002`) | `0.6132` / `0.6437` / `0.6281` |

*Champion Confusion Matrix on Locked `2014J` (`tau* = 0.4279`):* `TN = 3,755`, `FP = 1,700`, `FN = 1,183`, `TP = 2,575`.

### 4.2 Calibration Interpretation (`AUD-B5-02` & `FA-07`)
- **Brier Score vs Calibration Quality:** Brier Score is a proper scoring rule decomposing into *uncertainty − resolution (discrimination) + reliability (calibration error)*. **A lower Brier Score does NOT automatically imply better calibration.** For example, on `val_calib`, `isotonic` achieved slightly lower OOF ECE (`0.01362`) than raw LightGBM (`0.01429`), but worse OOF Brier (`0.18064` vs `0.17912`).
- **Why LightGBM Is Uncalibrated:** LightGBM was kept `uncalibrated` strictly because the pre-specified selection criterion minimized OOF Brier on `val_calib`. **We do NOT claim that LightGBM's probabilities are well-calibrated or perfectly calibrated.**
- **Seasonal Prior Shift (`FA-07`):** Between February `2014B` (`val_calib`, `48.43%` at-risk) and October `2014J` (`40.79%` at-risk, a **`-7.64 pp`** prior shift), LightGBM's locked test **ECE = `0.0542`** (`Brier = 0.2040`), with predicted probabilities in `0.55–0.85` visibly above observed positive rates (`reports/figures/calibration_curve.png`). Similarly, `sigmoid` calibrators fitted on `2014B` slightly increased test Brier on `2014J` (`+0.0007` LogReg; `+0.0058` MLP), and LightGBM's locked threshold shifted from `val_calib` OOF Precision `0.7011` to `2014J` Precision `0.6023`.

### 4.3 Illustrative Cost–Benefit Analysis (`AUD-B5-01`)

| Assumed Cost Ratio (`FP:FN`) | Threshold Policy | Threshold | FP | FN | Total Illustrative Cost | Cost / Student |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `1:5` | `locked_tau` (from `val_calib` OOF) | `0.4279` | `1,700` | `1,183` | `7,615` | `0.8265` |
| `1:5` | Fixed cost-ratio cutoff `C_FP/(C_FP+C_FN)` | `0.1667` | `4,216` | `180` | `5,116` | `0.5553` |
| `1:10` | `locked_tau` (from `val_calib` OOF) | `0.4279` | `1,700` | `1,183` | `13,530` | `1.4686` |
| `1:10` | Fixed cost-ratio cutoff `C_FP/(C_FP+C_FN)` | `0.0909` | `5,247` | `12` | `5,367` | `0.5825` |

- **Strict Caveat (`AUD-B5-01`):**
  1. Ratios `1:5` and `1:10` are **illustrative assumptions only**, not real institutional intervention costs.
  2. Because the locked LightGBM champion is **`uncalibrated`**, thresholds `0.1667` and `0.0909` are **NOT claimed to be Bayes-optimal**. The formula `C_FP / (C_FP + C_FN)` carries a Bayes decision-theoretic interpretation **only** when input scores are well-calibrated posterior probabilities under matching deployment priors and verified operational cost assumptions.
  3. *Historical Artifact Note:* The frozen JSON artifact `reports/metrics_test_locked.json` (`P1-B5` commit `2c8bab8`) is preserved verbatim without re-running `2014J` and retains the original key name `"bayes_cost_tau"`; `src/oulad/evaluate.py` has been updated to label this field `"cost_ratio_tau"` with the non-Bayes disclaimer above.

### 4.4 Returning vs New & Sensitivity Findings (`AUD-B5-04`)

- **Returning vs New (`2014J`, LightGBM):** Returning students (`n = 782`, `8.49%`, `pos_rate = 57.93%`) achieved `PR-AUC = 0.7987`, `ROC-AUC = 0.7442`, `Brier = 0.2037`; New students (`n = 8,431`, `91.51%`, `pos_rate = 39.20%`) achieved `PR-AUC = 0.6640`, `ROC-AUC = 0.7428`, `Brier = 0.2040`.
- **Cutoff Progression (`reports/cutoff_comparison_test.csv`):** LightGBM `PR-AUC` rises monotonically across `T=14 → 21 → 28` on both Operational (`0.6312 → 0.6526 → 0.6813`) and Safe Common (`0.6239 → 0.6501 → 0.6798`).
- **Score-Free Sensitivity Check (`FA-03`, `AUD-B5-04`):** Dropping `mean_score` and `has_score` yielded `PR-AUC = 0.7008 (+0.0195)`, `ROC-AUC = 0.7525 (+0.0058)`, and `Brier = 0.1948` on `2014J` (versus `val_select` where dropping scores reduced `PR-AUC` by `-0.0301`). **Per locked-test holdout discipline (`FA-02`), Full LightGBM remains the locked champion** (no model, feature set, calibrator, or threshold was modified after committing the historical `2014J` report in `2c8bab8`). Score-Free `PR-AUC = 0.7008` is recorded strictly as an **exploratory test finding** (reflecting OULAD's missing `date_graded` and cross-season formative score anomalies) and is **NOT** selected for production based on `2014J`; any follow-up comparison must use a **fresh, unused holdout cohort**.

---

## 5. Subgroup Performance Audit & Fairness Limitations (`AUD-B5-03`)

Full 18-subgroup audit on locked `2014J` (`tau* = 0.4279`, Champion LightGBM) is stored in `reports/fairness_test.csv`. Key observed ranges:

| Dimension | Subgroup Highlights (`n`, `n_pos`, `Positive Rate`) | Observed Recall | Observed Precision | Observed FPR |
| :--- | :--- | :---: | :---: | :---: |
| **`gender`** | `F` (`n=3,872`, `n_pos=1,490`, `38.48%`) vs `M` (`n=5,341`, `n_pos=2,268`, `42.46%`) | `0.6161` vs `0.7306` | `0.5334` vs `0.6488` | `0.3371` vs `0.2919` |
| **`disability`** | `Y` (`n=855`, `n_pos=451`, `52.75%`) vs `N` (`n=8,358`, `n_pos=3,307`, `39.57%`) | `0.7716` vs `0.6734` | `0.6988` vs `0.5896` | `0.3713` vs `0.3069` |
| **`imd_band`** | `0-10%` (`n=820`, `49.27%`) vs `90-100%` (`n=747`, `33.20%`) vs `MISSING` (`n=366`, `28.69%`) | `0.7376` vs `0.5847` vs `0.4857` | `0.6221` vs `0.5642` vs `0.5258` | `0.4351` vs `0.2244` vs `0.1762` |
| **`age_band`** | `0-35` (`n=6,312`, `42.22%`) vs `35-55` (`n=2,814`, `37.74%`) vs `55<=` (`n=87`, `35.63%`) | `0.7107` vs `0.6252` vs `0.5484` | `0.6067` vs `0.5881` vs `0.7083` | `0.3367` vs `0.2654` vs `0.1250` |

### Explicit Fairness Limitations (`AUD-B5-03`)
1. **Subgroup Performance Audit ≠ Fairness Certification:** Reporting per-group metrics is an observational audit to surface disparities, **not** a certification that the model is fair or legally compliant.
2. **Demographic Ablation Does Not Prove Fairness:** Dropping `gender`, `disability`, and `imd_band` changed LightGBM `2014J` `PR-AUC` by only `-0.0014` (`0.6813 → 0.6799`), but correlated proxy variables (module choice, region, prior education, early VLE access) still encode socioeconomic and demographic differences.
3. **Passing `n >= 30` Does Not Imply Fairness:** Although `0 / 18` subgroups triggered `low_sample_warning` (`n < 30`) on `2014J` (smallest group: `age_band = 55<=`, `n = 87`, `n_pos = 31`), substantial Recall (`11.45 pp` gender gap; `25.19 pp` `imd_band` `0-10%` vs `MISSING` gap) and FPR (`21.07 pp` `imd_band 0-10%` vs `90-100%` gap) disparities persist.
4. **Uncertainty & Drift Monitoring Needed:** Point estimates lack confidence intervals (especially critical for smaller groups like `age_band = 55<=` and `imd_band = MISSING`). Any real-world pilot requires bootstrap confidence intervals, stakeholder-defined fairness criteria, and continuous cohort monitoring.

---

## 6. Post-Lock Error Analysis & Ethical Risks

- **Observational Error Audit (`min(20, len)` FP/FN on `2014J`, zero post-unlock edits):**
  - **False Positives (`total_fp = 1,700`, `sampled = 20`):** Students flagged at `pred_prob = 0.91–0.95` who ultimately passed (`target = 0`) — predominantly students who missed an early Day-28 due assessment (`submission_rate_due = 0.0`) or recorded a `1.0` early score in `BBB 2014J` yet recovered later in the course.
  - **False Negatives (`total_fn = 1,183`, `sampled = 20`):** Students scored `pred_prob = 0.06–0.10` who still failed/withdrew (`target = 1`) — highly active Day-28 learners (`clicks_total = 109–3,300`, `submission_rate_due = 1.0`, `mean_score = 85–100`) affected by late-term shocks outside the Day-28 horizon.
- **Ethical Risks & Mitigations:**
  - *Stigmatization / Self-Fulfilling Prophecy:* Never expose raw risk labels directly to students or instructors as fixed ability judgments; frame alerts exclusively as prompts for supportive check-ins.
  - *Disparate False-Alarm Burden:* Higher FPR among socioeconomically deprived (`imd_band 0-10%`: `0.4351`) and disabled (`0.3713`) students means alerts must trigger helpful, opt-in resources rather than surveillance or penalties.
  - *Missing `date_graded` & Label Noise:* Early scores may not always be graded before Day 28 in real-time operations; operational deployments should verify LMS timestamp availability before enabling score-based features.
