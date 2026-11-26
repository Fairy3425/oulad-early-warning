# Reproducibility, Environments & Testing Guide

This document explains dependency locking, supported execution workflows, exact `2014J` data-access boundaries, post-hoc historical reproduction rules, and the automated test suite for `oulad-early-warning`.

For the project overview and headline results, see [README.md](../README.md).

---

## 1. Dependency & Lockfile Strategy

| File | Target Platform | Role | Notes |
| :--- | :--- | :--- | :--- |
| [requirements-cpu.in](../requirements-cpu.in) | Linux `x86_64` · Python `3.11` · CPU | Direct dependency declarations | Input to `uv pip compile` for CI lockfile generation. |
| [requirements-cpu.lock](../requirements-cpu.lock) | Linux `x86_64` · Python `3.11` · CPU | **CI-verified pinned lockfile** (`42` packages) | Pins `torch==2.15.0+cpu`, `duckdb==1.5.6`, `lightgbm==4.7.0`, `scikit-learn==1.9.1`, `shap==0.51.0`; `0` CUDA packages and `0` `@ file://` entries. Installed directly by [ci.yml](../.github/workflows/ci.yml). |
| [requirements-local.lock](../requirements-local.lock) | Windows workstation snapshot | Provenance record only *(non-portable)* | Snapshot of the original Windows dev `.venv` (includes Jupyter, MLflow, `duckdb==2.0.1`). Not intended as a cross-OS install target. |

- **Optional MLflow Tracking:** `mlflow` is omitted from `requirements-cpu.lock` to keep CI lightweight. When installed locally (`pip install mlflow`), Stage 5 logs 4 core runs to local `sqlite:///mlflow.db` (gitignored); when absent, evaluation degrades gracefully (`logged: false`).

### Installing on the CI Target (Linux `x86_64`, Python `3.11`, CPU)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.lock
pip install --no-deps --no-build-isolation -e .
```

*(Verified automatically in [OULAD GitHub Actions CI](https://github.com/Fairy3425/oulad-early-warning/actions/workflows/ci.yml), e.g., passing run `#45787412942` on `1d7fe6f` and `#45781213279` on `cfd9e6b`.)*

---

## 2. Supported Execution Workflows

Raw OULAD CSVs (`data/raw/`) and trained model binaries (`artifacts/champion_locked.joblib`, `artifacts/tabular_mlp_best.pt`) are intentionally gitignored. Place the 7 OULAD CSV files (UCI ID 349, CC BY 4.0) into `data/raw/` before running pipeline commands.

### 2.1 Dry-Run Validation (Writes Nothing)

```bash
python -m oulad.run_all --skip-download --dry-run
```

Validates that all 7 OULAD CSVs are present and prints the planned safe-by-default execution stages without modifying files.

### 2.2 Safe Fresh-Clone Reproduction & CLI Demo (Default — Skips `2014J` Evaluation)

```bash
# Rebuilds DuckDB, 6 Parquets, validation models, and artifacts/champion_locked.joblib
# Strictly skips Stage 5 (2014J evaluation)
python -m oulad.run_all --skip-download --skip-locked-test

# Optional: isolate outputs without touching committed reports/ or figures/
python -m oulad.run_all --skip-download --skip-locked-test \
  --reports-dir ./tmp_run/reports --figures-dir ./tmp_run/figures

# Score two self-contained synthetic illustrative Day-28 student profiles
python -m oulad.predict --sample
```

- Running `python -m oulad.predict --sample` before building `artifacts/champion_locked.joblib` raises a clear `FileNotFoundError` pointing to the safe `--skip-locked-test` command above.
- **Exact `2014J` Data-Access Boundary:**
  - *Stages 1–2 (`db.py`, `features.py`):* All 7 raw CSVs (including `2014J` rows) are ingested into DuckDB for schema quality checks and row-wise Start-of-Day feature Parquet export. Top-5 VLE activity types are derived **strictly from Train (`2013B + 2013J`)**, and `cohort_summary.json` records aggregate counts only.
  - *Stages 3–4b (`baselines.py`, `train_torch.py`, `fit_and_lock_phase_a_champion`):* `2014J` is **strictly excluded** from training, feature selection, early stopping, model selection, OOF calibration, threshold selection, and evaluation, and `run_p1_b5_locked_evaluation_suite()` is unreachable.

### 2.3 Post-Hoc Historical Reproduction (`--include-locked-test`, Explicit Opt-In)

> **Scientific Integrity Warning:**
> - **Default execution skips `2014J` evaluation** (`include_locked_test = False`). Passing both `--skip-locked-test` and `--include-locked-test` raises a `ValueError`.
> - Passing `--include-locked-test` runs a **post-hoc historical reproduction** of the `2014J` evaluation originally committed in `2c8bab8` (unchanged in subsequent Git history; repo records alone do not independently certify the exact number of local runs prior to `2c8bab8`).
> - **This is NOT a new blind test.** Its output must **never** be used to select, compare, or tune models, features, calibrators, or thresholds while claiming `2014J` as an independent holdout.

To reproduce historical `2014J` outputs into an isolated folder without modifying committed [reports/](../reports):

```bash
python -m oulad.run_all --skip-download --include-locked-test \
  --reports-dir ./tmp_repro/reports --figures-dir ./tmp_repro/figures
```

---

## 3. Automated Test Suite (`7` Python Files · `6` Test Modules · `86` Tests)

All automated checks run offline on synthetic in-memory DuckDB fixtures (no raw dataset download, no GPU required):

```bash
ruff check .
pytest -q
```

| File | Tests | Scope Verified |
| :--- | :---: | :--- |
| [tests/conftest.py](../tests/conftest.py) | — | Shared offline synthetic DuckDB & feature-table fixtures |
| [tests/test_quality.py](../tests/test_quality.py) | `13` | 9 SQL data quality checks, `nullstr='?'` parsing, Returning vs. New overlap |
| [tests/test_leakage.py](../tests/test_leakage.py) | `29` | Parameterized `[14, 21, 28]` × `{operational, common}` future-event & post-outcome invariance, registration/week bounds, `submission_rate_due ∈ [0, 1]`, train-only top activities, and `python -O`-safe `RuntimeError` guards |
| [tests/test_split_and_baselines.py](../tests/test_split_and_baselines.py) | `10` | Forward-split invariants, `subsample_freq=1`, unseen categories, `2014J` mutation isolation, and `python -O`-safe `RuntimeError` `2014J` guards |
| [tests/test_models.py](../tests/test_models.py) | `5` | PyTorch `TabularMLP` `Unknown/Missing -> 0`, raw logits, gradient update step, checkpoint roundtrip, and `2014J` isolation |
| [tests/test_metrics.py](../tests/test_metrics.py) | `13` | `n_calib < 1000` isotonic guard, explicit 5-fold disjoint OOF `FrozenEstimator` audit, `python -O`-safe `RuntimeError` guards, `2014J` isolation, `max_f1` fallback, cost-ratio disclaimer, `min(20, len)` FP/FN, and Fairness `low_sample_warning` |
| [tests/test_cli.py](../tests/test_cli.py) | `16` | `predict --sample` SHAP contract, missing-column rejection, extra-column reporting, SHA-256 artifact-fingerprint ECE verification, missing-bundle clean-workspace error, safe default `run_all`, conflicting-flag rejection, post-hoc warning, and isolated Stage 1–2 execution |
| **Total** | **`86`** | **Verified passing locally (`ruff` 0 errors) and on Linux CPU GitHub Actions CI** |

---

## 4. Artifact & Report Map

| Artifact | Tracked in Git | Role |
| :--- | :---: | :--- |
| [reports/data_quality.json](../reports/data_quality.json) | Yes | 9 SQL quality checks, registration stats, Returning/New overlap |
| [reports/cohort_summary.json](../reports/cohort_summary.json) | Yes | Row counts, feature counts, positive rates across all 6 cutoff Parquets |
| [reports/cutoff_comparison_val.csv](../reports/cutoff_comparison_val.csv) | Yes | 12 validation models (`14/21/28` × `Operational/Common`) on `val_select` |
| [reports/metrics_val.json](../reports/metrics_val.json) | Yes | 3 primary models, Returning/New, SHAP top 10, sensitivity checks on `val_select` |
| [reports/metrics_test_locked.json](../reports/metrics_test_locked.json) | Yes | Frozen historical `2014J` evaluation report (unchanged since `2c8bab8`) |
| [reports/cutoff_comparison_test.csv](../reports/cutoff_comparison_test.csv) | Yes | Frozen historical `2014J` cutoff comparison for 12 models (since `2c8bab8`) |
| [reports/fairness_test.csv](../reports/fairness_test.csv) | Yes | Frozen historical `2014J` 18-subgroup performance audit (since `2c8bab8`) |
| `reports/figures/*.png` | Yes | `shap_summary.png`, `learning_curve.png`, `calibration_curve.png` |
| `data/`, `artifacts/`, `mlflow.db` | No (gitignored) | Raw OULAD CSVs, DuckDB DB, Parquets, model bundles, local MLflow SQLite DB |
