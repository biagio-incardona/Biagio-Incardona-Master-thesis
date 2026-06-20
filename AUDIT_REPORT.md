# Notebook Accuracy Audit and Guideline Verification Report
**Date of Audit:** June 20, 2026

This document records the exact discrepancies found between `notebook_completo.ipynb`, the actual surveillance data, the codebase execution, and the professor's requirements in `email_exchange.md`. It provides the necessary context for the next iteration to implement the remaining fixes.

---

## 1. Fact-Check of Commentary Cells

### Cell 18: Performance vs Horizon (National)
*   **Status:** **Discrepancy (Data Indexing Mismatch).**
*   **Details:** The numerical values in the Cell 18 table (and the text comments referring to them) match the results of a run performed under **`calendar`** time-indexing (where summer gaps are filled, yielding `1124` weeks and `80` origins at step=12).
*   **Mismatches with Execution:** When executing the notebook/scripts with their default arguments (which use **`epidemic`** time-indexing, yielding `615` weeks and `38` origins at step=12), the CatBoost model generates an average MAE of **`1.527`** (with tuning) or **`1.298`** (without tuning). However, the hardcoded markdown table lists CatBoost's average MAE as **`1.578`** (ranging from 0.736 at h=1 to 2.470 at h=8).
*   **Qualitative Observations:** The observations regarding error propagation (e.g., ARIMA/SARIMA deteriorating by 700% at $h=8$, Chronos-V2 and TimesFM exhibiting flatter error growth, and SeasonalNaive being a strong baseline) are qualitatively valid, but the exact numbers only align if the data is processed in `calendar` mode (or if `epidemic` mode metrics are updated).

### Cell 21: Heatmap Insights
*   **Status:** **Discrepancy (Model Listing Mismatch).**
*   **Details:** The commentary discusses the strong performance of **`CatBoost`** and **`Ridge`** regression models.
*   **Mismatches with Execution:** In the master notebook `tesi_biagio_incardona_(master).ipynb`, the stored leaderboard output for Cell 25 **completely excludes `CatBoost` and `Ridge`**. This is because they failed to run or had not been implemented at the time the cell outputs were generated on June 16, 2026.

### Cell 24: National Forecast (October 2023)
*   **Status:** **Verified Correct.**
*   **Details:** The text notes that the national flu season peaked at **`18.45`** on **December 31, 2023**.
*   **Fact-Check:** The raw data for the `2023-2024` season verifies that the peak incidence was indeed exactly **`18.45`** on Sunday, December 31, 2023 (`calendar_ds`).

### Cell 28: Peak Performance & Calibration
*   **Status:** **Potential Visual Discrepancy.**
*   **Details:** The commentary discusses the calibration and peak timing errors of `TimesFM` and `TiRex`.
*   **Mismatches with Execution:** `TimesFM` and `TiRex` are not installed in the python environment. If a user runs the notebook cells, these models fail silently and are excluded from the output plots, meaning the commentary will refer to models that are visually missing from the curves.

### Cell 37: Regional Performance Analysis
*   **Status:** **Discrepancy (Incomplete Scope).**
*   **Details:** Shows a heatmap aggregating Average MAE by region and model.
*   **Mismatches with Execution:** The regional benchmark directory (`results/regional/`) only contains metrics and forecasts for **two regions** (`abruzzo` and `basilicata`) because it was run with the `--dry-run` flag. Thus, the heatmap is a 2-row plot, not a full 22-region visualization.

### Cell 40 & 41: Final Summary and Lessons Learned
*   **Status:** **Verified Correct.**
*   **Details:** The lessons learned and takeaways incorporate the cautious, nuanced academic tone requested by the professor (using phrasing like *"results suggest that some foundation models can be competitive... interpret these findings with caution"*).

---

## 2. Guideline Deviations & Remaining Technical Gaps

Based on the last email in `email_exchange.md`, the following requirements have discrepancies or are missing:

### 1. Environment & Package Issues
*   **`TimesFM`** and **`TiRex`** are declared in `requirements.txt` but are not installed in the current environment (`ImportError` on import).
*   **`TimeGPT`** requires `NIXTLA_API_KEY` to run. Since no key is present in `.env`, the model fails backtest execution and is excluded from the final outputs.
*   **`MovingAverage`** fails in regional runs because the default `--min-train 156` parameter provides history that is too short to calibrate the conformal prediction intervals (which require `n_windows * h + window_size = 5 * 26 + 52 = 182` points).

### 2. Standardization of Output Filenames
*   The professor explicitly requested that the national benchmark save:
    1.  `backtest_summary.csv`
    2.  `backtest_predictions.csv`
*   The current script `benchmark_ili_national.py` saves them as:
    1.  `all_models_metrics.csv`
    2.  `all_models_forecasts.csv`

### 3. Missing Regional Deliverables
*   The professor requested **"best model per region"** and a **"map synthetic of best model per region"** (e.g., showing a geographical representation of the best model across Italy).
*   The regional benchmark script `benchmark_ili_regional.py` does not compute or save regional peak metrics. The professor requested peak metrics (timing/intensity errors) for each region and season.
*   Neither of these are currently implemented. The visualization suite only generates the MAE performance heatmap.

### 4. Outdated Documentation
*   `BUILD_DOCS.md` (Line 15) states that the ingestion script has a *hardcoded* targeting of regions (Lombardia, Lazio, etc.).
*   In reality, `src/data/ingestion.py` was refactored to perform *automated regional discovery* across all 22 regions/provinces. The documentation in `BUILD_DOCS.md` is outdated and should be updated.

---

## 3. Recommended Roadmap for Next Iteration

To resolve the remaining gaps, the next iteration should implement the following steps:

1.  **Align Time-Indexing Strategy:**
    *   Since the notebook commentary is based on `calendar` mode metrics, update the default preprocessing command in the notebook (or default arguments in the scripts) to run in `calendar` mode, or update the commentaries to reflect the `epidemic` mode metrics.
2.  **Fix Filenames:**
    *   Modify `benchmark_ili_national.py` and the notebook cell loads to write/load `backtest_summary.csv` and `backtest_predictions.csv` instead of `all_models_metrics.csv` and `all_models_forecasts.csv`.
3.  **Implement Regional Peak Metrics:**
    *   Update `benchmark_ili_regional.py` to calculate and save peak timing and intensity errors per region and season, similar to `benchmark_ili_national.py`.
4.  **Implement Best Model per Region and Map:**
    *   Update `src/utils/visualizations.py` and `benchmark_ili_regional.py` to identify the best model (lowest average MAE/WIS) for each region and output it as a text table and a clean graphical layout (synthetic map).
5.  **Fix Ingest Documentation:**
    *   Update `BUILD_DOCS.md` to remove the mention of hardcoded regions and explain the recursive directory discovery mechanism.
6.  **Address Conformal Calibration History Limit:**
    *   For the regional runs, either increase the `--min-train` parameter for `MovingAverage` to `182` or adjust the conformal prediction interval parameters to fit within a `156`-week history.
