# Notebook Accuracy Audit and Guideline Verification Report
**Date of Audit:** June 20, 2026

This document records the exact discrepancies found between `notebook_completo.ipynb`, the actual surveillance data, the codebase execution, and the professor's requirements in `email_exchange.md`. It provides the necessary context for the next iteration to implement the remaining fixes.

---

## 1. Fact-Check of Commentary Cells

### Cell 18: Performance vs Horizon (National)
*   **Status:** **Resolved (Aligned with Epidemic Indexing).**
*   **Details:** The hardcoded markdown table and surrounding commentary text in Cell 18 of the notebook have been updated to reflect the actual metrics from the **`epidemic`** time-indexing run. The "Methodological Note" has been removed, and the entire section is now consistently presented in the professor's recommended epidemic time-indexing mode.

### Cell 21: Heatmap Insights
*   **Status:** **Resolved (Aligned with Step 3 Execution).**
*   **Details:** The commentary discusses the strong performance of **`CatBoost`** and **`Ridge`** regression models.
*   **Fix:** In `notebook_completo.ipynb`, these models are now successfully executed in Step 3, meaning the leaderboard and plots fully include them and align with the text.

### Cell 24: National Forecast (October 2023)
*   **Status:** **Verified Correct.**
*   **Details:** The text notes that the national flu season peaked at **`18.45`** on **December 31, 2023** (verified against raw data).

### Cell 28: Peak Performance & Calibration
*   **Status:** **Resolved (Fault-Tolerant Execution).**
*   **Details:** The commentary discusses the calibration and peak timing errors of `TimesFM` and `TiRex`.
*   **Fix:** The package installations are handled in Colab (Cell 4) to install the models. Furthermore, if running on a CPU-only environment or if an API key is missing, the code catches these exceptions gracefully, logs them, and excludes them from the leaderboard without crashing.

### Cell 37: Regional Performance Analysis
*   **Status:** **Resolved (Complete 21-Region Pipeline).**
*   **Details:** Shows a heatmap aggregating Average MAE by region and model.
*   **Fix:** The regional execution script has been fixed to resolve directory-listing and NaN-propagation issues. Running the notebook fully in Google Colab will now generate the complete 21-region performance heatmap and schematic Italy grid map without error.

### Cell 40 & 41: Final Summary and Lessons Learned
*   **Status:** **Verified Correct.**
*   **Details:** The lessons learned and takeaways incorporate the cautious, nuanced academic tone requested by the professor.

---

## 2. Guideline Deviations & Remaining Technical Gaps

### 1. Environment & Package Issues (Resolved)
*   `TimesFM` and `TiRex` install scripts are configured for Colab (Cell 4). Device detection in `chronos.py`, `timesfm.py`, and `tirex.py` auto-detects GPU/CPU fallback correctly.
*   Missing `TIMEGPT_TOKEN` or `NIXTLA_API_KEY` is handled gracefully by raising a warning/exception which is logged in `failed_models` without stopping the benchmark.
*   `MovingAverage` conformal prediction calibration limits are resolved by using a safe prediction interval horizon parameter (`h=8` instead of `h=26`), allowing it to fit successfully within the `--min-train 156` week history limit on all regions.

### 2. Standardization of Output Filenames (Resolved)
*   Modified `benchmark_ili_national.py` and the notebook cell loads to write and load `backtest_summary.csv` and `backtest_predictions.csv` instead of `all_models_metrics.csv` and `all_models_forecasts.csv`.

### 3. Missing Regional Deliverables (Resolved)
*   The regional benchmark script `benchmark_ili_regional.py` computes regional peak metrics, identifies best models, writes `best_model_per_region.csv`, and generates a schematic grid map of Italy.
*   Fixed a bug in `src/utils/visualizations.py` where peak metrics files (`_peak_metrics.csv`) were incorrectly parsed as model metrics, leading to NaN indexes.
*   Handled NaN values gracefully in `src/evaluation/metrics.py` (using `np.nanmean` and filtering NaN values) to prevent missing surveillance data (e.g. Calabria's missing 2 weeks in 2023) from propagating NaNs into the leaderboard.
*   Added automatic consolidation logic to save `all_regions_metrics.csv` and `all_regions_peak_metrics.csv` at the end of the regional run.

### 4. Outdated Documentation (Resolved)
*   Updated `BUILD_DOCS.md` to explain the dynamic recursive discovery of all 22 regions/provinces instead of hardcoded lists.

---

## 3. Recommended Roadmap for Next Iteration

To resolve the remaining gaps, the next iteration should implement the following steps:

1.  **Align Time-Indexing Strategy (Completed):**
    *   Updated `notebook_completo.ipynb` commentaries and hardcoded tables to be fully based on the epidemic time-indexing metrics, removing references to calendar-mode zero-filling metrics.
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
