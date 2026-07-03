<!-- generated-by: gsd-doc-writer -->
# Italian ILI Zero-Shot Forecasting Evaluation

This project evaluates zero-shot foundation models (Chronos, TimesFM, etc.) against classical and ML baselines for Italian Influenza-Like Illness (ILI) forecasting.

## Project Structure

- `data/`: ILI data storage.
    - `raw/`: Historical CSV files from Influcast organized by season.
    - `processed/`: 
        - `ili_gold.csv`: Cleaned and aggregated master dataset.
        - `source_files_index.csv`: Manifest tracking the exact source file used for every season and region.
- `src/`: Source code.
    - `data/`: Scripts for data `ingestion.py` and `preprocessing.py`.
    - `models/`: Implementations for `baselines.py` (statistical), `ml.py` (LightGBM/XGBoost/CatBoost/Ridge), and Foundation Models (`chronos.py`, `timesfm.py`, `tirex.py`, `timegpt.py`).
    - `evaluation/`: Backtesting engine (`backtest.py`), accuracy `metrics.py`, and epidemic `peak_metrics.py`.
    - `utils/`: Common utilities like `quantiles.py` and `visualizations.py`.
- `benchmark_ili_national.py`: Main entry point for national benchmarks.
- `benchmark_ili_regional.py`: Entry point for regional benchmarks across all 21 regions/provinces.
- `requirements.txt`: Python dependencies.
- `results/`: (Generated) Output directory for forecasts and evaluation summaries.
    - `national/`: Standardized outputs for national runs (`RUN_REPORT.md`, `run_info.json`, `backtest_summary.csv`, `backtest_metrics_by_origin.csv`, `backtest_predictions.csv`, `all_models_peak_metrics.csv`).
        - `plots/`: Automated PNG/PDF visualisations.
    - `regional/`: Standardized outputs for regional runs (`RUN_REPORT.md`, `run_info.json`, `all_regions_metrics.csv`, `all_regions_peak_metrics.csv`, `best_model_per_region.csv`, and individual `{region}_forecasts.csv`, `{region}_metrics.csv`, and `{region}_peak_metrics.csv` files).
        - `plots/`: Automated PNG/PDF visualisations.

## Setup

### 1. Environment and Python Dependencies
This project requires **Python 3.11** (required for TimesFM 2.5). 
Create a virtual environment and install the required Python packages:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
pip install git+https://github.com/NX-AI/tirex.git
```

### 2. System Dependencies
Some models require additional system-level libraries:

- **LightGBM (macOS)**: Requires OpenMP. Install it via Homebrew:
  ```bash
  brew install libomp
  ```

### 3. Model Downloads & API Keys
Foundation models are downloaded automatically from the HuggingFace Hub during their first execution. 

- **Chronos (small, v2, bolt-small, large, etc.)**: ~3GB for large, smaller for other variants.
- **TimesFM (2.5)**: ~1GB.
- **TiRex**: ~150MB.

**TimeGPT** requires an API token from Nixtla. You must set the `TIMEGPT_TOKEN` environment variable before running benchmarks:
```bash
export TIMEGPT_TOKEN="your_nixtla_api_token_here"
```

### 4. Data Preparation
Before running benchmarks, ingest and preprocess the latest data from the Influcast repository:

> [!NOTE]
> **Data Scope (ILI vs ARI):** This benchmark focuses exclusively on historical **ILI** (Influenza-Like Illness) data. While the Italian surveillance system transitioned to tracking **ARI** (Acute Respiratory Infections) starting from the 2025-2026 season, this pipeline maintains a strict distinction and restricts evaluation to the long-term ILI historical series to ensure modeling consistency over the 20+ year timeframe.

```bash
# 1. Automated ingestion of all 22 geographic entities (national + 21 regions) (2003-present)
python3 src/data/ingestion.py

# 2. Preprocess with Epidemic Time-Indexing (Default)
python3 src/data/preprocessing.py --time-index epidemic

# Note: For traditional time-series indexing with optional zero-filling during summer off-seasons, use:
# python3 src/data/preprocessing.py --time-index calendar --fill-zeros
```

## Benchmarking CLI

The primary benchmarking scripts are `benchmark_ili_national.py` and `benchmark_ili_regional.py`. They support several flags for customization:

- `--model <name>`: Run only a specific model (e.g., `Chronos`) or a comma-separated list of models (e.g., `Naive,ARIMA,CatBoost`). Supported: `Naive`, `SeasonalNaive`, `Drift`, `MovingAverage`, `ETS`, `ARIMA`, `SARIMA`, `Prophet`, `Ridge`, `LightGBM`, `XGBoost`, `CatBoost`, `Chronos`, `TimesFM`, `TiRex`, `TimeGPT`.
- `--region <name>`: Run only a specific region (only applicable for `benchmark_ili_regional.py`).
- `--append`: Merge new results into existing `results/national/backtest_predictions.csv` without overwriting results for other models.
- `--tune`: Enable Optuna hyperparameter tuning for ML models.
- `--n-jobs <N>`: Control parallel execution (default: -1).
- `--min-train <N>`: Minimum training weeks before the first forecast origin (default: 156).
- `--step <N>`: Step size (weeks) between rolling origins (default: 4 for national, 8 for regional).
- `--horizons <h1,h2>`: Comma-separated horizons (default: 1,2,4,8).
- `--model-size <size>`: Choose FM size (e.g., Chronos). Options: `tiny`, `mini`, `small`, `base`, `large`, `v2`, `bolt-tiny`, `bolt-mini`, `bolt-small`, `bolt-base`.
- `--num-samples <N>`: Number of samples for foundation models (default: 1000).
- `--batch-size <N>`: Batch size for foundation model inference (default: 1 for national, 8 for regional).
- `--device <type>`: Override default device for deep learning models (`cpu`, `cuda`, `mps`).
- `--dry-run`: Quick execution with minimal origins/horizons.

### Usage Examples

**Full National Benchmark:**
```bash
python3 benchmark_ili_national.py --n-jobs -1 --min-train 156 --step 4
```

**Full Regional Benchmark (Speed-Optimized):**
```bash
python3 benchmark_ili_regional.py --step 8 --n-jobs -1
```

**Incremental Model Update:**
```bash
python3 benchmark_ili_national.py --model CatBoost --tune --append
```

*Note: On Apple Silicon Macs, models automatically use the **MPS (Metal)** backend. On NVIDIA GPUs, they use **CUDA** with float16 optimization.*

## Performance & Resource Management

- **Vectorized Backtesting**: Foundation models (Chronos, TimesFM) use a custom `predict_batch` implementation that processes multiple origins in parallel on the GPU/MPS.
- **Memory Chunking**: To prevent Out-Of-Memory (OOM) errors, batch processing is performed in chunks with explicit cache clearing (`torch.cuda.empty_cache()` or `torch.mps.empty_cache()`).
- **Sequential Execution Guard**: To prevent OOM errors on machines with limited RAM (e.g., 16GB), Foundation Models are restricted to sequential execution (`n_jobs=1`) across CPU cores.
- **SARIMA Optimization**: Seasonal Auto-ARIMA is optimized for speed and stability by setting `max_P=1`, `max_Q=1`, and `max_D=1`.
- **Explicit Garbage Collection**: The benchmarking suite performs explicit garbage collection (`gc.collect()`) between model runs.

## Goals

- Ingest Italian ILI data from Influcast.
- Establish classical baselines (ARIMA, SARIMA, ETS, etc.).
- Evaluate zero-shot performance of Time Series Foundation Models (Chronos, TimesFM 2.5, TiRex, TimeGPT).
- Analyze regional vs. national generalizability.

## Technical Decisions & Optimizations

### Data Engineering Layer
To meet the professor's requirements for a robust and reproducible pipeline, we implemented:
- **Recursive Regional Discovery**: The ingestion engine automatically identifies all 22 geographic entities in the source repository, ensuring the benchmark is not limited to hardcoded subsets.
- **Epidemic Time-Indexing**: To avoid artificial seasonality artifacts, the pipeline defaults to concatenating observed weeks. This is critical for Foundation Models, which can be sensitive to large zero-filled gaps during summer off-seasons.
- **Source Transparency**: The `source_files_index.csv` provides a full audit trail of whether a `latest` snapshot or a weekly fallback was used for each data point.

### Backtest Engine Optimization
To handle the large number of forecast origins across multiple regions, we implemented a highly optimized backtest engine:
- **Batch Processing**: For statistical models (ARIMA, ETS, Naive), we leverage `StatsForecast`'s ability to handle multiple series in parallel.
- **Parallel Execution**: For models not compatible with batching (e.g., Prophet), the engine uses `joblib` for parallel processing across available CPU cores.
- **Future Leakage Prevention**: Strict slicing of history at each origin ensures no future information is used during model fitting or prediction.

### Model Configurations
- **AutoARIMA/SARIMA**: Optimized for speed by enabling `stepwise` search and `approximation`.
- **ML Baselines (LightGBM/XGBoost/CatBoost/Ridge)**: Implemented using `MLForecast` with recursive lags and **Conformal Prediction** (5 windows) to generate probabilistic outputs.
- **Foundation Models**: Includes **Chronos**, **TimesFM (2.5)**, **TiRex (xLSTM)**, and **TimeGPT**.
- **Probabilistic Forecasting**: All models produce a standardized **23-quantile output** (from 0.01 to 0.99), aligned with the CDC/Influcast standard for epidemic forecasting.
- **Evaluation Metrics**: Comprehensive suite including point accuracy (MAE, RMSE, sMAPE, MASE) and probabilistic calibration (WIS, CRPS, Pinball Loss, Coverage distance from nominal).
