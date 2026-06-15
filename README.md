<!-- generated-by: gsd-doc-writer -->
# Italian ILI Zero-Shot Forecasting Evaluation

This project evaluates zero-shot foundation models (Chronos, TimesFM, etc.) against classical and ML baselines for Italian Influenza-Like Illness (ILI) forecasting.

## Project Structure

- `data/`: Contains raw and processed ILI data.
- `src/`: Source code for the project.
    - `data/`: Data ingestion and preprocessing scripts.
    - `models/`: Implementation of baseline and foundation models.
    - `evaluation/`: Backtesting engine and metrics.
    - `utils/`: Utility functions.
- `results/`: Output directory for evaluation results and forecasts (created during execution).
- `benchmark_ili_national.py`: Main unified benchmarking script for national ILI series.
- `colab_benchmarking.ipynb`: Notebook for running experiments on Google Colab.
- `requirements.txt`: Python dependencies.

## Setup

### 1. Environment and Python Dependencies
This project requires **Python 3.11** (required for TimesFM 2.0). 
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

### 3. Model Downloads
Foundation models are downloaded automatically from the HuggingFace Hub during their first execution. 

- **Chronos (Large)**: ~3GB.
- **TimesFM (2.5)**: ~1GB.
- **TiRex**: ~150MB.

## Benchmarking CLI

The primary benchmarking script is `benchmark_ili_national.py`. It supports several flags for customization:

- `--model <name>`: Run only a specific model (e.g., `Chronos`, `LightGBM`).
- `--append`: Merge new results into existing `all_models_forecasts.csv` without overwriting results for other models. Useful for incremental testing.
- `--tune`: Enable Optuna hyperparameter tuning for ML models (LightGBM/XGBoost). This significantly increases execution time but improves accuracy.
- `--n-jobs <N>`: Control the number of CPU cores used for parallel execution. Use `-1` for all available cores.
- `--model-size <size>`: Choose the size of foundation models where applicable (e.g., Chronos). Options: `tiny`, `small`, `base`, `large` (default: `large`).
- `--num-samples <N>`: Control the precision of foundation models (default: 1000). Lower values (e.g., 200) are faster for testing.
- `--batch-size <N>`: Control the memory/speed trade-off (default: 1). Use 4-8 on Colab for max speed.
- `--dry-run`: Run a quick execution with minimal origins and horizons to verify the pipeline.

### Usage Examples

**Full National Benchmark (Background):**
```bash
PYTHONPATH=. nohup python3 benchmark_ili_national.py --n-jobs -1 > benchmark.log 2>&1 &
```

**High-Performance Cloud Execution (Google Colab T4):**
```bash
PYTHONPATH=. python3 benchmark_ili_national.py --model Chronos --num-samples 1000 --batch-size 4
```

**Local Mac Execution (Optimized for Apple Silicon):**
```bash
PYTHONPATH=. python3 benchmark_ili_national.py --model-size small --num-samples 200 --batch-size 1
```

*Note: On Apple Silicon Macs, models automatically use the **MPS (Metal)** backend. On NVIDIA GPUs, they use **CUDA** with float16 optimization.*

## Performance & Resource Management

- **Vectorized Backtesting**: Foundation models use a custom `predict_batch` implementation that processes multiple origins in parallel on the GPU/MPS, drastically reducing execution time.
- **Memory Chunking**: To prevent Out-Of-Memory (OOM) errors, batch processing is performed in chunks with explicit cache clearing (`torch.cuda.empty_cache()` or `torch.mps.empty_cache()`).
- **Sequential Execution Guard**: To prevent OOM errors on machines with limited RAM (e.g., 16GB), Foundation Models (Chronos, TimesFM, TiRex) are automatically restricted to sequential execution (`n_jobs=1`) across CPU cores, as they leverage internal GPU parallelism instead.
- **SARIMA Optimization**: Seasonal Auto-ARIMA is optimized for speed and stability by setting `max_P=1`, `max_Q=1`, and `max_D=1`. This prevents the model from exploring excessively complex seasonal structures that often lead to convergence failures or extreme slowdowns on epidemic data.
- **Explicit Garbage Collection**: The benchmarking suite performs explicit garbage collection (`gc.collect()`) and clears the Torch cache between model runs to ensure a clean memory state for the next model.

## Goals

- Ingest Italian ILI data from Influcast.
- Establish classical baselines (ARIMA, SARIMA, ETS, etc.).
- Evaluate zero-shot performance of Time Series Foundation Models.
- Analyze regional vs. national generalizability.

## Technical Decisions & Optimizations

### Backtest Engine Optimization
To handle the large number of forecast origins (weekly rolling origins over multiple seasons) across multiple regions, we implemented a highly optimized backtest engine:
- **Batch Processing**: For statistical models (ARIMA, ETS, Naive), we leverage `StatsForecast`'s ability to handle multiple series in parallel. Instead of a manual loop that re-initializes models for each origin, we restructure the backtest as a multi-series forecasting task where each origin's history is treated as a unique series. This reduces execution time from hours to minutes.
- **Parallel Execution**: For models not compatible with batching (e.g., Prophet), the engine uses `joblib` for parallel processing across available CPU cores.
- **Future Leakage Prevention**: Strict slicing of history at each origin ensures no future information is used during model fitting or prediction, maintaining the integrity of the rolling-origin evaluation.

### Model Configurations
- **AutoARIMA/SARIMA**: Optimized for speed by enabling `stepwise` search and `approximation`.
- **ML Baselines (LightGBM/XGBoost)**: Implemented using `MLForecast` with recursive lags and **Conformal Prediction** to generate probabilistic outputs.
- **Foundation Models**: Includes **Chronos**, **TimesFM (2.0)**, and **TiRex (xLSTM)**. (TimeGPT is excluded due to closed-beta/sales API requirements).
- **Probabilistic Forecasting**: All models produce a standardized **23-quantile output** (from 0.01 to 0.99), aligned with the CDC/Influcast standard for epidemic forecasting. This allows for rigorous evaluation using CRPS, WIS, and Pinball Loss.

### Scalability
The pipeline is designed to scale from the national aggregate to 21 regional series. The optimization of the baseline models is a critical component in meeting the project's technical deadlines.
