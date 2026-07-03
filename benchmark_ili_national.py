"""
Unified benchmarking script for Phase 2: ML & Foundation Models.
Executes backtests for all models (Baselines, ML, and TSFMs) on the national ILI series.
"""

import torch
import os
import argparse
import pandas as pd
import numpy as np
import gc
import warnings
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from src.models import (
    NaiveForecaster, 
    SeasonalNaiveForecaster, 
    DriftForecaster,
    MovingAverageForecaster,
    ETSForecaster, 
    ARIMAForecaster, 
    SARIMAForecaster, 
    ProphetForecaster,
    LightGBMForecaster,
    XGBoostForecaster,
    CatBoostForecaster,
    RidgeForecaster,
    ChronosForecaster,
    TimesFMForecaster,
    TiRexForecaster,
    TimeGPTForecaster
)
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import evaluate_forecasts
from src.evaluation.peak_metrics import calculate_seasonal_peak_metrics
from src.utils.quantiles import INFLUCAST_QUANTILES
from src.utils.visualizations import plot_national_trajectories, plot_best_model_heatmap

def main():
    parser = argparse.ArgumentParser(description='National ILI Model Benchmarking')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick check with one model and few origins')
    parser.add_argument('--model', type=str, default=None, help='Run only a specific model')
    parser.add_argument('--append', action='store_true', help='Append to existing results/national/backtest_predictions.csv')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs for parallel execution (default: -1)')
    parser.add_argument('--min-train', type=int, default=156, help='Minimum training weeks (default: 156 = 3 years)')
    parser.add_argument('--step', type=int, default=4, help='Step size between origins (default: 4)')
    parser.add_argument('--horizons', type=str, default='1,2,4,8', help='Comma-separated horizons (default: 1,2,4,8)')
    parser.add_argument('--model-size', type=str, default='large', 
                        choices=['tiny', 'mini', 'small', 'base', 'large', 
                                 'v2', 'bolt-tiny', 'bolt-mini', 'bolt-small', 'bolt-base'],
                        help='Size of foundation models (Chronos, etc.)')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples for foundation models')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for foundation model inference')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters for ML models')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='Override default device for deep learning models')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df = pd.read_csv('data/processed/ili_gold.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Filter for national ILI
    df_nat = df[df['region'] == 'italia'].copy().sort_values('ds')
    print(f"Loaded {len(df_nat)} weeks of national data.")
        
    # Dynamic Origin Generation
    horizons = [int(h) for h in args.horizons.split(',')]
    max_horizon = max(horizons)
    
    # Origins start after min-train and end before max-horizon from the end
    available_dates = df_nat['ds'].values
    start_idx = args.min_train
    end_idx = len(available_dates) - max_horizon
    
    if start_idx >= end_idx:
        print(f"Error: Not enough data for min_train={args.min_train} and max_horizon={max_horizon}")
        return
        
    origin_indices = range(start_idx, end_idx, args.step)
    origins = pd.to_datetime(available_dates[origin_indices])
    
    if args.dry_run:
        print("!!! DRY RUN MODE ENABLED !!!")
        origins = origins[:2]
        horizons = [1, 2]
        
    print(f"Total origins to evaluate: {len(origins)}")
    print(f"Horizons: {horizons}")
    
    quantiles = INFLUCAST_QUANTILES
    
    # Model Registry
    all_models = {
        'Naive': (NaiveForecaster, {}),
        'SeasonalNaive': (SeasonalNaiveForecaster, {}),
        'Drift': (DriftForecaster, {}),
        'MovingAverage': (MovingAverageForecaster, {'window_size': 52}),
        'ETS': (ETSForecaster, {}),
        'ARIMA': (ARIMAForecaster, {}),
        'SARIMA': (SARIMAForecaster, {}),
        'Prophet': (ProphetForecaster, {}),
        'LightGBM': (LightGBMForecaster, {}),
        'XGBoost': (XGBoostForecaster, {}),
        'CatBoost': (CatBoostForecaster, {}),
        'Ridge': (RidgeForecaster, {}),
        'Chronos': (ChronosForecaster, {'model_name': args.model_size, 'num_samples': args.num_samples, 'batch_size': args.batch_size, 'device': args.device}),
        'TimesFM': (TimesFMForecaster, {'batch_size': args.batch_size, 'device': args.device}),
        'TiRex': (TiRexForecaster, {'batch_size': args.batch_size, 'device': args.device}),
        'TimeGPT': (TimeGPTForecaster, {})
    }
    
    # Filter if specific model requested
    if args.model:
        requested_models = [m.strip() for m in args.model.split(',')]
        models_to_run = {m: all_models[m] for m in requested_models if m in all_models}
        if not models_to_run:
            print(f"None of the requested models {args.model} were found in the registry.")
            return
    elif args.dry_run:
        models_to_run = {'Naive': all_models['Naive']}
    else:
        models_to_run = all_models

    output_dir = 'results/national'
    os.makedirs(output_dir, exist_ok=True)
    
    existing_forecasts = None
    if args.append:
        full_forecast_file = os.path.join(output_dir, "backtest_predictions.csv")
        if os.path.exists(full_forecast_file):
            print(f"Loading existing forecasts from {full_forecast_file} for appending...")
            existing_forecasts = pd.read_csv(full_forecast_file)
            existing_forecasts['target_date'] = pd.to_datetime(existing_forecasts['target_date'])
            print(f"Loaded {len(existing_forecasts)} existing forecast rows.")
        else:
            print(f"Warning: --append requested but {full_forecast_file} does not exist. Starting fresh.")

    all_forecast_dfs = []
    failed_models = []
    completed_models = []
    
    # Foundation models that require sequential execution to avoid OOM
    sequential_models = ['Chronos', 'TimesFM', 'TiRex', 'TimeGPT']

    for name, (model_cls, kwargs) in models_to_run.items():
        print(f"\n" + "="*50)
        print(f"RUNNING BACKTEST FOR: {name}")
        print(f"DEBUG: kwargs={kwargs}")
        print("="*50)
        
        # Enforce sequential execution for foundation models
        current_n_jobs = 1 if name in sequential_models else args.n_jobs
        print(f"Parallelism: n_jobs={current_n_jobs}")

        try:
            # Initialize model
            model = model_cls(**kwargs)
            
            # Tuning logic for ML models
            if args.tune and name in ['LightGBM', 'XGBoost', 'CatBoost', 'Ridge']:
                print(f"Hyperparameter tuning enabled for {name}...")
                # Training data for tuning is everything before the first backtest origin
                train_df = df_nat[df_nat['ds'] < origins[0]].copy()
                if not train_df.empty:
                    model.tune(train_df, n_trials=20)
                else:
                    print(f"Warning: No training data before origin {origins[0]} for tuning.")

            # Run backtest with warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
                # Statsforecast uses specific convergence warnings
                warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
                
                forecasts = run_backtest(
                    df_nat, 
                    model, 
                    origins, 
                    horizons, 
                    quantiles,
                    n_jobs=current_n_jobs
                )
            
            model_display_name = name
            if name == 'Chronos':
                model_display_name = f"{name}-{args.model_size.capitalize()}"
            
            forecasts['model'] = model_display_name
            all_forecast_dfs.append(forecasts)
            
            # Save intermediate results for this model
            model_file = os.path.join(output_dir, f"{model_display_name.lower()}_forecasts.csv")
            forecasts.to_csv(model_file, index=False)
            print(f"Saved {model_display_name} forecasts to {model_file}")
            
            completed_models.append(model_display_name)
            
            # Explicitly free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"CRITICAL ERROR running {name}: {e}")
            failed_models.append({'model': name, 'error': str(e)})
            import traceback
            traceback.print_exc()

    if not all_forecast_dfs:
        print("No results generated.")
        return

    # Consolidate and evaluate
    print("\nConsolidating all results...")
    full_forecasts = pd.concat(all_forecast_dfs, ignore_index=True)
    
    if args.append and existing_forecasts is not None:
        # Get names of models just run
        newly_run_models = full_forecasts['model'].unique()
        print(f"Merging with existing results. Overwriting models: {newly_run_models}")
        
        # Filter existing: remove models that were just re-run
        filtered_existing = existing_forecasts[~existing_forecasts['model'].isin(newly_run_models)]
        
        # Combine
        full_forecasts = pd.concat([filtered_existing, full_forecasts], ignore_index=True)

    full_forecasts['target_date'] = pd.to_datetime(full_forecasts['target_date'])
    
    full_forecast_file = os.path.join(output_dir, "backtest_predictions.csv")
    full_forecasts.to_csv(full_forecast_file, index=False)
    
    print("Calculating metrics...")
    # Pass train_data for MASE (strictly pre-backtest to avoid leakage)
    # Using data before the first origin to calculate the naive seasonal scale
    train_slice = df_nat[df_nat['ds'] < origins[0]].copy()
    metrics_df = evaluate_forecasts(full_forecasts, df_nat, train_data=train_slice, aggregate=True)
    
    # Save per-origin metrics
    print("Calculating metrics by origin...")
    metrics_by_origin_df = evaluate_forecasts(full_forecasts, df_nat, train_data=train_slice, aggregate=False)
    metrics_by_origin_file = os.path.join(output_dir, "backtest_metrics_by_origin.csv")
    metrics_by_origin_df.to_csv(metrics_by_origin_file, index=False)

    # Save run_info.json
    import json
    run_info = {
        'period_covered': f"{df_nat['ds'].min().date()} to {df_nat['ds'].max().date()}",
        'num_origins': len(origins),
        'horizons': horizons,
        'completed_models': completed_models,
        'failed_models': failed_models
    }
    with open(os.path.join(output_dir, "run_info.json"), 'w') as f:
        json.dump(run_info, f, indent=4)
    
    # Calculate Peak Metrics
    print("Calculating seasonal peak metrics...")
    # Prepare truth for peak metrics (needs target_date and true_value)
    peak_truth = df_nat.rename(columns={'ds': 'target_date', 'y': 'true_value'})
    
    # Calculate for each horizon and for 'latest'
    peak_results = []
    for h in [None, 1, 2, 4, 8]:
        h_peak = calculate_seasonal_peak_metrics(full_forecasts, peak_truth, horizon=h)
        peak_results.append(h_peak)
    peak_metrics_df = pd.concat(peak_results, ignore_index=True)

    # Generate RUN_REPORT.md
    report_file = os.path.join(output_dir, "RUN_REPORT.md")
    with open(report_file, 'w') as f:
        f.write("# National ILI Benchmark Run Report\n\n")
        f.write(f"- **Period Covered:** {df_nat['ds'].min().date()} to {df_nat['ds'].max().date()}\n")
        f.write(f"- **Number of Origins:** {len(origins)}\n")
        f.write(f"- **Horizons:** {horizons}\n")
        f.write(f"- **Completed Models:** {', '.join(completed_models)}\n")
        f.write(f"- **Failed Models:** {len(failed_models)}\n")
        for fm in failed_models:
            f.write(f"  - {fm['model']}: {fm['error']}\n")
        f.write("\n## Summary Metrics (MAE/WIS)\n\n")
        if not metrics_df.empty:
            summary = metrics_df.groupby('model')[['MAE', 'WIS']].mean().sort_values('MAE')
            f.write(summary.to_markdown())
        
    print(f"Run report saved to {report_file}")
    
    # Generate Visualizations
    print("Generating visualizations...")
    vis_dir = os.path.join(output_dir, "plots")
    # For trajectories, use truth with renamed columns
    plot_truth = df_nat.rename(columns={'ds': 'target_date', 'y': 'true_value'})
    plot_national_trajectories(full_forecasts, plot_truth, vis_dir, horizon=4)
    plot_best_model_heatmap(metrics_df, vis_dir, metric_name='MAE')
    plot_best_model_heatmap(metrics_df, vis_dir, metric_name='WIS')
    print(f"Visualizations saved to {vis_dir}")

    if not metrics_df.empty:
        metrics_file = os.path.join(output_dir, "backtest_summary.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        if not peak_metrics_df.empty:
            peak_file = os.path.join(output_dir, "all_models_peak_metrics.csv")
            peak_metrics_df.to_csv(peak_file, index=False)
            print(f"Peak Metrics: {peak_file}")
        
        print("\n" + "#"*40)
        print("NATIONAL ILI BENCHMARKING COMPLETE")
        print("#"*40)
        print(f"Forecasts: {full_forecast_file}")
        print(f"Metrics:   {metrics_file}")
        
        print("\nOverall Summary (MAE/WIS averaged):")
        summary = metrics_df.groupby('model')[['MAE', 'WIS']].mean().sort_values('MAE')
        print(summary)
    else:
        print("Warning: No metrics calculated. Check truth data overlap.")

if __name__ == "__main__":
    main()
