"""
Regional benchmarking script for ILI forecasting.
Executes backtests for all models across all available regions.
"""

import torch
import os
import argparse
import pandas as pd
import numpy as np
import gc
import warnings
import logging
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
from src.utils.visualizations import plot_regional_performance, plot_best_model_map

def main():
    parser = argparse.ArgumentParser(description='Regional ILI Model Benchmarking')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick check with few regions/origins')
    parser.add_argument('--model', type=str, default=None, help='Run only a specific model')
    parser.add_argument('--region', type=str, default=None, help='Run only a specific region')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs for parallel execution')
    parser.add_argument('--min-train', type=int, default=156, help='Minimum training weeks')
    parser.add_argument('--step', type=int, default=8, help='Step size between origins (default 8 for regional speed)')
    parser.add_argument('--horizons', type=str, default='1,2,4,8', help='Comma-separated horizons')
    parser.add_argument('--append', action='store_true', help='Append to existing regional forecasts')
    parser.add_argument('--model-size', type=str, default='small', 
                        choices=['tiny', 'mini', 'small', 'base', 'large', 
                                 'v2', 'bolt-tiny', 'bolt-mini', 'bolt-small', 'bolt-base'],
                        help='Size of foundation models (Chronos, etc.)')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples for foundation models')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for foundation model inference')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='Override default device for deep learning models')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv('data/processed/ili_gold.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Exclude national
    df_reg = df[df['region'] != 'italia'].copy()
    regions = sorted(df_reg['region'].unique())
    
    if args.region:
        if args.region in regions:
            regions = [args.region]
        else:
            print(f"Region {args.region} not found.")
            return
            
    if args.dry_run:
        regions = regions[:2]
        
    print(f"Total regions to evaluate: {len(regions)}")
    
    horizons = [int(h) for h in args.horizons.split(',')]
    quantiles = INFLUCAST_QUANTILES
    
    output_dir = 'results/regional'
    os.makedirs(output_dir, exist_ok=True)
    
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
        'TiRex': (TiRexForecaster, {'device': args.device}),
        'TimeGPT': (TimeGPTForecaster, {}),
    }

    if args.model:
        requested_models = [m.strip() for m in args.model.split(',')]
        models_to_run = {m: all_models[m] for m in requested_models if m in all_models}
        if not models_to_run:
            print(f"None of the requested models {args.model} were found in the registry.")
            return
    else:
        models_to_run = all_models

    sequential_models = ['Chronos', 'TimesFM', 'TiRex', 'TimeGPT']
    
    completed_models = []
    failed_models = []

    for region in regions:
        print(f"\n" + "#"*60)
        print(f"PROCESSING REGION: {region.upper()}")
        print("#"*60)
        
        region_df = df_reg[df_reg['region'] == region].copy().sort_values('ds')
        
        # Dynamic Origin Generation
        max_horizon = max(horizons)
        available_dates = region_df['ds'].values
        start_idx = args.min_train
        end_idx = len(available_dates) - max_horizon
        
        if start_idx >= end_idx:
            print(f"Skipping {region}: Not enough data (Total: {len(region_df)})")
            continue
            
        origin_indices = range(start_idx, end_idx, args.step)
        origins = pd.to_datetime(available_dates[origin_indices])
        
        if args.dry_run:
            origins = origins[:2]

        region_forecasts = []
        
        # Load existing forecasts if appending
        reg_file = os.path.join(output_dir, f"{region}_forecasts.csv")
        existing_fcsts = None
        if args.append and os.path.exists(reg_file):
            print(f"Loading existing forecasts from {reg_file} for appending...")
            existing_fcsts = pd.read_csv(reg_file)
            existing_fcsts['origin'] = pd.to_datetime(existing_fcsts['origin'])
            existing_fcsts['target_date'] = pd.to_datetime(existing_fcsts['target_date'])
        
        for name, (model_cls, kwargs) in models_to_run.items():
            model_display_name = name
            if name == 'Chronos':
                model_display_name = f"{name}-{args.model_size.capitalize()}"
                
            print(f"--- Running {model_display_name} for {region} ---")
            current_n_jobs = 1 if name in sequential_models else args.n_jobs
            
            try:
                model = model_cls(**kwargs)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
                    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
                    forecasts = run_backtest(region_df, model, origins, horizons, quantiles, n_jobs=current_n_jobs)
                forecasts['model'] = model_display_name
                forecasts['region'] = region
                region_forecasts.append(forecasts)
                
                if model_display_name not in completed_models:
                    completed_models.append(model_display_name)
                
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error in {name} for {region}: {e}")
                failed_models.append({'model': model_display_name, 'region': region, 'error': str(e)})

        # Combine new forecasts with existing ones if applicable
        combined_forecasts = []
        if existing_fcsts is not None:
            # Filter out models we just ran
            newly_run = [f['model'].iloc[0] for f in region_forecasts] if region_forecasts else []
            existing_fcsts = existing_fcsts[~existing_fcsts['model'].isin(newly_run)]
            
            # Add existing models to completed list
            for m in existing_fcsts['model'].unique():
                if m not in completed_models:
                    completed_models.append(m)
                    
            combined_forecasts.append(existing_fcsts)
            
        if region_forecasts:
            combined_forecasts.append(pd.concat(region_forecasts, ignore_index=True))
            
        if combined_forecasts:
            reg_full_fcst = pd.concat(combined_forecasts, ignore_index=True)
            reg_full_fcst.to_csv(reg_file, index=False)
            
            # Evaluate locally for this region
            train_slice = region_df[region_df['ds'] < origins[0]].copy()
            reg_metrics = evaluate_forecasts(reg_full_fcst, region_df, train_data=train_slice)
            reg_metrics['region'] = region
            reg_metrics.to_csv(os.path.join(output_dir, f"{region}_metrics.csv"), index=False)

            # Calculate and save regional peak metrics
            print(f"Calculating seasonal peak metrics for {region}...")
            peak_truth = region_df.rename(columns={'ds': 'target_date', 'y': 'true_value'})
            peak_results = []
            for h in [None, 1, 2, 4, 8]:
                h_peak = calculate_seasonal_peak_metrics(reg_full_fcst, peak_truth, horizon=h)
                peak_results.append(h_peak)
            reg_peak_metrics_df = pd.concat(peak_results, ignore_index=True)
            reg_peak_metrics_df['region'] = region
            reg_peak_metrics_df.to_csv(os.path.join(output_dir, f"{region}_peak_metrics.csv"), index=False)

    # Consolidate regional metrics and peak metrics
    all_metrics = []
    all_peak_metrics = []
    for r in regions:
        metrics_file = os.path.join(output_dir, f"{r}_metrics.csv")
        peak_file = os.path.join(output_dir, f"{r}_peak_metrics.csv")
        if os.path.exists(metrics_file):
            all_metrics.append(pd.read_csv(metrics_file))
        if os.path.exists(peak_file):
            all_peak_metrics.append(pd.read_csv(peak_file))
            
    if all_metrics:
        consolidated_metrics_df = pd.concat(all_metrics, ignore_index=True)
        consolidated_metrics_df.to_csv(os.path.join(output_dir, "all_regions_metrics.csv"), index=False)
        print(f"Consolidated regional metrics saved to {os.path.join(output_dir, 'all_regions_metrics.csv')}")
        
    if all_peak_metrics:
        consolidated_peak_df = pd.concat(all_peak_metrics, ignore_index=True)
        consolidated_peak_df.to_csv(os.path.join(output_dir, "all_regions_peak_metrics.csv"), index=False)
        print(f"Consolidated regional peak metrics saved to {os.path.join(output_dir, 'all_regions_peak_metrics.csv')}")

    # Save run_info.json
    import json
    run_info = {
        'period_covered': f"{df_reg['ds'].min().date()} to {df_reg['ds'].max().date()}",
        'num_regions': len(regions),
        'horizons': horizons,
        'completed_models': completed_models,
        'failed_models': failed_models
    }
    with open(os.path.join(output_dir, "run_info.json"), 'w') as f:
        json.dump(run_info, f, indent=4)

    # Generate RUN_REPORT.md
    report_file = os.path.join(output_dir, "RUN_REPORT.md")
    with open(report_file, 'w') as f:
        f.write("# Regional ILI Benchmark Run Report\n\n")
        f.write(f"- **Period Covered:** {df_reg['ds'].min().date()} to {df_reg['ds'].max().date()}\n")
        f.write(f"- **Regions Evaluated:** {len(regions)}\n")
        f.write(f"- **Horizons:** {horizons}\n")
        f.write(f"- **Completed Models:** {', '.join(completed_models)}\n")
        f.write(f"- **Failed Instances:** {len(failed_models)}\n")
        for fm in failed_models:
            f.write(f"  - {fm['model']} in {fm['region']}: {fm['error']}\n")

    # Generate Aggregate Regional Plots
    print("\nGenerating regional aggregate visualizations...")
    plot_regional_performance(output_dir, os.path.join(output_dir, "plots"))
    plot_best_model_map(output_dir, os.path.join(output_dir, "plots"))
    print(f"Regional plots saved to {os.path.join(output_dir, 'plots')}")

if __name__ == "__main__":
    main()
