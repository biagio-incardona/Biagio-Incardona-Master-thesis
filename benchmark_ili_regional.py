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
        'Chronos': (ChronosForecaster, {'model_name': 'small'}), # Use small for regional to save time
        'TimesFM': (TimesFMForecaster, {}),
        'TiRex': (TiRexForecaster, {}),
    }

    if args.model:
        models_to_run = {args.model: all_models[args.model]}
    else:
        models_to_run = all_models

    sequential_models = ['Chronos', 'TimesFM', 'TiRex']
    
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
        
        for name, (model_cls, kwargs) in models_to_run.items():
            print(f"--- Running {name} for {region} ---")
            current_n_jobs = 1 if name in sequential_models else args.n_jobs
            
            try:
                model = model_cls(**kwargs)
                forecasts = run_backtest(region_df, model, origins, horizons, quantiles, n_jobs=current_n_jobs)
                forecasts['model'] = name
                forecasts['region'] = region
                region_forecasts.append(forecasts)
                
                if name not in completed_models:
                    completed_models.append(name)
                
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error in {name} for {region}: {e}")
                failed_models.append({'model': name, 'region': region, 'error': str(e)})

        if region_forecasts:
            reg_full_fcst = pd.concat(region_forecasts, ignore_index=True)
            reg_file = os.path.join(output_dir, f"{region}_forecasts.csv")
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
