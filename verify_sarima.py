import os
import sys
import pandas as pd

# Add the current directory to python path so it can import from src
sys.path.append(os.path.abspath('.'))
from src.evaluation.metrics import evaluate_forecasts

# Load SARIMA forecasts
forecasts_path = 'results/national/sarima_forecasts.csv'
if not os.path.exists(forecasts_path):
    print(f"Error: Forecast file not found at {forecasts_path}")
    sys.exit(1)

forecasts = pd.read_csv(forecasts_path)
forecasts['target_date'] = pd.to_datetime(forecasts['target_date'])
forecasts['model'] = 'SARIMA'

# Load truth data (Gold Dataset)
truth_path = 'data/processed/ili_gold.csv'
if not os.path.exists(truth_path):
    print(f"Error: Gold dataset not found at {truth_path}")
    sys.exit(1)

truth = pd.read_csv(truth_path)
truth['ds'] = pd.to_datetime(truth['ds'])
truth_ita = truth[truth['region'] == 'italia']

# Get the first origin date to partition training data for metrics scale (e.g. for MASE)
first_origin = pd.to_datetime(forecasts['origin'].min())
train_slice = truth_ita[truth_ita['ds'] < first_origin].copy()

# Run the evaluation metrics calculation
metrics = evaluate_forecasts(forecasts, truth_ita, train_data=train_slice)

print("\n=== SARIMA METRICS VS FORECAST HORIZON ===")
print(metrics[['model', 'horizon', 'MAE', 'WIS']].to_string(index=False))
