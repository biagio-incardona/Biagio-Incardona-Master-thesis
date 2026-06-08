"""Seasonal Peak Metrics for epidemic forecasting."""

import pandas as pd
import numpy as np
from typing import Dict

def calculate_seasonal_peak_metrics(forecasts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Peak Timing and Peak Intensity error for each model and each season.
    
    Args:
        forecasts: Long DataFrame [model, origin, target_date, quantile, value]
        truth: DataFrame [target_date, true_value]
        
    Returns:
        DataFrame with peak metrics per model and season.
    """
    # Use only median forecasts
    point_fcsts = forecasts[np.isclose(forecasts['quantile'], 0.5)].copy()
    
    # Identify seasons (standard Oct-May roughly)
    # For simplicity, let's group by the year of the peak
    results = []
    
    # Filter truth for the evaluation period
    truth = truth.set_index('target_date').sort_index()
    
    for model, m_group in point_fcsts.groupby('model'):
        # We need continuous trajectories. 
        # For rolling origins, we might have multiple forecasts for the same date.
        # Let's take the latest forecast available for each date (shortest horizon).
        m_group = m_group.sort_values('horizon').drop_duplicates('target_date')
        m_group = m_group.set_index('target_date').sort_index()
        
        # Calculate peaks per season (roughly)
        # For this project, we have 2023-2024 and 2024-2025
        seasons = [
            ('2023-10-01', '2024-06-01'),
            ('2024-10-01', '2025-06-01')
        ]
        
        for start, end in seasons:
            s_truth = truth.loc[start:end]
            s_pred = m_group.loc[start:end]
            
            if s_truth.empty or s_pred.empty:
                continue
                
            true_peak_date = s_truth['true_value'].idxmax()
            true_peak_val = s_truth['true_value'].max()
            
            pred_peak_date = s_pred['value'].idxmax()
            pred_peak_val = s_pred['value'].max()
            
            timing_error = (pred_peak_date - true_peak_date).days / 7.0 # weeks
            intensity_error = pred_peak_val - true_peak_val # relative error
            
            results.append({
                'model': model,
                'season': f"{start[:4]}-{end[:4]}",
                'peak_timing_error_weeks': timing_error,
                'peak_intensity_error_abs': intensity_error
            })
            
    return pd.DataFrame(results)
