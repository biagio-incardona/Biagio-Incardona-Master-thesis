"""Seasonal Peak Metrics for epidemic forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List

def calculate_seasonal_peak_metrics(forecasts: pd.DataFrame, truth: pd.DataFrame, horizon: Optional[int] = None) -> pd.DataFrame:
    """
    Calculates Peak Timing and Peak Intensity error for each model and each season.
    
    Args:
        forecasts: Long DataFrame [model, origin, target_date, quantile, value, horizon]
        truth: DataFrame [target_date, true_value]
        horizon: Optional horizon to filter for (e.g. 1, 2, 4, 8). If None, takes 
                the latest available forecast for each date.
        
    Returns:
        DataFrame with peak metrics per model and season.
    """
    # Use only median forecasts
    point_fcsts = forecasts[np.isclose(forecasts['quantile'], 0.5)].copy()
    
    results = []
    
    # Filter truth for the evaluation period
    truth = truth.set_index('target_date').sort_index()
    
    for model, m_group in point_fcsts.groupby('model'):
        # Selection logic based on horizon
        if horizon is not None:
            # Filter for specific horizon
            m_group = m_group[m_group['horizon'] == horizon]
        else:
            # Take the latest forecast available for each date (shortest horizon)
            m_group = m_group.sort_values('horizon').drop_duplicates('target_date')
        
        m_group = m_group.set_index('target_date').sort_index()
        
        # Automatically detect seasons: October to May
        years = sorted(list(set(truth.index.year)))
        seasons = []
        for y in years:
            start = f"{y}-10-01"
            end = f"{y+1}-06-01"
            # Only add if we have some data in this range
            if not truth.loc[start:end].empty:
                seasons.append((start, end))
        
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
            intensity_error_rel = (pred_peak_val - true_peak_val) / true_peak_val
            
            results.append({
                'model': model,
                'season': f"{start[:4]}-{end[:4]}",
                'horizon': horizon if horizon is not None else 'latest',
                'peak_timing_error_weeks': timing_error,
                'peak_intensity_error_rel': intensity_error_rel
            })
            
    return pd.DataFrame(results)
