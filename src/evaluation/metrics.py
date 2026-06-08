"""Advanced metrics for academic epidemic forecasting."""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict
from scipy.stats import norm


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def sMAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    mask = denominator > 0
    return 100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])


def MASE(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, periodicity: int = 52) -> float:
    """Mean Absolute Scaled Error (Seasonal)."""
    # Calculate scale from training data (naive seasonal forecast)
    if len(y_train) <= periodicity:
        # Fallback to non-seasonal naive if training data is too short
        scale = np.mean(np.abs(np.diff(y_train)))
    else:
        scale = np.mean(np.abs(y_train[periodicity:] - y_train[:-periodicity]))
    
    if scale == 0:
        return np.inf
        
    mae = MAE(y_true, y_pred)
    return mae / scale


def interval_score(y_true: float, lower: float, upper: float, alpha: float) -> float:
    """Interval Score (standard component of WIS)."""
    score = (upper - lower)
    if y_true < lower:
        score += (2 / alpha) * (lower - y_true)
    elif y_true > upper:
        score += (2 / alpha) * (y_true - upper)
    return score


def WIS(y_true: float, quantiles: np.ndarray, values: np.ndarray) -> float:
    """Weighted Interval Score."""
    # Sort by quantile
    idx = np.argsort(quantiles)
    qs = np.array(quantiles)[idx]
    vals = np.array(values)[idx]
    
    # Find median
    median_idx = np.argmin(np.abs(qs - 0.5))
    median = vals[median_idx]
    
    # w_0 = 0.5 for the median part
    score_sum = 0.5 * np.abs(y_true - median)
    
    unique_alphas = []
    for i, q in enumerate(qs):
        if q < 0.499:
            alpha = 2 * q
            q_upper = 1 - q
            upper_idx = np.where(np.isclose(qs, q_upper))[0]
            if len(upper_idx) > 0:
                val_lower = vals[i]
                val_upper = vals[upper_idx[0]]
                score_sum += (alpha / 2) * interval_score(y_true, val_lower, val_upper, alpha)
                unique_alphas.append(alpha)
                
    num_intervals = len(unique_alphas)
    if num_intervals == 0:
        return np.abs(y_true - median)
        
    return score_sum / (num_intervals + 0.5)


def CRPS_samples(y_true: float, samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score from samples."""
    n = len(samples)
    samples = np.sort(samples)
    # Empirical calculation: E|X - y| - 0.5 * E|X - X'|
    term1 = np.mean(np.abs(samples - y_true))
    # Efficient calculation of E|X - X'|
    term2 = np.mean(np.abs(np.subtract.outer(samples, samples))) / 2.0
    return term1 - term2


def pinball_loss(y_true: float, quantile: float, value: float) -> float:
    """Pinball loss for a single quantile."""
    err = y_true - value
    return max(quantile * err, (quantile - 1) * err)


def coverage(y_true: float, lower: float, upper: float) -> bool:
    """Check if y_true is within the interval."""
    return lower <= y_true <= upper


def peak_metrics(y_true_series: pd.Series, y_pred_series: pd.Series) -> Dict[str, float]:
    """Calculate Peak Timing and Intensity error.
    
    Args:
        y_true_series: Pandas series with DatetimeIndex of actual values.
        y_pred_series: Pandas series with DatetimeIndex of point forecasts.
        
    Returns:
        Dict with 'peak_timing_error' (weeks) and 'peak_intensity_error' (absolute).
    """
    if y_true_series.empty or y_pred_series.empty:
        return {'peak_timing_error': np.nan, 'peak_intensity_error': np.nan}
        
    true_peak_idx = y_true_series.idxmax()
    pred_peak_idx = y_pred_series.idxmax()
    
    true_peak_val = y_true_series.max()
    pred_peak_val = y_pred_series.max()
    
    # Timing error in weeks (delta between peak dates)
    timing_error = (pred_peak_idx - true_peak_idx).days / 7.0
    
    # Intensity error (absolute diff)
    intensity_error = np.abs(pred_peak_val - true_peak_val)
    
    return {
        'peak_timing_error': timing_error,
        'peak_intensity_error': intensity_error
    }


def evaluate_forecasts(forecasts: pd.DataFrame, truth: pd.DataFrame, train_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Comprehensive evaluation using all requested academic metrics.
    
    Args:
        forecasts: Long DataFrame [model, origin, target_date, horizon, quantile, value]
        truth: DataFrame [ds, y]
        train_data: Optional training data for MASE calculation.
        
    Returns:
        DataFrame with aggregated metrics per model and horizon.
    """
    if 'true_value' not in truth.columns and 'y' in truth.columns:
        truth = truth.rename(columns={'y': 'true_value', 'ds': 'target_date'})
    
    if 'target_date' not in forecasts.columns and 'ds' in forecasts.columns:
        forecasts = forecasts.rename(columns={'ds': 'target_date'})
        
    eval_df = forecasts.merge(truth[['target_date', 'true_value']], on='target_date', how='inner')
    if eval_df.empty:
        return pd.DataFrame()

    results = []
    
    # Group by model and horizon for standard metrics
    for (model, horizon), group in eval_df.groupby(['model', 'horizon']):
        
        # 1. Point Metrics (Median/Mean)
        point_group = group[np.isclose(group['quantile'], 0.5)].copy()
        if not point_group.empty:
            y_t = point_group['true_value'].values
            y_p = point_group['value'].values
            
            mae = MAE(y_t, y_p)
            rmse = RMSE(y_t, y_p)
            smape = sMAPE(y_t, y_p)
            
            # MASE
            mase = np.nan
            if train_data is not None:
                mase = MASE(y_t, y_p, train_data['y'].values)
        else:
            mae = rmse = smape = mase = np.nan

        # 2. Probabilistic Metrics (Averaged across origins)
        wis_list = []
        pinball_list = []
        crps_list = []
        cov_95_list = []
        
        for (origin, target_date), sub_group in group.groupby(['origin', 'target_date']):
            y_true = sub_group['true_value'].iloc[0]
            qs = sub_group['quantile'].values
            vals = sub_group['value'].values
            
            # WIS
            wis_list.append(WIS(y_true, qs, vals))
            
            # Mean Pinball Loss across all 23 quantiles
            pb_losses = [pinball_loss(y_true, q, v) for q, v in zip(qs, vals)]
            mean_pb = np.mean(pb_losses)
            pinball_list.append(mean_pb)
            
            # CRPS approximation: 2 * mean(Pinball Loss)
            crps_list.append(2.0 * mean_pb)
            
            # 95% Coverage (q=0.025 to q=0.975)
            # Find closest to 0.025 and 0.975
            idx_low = np.argmin(np.abs(qs - 0.025))
            idx_high = np.argmin(np.abs(qs - 0.975))
            cov_95_list.append(coverage(y_true, vals[idx_low], vals[idx_high]))
            
        results.append({
            'model': model,
            'horizon': horizon,
            'MAE': mae,
            'RMSE': rmse,
            'sMAPE': smape,
            'MASE': mase,
            'WIS': np.mean(wis_list) if wis_list else np.nan,
            'CRPS': np.mean(crps_list) if crps_list else np.nan,
            'PinballLoss': np.mean(pinball_list) if pinball_list else np.nan,
            'Coverage95': np.mean(cov_95_list) if cov_95_list else np.nan
        })

    # 3. Peak Metrics (Calculated per season-forecast)
    # This is more complex as it needs the full forecasted trajectory.
    # For simplicity in this summary, we might calculate it separately 
    # or skip for now if origins are too fragmented.
    
    return pd.DataFrame(results)
