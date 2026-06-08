"""
Utilities for quantile-based probabilistic forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


# Standard 23 quantiles used by CDC and Influcast
INFLUCAST_QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
]


def samples_to_quantiles(
    samples: np.ndarray, 
    quantiles: List[float], 
    target_dates: Optional[List] = None
) -> pd.DataFrame:
    """
    Converts model samples to a long-format DataFrame of quantiles.

    Args:
        samples (np.ndarray): Array of shape (num_samples, horizon).
        quantiles (List[float]): List of quantiles to calculate (0.0 to 1.0).
        target_dates (Optional[List]): List of dates corresponding to the horizon.
            If None, defaults to integer steps [0, 1, ..., horizon-1].

    Returns:
        pd.DataFrame: Long-format DataFrame with columns ['ds', 'quantile', 'value'].
    """
    num_samples, horizon = samples.shape
    
    if target_dates is None:
        target_dates = list(range(horizon))
    
    if len(target_dates) != horizon:
        raise ValueError(f"Length of target_dates ({len(target_dates)}) must match horizon ({horizon})")

    results = []
    for q in quantiles:
        # Calculate quantile across the samples axis (axis 0)
        q_values = np.quantile(samples, q, axis=0)
        
        df_q = pd.DataFrame({
            'ds': target_dates,
            'quantile': q,
            'value': q_values
        })
        results.append(df_q)
    
    return pd.concat(results, ignore_index=True)
