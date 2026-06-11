"""TimeGPT Foundation Model wrapper for ILI forecasting."""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Any
import logging

try:
    from nixtla import NixtlaClient
    HAS_NIXTLA = True
except ImportError:
    HAS_NIXTLA = False

from src.models.base import BaseForecaster


class TimeGPTForecaster(BaseForecaster):
    """Wrapper for Nixtla TimeGPT Foundation Model via API.
    
    Attributes:
        api_key: Nixtla API key.
        client: NixtlaClient instance.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initializes the wrapper.
        
        Args:
            api_key: Nixtla API key. If None, looks for TIMEGPT_TOKEN env var.
        """
        if not HAS_NIXTLA:
            raise ImportError(
                "Nixtla is not installed. Please install it with: pip install nixtla"
            )
            
        self.api_key = api_key or os.environ.get('TIMEGPT_TOKEN')
        if not self.api_key:
            logging.warning("TimeGPT API key not found. Ensure TIMEGPT_TOKEN is set.")
            
        self.client = NixtlaClient(api_key=self.api_key)

    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """Generates forecasts using TimeGPT.
        
        Args:
            history: DataFrame with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Not used directly by TimeGPT.
            
        Returns:
            DataFrame with columns 'ds', 'quantile', 'value'.
        """
        # Prepare data for TimeGPT
        df = history[['ds', 'y']].copy()
        df['unique_id'] = 'ILI'
        
        if quantiles:
            # Map quantiles to levels (1-99)
            # TimeGPT uses level=[80, 90] etc.
            levels = []
            for q in quantiles:
                if q == 0.5:
                    continue
                level = round(abs(q - 0.5) * 2 * 100, 1)
                levels.append(level)
            levels = sorted(list(set(levels)))
            
            forecasts = self.client.forecast(
                df=df,
                h=horizon,
                level=levels,
                freq='W-SUN',
                model='timegpt-1'
            )
        else:
            forecasts = self.client.forecast(
                df=df,
                h=horizon,
                freq='W-SUN',
                model='timegpt-1'
            )
            
        # Convert to long format
        res_dfs = []
        
        # Point forecast (TimeGPT returns it in column 'TimeGPT')
        res_dfs.append(pd.DataFrame({
            'ds': forecasts['ds'],
            'quantile': 0.5,
            'value': forecasts['TimeGPT']
        }))
        
        if quantiles:
            for q in quantiles:
                if q == 0.5:
                    continue
                level = round(abs(q - 0.5) * 2 * 100, 1)
                suffix = 'lo' if q < 0.5 else 'hi'
                
                # Check for various possible column name formats
                # e.g., TimeGPT-lo-95 or TimeGPT-lo-95.0
                level_int = int(level) if level == int(level) else level
                col_name = f"TimeGPT-{suffix}-{level_int}"
                
                if col_name not in forecasts.columns:
                    # Try with .0
                    col_name = f"TimeGPT-{suffix}-{float(level_int)}"
                
                if col_name in forecasts.columns:
                    res_dfs.append(pd.DataFrame({
                        'ds': forecasts['ds'],
                        'quantile': q,
                        'value': forecasts[col_name]
                    }))
                else:
                    # Fallback to closest
                    potential = [c for c in forecasts.columns if f"TimeGPT-{suffix}-" in c]
                    if potential:
                        # Extract numeric part
                        avail = [float(c.split('-')[-1]) for c in potential]
                        best_idx = np.argmin([abs(a - level) for a in avail])
                        res_dfs.append(pd.DataFrame({
                            'ds': forecasts['ds'],
                            'quantile': q,
                            'value': forecasts[potential[best_idx]]
                        }))

        return pd.concat(res_dfs, ignore_index=True)
