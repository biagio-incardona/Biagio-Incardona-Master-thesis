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
        results = self.predict_batch([history], horizon, quantiles, num_samples)
        return results[0]

    def predict_batch(
        self, 
        histories: List[pd.DataFrame], 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000,
        batch_size: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """Generates forecasts for a batch of histories using TimeGPT.
        
        Args:
            histories: List of DataFrames, each with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Not used directly.
            batch_size: Not used directly (API handles multiple series).
            
        Returns:
            List of DataFrames, each with columns 'ds', 'quantile', 'value'.
        """
        # Prepare data for TimeGPT in Nixtla format
        dfs = []
        for i, h in enumerate(histories):
            temp_df = h[['ds', 'y']].copy()
            temp_df['unique_id'] = f'series_{i}'
            dfs.append(temp_df)
        
        input_df = pd.concat(dfs, ignore_index=True)
        
        # Determine levels for TimeGPT
        levels = []
        if quantiles:
            for q in quantiles:
                if q == 0.5:
                    continue
                level = round(abs(q - 0.5) * 2 * 100, 1)
                levels.append(level)
            levels = sorted(list(set(levels)))
        
        # API call
        forecasts = self.client.forecast(
            df=input_df,
            h=horizon,
            level=levels if levels else None,
            freq='W-SUN',
            model='timegpt-1'
        )
        
        # Ensure 'ds' is datetime
        forecasts['ds'] = pd.to_datetime(forecasts['ds'])
        
        results = []
        for i, history in enumerate(histories):
            # Filter for this series
            series_id = f'series_{i}'
            f = forecasts[forecasts['unique_id'] == series_id].copy()
            
            # Recalculate target dates locally to ensure alignment with our expected freq
            # even if the API returns slightly different timestamps
            last_date = pd.to_datetime(history['ds'].iloc[-1])
            expected_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1), 
                periods=horizon, 
                freq='W-SUN'
            )
            
            # Map API results to expected dates to handle potential API frequency shifts
            f['ds'] = expected_dates[:len(f)]
            
            res_dfs = []
            
            # Point forecast
            res_dfs.append(pd.DataFrame({
                'ds': f['ds'],
                'quantile': 0.5,
                'value': f['TimeGPT']
            }))
            
            if quantiles:
                for q in quantiles:
                    if q == 0.5:
                        continue
                        
                    level = round(abs(q - 0.5) * 2 * 100, 1)
                    suffix = 'lo' if q < 0.5 else 'hi'
                    
                    # Try possible column names
                    level_int = int(level) if level == int(level) else level
                    col_name = f"TimeGPT-{suffix}-{level_int}"
                    
                    if col_name not in f.columns:
                        col_name = f"TimeGPT-{suffix}-{float(level_int)}"
                    
                    if col_name in f.columns:
                        res_dfs.append(pd.DataFrame({
                            'ds': f['ds'],
                            'quantile': q,
                            'value': f[col_name]
                        }))
                    else:
                        # Fallback to nearest available or point forecast if all fails
                        potential = [c for c in f.columns if f"TimeGPT-{suffix}-" in c]
                        if potential:
                            avail = [float(c.split('-')[-1]) for c in potential]
                            best_idx = np.argmin([abs(a - level) for a in avail])
                            res_dfs.append(pd.DataFrame({
                                'ds': f['ds'],
                                'quantile': q,
                                'value': f[potential[best_idx]]
                            }))
            
            results.append(pd.concat(res_dfs, ignore_index=True))
            
        return results
