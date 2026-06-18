"""Statistical baseline models for ILI forecasting using statsforecast."""

import pandas as pd
import numpy as np
from typing import List, Optional, Any
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive, 
    SeasonalNaive, 
    AutoETS, 
    AutoARIMA,
    RandomWalkWithDrift,
    WindowAverage
)
from statsforecast.utils import ConformalIntervals
from prophet import Prophet

from src.models.base import BaseForecaster


class StatsForecastWrapper(BaseForecaster):
    """Base wrapper for statsforecast models.
    
    Attributes:
        model_obj: A statsforecast model instance (e.g., Naive(), AutoARIMA()).
    """
    
    def __init__(self, model_obj: Any):
        """Initializes the wrapper with a specific statsforecast model object.
        
        Args:
            model_obj: A statsforecast model instance.
        """
        self.model_obj = model_obj

    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """Generates forecasts using the wrapped statsforecast model.
        
        Args:
            history: DataFrame with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Number of samples (not used by all models).
            
        Returns:
            DataFrame with columns 'ds', 'quantile', 'value'.
        """
        # Prepare data for StatsForecast
        # It expects columns: unique_id, ds, y
        sf_df = history[['ds', 'y']].copy()
        sf_df['unique_id'] = 'ILI'
        
        # Ensure ds is datetime
        sf_df['ds'] = pd.to_datetime(sf_df['ds'])
        
        # Initialize StatsForecast
        sf = StatsForecast(
            models=[self.model_obj],
            freq='W-SUN',
            n_jobs=1
        )
        
        if quantiles:
            # Statsforecast level parameter: 
            # level=[80, 90] produces quantiles [0.1, 0.9, 0.05, 0.95]
            levels = []
            for q in quantiles:
                level = abs(q - 0.5) * 2 * 100
                levels.append(level)
            levels = sorted(list(set(levels)))
            
            try:
                forecasts = sf.forecast(df=sf_df, h=horizon, level=levels)
            except Exception as e:
                # Fallback to point forecast if intervals fail
                forecasts = sf.forecast(df=sf_df, h=horizon)
                quantiles = None
        else:
            forecasts = sf.forecast(df=sf_df, h=horizon)
            
        # Convert to long format [ds, quantile, value]
        model_name = repr(self.model_obj)
        if '(' in model_name:
            model_name = model_name.split('(')[0]

        res_dfs = []
        
        # Point forecast as quantile 0.5 (approximately)
        res_dfs.append(pd.DataFrame({
            'ds': forecasts['ds'],
            'quantile': 0.5,
            'value': forecasts[model_name]
        }))
        
        if quantiles:
            for q in quantiles:
                if q == 0.5:
                    continue
                level = abs(q - 0.5) * 2 * 100
                suffix = 'lo' if q < 0.5 else 'hi'
                col_name = f"{model_name}-{suffix}-{level}"
                
                # Check if column exists (might be slight rounding diffs)
                if col_name not in forecasts.columns:
                    # find closest level
                    available_levels = [float(c.split('-')[-1]) for c in forecasts.columns if '-' in c]
                    if available_levels:
                        closest_level = min(available_levels, key=lambda x: abs(x - level))
                        col_name = f"{model_name}-{suffix}-{closest_level}"

                if col_name in forecasts.columns:
                    res_dfs.append(pd.DataFrame({
                        'ds': forecasts['ds'],
                        'quantile': q,
                        'value': forecasts[col_name]
                    }))

        return pd.concat(res_dfs, ignore_index=True)


class NaiveForecaster(StatsForecastWrapper):
    """Naive forecaster (persistence model)."""
    
    def __init__(self):
        """Initializes NaiveForecaster."""
        super().__init__(Naive())


class SeasonalNaiveForecaster(StatsForecastWrapper):
    """Seasonal Naive forecaster (seasonality=52)."""
    
    def __init__(self):
        """Initializes SeasonalNaiveForecaster."""
        super().__init__(SeasonalNaive(season_length=52))


class ETSForecaster(StatsForecastWrapper):
    """Auto-ETS forecaster."""
    
    def __init__(self):
        """Initializes ETSForecaster."""
        super().__init__(AutoETS(season_length=52))


class ARIMAForecaster(StatsForecastWrapper):
    """Auto-ARIMA forecaster."""
    
    def __init__(self):
        """Initializes ARIMAForecaster."""
        super().__init__(AutoARIMA(stepwise=True, approximation=True))


class SARIMAForecaster(StatsForecastWrapper):
    """Seasonal Auto-ARIMA forecaster (seasonality=52)."""
    
    def __init__(self):
        """Initializes SARIMAForecaster."""
        import warnings
        # Silence the matmul/overflow warnings from AutoARIMA optimization
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsforecast.arima")
        
        super().__init__(AutoARIMA(
            season_length=52, 
            stepwise=True, 
            approximation=True,
            max_P=1,
            max_Q=1,
            max_D=1
        ))


class DriftForecaster(StatsForecastWrapper):
    """Drift forecaster (random walk with drift)."""
    
    def __init__(self):
        """Initializes DriftForecaster."""
        super().__init__(RandomWalkWithDrift())


class MovingAverageForecaster(StatsForecastWrapper):
    """Moving Average forecaster (Window=52)."""
    
    def __init__(self, window_size: int = 52):
        """Initializes MovingAverageForecaster.
        
        Args:
            window_size: Size of the moving window.
        """
        # We use ConformalIntervals to provide prediction intervals for simple WindowAverage.
        # We set h=26 (half a year) as a safe calibration buffer for any reasonable ILI horizon.
        super().__init__(WindowAverage(
            window_size=window_size, 
            prediction_intervals=ConformalIntervals(n_windows=5, h=26)
        ))


class ProphetForecaster(BaseForecaster):
    """Prophet forecaster wrapper."""
    
    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """Generates forecasts using Prophet.
        
        Args:
            history: DataFrame with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Number of samples to draw for uncertainty.
            
        Returns:
            DataFrame with columns 'ds', 'quantile', 'value'.
        """
        # Prepare data
        m_df = history[['ds', 'y']].copy()
        m_df['ds'] = pd.to_datetime(m_df['ds'])
        
        # Initialize and fit
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=num_samples
        )
        m.fit(m_df)
        
        # Future dataframe
        future = m.make_future_dataframe(periods=horizon, freq='W-SUN', include_history=False)
        
        # Predict
        forecast_samples = m.predictive_samples(future)
        samples = forecast_samples['yhat'].T # (num_samples, horizon)
        
        from src.utils.quantiles import samples_to_quantiles
        
        if quantiles is None:
            quantiles = [0.5]
            
        return samples_to_quantiles(samples, quantiles, target_dates=future['ds'].tolist())
