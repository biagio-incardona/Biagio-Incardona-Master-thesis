"""
Rolling-origin backtest engine for model evaluation.
"""

import pandas as pd
from typing import List, Union, Optional
from src.models.base import BaseForecaster
from joblib import Parallel, delayed
from tqdm import tqdm


def run_backtest(
    df: pd.DataFrame, 
    model: BaseForecaster, 
    origins: List[Union[str, pd.Timestamp]], 
    horizons: List[int],
    quantiles: List[float],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Executes a rolling-origin backtest for a given model.

    Args:
        df (pd.DataFrame): Time series data with 'ds' and 'y' columns.
        model (BaseForecaster): An instance of a class implementing BaseForecaster.
        origins (List[Union[str, pd.Timestamp]]): List of dates to use as forecast origins.
        horizons (List[int]): List of horizons (in weeks) to evaluate.
        quantiles (List[float]): List of quantiles to predict.
        n_jobs (int): Number of jobs for parallel execution. Defaults to -1 (all cores).

    Returns:
        pd.DataFrame: Long-format DataFrame with columns:
            ['origin', 'target_date', 'horizon', 'quantile', 'value'].
    
    Raises:
        AssertionError: If future data leakage is detected during slicing.
    """
    # Ensure ds is datetime
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    origins = pd.to_datetime(origins)
    
    max_horizon = max(horizons)

    # Optimization for StatsForecast models
    from src.models.baselines import StatsForecastWrapper
    if isinstance(model, StatsForecastWrapper):
        from statsforecast import StatsForecast
        
        # Create a multi-series DataFrame where each series is one origin's history.
        # This allows StatsForecast to handle all origins in parallel efficiently.
        sf_df_list = []
        for origin in origins:
            # Slicing ensures no future leakage
            origin_history = df[df['ds'] <= origin][['ds', 'y']].copy()
            origin_history['unique_id'] = str(origin)
            sf_df_list.append(origin_history)
        
        sf_df = pd.concat(sf_df_list, ignore_index=True)
        
        # Map quantiles to levels
        levels = sorted(list(set([round(abs(q - 0.5) * 2 * 100, 2) for q in quantiles if q != 0.5])))
        
        sf = StatsForecast(
            models=[model.model_obj],
            freq='W-SUN',
            n_jobs=n_jobs
        )
        
        # Batch forecast all origins
        cv_res = sf.forecast(
            df=sf_df,
            h=max_horizon,
            level=levels
        )
        
        model_name = repr(model.model_obj).split('(')[0]
        res_dfs = []
        
        # Point forecast (0.5 quantile)
        point_df = cv_res[['unique_id', 'ds', model_name]].copy()
        point_df.columns = ['origin', 'target_date', 'value']
        point_df['quantile'] = 0.5
        res_dfs.append(point_df)
        
        # Probabilistic forecasts
        for level in levels:
            for side, q_factor in [('lo', -1), ('hi', 1)]:
                col = f"{model_name}-{side}-{level}"
                if col in cv_res.columns:
                    q = 0.5 + q_factor * (level / 200)
                    q_match = min(quantiles, key=lambda x: abs(x - q))
                    
                    q_df = cv_res[['unique_id', 'ds', col]].copy()
                    q_df.columns = ['origin', 'target_date', 'value']
                    q_df['quantile'] = q_match
                    res_dfs.append(q_df)
        
        forecast = pd.concat(res_dfs, ignore_index=True)
        forecast['origin'] = pd.to_datetime(forecast['origin'])
        forecast['target_date'] = pd.to_datetime(forecast['target_date'])
        forecast['horizon'] = ((forecast['target_date'] - forecast['origin']).dt.days / 7).round().astype(int)
        
        # Filter for requested horizons and quantiles
        forecast = forecast[forecast['horizon'].isin(horizons)]
        forecast = forecast[forecast['quantile'].isin(quantiles)]
        
        return forecast[['origin', 'target_date', 'horizon', 'quantile', 'value']].sort_values(['origin', 'target_date', 'quantile'])

    # Fallback for other models (e.g., Prophet)
    def process_origin(origin):
        # T-01-04: Mitigate Information Disclosure (Future Leakage)
        # Strictly slice history to ds <= origin
        history = df[df['ds'] <= origin].copy()
        
        # Verify no future leakage
        if history['ds'].max() > origin:
             raise ValueError(f"Leakage detected: history max date {history['ds'].max()} > origin {origin}")
        
        # Generate forecasts
        forecast = model.predict(
            history=history, 
            horizon=max_horizon, 
            quantiles=quantiles
        )
        
        # Add origin and calculate horizon
        forecast['origin'] = origin
        forecast['horizon'] = ((forecast['ds'] - origin).dt.days / 7).round().astype(int)
        
        # Filter and rename
        forecast = forecast[forecast['horizon'].isin(horizons)]
        forecast = forecast.rename(columns={'ds': 'target_date'})
        
        return forecast[['origin', 'target_date', 'horizon', 'quantile', 'value']]

    # Use Parallel to speed up backtesting across origins
    if n_jobs == 1:
        # Optimization: Use predict_batch if model supports it to leverage vectorization/GPU
        if hasattr(model, 'predict_batch'):
            print(f"Using batch prediction for {type(model).__name__} across {len(origins)} origins...")
            histories = [df[df['ds'] <= origin].copy() for origin in origins]
            
            # Generate forecasts in one go
            batch_results = model.predict_batch(
                histories=histories, 
                horizon=max_horizon, 
                quantiles=quantiles
            )
            
            all_forecasts = []
            for origin, forecast in zip(origins, batch_results):
                # Add origin and calculate horizon
                forecast['origin'] = origin
                forecast['horizon'] = ((forecast['ds'] - origin).dt.days / 7).round().astype(int)
                
                # Filter and rename
                forecast = forecast[forecast['horizon'].isin(horizons)]
                forecast = forecast.rename(columns={'ds': 'target_date'})
                all_forecasts.append(forecast[['origin', 'target_date', 'horizon', 'quantile', 'value']])
        else:
            all_forecasts = [process_origin(origin) for origin in tqdm(origins, desc="Backtesting")]
    else:
        all_forecasts = Parallel(n_jobs=n_jobs)(
            delayed(process_origin)(origin) for origin in tqdm(origins, desc="Backtesting")
        )

    if not all_forecasts:
        return pd.DataFrame(columns=['origin', 'target_date', 'horizon', 'quantile', 'value'])

    return pd.concat(all_forecasts, ignore_index=True).sort_values(['origin', 'target_date', 'quantile'])
