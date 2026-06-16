"""Machine Learning baseline models for ILI forecasting using mlforecast."""

import pandas as pd
import numpy as np
from typing import List, Optional, Any
import optuna
from sklearn.metrics import mean_squared_error
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from utilsforecast.preprocessing import fill_gaps
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

from src.models.base import BaseForecaster


class MLForecastWrapper(BaseForecaster):
    """Wrapper for mlforecast models.
    
    Attributes:
        model_obj: A regression model (e.g., LGBMRegressor(), XGBRegressor()).
    """
    
    def __init__(self, model_obj: Any):
        """Initializes the wrapper with a specific regression model.
        
        Args:
            model_obj: A regression model instance.
        """
        self.model_obj = model_obj
        self.lags = [1, 2, 3, 4, 8, 12, 52]
        self.date_features = ['week', 'month']

    def tune(self, df: pd.DataFrame, n_trials: int = 20):
        """Tunes hyperparameters using Optuna.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns.
            n_trials: Number of Optuna trials.
        """
        # Prepare data
        df = df[['ds', 'y']].copy()
        df['unique_id'] = 'ILI'
        df['ds'] = pd.to_datetime(df['ds'])
        df = fill_gaps(df, freq='W-SUN')
        
        # Split into train and validation (last 8 weeks)
        if len(df) <= 12:
            print("Dataset too small for tuning, skipping.")
            return

        train_df = df.iloc[:-8]
        val_df = df.iloc[-8:]
        
        def objective(trial):
            params = self._suggest_params(trial)
            
            # Update model with suggested params
            model_class = type(self.model_obj)
            # Create a new instance of the model with tuned params
            base_params = {
                'random_state': 42,
                'n_jobs': 1
            }
            if isinstance(self.model_obj, lgb.LGBMRegressor):
                base_params['verbosity'] = -1
            
            current_model = model_class(**{**base_params, **params})
            
            fcst = MLForecast(
                models={'model': current_model},
                freq='W-SUN',
                lags=self.lags,
                date_features=self.date_features,
            )
            
            fcst.fit(train_df)
            preds = fcst.predict(h=8)
            
            mse = mean_squared_error(val_df['y'], preds['model'])
            return np.sqrt(mse)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Tuning complete. Best params: {study.best_params}")
        
        # Update self.model_obj with best params
        model_class = type(self.model_obj)
        base_params = {
            'random_state': 42,
            'n_jobs': 1
        }
        if isinstance(self.model_obj, lgb.LGBMRegressor):
            base_params['verbosity'] = -1
            
        self.model_obj = model_class(**{**base_params, **study.best_params})

    def _suggest_params(self, trial):
        """Suggests parameters for the specific model type. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _suggest_params")

    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """Generates forecasts using the wrapped ML model via mlforecast.
        
        Args:
            history: DataFrame with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Not used directly, as we use conformal prediction.
            
        Returns:
            DataFrame with columns 'ds', 'quantile', 'value'.
        """
        # Prepare data for MLForecast
        df = history[['ds', 'y']].copy()
        df['unique_id'] = 'ILI'
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure no gaps in weekly data (required for lags)
        df = fill_gaps(df, freq='W-SUN')
        
        # Initialize MLForecast
        fcst = MLForecast(
            models={'model': self.model_obj},
            freq='W-SUN',
            lags=self.lags,
            date_features=self.date_features,
        )
        
        if quantiles:
            # Map quantiles to levels for conformal prediction
            # PredictionIntervals uses levels (e.g., 80 means 10th and 90th quantiles)
            levels = []
            for q in quantiles:
                if q == 0.5:
                    continue
                level = round(abs(q - 0.5) * 2 * 100, 2)
                levels.append(level)
            levels = sorted(list(set(levels)))
            
            # Use Conformal Prediction (Recursive)
            # We use a small number of windows for speed in this baseline
            fcst.fit(
                df, 
                prediction_intervals=PredictionIntervals(n_windows=5, h=horizon)
            )
            forecasts = fcst.predict(h=horizon, level=levels)
        else:
            fcst.fit(df)
            forecasts = fcst.predict(h=horizon)
            
        # Convert to long format [ds, quantile, value]
        res_dfs = []
        
        # Point forecast (quantile 0.5)
        res_dfs.append(pd.DataFrame({
            'ds': forecasts['ds'],
            'quantile': 0.5,
            'value': forecasts['model']
        }))
        
        if quantiles:
            for q in quantiles:
                if q == 0.5:
                    continue
                level = round(abs(q - 0.5) * 2 * 100, 2)
                suffix = 'lo' if q < 0.5 else 'hi'
                
                # mlforecast format is model-lo-level or model-hi-level
                # It might format level as int if it's whole
                level_str = str(int(level)) if level == int(level) else str(level)
                col_name = f"model-{suffix}-{level_str}"
                
                if col_name in forecasts.columns:
                    res_dfs.append(pd.DataFrame({
                        'ds': forecasts['ds'],
                        'quantile': q,
                        'value': forecasts[col_name]
                    }))
                else:
                    # Fallback for slight rounding issues in column names
                    potential_cols = [c for c in forecasts.columns if f"model-{suffix}-" in c]
                    if potential_cols:
                        # Extract levels from names
                        avail_levels = [float(c.split('-')[-1]) for c in potential_cols]
                        closest_idx = np.argmin([abs(l - level) for l in avail_levels])
                        best_col = potential_cols[closest_idx]
                        res_dfs.append(pd.DataFrame({
                            'ds': forecasts['ds'],
                            'quantile': q,
                            'value': forecasts[best_col]
                        }))

        return pd.concat(res_dfs, ignore_index=True)


class LightGBMForecaster(MLForecastWrapper):
    """LightGBM forecaster."""
    
    def __init__(self):
        """Initializes LightGBMForecaster with default params."""
        model = lgb.LGBMRegressor(
            verbosity=-1,
            random_state=42,
            n_jobs=1
        )
        super().__init__(model)

    def _suggest_params(self, trial):
        """Suggests hyperparameters for LightGBM."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }


class XGBoostForecaster(MLForecastWrapper):
    """XGBoost forecaster."""
    
    def __init__(self):
        """Initializes XGBoostForecaster with default params."""
        model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=1
        )
        super().__init__(model)

    def _suggest_params(self, trial):
        """Suggests hyperparameters for XGBoost."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }


class CatBoostForecaster(MLForecastWrapper):
    """CatBoost forecaster."""
    
    def __init__(self):
        """Initializes CatBoostForecaster with default params."""
        model = CatBoostRegressor(
            silent=True,
            random_state=42,
            thread_count=1
        )
        super().__init__(model)

    def _suggest_params(self, trial):
        """Suggests hyperparameters for CatBoost."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        }


class RidgeForecaster(MLForecastWrapper):
    """Ridge regression forecaster (Linear model on lags)."""
    
    def __init__(self):
        """Initializes RidgeForecaster with default params."""
        model = Ridge(
            random_state=42
        )
        super().__init__(model)

    def _suggest_params(self, trial):
        """Suggests hyperparameters for Ridge."""
        return {
            'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
        }
