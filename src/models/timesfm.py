"""TimesFM Foundation Model wrapper for ILI forecasting."""

import pandas as pd
import numpy as np
from typing import List, Optional, Any
import logging

try:
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    HAS_TIMESFM = True
    
    # Fix for TypeError: TimesFM_2p5_200M_torch.__init__() got an unexpected keyword argument 'proxies'
    # This happens in some environments (e.g. Colab) where huggingface_hub passes proxies to _from_pretrained
    import inspect
    sig = inspect.signature(TimesFM_2p5_200M_torch.__init__)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_kwargs:
        _original_init = TimesFM_2p5_200M_torch.__init__
        def _patched_init(self, *args, **kwargs):
            # Filter kwargs to only those accepted by _original_init
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters and k != 'self'}
            return _original_init(self, *args, **valid_kwargs)
        TimesFM_2p5_200M_torch.__init__ = _patched_init
except ImportError:
    HAS_TIMESFM = False

from src.models.base import BaseForecaster
from src.utils.quantiles import INFLUCAST_QUANTILES


class TimesFMForecaster(BaseForecaster):
    """Wrapper for Google TimesFM Foundation Model.
    
    Attributes:
        model_name: Name of the TimesFM model (e.g., "google/timesfm-2.5-200m-pytorch").
        tfm: The loaded TimesFM model instance.
    """
    
    def __init__(self, model_name: str = "google/timesfm-2.5-200m-pytorch", device: Optional[str] = None, batch_size: int = 16):
        """Initializes the wrapper and loads the model.
        
        Args:
            model_name: The HuggingFace model ID.
            device: Device to use. If None, auto-detects.
            batch_size: Default batch size for inference.
        """
        if not HAS_TIMESFM:
            raise ImportError(
                "TimesFM is not installed. Please install it with: "
                "pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'"
            )
            
        self.model_name = model_name
        
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size

        print(f"Loading TimesFM model: {model_name} on {self.device}...")
        
        # Initialize and load
        # Force torch_compile=False for stability on macOS
        self.tfm = TimesFM_2p5_200M_torch.from_pretrained(
            model_name, 
            torch_compile=False
        )
        
        # Move model to device
        self.tfm.model.to(self.device)
        self.tfm.model.device = self.device
        
        # Default compilation (can be re-compiled in predict if needed)
        self.current_max_context = 512
        self.current_max_horizon = 32
        self.tfm.compile(ForecastConfig(
            max_context=self.current_max_context, 
            max_horizon=self.current_max_horizon
        ))

    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """Generates forecasts using TimesFM.
        
        Args:
            history: DataFrame with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Not used directly by TimesFM 2.5 (it's quantile-based).
            
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
        """Generates forecasts for a batch of histories.
        
        Args:
            histories: List of DataFrames, each with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Not used directly.
            batch_size: Number of histories to process in one call.
            
        Returns:
            List of DataFrames, each with columns 'ds', 'quantile', 'value'.
        """
        import torch
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # Re-compile if horizon exceeds current max or any history exceeds current context
        max_h_len = max([len(h) for h in histories])
        if horizon > self.current_max_horizon or max_h_len > self.current_max_context:
            self.current_max_horizon = max(horizon, self.current_max_horizon)
            self.current_max_context = max(max_h_len, self.current_max_context)
            print(f"Re-compiling TimesFM for context={self.current_max_context}, horizon={self.current_max_horizon}...")
            self.tfm.compile(ForecastConfig(
                max_context=self.current_max_context, 
                max_horizon=self.current_max_horizon
            ))

        all_means = []
        all_tfm_quantiles = []
        
        # Generate forecasts in chunks
        for i in range(0, len(histories), batch_size):
            chunk = histories[i : i + batch_size]
            inputs = [h['y'].values for h in chunk]
            
            print(f"TimesFM batch inference for {len(chunk)} series (origin {i}-{i+len(chunk)})...")
            mean, tfm_qs = self.tfm.forecast(horizon, inputs)
            all_means.append(mean)
            all_tfm_quantiles.append(tfm_qs)
            
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate results
        means = np.concatenate(all_means, axis=0) # (num_series, horizon)
        tfm_quantiles = np.concatenate(all_tfm_quantiles, axis=0) # (num_series, horizon, 10)
        
        tfm_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        for idx, history in enumerate(histories):
            # Extract for this series
            y_deciles = tfm_quantiles[idx, :, 1:10] # (horizon, 9)
            y_mean = means[idx, :] # (horizon,)
            
            # Calculate target dates
            last_date = pd.to_datetime(history['ds'].iloc[-1])
            target_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1), 
                periods=horizon, 
                freq='W-SUN'
            ).tolist()
            
            if quantiles is None:
                quantiles = [0.5]
                
            res_dfs = []
            for i, target_date in enumerate(target_dates):
                step_deciles = y_deciles[i, :]
                for q in quantiles:
                    if q == 0.5:
                        val = y_mean[i]
                    else:
                        val = np.interp(q, tfm_levels, step_deciles)
                    
                    res_dfs.append(pd.DataFrame({
                        'ds': [target_date],
                        'quantile': [q],
                        'value': [val]
                    }))
            results.append(pd.concat(res_dfs, ignore_index=True))
                
        return results
