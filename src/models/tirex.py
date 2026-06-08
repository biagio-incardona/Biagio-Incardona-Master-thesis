import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Any
import logging

try:
    from tirex import load_model
    HAS_TIREX = True
except ImportError:
    HAS_TIREX = False

from src.models.base import BaseForecaster


class TiRexForecaster(BaseForecaster):
    """Wrapper for NX-AI TiRex (xLSTM-based) Foundation Model.
    
    Attributes:
        model_name: Name of the TiRex model (e.g., "NX-AI/TiRex").
        model: The loaded TiRex model.
    """
    
    def __init__(self, model_name: str = "NX-AI/TiRex", device: Optional[str] = None, batch_size: int = 16):
        """Initializes the wrapper and loads the model.
        
        Args:
            model_name: The HuggingFace model ID.
            device: Device to use. If None, auto-detects.
            batch_size: Default batch size for inference.
        """
        if not HAS_TIREX:
            raise ImportError(
                "TiRex is not installed. Please install it with: "
                "pip install git+https://github.com/NX-AI/tirex.git"
            )
            
        self.model_name = model_name
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        
        print(f"Loading TiRex model: {model_name} on {self.device}...")
        self.model = load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """Generates forecasts using TiRex.
        
        Args:
            history: DataFrame with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Number of samples to draw.
            
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
            num_samples: Number of samples to draw.
            batch_size: Number of histories to process in one call.
            
        Returns:
            List of DataFrames, each with columns 'ds', 'quantile', 'value'.
        """
        if batch_size is None:
            batch_size = self.batch_size
        all_res_quantiles = []
        all_res_means = []
        
        # TiRex expects contexts of same length for true batching, or a list if supported.
        # Given the rolling origin, lengths differ. 
        # For now, we process 1-by-1 but on the correct device.
        # Actually, let's just do sequential on the device to be safe with lengths.
        
        for i, history in enumerate(histories):
            print(f"TiRex inference for series {i+1}/{len(histories)}...")
            context = torch.tensor(history['y'].values, dtype=torch.float32).unsqueeze(0)
            context = context.to(self.device)
            
            with torch.no_grad():
                res_quantiles, res_mean = self.model.forecast(
                    context=context, 
                    prediction_length=horizon
                )
                all_res_quantiles.append(res_quantiles.cpu())
                all_res_means.append(res_mean.cpu())
                
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
                
        # Standard TiRex quantiles are [0.1, 0.2, ..., 0.9]
        model_q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        for idx, history in enumerate(histories):
            target_dates = pd.date_range(
                start=pd.to_datetime(history['ds'].iloc[-1]) + pd.Timedelta(weeks=1),
                periods=horizon,
                freq='W-SUN'
            ).tolist()
            
            if quantiles is None:
                quantiles = [0.5]
                
            res_dfs = []
            for i, target_date in enumerate(target_dates):
                y_model_qs = all_res_quantiles[idx][0, i, :].numpy()
                y_mean = all_res_means[idx][0, i].numpy()
                
                for q in quantiles:
                    if q == 0.5:
                        val = y_mean
                    else:
                        val = np.interp(q, model_q_levels, y_model_qs)
                    
                    res_dfs.append(pd.DataFrame({
                        'ds': [target_date],
                        'quantile': [q],
                        'value': [float(val)]
                    }))
            results.append(pd.concat(res_dfs, ignore_index=True))
                
        return results
