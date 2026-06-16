"""Chronos Foundation Model wrapper for ILI forecasting."""

import pandas as pd
import numpy as np
import torch
from typing import List, Optional, Any
from chronos import ChronosPipeline

from src.models.base import BaseForecaster
from src.utils.quantiles import samples_to_quantiles


class ChronosForecaster(BaseForecaster):
    """Wrapper for Amazon Chronos Foundation Model.
    
    Attributes:
        model_name: Name of the Chronos model (e.g., "amazon/chronos-t5-large").
        pipeline: The loaded ChronosPipeline.
    """
    
    SIZE_MAP = {
        "tiny": "amazon/chronos-t5-tiny",
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
        "v2-tiny": "amazon/chronos-v2-t5-tiny",
        "v2-small": "amazon/chronos-v2-t5-small",
        "v2-base": "amazon/chronos-v2-t5-base",
        "v2-large": "amazon/chronos-v2-t5-large",
        "bolt-tiny": "amazon/chronos-bolt-tiny",
        "bolt-small": "amazon/chronos-bolt-small",
        "bolt-base": "amazon/chronos-bolt-base",
    }
    
    def __init__(self, model_name: str = "small", device: Optional[str] = None, num_samples: int = 1000, batch_size: int = 8):
        """Initializes the wrapper and loads the model.
        
        Args:
            model_name: The HuggingFace model ID or a size key ('tiny', 'small', 'base', 'large').
            device: Device to use (e.g., "cpu", "cuda", "mps"). If None, auto-detects.
            num_samples: Default number of samples to draw during prediction.
            batch_size: Default batch size for inference.
        """
        # Map size keywords to actual HF model IDs
        if model_name in self.SIZE_MAP:
            self.model_name = self.SIZE_MAP[model_name]
        else:
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
            
        self.num_samples = num_samples
        self.batch_size = batch_size
            
        # Load the pipeline
        # Use float32 for CPU/MPS, float16 for CUDA if available
        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.float16
        
        print(f"Loading Chronos model: {self.model_name} on {self.device}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=dtype,
        )


    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Generates forecasts using Chronos.
        
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
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """Generates forecasts for a batch of histories.
        
        Args:
            histories: List of DataFrames, each with columns 'ds' and 'y'.
            horizon: Number of steps to forecast.
            quantiles: List of quantiles to produce.
            num_samples: Number of samples to draw.
            batch_size: Number of histories to process in one model call.
            
        Returns:
            List of DataFrames, each with columns 'ds', 'quantile', 'value'.
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        if batch_size is None:
            batch_size = self.batch_size

        if quantiles is None:
            quantiles = [0.5]

        # Prepare contexts
        contexts = [torch.tensor(h['y'].values, dtype=torch.float32) for h in histories]
        
        all_forecast_samples = []
        
        # Generate forecasts in chunks to avoid memory issues
        for i in range(0, len(contexts), batch_size):
            chunk = contexts[i : i + batch_size]
            print(f"Chronos batch inference for {len(chunk)} series (origin {i}-{i+len(chunk)})...")
            with torch.no_grad():
                chunk_samples = self.pipeline.predict(
                    chunk,
                    prediction_length=horizon,
                    num_samples=num_samples,
                )
                all_forecast_samples.append(chunk_samples.numpy())
            
            # Explicitly clear cache
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
        # Concatenate all chunks
        forecast_samples = np.concatenate(all_forecast_samples, axis=0) # (num_series, num_samples, horizon)
        
        results = []
        for i, history in enumerate(histories):
            samples = forecast_samples[i] # (num_samples, horizon)
            
            # Calculate target dates
            last_date = pd.to_datetime(history['ds'].iloc[-1])
            target_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1), 
                periods=horizon, 
                freq='W-SUN'
            ).tolist()
            
            # Convert samples to the standardized quantile format
            results.append(samples_to_quantiles(
                samples, 
                quantiles, 
                target_dates=target_dates
            ))
            
        return results
