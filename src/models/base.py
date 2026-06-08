"""
Abstract base classes for forecasters.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    This interface ensures that both classical statistical models and 
    modern foundation models can be evaluated using a unified engine.
    """

    @abstractmethod
    def predict(
        self, 
        history: pd.DataFrame, 
        horizon: int, 
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Generates forecasts for a given horizon.
        
        Args:
            history (pd.DataFrame): Training data with at least 'ds' and 'y' columns.
            horizon (int): Number of time steps to forecast into the future.
            quantiles (Optional[List[float]]): List of quantiles to predict (e.g., [0.05, 0.5, 0.95]).
                If None, the model should return point forecasts or samples.
            num_samples (int): Number of samples to draw if the model is probabilistic 
                and quantiles are not explicitly requested.

        Returns:
            pd.DataFrame: Long-format DataFrame with columns ['ds', 'quantile', 'value'].
                If the model returns samples instead of quantiles, the 'quantile' 
                column may contain sample indices or be replaced by a 'sample' column 
                depending on implementation, but 'ds' and 'value' are required.
        """
        pass
