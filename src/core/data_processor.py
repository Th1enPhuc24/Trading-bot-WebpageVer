"""
Data Processing Module
Handles normalization, bar windowing, and data preparation
"""

import numpy as np
from typing import Tuple, Optional, Dict
import pandas as pd


class DataProcessor:
    """
    Handles data normalization and preprocessing for neural network input
    Normalization: 2 × (price − min) / (max − min) − 1 → range [-1, 1]
    """
    
    def __init__(self, window_size: int = 112):
        self.window_size = window_size
        self.scalers = {}  # Store min/max per symbol for running normalization
    
    def normalize_prices(self, prices: np.ndarray, symbol: str, 
                        update_scaler: bool = True) -> np.ndarray:
        """
        Normalize prices to [-1, 1] range using min-max scaling
        Formula: normalized_price = 2 × (price − min) / (max − min) − 1
        
        Args:
            prices: Array of closing prices
            symbol: Trading symbol for per-symbol normalization
            update_scaler: Whether to update running min/max values
        
        Returns:
            Normalized prices in range [-1, 1]
        """
        if len(prices) == 0:
            return np.array([])
        
        # Get current min/max
        price_min = np.min(prices)
        price_max = np.max(prices)
        
        # Update running scaler for this symbol
        if update_scaler:
            if symbol not in self.scalers:
                self.scalers[symbol] = {'min': price_min, 'max': price_max}
            else:
                self.scalers[symbol]['min'] = min(self.scalers[symbol]['min'], price_min)
                self.scalers[symbol]['max'] = max(self.scalers[symbol]['max'], price_max)
        
        # Use stored scaler if available, otherwise current min/max
        if symbol in self.scalers:
            min_val = self.scalers[symbol]['min']
            max_val = self.scalers[symbol]['max']
        else:
            min_val = price_min
            max_val = price_max
        
        # Avoid division by zero
        if max_val - min_val < 1e-10:
            return np.zeros_like(prices)
        
        # Apply normalization formula: 2 × (price − min) / (max − min) − 1
        normalized = 2 * (prices - min_val) / (max_val - min_val) - 1
        
        return normalized
    
    def create_input_window(self, prices: np.ndarray, symbol: str) -> Optional[np.ndarray]:
        """
        Create normalized input window of last 112 bars
        
        Args:
            prices: Array of closing prices (must have at least 112 bars)
            symbol: Trading symbol
        
        Returns:
            Normalized window of shape (112,) or None if insufficient data
        """
        if len(prices) < self.window_size:
            print(f"Insufficient data: need {self.window_size} bars, got {len(prices)}")
            return None
        
        # Take last 112 bars
        window = prices[-self.window_size:]
        
        # Normalize
        normalized_window = self.normalize_prices(window, symbol, update_scaler=True)
        
        return normalized_window
    
    def create_training_dataset(self, prices: np.ndarray, symbol: str, 
                               training_bars: int = 340) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset with 112-bar windows and binary targets
        
        Args:
            prices: Array of closing prices (must have at least training_bars)
            symbol: Trading symbol
            training_bars: Number of bars to use for training (default 340)
        
        Returns:
            Tuple of (X, y) where:
                X: shape (n_samples, 112) - normalized price windows
                y: shape (n_samples, 1) - binary targets (+1 or -1)
        """
        if len(prices) < training_bars:
            raise ValueError(f"Insufficient data: need {training_bars} bars, got {len(prices)}")
        
        # Take most recent training_bars
        recent_prices = prices[-training_bars:]
        
        X_list = []
        y_list = []
        
        # Create windows and targets
        # Need window_size + 1 bars to create one sample (window + next bar for target)
        for i in range(len(recent_prices) - self.window_size):
            # Input window: 112 bars starting at position i
            window = recent_prices[i:i + self.window_size]
            normalized_window = self.normalize_prices(window, symbol, update_scaler=False)
            
            # Target: binary direction of next bar
            # target = (Close[i-1] > Close[i]) ? +1 : -1
            # In forward indexing: (Close[i+window_size] > Close[i+window_size-1]) ? +1 : -1
            current_close = recent_prices[i + self.window_size - 1]
            next_close = recent_prices[i + self.window_size]
            
            target = 1.0 if next_close > current_close else -1.0
            
            X_list.append(normalized_window)
            y_list.append([target])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def get_scaler_info(self, symbol: str) -> Optional[Dict]:
        """Get normalization scaler info for a symbol"""
        return self.scalers.get(symbol)
    
    def reset_scaler(self, symbol: str):
        """Reset scaler for a symbol"""
        if symbol in self.scalers:
            del self.scalers[symbol]
    
    def prepare_prediction_input(self, prices: np.ndarray, symbol: str) -> Optional[np.ndarray]:
        """
        Prepare input for prediction (same as create_input_window but clearer naming)
        
        Args:
            prices: Array of closing prices
            symbol: Trading symbol
        
        Returns:
            Normalized input ready for neural network, shape (1, 112)
        """
        window = self.create_input_window(prices, symbol)
        if window is None:
            return None
        
        # Reshape to (1, 112) for batch processing
        return window.reshape(1, -1)
