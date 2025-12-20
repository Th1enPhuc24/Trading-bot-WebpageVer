"""
Data Processing Module with Technical Indicators
Handles normalization, bar windowing, and feature engineering
Enhanced with RSI, MACD, Bollinger Bands, and other indicators
OPTIMIZED: Vectorized calculations for better performance
DATA LOGGING: Saves processed data with timestamps
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings
import os
from datetime import datetime


class DataProcessor:
    """
    Handles data normalization and preprocessing for SVM model input
    Enhanced with technical indicators for better prediction
    """
    
    def __init__(self, window_size: int = 112, use_indicators: bool = True):
        self.window_size = window_size
        self.use_indicators = use_indicators
        self.scalers = {}
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI - returns single value for the last bar"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # Use simple moving average for stability
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss < 1e-10:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(rsi)
    
    def calculate_macd(self, prices: np.ndarray, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Calculate MACD histogram - returns single value"""
        if len(prices) < slow + signal:
            return 0.0
        
        # EMA calculation using vectorized approach
        def ema(data, period):
            alpha = 2.0 / (period + 1)
            result = np.zeros(len(data))
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return float(histogram[-1])
    
    def calculate_bollinger_position(self, prices: np.ndarray, 
                                      period: int = 20, std_dev: float = 2.0) -> float:
        """Calculate position within Bollinger Bands - returns single value [-1, 1]"""
        if len(prices) < period:
            return 0.0
        
        window = prices[-period:]
        middle = np.mean(window)
        std = np.std(window)
        
        if std < 1e-10:
            return 0.0
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        band_width = upper - lower
        
        if band_width < 1e-10:
            return 0.0
        
        # Position: -1 (at lower) to +1 (at upper)
        position = (prices[-1] - middle) / (band_width / 2)
        return float(np.clip(position, -1.0, 1.0))
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate normalized price momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        momentum = prices[-1] - prices[-period-1]
        
        # Normalize by recent volatility
        std = np.std(prices[-period:])
        if std < 1e-10:
            return 0.0
        
        normalized = momentum / std
        return float(np.clip(normalized / 3.0, -1.0, 1.0))  # Scale and clip
    
    def calculate_volatility(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate normalized volatility"""
        if len(prices) < period + 1:
            return 0.0
        
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        
        # Handle any inf or nan
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        volatility = np.std(returns) * 100  # Scale up
        return float(np.clip(volatility, 0.0, 1.0))
    
    def normalize_prices(self, prices: np.ndarray, symbol: str, 
                        update_scaler: bool = True) -> np.ndarray:
        """Z-score normalize prices"""
        if len(prices) == 0:
            return np.array([])
        
        mean_val = np.mean(prices)
        std_val = np.std(prices)
        
        if update_scaler:
            self.scalers[symbol] = {'mean': mean_val, 'std': std_val}
        
        if std_val < 1e-10:
            return np.zeros_like(prices)
        
        return (prices - mean_val) / std_val
    
    def create_features(self, prices: np.ndarray, symbol: str) -> np.ndarray:
        """
        Create feature vector with technical indicators
        
        Returns:
            Feature vector combining:
            - Normalized price window (112 values)
            - RSI (normalized, 1 value)
            - MACD histogram (normalized, 1 value)
            - Bollinger Band position (1 value, -1 to 1)
            - Momentum (normalized, 1 value)
            - Volatility (normalized, 1 value)
            Total: 117 features
        """
        # Get normalized prices from window
        window_prices = prices[-self.window_size:]
        normalized_prices = self.normalize_prices(window_prices, symbol)
        
        if not self.use_indicators:
            return normalized_prices
        
        # Calculate indicators (all return single values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            rsi = self.calculate_rsi(prices)
            rsi_normalized = (rsi - 50.0) / 50.0  # Normalize to [-1, 1]
            
            macd_hist = self.calculate_macd(prices)
            # Normalize MACD by recent price std
            price_std = np.std(prices[-26:]) if len(prices) >= 26 else 1.0
            macd_normalized = macd_hist / price_std if price_std > 0.01 else 0.0
            macd_normalized = np.clip(macd_normalized / 3.0, -1.0, 1.0)
            
            bb_position = self.calculate_bollinger_position(prices)
            
            momentum = self.calculate_momentum(prices)
            
            volatility = self.calculate_volatility(prices)
        
        # Combine all features
        indicators = np.array([
            rsi_normalized,
            macd_normalized,
            bb_position,
            momentum,
            volatility
        ])
        
        features = np.concatenate([normalized_prices, indicators])
        return features
    
    def create_input_window(self, prices: np.ndarray, symbol: str) -> Optional[np.ndarray]:
        """Create normalized input window with indicators"""
        min_required = self.window_size + 26  # Need extra for MACD(26)
        if len(prices) < min_required:
            print(f"Insufficient data: need {min_required} bars, got {len(prices)}")
            return None
        
        return self.create_features(prices, symbol)
    
    def create_training_dataset(self, prices: np.ndarray, symbol: str, 
                               training_bars: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset with features and binary targets
        OPTIMIZED: Reduced iterations and cleaner logic
        """
        min_required = training_bars + 50  # Extra for indicators
        if len(prices) < min_required:
            raise ValueError(f"Need {min_required} bars, got {len(prices)}")
        
        recent_prices = prices[-training_bars:]
        
        X_list = []
        y_list = []
        
        # Need at least window_size + 26 bars for one sample
        start_idx = self.window_size + 26
        
        print(f"Creating training dataset...")
        total_samples = len(recent_prices) - start_idx - 1
        
        for i in range(start_idx, len(recent_prices) - 1):
            # Progress indicator every 500 samples
            if (i - start_idx) % 500 == 0:
                print(f"  Processing sample {i - start_idx}/{total_samples}...")
            
            # Get all prices up to current point
            price_window = recent_prices[:i+1]
            
            # Create features
            features = self.create_features(price_window, symbol)
            
            # Target: direction of next bar
            current_close = recent_prices[i]
            next_close = recent_prices[i + 1]
            target = 1.0 if next_close > current_close else -1.0
            
            X_list.append(features)
            y_list.append([target])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Created dataset with {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def get_scaler_info(self, symbol: str) -> Optional[Dict]:
        """Get normalization scaler info for a symbol"""
        return self.scalers.get(symbol)
    
    def reset_scaler(self, symbol: str):
        """Reset scaler for a symbol"""
        if symbol in self.scalers:
            del self.scalers[symbol]
    
    def prepare_prediction_input(self, prices: np.ndarray, symbol: str) -> Optional[np.ndarray]:
        """Prepare input for prediction"""
        features = self.create_input_window(prices, symbol)
        if features is None:
            return None
        
        return features.reshape(1, -1)
    
    def save_normalized_data(self, prices: np.ndarray, normalized: np.ndarray, 
                              symbol: str, data_type: str = "normalized"):
        """
        Save normalized data to CSV with timestamp
        
        Args:
            prices: Original prices
            normalized: Normalized prices
            symbol: Trading symbol
            data_type: Type of data (normalized, features, etc.)
        """
        # Create outputs/data directory if not exists
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{data_type}_{symbol}_{timestamp_str}.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Create DataFrame with both original and normalized data
        import pandas as pd
        df = pd.DataFrame({
            'original_price': prices,
            'normalized': normalized
        })
        
        # Add scaler info as metadata
        scaler = self.scalers.get(symbol, {})
        
        # Save to CSV with metadata header
        with open(filepath, 'w') as f:
            f.write(f"# Symbol: {symbol}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# Mean: {scaler.get('mean', 'N/A')}\n")
            f.write(f"# Std: {scaler.get('std', 'N/A')}\n")
            f.write(f"# Window Size: {self.window_size}\n")
        
        df.to_csv(filepath, mode='a', index=True)
        print(f"💾 Normalized data saved to: {filepath}")
    
    def save_training_dataset(self, X: np.ndarray, y: np.ndarray, symbol: str):
        """
        Save training dataset (features and targets) to files
        
        Args:
            X: Feature matrix
            y: Target vector
            symbol: Trading symbol
        """
        # Create outputs/data directory if not exists
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save features
        features_file = os.path.join(data_dir, f"training_features_{symbol}_{timestamp_str}.csv")
        np.savetxt(features_file, X, delimiter=',', 
                   header=f"Training Features - Symbol: {symbol}, Timestamp: {datetime.now().isoformat()}, Shape: {X.shape}")
        
        # Save targets
        targets_file = os.path.join(data_dir, f"training_targets_{symbol}_{timestamp_str}.csv")
        np.savetxt(targets_file, y, delimiter=',',
                   header=f"Training Targets - Symbol: {symbol}, Timestamp: {datetime.now().isoformat()}, Shape: {y.shape}")
        
        print(f"💾 Training features saved to: {features_file}")
        print(f"💾 Training targets saved to: {targets_file}")

