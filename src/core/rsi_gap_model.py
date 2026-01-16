"""
MTF RSI GAP Analysis Model for Trading Signal Classification
Enhanced with Multi-Timeframe Analysis and SHORT support

Strategy:
- Higher Timeframe (1H): Determines trend direction via EMA crossover
- Lower Timeframe (5min): Finds entries via RSI
- LONG: RSI < 45 in UPTREND
- SHORT: RSI > 55 in DOWNTREND
- Exit: TP/SL using GAP Analysis (8 points each)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Optional, Tuple, Dict
from enum import IntEnum


class TrendDirection(IntEnum):
    DOWNTREND = -1
    SIDEWAY = 0
    UPTREND = 1


class SignalType(IntEnum):
    SHORT = -1
    HOLD = 0
    LONG = 1


class RSIGapModel:
    """
    MTF RSI-based trading model with GAP analysis
    
    Strategy:
    - HTF (1H): EMA20 > EMA50 → UPTREND, EMA20 < EMA50 → DOWNTREND
    - LTF (5min): RSI < 45 in uptrend → LONG, RSI > 55 in downtrend → SHORT
    - Exit: TP/SL = 8 points using GAP analysis
    """
    
    def __init__(self, config: dict, **kwargs):
        self.config = config
        
        # Load config
        rsi_config = config.get('rsi_gap', {})
        self.rsi_threshold_long = rsi_config.get('rsi_threshold_long', 45)
        self.rsi_threshold_short = rsi_config.get('rsi_threshold_short', 55)
        self.tp_points = rsi_config.get('tp_points', 8)
        self.sl_points = rsi_config.get('sl_points', 8)
        self.rsi_period = rsi_config.get('rsi_period', 14)
        
        # EMA periods for HTF trend
        self.ema_fast = rsi_config.get('ema_fast', 20)
        self.ema_slow = rsi_config.get('ema_slow', 50)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Store HTF trend for reference
        self.current_htf_trend = TrendDirection.SIDEWAY
        
        print(f"MTF RSI GAP Model initialized:")
        print(f"  LONG: RSI<{self.rsi_threshold_long} in UPTREND")
        print(f"  SHORT: RSI>{self.rsi_threshold_short} in DOWNTREND")
        print(f"  TP={self.tp_points}pts, SL={self.sl_points}pts")
    
    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
            
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA indicator"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    def determine_htf_trend(self, htf_prices: np.ndarray) -> np.ndarray:
        """
        Determine Higher Timeframe trend using EMA crossover
        
        Returns array of trend values:
            1 = UPTREND (EMA20 > EMA50)
           -1 = DOWNTREND (EMA20 < EMA50)
            0 = SIDEWAY (EMAs close together)
        """
        ema_fast = self.calculate_ema(htf_prices, self.ema_fast)
        ema_slow = self.calculate_ema(htf_prices, self.ema_slow)
        
        # Calculate trend strength as percentage difference
        ema_diff = (ema_fast - ema_slow) / ema_slow * 100
        
        trends = np.zeros(len(htf_prices))
        
        # Define threshold for trend confirmation (0.1% difference)
        trend_threshold = 0.1
        
        for i in range(len(htf_prices)):
            if np.isnan(ema_diff[i]):
                trends[i] = TrendDirection.SIDEWAY
            elif ema_diff[i] > trend_threshold:
                trends[i] = TrendDirection.UPTREND
            elif ema_diff[i] < -trend_threshold:
                trends[i] = TrendDirection.DOWNTREND
            else:
                trends[i] = TrendDirection.SIDEWAY
        
        return trends
    
    def create_features(self, prices: np.ndarray, highs: np.ndarray = None, 
                        lows: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix from price data"""
        df = pd.DataFrame({'close': prices})
        
        # Returns
        for i in [1, 2, 3, 5]:
            df[f'ret_{i}'] = df['close'].pct_change(i)
        
        # Volatility
        df['vol'] = df['ret_1'].rolling(5).std()
        
        # RSI
        df['rsi'] = self.calculate_rsi(prices)
        
        # Price position relative to recent range
        if highs is not None and lows is not None:
            range_high = pd.Series(highs).rolling(14).max()
            range_low = pd.Series(lows).rolling(14).min()
            df['price_position'] = (prices - range_low) / (range_high - range_low + 1e-10)
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_ratio'] = df['ema_9'] / df['ema_21'] - 1
        
        df = df.dropna()
        feature_cols = [c for c in df.columns if c != 'close']
        
        return df[feature_cols].values, df['rsi'].values
    
    def fit(self, prices: np.ndarray, highs: np.ndarray = None, 
            lows: np.ndarray = None, htf_prices: np.ndarray = None,
            verbose: bool = True) -> Dict:
        """
        Fit the model
        """
        features, rsi = self.create_features(prices, highs, lows)
        
        # Fit scaler
        self.scaler.fit(features)
        self.is_fitted = True
        
        # Calculate expected signals
        n_long_signals = np.sum(rsi < self.rsi_threshold_long)
        n_short_signals = np.sum(rsi > self.rsi_threshold_short)
        
        if verbose:
            print(f"Model fitted on {len(prices)} samples")
            print(f"Potential LONG signals (RSI < {self.rsi_threshold_long}): {n_long_signals}")
            print(f"Potential SHORT signals (RSI > {self.rsi_threshold_short}): {n_short_signals}")
        
        return {
            'samples': len(prices),
            'potential_long_signals': n_long_signals,
            'potential_short_signals': n_short_signals
        }
    
    def predict(self, prices: np.ndarray, htf_trend: int = None,
                highs: np.ndarray = None, lows: np.ndarray = None) -> np.ndarray:
        """
        Predict trading signals with MTF filtering
        
        Args:
            prices: LTF price array
            htf_trend: Single HTF trend value (1=uptrend, -1=downtrend, 0=sideway)
            
        Returns:
            Array of signals:
             1 = LONG (buy)
            -1 = SHORT (sell)
             0 = HOLD
        """
        rsi = self.calculate_rsi(prices)
        signals = np.zeros(len(prices))
        
        # If no HTF trend provided, determine from price data
        if htf_trend is None:
            htf_trends = self.determine_htf_trend(prices)
        else:
            htf_trends = np.full(len(prices), htf_trend)
        
        for i in range(self.rsi_period + 1, len(prices)):
            if np.isnan(rsi[i]):
                continue
                
            trend = htf_trends[i]
            
            # LONG signal: RSI < threshold in UPTREND
            if trend == TrendDirection.UPTREND and rsi[i] < self.rsi_threshold_long:
                signals[i] = SignalType.LONG
            
            # SHORT signal: RSI > threshold in DOWNTREND
            elif trend == TrendDirection.DOWNTREND and rsi[i] > self.rsi_threshold_short:
                signals[i] = SignalType.SHORT
        
        return signals
    
    def predict_with_htf_array(self, ltf_prices: np.ndarray, htf_trends: np.ndarray,
                                ltf_to_htf_map: np.ndarray = None) -> np.ndarray:
        """
        Predict signals using separate HTF trend array
        
        Args:
            ltf_prices: Lower timeframe prices
            htf_trends: Higher timeframe trend values (aligned to LTF bars)
            ltf_to_htf_map: Optional mapping from LTF index to HTF trend index
        """
        rsi = self.calculate_rsi(ltf_prices)
        signals = np.zeros(len(ltf_prices))
        
        for i in range(self.rsi_period + 1, len(ltf_prices)):
            if np.isnan(rsi[i]):
                continue
            
            # Get HTF trend for this LTF bar
            if ltf_to_htf_map is not None:
                htf_idx = ltf_to_htf_map[i]
                trend = htf_trends[htf_idx] if htf_idx < len(htf_trends) else 0
            else:
                # Assume htf_trends is already aligned to LTF bars
                trend = htf_trends[i] if i < len(htf_trends) else 0
            
            # LONG signal
            if trend == TrendDirection.UPTREND and rsi[i] < self.rsi_threshold_long:
                signals[i] = SignalType.LONG
            
            # SHORT signal
            elif trend == TrendDirection.DOWNTREND and rsi[i] > self.rsi_threshold_short:
                signals[i] = SignalType.SHORT
        
        return signals
    
    def check_exit_gap_analysis(self, entry_price: float, bar_open: float, 
                                 bar_high: float, bar_low: float,
                                 is_long: bool = True) -> Tuple[bool, bool, bool]:
        """
        Check exit conditions using GAP analysis
        
        Args:
            is_long: True for LONG position, False for SHORT
            
        Returns:
            (tp_hit, sl_hit, ambiguous)
        """
        tp_move = self.tp_points * 0.1
        sl_move = self.sl_points * 0.1
        
        if is_long:
            tp_level = entry_price + tp_move
            sl_level = entry_price - sl_move
            
            # Check gap first
            if bar_open >= tp_level:
                return (True, False, False)
            if bar_open <= sl_level:
                return (False, True, False)
            
            # Check intrabar
            tp_in_range = bar_high >= tp_level
            sl_in_range = bar_low <= sl_level
        else:
            # SHORT position - reversed levels
            tp_level = entry_price - tp_move
            sl_level = entry_price + sl_move
            
            # Check gap first
            if bar_open <= tp_level:
                return (True, False, False)
            if bar_open >= sl_level:
                return (False, True, False)
            
            # Check intrabar
            tp_in_range = bar_low <= tp_level
            sl_in_range = bar_high >= sl_level
        
        if tp_in_range and sl_in_range:
            # Use open direction to determine which hits first
            if is_long:
                if bar_open >= entry_price:
                    return (True, False, True)  # Bullish → TP first
                else:
                    return (False, True, True)  # Bearish → SL first
            else:
                if bar_open <= entry_price:
                    return (True, False, True)  # Bearish → TP first for SHORT
                else:
                    return (False, True, True)  # Bullish → SL first for SHORT
        elif tp_in_range:
            return (True, False, False)
        elif sl_in_range:
            return (False, True, False)
        
        return (False, False, False)
    
    def get_exit_price(self, entry_price: float, is_tp: bool, is_long: bool) -> float:
        """Calculate exit price based on position type and exit reason"""
        move = self.tp_points * 0.1 if is_tp else self.sl_points * 0.1
        
        if is_long:
            return entry_price + move if is_tp else entry_price - move
        else:
            return entry_price - move if is_tp else entry_price + move
    
    def save(self, filepath: str):
        """Save model configuration"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'rsi_threshold_long': self.rsi_threshold_long,
            'rsi_threshold_short': self.rsi_threshold_short,
            'tp_points': self.tp_points,
            'sl_points': self.sl_points,
            'rsi_period': self.rsi_period,
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            model_data['scaler_mean'] = self.scaler.mean_.tolist()
            model_data['scaler_scale'] = self.scaler.scale_.tolist()
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model configuration"""
        model_data = joblib.load(filepath)
        
        self.rsi_threshold_long = model_data.get('rsi_threshold_long', 45)
        self.rsi_threshold_short = model_data.get('rsi_threshold_short', 55)
        self.tp_points = model_data['tp_points']
        self.sl_points = model_data['sl_points']
        self.rsi_period = model_data['rsi_period']
        self.ema_fast = model_data.get('ema_fast', 20)
        self.ema_slow = model_data.get('ema_slow', 50)
        self.is_fitted = model_data['is_fitted']
        
        if self.is_fitted and 'scaler_mean' in model_data:
            self.scaler.mean_ = np.array(model_data['scaler_mean'])
            self.scaler.scale_ = np.array(model_data['scaler_scale'])
        
        print(f"Model loaded from {filepath}")
        print(f"Config: LONG RSI<{self.rsi_threshold_long}, SHORT RSI>{self.rsi_threshold_short}")

