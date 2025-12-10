"""
Multi-Timeframe Analysis Module
Implements D/H1/M5 logic:
- D timeframe: Determine buy/sell bias (daily trend)
- H1 timeframe: Assess strength to determine M5 duration (long/short)
- M5 timeframe: Entry timing
- Model: Predict green/red day
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from datetime import datetime


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis for trading decisions
    - Daily (D): Determines overall bias (buy/sell direction)
    - Hourly (H1): Measures strength to determine M5 trade duration
    - 5-minute (M5): Precise entry timing
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.timeframes = config['trading']['timeframes']
    
    def analyze_daily_bias(self, daily_data: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        Analyze daily timeframe to determine buy/sell bias
        Predicts if today will be a green or red day
        
        Args:
            daily_data: Daily OHLCV data
            lookback: Number of days to analyze
        
        Returns:
            Dictionary with bias information
        """
        if len(daily_data) < lookback + 1:
            return {'bias': 'NEUTRAL', 'strength': 0.0, 'reason': 'insufficient_data'}
        
        recent_days = daily_data.iloc[-lookback:]
        
        # Calculate green/red days
        green_days = 0
        red_days = 0
        
        for idx in range(len(recent_days)):
            close = recent_days.iloc[idx]['close']
            open_price = recent_days.iloc[idx]['open']
            
            if close > open_price:
                green_days += 1
            else:
                red_days += 1
        
        # Determine bias
        if green_days > red_days:
            bias = 'BUY'
            strength = green_days / lookback
        elif red_days > green_days:
            bias = 'SELL'
            strength = red_days / lookback
        else:
            bias = 'NEUTRAL'
            strength = 0.5
        
        # Check current day trend
        current_day = daily_data.iloc[-1]
        current_open = current_day['open']
        current_close = current_day['close']
        current_high = current_day['high']
        current_low = current_day['low']
        
        # Position within daily range
        if current_high != current_low:
            position_in_range = (current_close - current_low) / (current_high - current_low)
        else:
            position_in_range = 0.5
        
        # Predict today's outcome
        predicted_day = 'GREEN' if bias == 'BUY' else 'RED' if bias == 'SELL' else 'NEUTRAL'
        
        return {
            'bias': bias,
            'strength': strength,
            'green_days': green_days,
            'red_days': red_days,
            'lookback': lookback,
            'predicted_day': predicted_day,
            'current_position_in_range': position_in_range,
            'current_intraday_change': current_close - current_open
        }
    
    def analyze_h1_strength(self, h1_data: pd.DataFrame, lookback: int = 24) -> Dict:
        """
        Analyze H1 strength to determine M5 trade duration
        Strong H1 → longer M5 trades
        Weak H1 → shorter M5 trades
        
        Args:
            h1_data: Hourly OHLCV data
            lookback: Number of hours to analyze (default 24 = 1 day)
        
        Returns:
            Dictionary with strength information
        """
        if len(h1_data) < lookback + 1:
            return {'strength': 'WEAK', 'score': 0.0, 'reason': 'insufficient_data'}
        
        recent_hours = h1_data.iloc[-lookback:]
        
        # Calculate momentum indicators
        closes = recent_hours['close'].values
        
        # Price change over period
        price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
        
        # Average true range (volatility)
        highs = recent_hours['high'].values
        lows = recent_hours['low'].values
        ranges = highs - lows
        avg_range = np.mean(ranges)
        
        # Directional consistency (how many bars move in same direction)
        price_changes = np.diff(closes)
        positive_moves = np.sum(price_changes > 0)
        negative_moves = np.sum(price_changes < 0)
        
        directional_consistency = max(positive_moves, negative_moves) / len(price_changes) if len(price_changes) > 0 else 0.5
        
        # Volume trend (if available)
        volume_trend = 1.0
        if 'volume' in recent_hours.columns:
            volumes = recent_hours['volume'].values
            recent_vol = np.mean(volumes[-6:])  # Last 6 hours
            earlier_vol = np.mean(volumes[:6])   # First 6 hours
            
            if earlier_vol > 0:
                volume_trend = recent_vol / earlier_vol
        
        # Calculate strength score (0-1)
        strength_score = (
            abs(price_change) * 10 +  # Price movement weight
            directional_consistency * 0.3 +  # Consistency weight
            min(volume_trend, 2.0) * 0.2  # Volume weight (capped at 2x)
        ) / 3
        
        strength_score = np.clip(strength_score, 0, 1)
        
        # Classify strength
        if strength_score > 0.7:
            strength = 'STRONG'
            m5_duration = 'LONG'  # 2-4 hours on M5
        elif strength_score > 0.4:
            strength = 'MODERATE'
            m5_duration = 'MEDIUM'  # 1-2 hours on M5
        else:
            strength = 'WEAK'
            m5_duration = 'SHORT'  # 30min-1 hour on M5
        
        return {
            'strength': strength,
            'score': strength_score,
            'price_change_pct': price_change * 100,
            'directional_consistency': directional_consistency,
            'avg_range': avg_range,
            'volume_trend': volume_trend,
            'm5_recommended_duration': m5_duration,
            'lookback': lookback
        }
    
    def analyze_m5_entry(self, m5_data: pd.DataFrame, h1_bias: str, 
                        lookback: int = 12) -> Dict:
        """
        Analyze M5 timeframe for precise entry timing
        
        Args:
            m5_data: 5-minute OHLCV data
            h1_bias: Bias from H1 analysis ('BUY' or 'SELL')
            lookback: Number of M5 bars to analyze (default 12 = 1 hour)
        
        Returns:
            Dictionary with entry timing information
        """
        if len(m5_data) < lookback + 1:
            return {'entry_signal': 'WAIT', 'quality': 0.0, 'reason': 'insufficient_data'}
        
        recent_m5 = m5_data.iloc[-lookback:]
        
        # Recent price action
        closes = recent_m5['close'].values
        current_close = closes[-1]
        
        # Short-term momentum
        short_momentum = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] != 0 else 0
        
        # Pullback detection (for better entry)
        if h1_bias == 'BUY':
            # Look for pullback in uptrend
            recent_high = np.max(closes[-6:])
            pullback_depth = (recent_high - current_close) / recent_high if recent_high != 0 else 0
            
            if 0.001 < pullback_depth < 0.005:  # Small pullback (0.1% - 0.5%)
                entry_signal = 'ENTER_BUY'
                quality = 0.8
            elif short_momentum > 0:
                entry_signal = 'ENTER_BUY'
                quality = 0.6
            else:
                entry_signal = 'WAIT'
                quality = 0.3
        
        elif h1_bias == 'SELL':
            # Look for pullback in downtrend
            recent_low = np.min(closes[-6:])
            pullback_depth = (current_close - recent_low) / recent_low if recent_low != 0 else 0
            
            if 0.001 < pullback_depth < 0.005:  # Small pullback (0.1% - 0.5%)
                entry_signal = 'ENTER_SELL'
                quality = 0.8
            elif short_momentum < 0:
                entry_signal = 'ENTER_SELL'
                quality = 0.6
            else:
                entry_signal = 'WAIT'
                quality = 0.3
        
        else:
            entry_signal = 'WAIT'
            quality = 0.0
        
        return {
            'entry_signal': entry_signal,
            'quality': quality,
            'short_momentum': short_momentum,
            'current_price': current_close,
            'lookback': lookback
        }
    
    def get_multi_timeframe_decision(self, daily_data: pd.DataFrame, 
                                    h1_data: pd.DataFrame, 
                                    m5_data: pd.DataFrame) -> Dict:
        """
        Combine all timeframes for final trading decision
        
        Args:
            daily_data: Daily OHLCV data
            h1_data: Hourly OHLCV data
            m5_data: 5-minute OHLCV data
        
        Returns:
            Comprehensive multi-timeframe analysis
        """
        # Analyze each timeframe
        daily_bias = self.analyze_daily_bias(daily_data)
        h1_strength = self.analyze_h1_strength(h1_data)
        m5_entry = self.analyze_m5_entry(m5_data, daily_bias['bias'])
        
        # Combine for final decision
        decision = {
            'timestamp': datetime.now(),
            'daily': daily_bias,
            'h1': h1_strength,
            'm5': m5_entry,
            'final_decision': self._make_final_decision(daily_bias, h1_strength, m5_entry)
        }
        
        return decision
    
    def _make_final_decision(self, daily: Dict, h1: Dict, m5: Dict) -> Dict:
        """Make final trading decision based on all timeframes"""
        
        # Daily must provide clear bias
        if daily['bias'] == 'NEUTRAL':
            return {
                'action': 'NO_TRADE',
                'reason': 'Daily bias neutral',
                'confidence': 0.0
            }
        
        # H1 strength determines trade aggressiveness
        if h1['strength'] == 'WEAK':
            confidence_multiplier = 0.5
        elif h1['strength'] == 'MODERATE':
            confidence_multiplier = 0.75
        else:  # STRONG
            confidence_multiplier = 1.0
        
        # M5 must confirm entry
        if m5['entry_signal'] == 'WAIT':
            return {
                'action': 'WAIT',
                'reason': 'M5 entry not confirmed',
                'confidence': 0.0
            }
        
        # Final decision
        action = daily['bias']  # BUY or SELL
        confidence = daily['strength'] * confidence_multiplier * m5['quality']
        
        return {
            'action': action,
            'reason': f"D:{daily['bias']}, H1:{h1['strength']}, M5:{m5['entry_signal']}",
            'confidence': confidence,
            'recommended_duration': h1.get('m5_recommended_duration', 'MEDIUM')
        }
