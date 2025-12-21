"""
Signal Generation System
Generates Buy/Sell signals based on neural network output
Buy: output > +SignalThreshold (default 0.0005)
Sell: output < -SignalThreshold (default -0.0005)
"""

import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime
from .svr_model import SVRModel
from .data_processor import DataProcessor


class SignalGenerator:
    """
    Generates trading signals from neural network predictions
    - Buy signal: output > +0.0005
    - Sell signal: output < -0.0005
    - No signal (hold): -0.0005 <= output <= +0.0005
    - Enforces one position per symbol
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.signal_threshold = config['signal']['threshold']
        self.one_position_per_symbol = config['trading']['one_position_per_symbol']
        
        # Track active positions per symbol
        self.active_positions = {}  # symbol -> position info
        
        # Signal history
        self.signal_history = {}  # symbol -> list of signals
    
    def generate_signal(self, model: SVRModel, normalized_input: np.ndarray, 
                       symbol: str, current_price: float) -> Dict:
        """
        Generate trading signal from neural network output
        
        Args:
            network: Trained neural network
            normalized_input: Normalized price window, shape (1, 112)
            symbol: Trading symbol
            current_price: Current market price
        
        Returns:
            Dictionary with signal information
        """
        # Forward pass through network
        output = network.predict(normalized_input)
        output_value = float(output[0, 0])  # Extract scalar value
        
        # Clip output to [-1, 1] for SVM compatibility (SVR output is unbounded unlike tanh)
        output_value = np.clip(output_value, -1.0, 1.0)
        
        # Determine signal based on threshold
        signal_type = None
        if output_value > self.signal_threshold:
            signal_type = 'BUY'
        elif output_value < -self.signal_threshold:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        # Check if position already exists (one position per symbol rule)
        position_allowed = True
        if self.one_position_per_symbol and symbol in self.active_positions:
            existing_direction = self.active_positions[symbol]['direction']
            
            # Don't open new position if one exists
            if signal_type in ['BUY', 'SELL']:
                print(f"️ Position already exists for {symbol} ({existing_direction}), signal ignored")
                position_allowed = False
                signal_type = 'HOLD'
        
        # Create signal object
        signal = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal_type,
            'output_value': output_value,
            'threshold': self.signal_threshold,
            'current_price': current_price,
            'position_allowed': position_allowed
        }
        
        # Store signal in history
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        self.signal_history[symbol].append(signal)
        
        # Print signal
        self._print_signal(signal)
        
        return signal
    
    def _print_signal(self, signal: Dict):
        """Print signal information"""
        timestamp = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        symbol = signal['symbol']
        signal_type = signal['signal']
        output = signal['output_value']
        price = signal['current_price']
        
        if signal_type == 'BUY':
            print(f"🟢 {timestamp} | {symbol} | BUY SIGNAL | Output: {output:+.6f} | Price: {price:.2f}")
        elif signal_type == 'SELL':
            print(f" {timestamp} | {symbol} | SELL SIGNAL | Output: {output:+.6f} | Price: {price:.2f}")
        else:
            print(f" {timestamp} | {symbol} | HOLD | Output: {output:+.6f} | Price: {price:.2f}")
    
    def register_position(self, symbol: str, direction: str, entry_price: float, 
                         lot_size: float, stop_loss: float, take_profit: float):
        """
        Register an active position (enforces one position per symbol)
        
        Args:
            symbol: Trading symbol
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            lot_size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        if self.one_position_per_symbol and symbol in self.active_positions:
            print(f"️ Overwriting existing position for {symbol}")
        
        self.active_positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'lot_size': lot_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        print(f" Position registered: {symbol} {direction} @ {entry_price:.2f}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'manual'):
        """
        Close an active position
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing ('TP', 'SL', 'timeout', 'manual')
        
        Returns:
            Dictionary with position result or None
        """
        if symbol not in self.active_positions:
            print(f"️ No active position found for {symbol}")
            return None
        
        position = self.active_positions[symbol]
        entry_price = position['entry_price']
        direction = position['direction']
        lot_size = position['lot_size']
        entry_time = position['entry_time']
        
        # Calculate P&L (simplified, actual calculation depends on instrument specifications)
        if direction == 'BUY':
            pnl_points = exit_price - entry_price
        else:  # SELL
            pnl_points = entry_price - exit_price
        
        hold_time = datetime.now() - entry_time
        
        result = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': datetime.now(),
            'hold_time': hold_time,
            'lot_size': lot_size,
            'pnl_points': pnl_points,
            'reason': reason
        }
        
        # Remove from active positions
        del self.active_positions[symbol]
        
        # Print result
        pnl_emoji = "" if pnl_points > 0 else ""
        print(f"{pnl_emoji} Position closed: {symbol} {direction} | "
              f"Entry: {entry_price:.2f} → Exit: {exit_price:.2f} | "
              f"P&L: {pnl_points:+.2f} points | Reason: {reason}")
        
        return result
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has an active position"""
        return symbol in self.active_positions
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get active position for symbol"""
        return self.active_positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """Get all active positions"""
        return self.active_positions.copy()
    
    def get_signal_history(self, symbol: str, n_recent: int = 10) -> list:
        """Get recent signal history for symbol"""
        if symbol not in self.signal_history:
            return []
        
        return self.signal_history[symbol][-n_recent:]
    
    def get_signal_stats(self, symbol: str) -> Optional[Dict]:
        """Get signal statistics for symbol"""
        if symbol not in self.signal_history or len(self.signal_history[symbol]) == 0:
            return None
        
        signals = self.signal_history[symbol]
        
        buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
        hold_count = sum(1 for s in signals if s['signal'] == 'HOLD')
        
        outputs = [s['output_value'] for s in signals]
        
        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'avg_output': np.mean(outputs),
            'std_output': np.std(outputs),
            'min_output': np.min(outputs),
            'max_output': np.max(outputs)
        }
