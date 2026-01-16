"""
Live Dashboard Integration Module
Wrapper class to integrate WebDashboardServer with trading bots
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.web_dashboard.server import WebDashboardServer


class LiveDashboard:
    """
    Live Dashboard wrapper for trading bot integration
    Provides simple interface for updating dashboard data
    """
    
    def __init__(self, port: int = 5000, max_bars: int = 500):
        """
        Initialize live dashboard
        
        Args:
            port: Port to run the web server on
            max_bars: Maximum number of price bars to keep in memory
        """
        self.port = port
        self.server = WebDashboardServer(port=port, max_bars=max_bars)
        self._is_running = False
        
        # Metrics tracking
        self.metrics = {
            'episode': 1,
            'step': 0,
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_reward': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_balance': 10000.0,
            'starting_balance': 10000.0
        }
    
    def start(self):
        """Start the dashboard server"""
        if self._is_running:
            print("Live dashboard is already running")
            return
        
        self.server.start()
        self._is_running = True
        
        print(f"\n{'='*60}")
        print(f"LIVE DASHBOARD RUNNING")
        print(f"   Open your browser at: http://localhost:{self.port}")
        print(f"{'='*60}\n")
    
    def stop(self):
        """Stop the dashboard server"""
        if self._is_running:
            self.server.stop()
            self._is_running = False
            print("Live dashboard stopped")
    
    def set_replay_mode(self, enabled: bool = True):
        """Enable replay mode for backtest visualization"""
        self.server._replay_mode = enabled
        if enabled:
            print("Replay mode enabled - data will be streamed progressively")
    
    def update_price(self, timestamp: datetime, price: float, emit_update: bool = True):
        """
        Update price data
        
        Args:
            timestamp: Price timestamp
            price: Current price
            emit_update: Whether to emit update to clients (False for batch loading)
        """
        self.server.update_price(timestamp, price, emit_update=emit_update)
        self.metrics['step'] += 1
    
    def update_equity(self, balance: float):
        """
        Update equity/balance
        
        Args:
            balance: Current account balance
        """
        self.server.update_equity(balance)
        self.metrics['current_balance'] = balance
    
    def add_signal(self, signal_type: str, timestamp: datetime, price: float):
        """
        Add a trading signal
        
        Args:
            signal_type: 'BUY', 'SELL', or 'CLOSE'
            timestamp: Signal timestamp
            price: Price at signal
        """
        self.server.add_signal(signal_type, timestamp, price)
    
    def add_ohlc(self, timestamp, open_price: float, high: float, low: float, close: float):
        """Add OHLC candle data for TradingView chart"""
        self.server.add_ohlc(timestamp, open_price, high, low, close)
    
    def batch_update_ohlc(self, ohlc_list: list):
        """Batch update OHLC data"""
        self.server.batch_update_ohlc(ohlc_list)
    
    def add_trade_event(self, timestamp, event_type: str, price: float, trade_id: str = None,
                        tp_price: float = None, sl_price: float = None, pnl: float = None,
                        reason: str = None):
        """Add trade event for chart markers"""
        self.server.add_trade_event(timestamp, event_type, price, trade_id, tp_price, sl_price, pnl, reason)
    
    def update_position(self, position: Optional[Dict]):
        """
        Update current position info
        
        Args:
            position: Position dict with keys:
                - symbol, direction, entry_price, entry_time, stop_loss, take_profit, lot_size
            Or None if no position
        """
        self.server.update_position(position)
    
    def add_trade(self, trade: Dict):
        """
        Add completed trade to history
        
        Args:
            trade: Trade dict with keys:
                - type, entry_price, exit_price, pnl, reason, duration
        """
        self.server.add_trade(trade)
        
        # Update local metrics
        self.metrics['total_trades'] += 1
        
        trade_type = trade.get('type', trade.get('direction', 'BUY'))
        if trade_type == 'BUY':
            self.metrics['long_trades'] += 1
        else:
            self.metrics['short_trades'] += 1
        
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        self.metrics['total_pnl'] += pnl
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (
                self.metrics['winning_trades'] / self.metrics['total_trades'] * 100
            )
    
    def update_metrics(self, metrics: Dict):
        """
        Update dashboard metrics
        
        Args:
            metrics: Dict of metrics to update
        """
        self.metrics.update(metrics)
        self.server.update_stats(self.metrics)
    
    def update_nn_output(self, output: float):
        """
        Update neural network output (for compatibility with existing dashboard)
        
        Args:
            output: Neural network output value
        """
        # Web dashboard doesn't currently display NN output,
        # but we keep this method for API compatibility
        pass
    
    def add_training_loss(self, epoch: int, loss: float):
        """
        Record training loss (for compatibility)
        
        Args:
            epoch: Training epoch
            loss: Loss value
        """
        # Web dashboard doesn't display training loss,
        # but we keep this for API compatibility
        pass
    
    def update(self):
        """
        Update dashboard (for compatibility with matplotlib dashboard)
        Web dashboard updates automatically via WebSocket
        """
        pass
    
    def show(self):
        """
        Show dashboard (for compatibility)
        Web dashboard is shown in browser
        """
        print(f"Dashboard is running at http://localhost:{self.port}")
    
    def save_figure(self, filepath: str):
        """
        Save dashboard (for compatibility)
        Web dashboard doesn't support saving as image directly
        
        Args:
            filepath: Would-be filepath (ignored)
        """
        print(f"Web dashboard doesn't support saving as image. View at http://localhost:{self.port}")
    
    # ============================================
    # TRAINING METHODS
    # ============================================
    
    def set_training_callback(self, callback):
        """Set callback function for web-triggered training"""
        self.server.set_training_callback(callback)
    
    def emit_train_log(self, message: str, log_type: str = 'info'):
        """Emit training log to web clients"""
        self.server.emit_train_log(message, log_type)
    
    def log(self, message: str, log_type: str = 'info'):
        """Shorthand for emit_train_log - use instead of print()"""
        self.server.emit_train_log(message, log_type)
    
    def save_train_history(self, timestamp: str = None) -> str:
        """Save trade history to CSV file"""
        return self.server.save_train_history(timestamp)
    
    def update_stats(self, stats: dict):
        """Update dashboard statistics"""
        self.server.update_stats(stats)
    
    def reset_data(self):
        """Reset all data for a fresh training session"""
        self.server.reset_data()
