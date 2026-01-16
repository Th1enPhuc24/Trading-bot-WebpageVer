"""
Flask-SocketIO Web Dashboard Server
Real-time trading dashboard with WebSocket support
"""

import threading
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit


class WebDashboardServer:
    """
    Web-based real-time trading dashboard server
    Uses Flask-SocketIO for WebSocket communication
    """
    
    def __init__(self, port: int = 5000, max_bars: int = 500):
        self.port = port
        self.max_bars = max_bars
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'trading_dashboard_secret'
        
        # SocketIO setup
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Data storage (thread-safe with deque)
        self.prices = deque(maxlen=max_bars)
        self.timestamps = deque(maxlen=max_bars)
        self.equity_curve = deque(maxlen=max_bars)
        
        # OHLC data for candlestick chart (TradingView Lightweight Charts)
        self.ohlc_data = []  # List of {time, open, high, low, close}
        
        # Signals storage
        self.signals = []  # List of {type, timestamp, price}
        
        # Trade events for chart markers (Unix timestamps)
        self.trade_events = []  # List of {time, type, price, trade_id, ...}
        
        # Current position
        self.current_position = None
        
        # Trade history
        self.trades = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'current_balance': 10000.0,
            'starting_balance': 10000.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Server thread
        self._server_thread = None
        self._is_running = False
        
        # Replay mode flag (for backtest visualization)
        self._replay_mode = False
        
        # Training callback (set by trading bot)
        self._training_callback = None
        
        # Flag to track if training has been done in this session
        self._session_has_data = False
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/state')
        def get_state():
            """Get current dashboard state"""
            return jsonify({
                'prices': list(self.prices),
                'timestamps': [t.isoformat() if isinstance(t, datetime) else t for t in self.timestamps],
                'equity': list(self.equity_curve),
                'signals': self.signals[-100:],  # Last 100 signals
                'position': self.current_position,
                'trades': self.trades[-50:],  # Last 50 trades
                'stats': self.stats
            })
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected to dashboard")
            
            # Only send existing data if training has been done in this session
            if self._session_has_data:
                # Convert timestamps to Unix for TradingView Lightweight Charts
                unix_timestamps = [self._to_unix_timestamp(t) for t in list(self.timestamps)]
                
                # Send current state to new client - send all data for backtest mode
                emit('initial_state', {
                    'prices': list(self.prices),
                    'timestamps': unix_timestamps,
                    'ohlc': self.ohlc_data,  # OHLC for candlestick
                    'trade_events': self.trade_events,  # Trade events for markers
                    'equity': list(self.equity_curve),
                    'signals': self.signals,  # All signals
                    'position': self.current_position,
                    'trades': self.trades,  # All trades
                    'stats': self.stats,
                    'replay_mode': self._replay_mode  # Flag for frontend to detect replay mode
                })
            else:
                # Send empty/default state for fresh session
                emit('initial_state', {
                    'prices': [],
                    'timestamps': [],
                    'ohlc': [],
                    'trade_events': [],
                    'equity': [],
                    'signals': [],
                    'position': None,
                    'trades': [],
                    'stats': {
                        'total_trades': 0,
                        'long_trades': 0,
                        'short_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'current_balance': 10000.0,
                        'starting_balance': 10000.0,
                        'max_drawdown': 0.0,
                        'sharpe_ratio': 0.0
                    },
                    'replay_mode': False
                })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected from dashboard")
        
        @self.socketio.on('start_training')
        def handle_start_training():
            print("Training request received from web client")
            # Emit to signal that training callback should be triggered
            if self._training_callback:
                # Run training in a separate thread
                import threading
                training_thread = threading.Thread(target=self._run_training)
                training_thread.start()
            else:
                self.emit_train_log("No training callback registered", "error")
        
        @self.socketio.on('save_results')
        def handle_save_results():
            print("Save results request received")
            self._save_all_results()
        
        @self.socketio.on('capture_screenshot')
        def handle_capture_screenshot():
            print("Screenshot request received")
            # Server-side screenshot is handled separately
            self.emit_train_log("Screenshot functionality requires browser-side capture", "warning")
    
    def start(self):
        """Start the dashboard server in a background thread"""
        if self._is_running:
            print("Dashboard server is already running")
            return
        
        self._is_running = True
        
        def run_server():
            print(f"\n{'='*60}")
            print(f"Web Dashboard starting on http://localhost:{self.port}")
            print(f"{'='*60}\n")
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, 
                            debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        # Give server time to start
        import time
        time.sleep(1)
        
        print(f"Dashboard server started at http://localhost:{self.port}")
    
    def stop(self):
        """Stop the dashboard server"""
        self._is_running = False
        print("Dashboard server stopped")
    
    def _to_unix_timestamp(self, timestamp) -> int:
        """Convert timestamp to Unix timestamp (seconds)"""
        if timestamp is None:
            return int(datetime.now().timestamp())
        if isinstance(timestamp, (int, float)):
            return int(timestamp)
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return int(dt.timestamp())
            except:
                return int(datetime.now().timestamp())
        if hasattr(timestamp, 'timestamp'):
            return int(timestamp.timestamp())
        return int(datetime.now().timestamp())
    
    def add_ohlc(self, timestamp, open_price: float, high: float, low: float, close: float):
        """Add OHLC candle data"""
        unix_time = self._to_unix_timestamp(timestamp)
        self.ohlc_data.append({
            'time': unix_time,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    def batch_update_ohlc(self, ohlc_list: list):
        """Batch update OHLC data from list of dicts with time, open, high, low, close"""
        for candle in ohlc_list:
            unix_time = self._to_unix_timestamp(candle.get('time') or candle.get('timestamp'))
            self.ohlc_data.append({
                'time': unix_time,
                'open': candle.get('open', candle.get('close', 0)),
                'high': candle.get('high', candle.get('close', 0)),
                'low': candle.get('low', candle.get('close', 0)),
                'close': candle.get('close', 0)
            })
    
    def add_trade_event(self, timestamp, event_type: str, price: float, trade_id: str = None,
                        tp_price: float = None, sl_price: float = None, pnl: float = None,
                        reason: str = None):
        """Add trade event for chart markers"""
        unix_time = self._to_unix_timestamp(timestamp)
        event = {
            'time': unix_time,
            'type': event_type,  # 'BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT'
            'price': price,
            'trade_id': trade_id or f"T{len(self.trade_events) + 1:03d}"
        }
        if tp_price is not None:
            event['tp_price'] = tp_price
        if sl_price is not None:
            event['sl_price'] = sl_price
        if pnl is not None:
            event['pnl'] = pnl
        if reason is not None:
            event['reason'] = reason
        self.trade_events.append(event)
    
    def update_price(self, timestamp: datetime, price: float, emit_update: bool = True):
        """Update price data and emit to clients"""
        self.timestamps.append(timestamp)
        self.prices.append(price)
        
        # Only emit for live updates, not batch loading
        if emit_update:
            self.socketio.emit('price_update', {
                'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                'price': price
            })
    
    def batch_update_prices(self, timestamps: list, prices: list):
        """Batch update prices without emitting individual updates (for backtest)"""
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            self.timestamps.append(ts)
            self.prices.append(price)
    
    def update_equity(self, balance: float):
        """Update equity curve"""
        self.equity_curve.append(balance)
        self.stats['current_balance'] = balance
        
        # Keep total_pnl in sync with current_balance
        self.stats['total_pnl'] = balance - self.stats['starting_balance']
        
        # Calculate max drawdown
        if len(self.equity_curve) > 1:
            equity_list = list(self.equity_curve)
            running_max = equity_list[0]
            max_dd = 0
            for eq in equity_list:
                if eq > running_max:
                    running_max = eq
                dd = (running_max - eq) / running_max * 100
                if dd > max_dd:
                    max_dd = dd
            self.stats['max_drawdown'] = max_dd
            
            # Debug: Log when max_drawdown changes
            if max_dd > 0 and len(self.equity_curve) % 100 == 0:
                print(f"[DEBUG] Max Drawdown: {max_dd:.4f}% (peak: {running_max:.2f}, current: {balance:.2f})")
        
        self.socketio.emit('equity_update', {
            'balance': balance,
            'max_drawdown': self.stats['max_drawdown'],
            'total_pnl': self.stats['total_pnl']
        })
    
    def add_signal(self, signal_type: str, timestamp: datetime, price: float):
        """Add a trading signal"""
        signal = {
            'type': signal_type,
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'price': price
        }
        self.signals.append(signal)
        
        self.socketio.emit('signal', signal)
    
    def update_position(self, position: Optional[Dict]):
        """Update current position"""
        if position:
            self.current_position = {
                'symbol': position.get('symbol', 'N/A'),
                'direction': position.get('direction', 'N/A'),
                'entry_price': position.get('entry_price', 0),
                'entry_time': position.get('entry_time').isoformat() if isinstance(position.get('entry_time'), datetime) else str(position.get('entry_time', '')),
                'stop_loss': position.get('stop_loss', 0),
                'take_profit': position.get('take_profit', 0),
                'lot_size': position.get('lot_size', 0)
            }
        else:
            self.current_position = None
        
        self.socketio.emit('position_update', {
            'position': self.current_position
        })
    
    def add_trade(self, trade: Dict):
        """Add completed trade to history"""
        # Use passed timestamp if available, otherwise use current time
        timestamp = trade.get('timestamp')
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        elif hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()
        # else: already a string, use as-is
        
        # Get entry and exit times
        entry_time = trade.get('entry_time')
        if entry_time and hasattr(entry_time, 'isoformat'):
            entry_time = entry_time.isoformat()
        exit_time = trade.get('exit_time', timestamp)
        if exit_time and hasattr(exit_time, 'isoformat'):
            exit_time = exit_time.isoformat()
        
        trade_record = {
            'timestamp': timestamp,
            'type': trade.get('type', trade.get('direction', 'N/A')),
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', trade.get('close_price', 0)),
            'pnl': trade.get('pnl', trade.get('pnl_points', 0)),
            'reason': trade.get('exit_reason', trade.get('reason', 'N/A')),
            'duration': str(trade.get('duration', 'N/A')),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'id': trade.get('id', trade.get('trade_id', len(self.trades) + 1)),
            'trade_id': trade.get('trade_id', trade.get('id', len(self.trades) + 1))
        }
        self.trades.append(trade_record)
        
        # Update statistics
        self.stats['total_trades'] += 1
        
        # Count long vs short trades
        trade_type = str(trade_record['type']).upper()
        if trade_type in ['BUY', 'LONG']:
            self.stats['long_trades'] += 1
        elif trade_type in ['SELL', 'SHORT']:
            self.stats['short_trades'] += 1
        
        if trade_record['pnl'] > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        self.stats['total_pnl'] += trade_record['pnl']
        
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
        
        # Calculate Sharpe ratio
        self.stats['sharpe_ratio'] = self._calculate_sharpe_ratio()
        
        # Mark that we have data in this session
        self._session_has_data = True
        
        self.socketio.emit('trade_closed', {
            'trade': trade_record,
            'stats': self.stats
        })
    
    def update_stats(self, stats: Dict):
        """Update dashboard statistics"""
        self.stats.update(stats)
        self.socketio.emit('stats_update', self.stats)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade returns"""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.get('pnl', 0) for t in self.trades]
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        
        # Standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5
        
        if std_return == 0:
            return 0.0
        
        # Sharpe ratio (simplified, not annualized since we're using trade returns)
        # Higher is better - indicates consistent returns relative to risk
        sharpe = avg_return / std_return
        return round(sharpe, 2)
    
    # ============================================
    # TRAINING METHODS
    # ============================================
    
    def set_training_callback(self, callback):
        """Set callback function for training"""
        self._training_callback = callback
    
    def emit_train_log(self, message: str, log_type: str = 'info'):
        """Emit training log to web clients"""
        self.socketio.emit('train_log', {
            'message': message,
            'type': log_type
        })
    
    def _run_training(self):
        """Run the training callback"""
        try:
            if self._training_callback:
                self._training_callback()
        except Exception as e:
            self.socketio.emit('train_error', {'message': str(e)})
    
    def _save_all_results(self):
        """Save trade history and screenshot"""
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trade history to CSV
        csv_path = self.save_train_history(timestamp)
        
        # Emit completion
        self.socketio.emit('train_complete', {
            'csv_path': csv_path,
            'screenshot_path': None  # Screenshot handled by browser
        })
    
    def save_train_history(self, timestamp: str = None) -> str:
        """Save trade history to CSV file"""
        import os
        import pandas as pd
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs', 'train-history')
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f'train_history_{timestamp}.csv')
        
        if self.trades:
            df = pd.DataFrame(self.trades)
            df.to_csv(filepath, index=False)
            self.emit_train_log(f"Train history saved: {filepath}", 'result')
        else:
            self.emit_train_log("No trades to save", 'warning')
        
        return filepath
    
    def reset_data(self):
        """Reset all data for a fresh training session"""
        from collections import deque
        
        # Clear data storage
        self.prices = deque(maxlen=self.max_bars)
        self.timestamps = deque(maxlen=self.max_bars)
        self.equity_curve = deque(maxlen=self.max_bars)
        
        # Clear OHLC data
        self.ohlc_data = []
        
        # Clear signals
        self.signals = []
        
        # Clear trade events
        self.trade_events = []
        
        # Clear current position
        self.current_position = None
        
        # Clear trade history
        self.trades = []
        
        # Reset statistics
        self.stats = {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'current_balance': 10000.0,
            'starting_balance': 10000.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Emit reset to clients
        self.socketio.emit('data_reset', {'message': 'Data cleared for new training session'})
