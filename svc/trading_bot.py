"""
RSI GAP Trading Bot with Dashboard
Runs backtest and displays results on web dashboard
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rsi_gap_model import RSIGapModel, TrendDirection, SignalType
from src.core.data_fetcher import TradingViewDataFetcher
from src.utils.live_dashboard import LiveDashboard


class RSIGapTradingBot:
    """
    RSI GAP trading bot with dashboard support
    - Fetches data
    - Runs backtest with RSI GAP strategy
    - Displays results on web dashboard
    """
    
    def __init__(self, config_path: str = 'config.json', use_web_dashboard: bool = True):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Add MTF RSI GAP config
        self.config['rsi_gap'] = {
            'rsi_threshold_long': 45,   # RSI < 45 for LONG in uptrend
            'rsi_threshold_short': 55,  # RSI > 55 for SHORT in downtrend
            'tp_points': 8,
            'sl_points': 8,
            'rsi_period': 14,
            'ema_fast': 20,
            'ema_slow': 50
        }
        
        print(f"{'='*60}")
        print(f" MTF RSI GAP Trading Bot - Backtest with Dashboard")
        print(f"{'='*60}")
        print(f"Symbol: {self.config['trading']['exchange']}:{self.config['trading']['symbol']}")
        print(f"Strategy: Multi-Timeframe Analysis")
        print(f"  LONG: RSI < {self.config['rsi_gap']['rsi_threshold_long']} in UPTREND")
        print(f"  SHORT: RSI > {self.config['rsi_gap']['rsi_threshold_short']} in DOWNTREND")
        print(f"TP: {self.config['rsi_gap']['tp_points']} pts | SL: {self.config['rsi_gap']['sl_points']} pts")
        print(f"{'='*60}\n")
        
        # Initialize components
        self.symbol = self.config['trading']['symbol']
        self.account_balance = 10000.0
        self.starting_balance = self.account_balance
        
        self.data_fetcher = TradingViewDataFetcher(self.config)
        self.model = RSIGapModel(self.config)
        
        # Dashboard
        self.use_web_dashboard = use_web_dashboard
        if use_web_dashboard:
            self.dashboard = LiveDashboard(port=5000, max_bars=500)
            # Register training callback for web-triggered training
            self.dashboard.set_training_callback(self._web_training_callback)
        else:
            self.dashboard = None
        
        # Trading state
        self.trades = []
        self.equity_curve = [self.account_balance]
    
    def log(self, message: str, log_type: str = 'info'):
        """Log message to both terminal and web dashboard"""
        print(message)
        if self.dashboard:
            self.dashboard.log(message, log_type)
    
    def _web_training_callback(self):
        """Callback for web-triggered training"""
        try:
            self.run_backtest_with_dashboard(from_web=True)
        except Exception as e:
            if self.dashboard:
                self.dashboard.emit_train_log(f"Error: {str(e)}", 'error')
        
    def run_backtest_with_dashboard(self, from_web: bool = False):
        """Run MTF backtest and display on dashboard"""
        self.log("=" * 50, 'info')
        self.log("Step 1: Fetching data...", 'step')
        
        # Fetch LTF (5min) data
        df = self.data_fetcher.fetch_data(timeframe='5', n_bars=5000)
        
        if df is None or len(df) < 1000:
            self.log("Failed to fetch LTF data", 'error')
            return False
        
        self.log(f"Fetched {len(df)} LTF (5min) bars", 'success')
        
        # Fetch HTF (1H) data for trend analysis (optional, for logging only)
        self.log("Fetching HTF (1H) data for trend analysis...", 'info')
        htf_df = self.data_fetcher.fetch_data(timeframe='60', n_bars=500)
        
        # Prepare data
        prices = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index.tolist()
        
        # Log HTF trend info if available (for analysis only)
        if htf_df is not None and len(htf_df) > 100:
            self.log(f"Fetched {len(htf_df)} HTF (1H) bars", 'success')
            htf_prices = htf_df['close'].values
            htf_trends = self.model.determine_htf_trend(htf_prices)
            uptrend_bars = np.sum(htf_trends == 1)
            downtrend_bars = np.sum(htf_trends == -1)
            sideway_bars = np.sum(htf_trends == 0)
            self.log(f"HTF Trend distribution: UP={uptrend_bars}, DOWN={downtrend_bars}, SIDE={sideway_bars}", 'info')
        else:
            self.log("HTF data unavailable (not required for simple strategy)", 'warning')
        
        # Fit model
        self.log("Step 2: Fitting model...", 'step')
        self.model.fit(prices, highs, lows)
        self.log("Model fitted successfully", 'success')
        
        # Get signals using MTF strategy with actual HTF trend
        # LONG: RSI < 45 in UPTREND
        # SHORT: RSI > 55 in DOWNTREND
        if htf_df is not None and len(htf_df) > 100:
            # Use actual HTF trend data
            htf_prices = htf_df['close'].values
            htf_timestamps = htf_df.index.tolist()
            htf_trends = self.model.determine_htf_trend(htf_prices)
            
            # Align HTF trends to LTF bars
            htf_trend_aligned = np.zeros(len(prices))
            htf_idx = 0
            for i in range(len(timestamps)):
                ltf_time = timestamps[i]
                while htf_idx < len(htf_timestamps) - 1 and htf_timestamps[htf_idx + 1] <= ltf_time:
                    htf_idx += 1
                if htf_idx < len(htf_trends):
                    htf_trend_aligned[i] = htf_trends[htf_idx]
            
            signals = self.model.predict_with_htf_array(prices, htf_trend_aligned)
        else:
            # Fallback: use LTF EMA trend if no HTF data
            signals = self.model.predict(prices, htf_trend=None, highs=highs, lows=lows)
        
        # Count how many signals we'll generate
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        self.log(f"Generated {long_signals} LONG signals, {short_signals} SHORT signals", 'info')
        
        # Start dashboard (only if not from_web, since dashboard is already running)
        if self.dashboard and not from_web:
            self.log("Starting web dashboard...", 'info')
            self.dashboard.start()
            self.log("Dashboard available at: http://localhost:5000", 'info')
        
        # Load OHLC data for candlestick chart (for both from_web and normal cases)
        if self.dashboard:
            self.log("Loading chart data...", 'info')
            ohlc_list = []
            for i in range(len(df)):
                ts = timestamps[i]
                # Convert to ISO format string for proper JSON serialization
                if hasattr(ts, 'isoformat'):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts)
                ohlc_list.append({
                    'time': ts_str,
                    'open': float(opens[i]),
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'close': float(prices[i])
                })
            self.dashboard.batch_update_ohlc(ohlc_list)
            self.log(f"Loaded {len(ohlc_list)} candles", 'success')
        
        # Run backtest
        self.log("Step 3: Running backtest...", 'step')
        position = None
        trade_id = 0
        
        tp_move = self.model.tp_points * 0.1
        sl_move = self.model.sl_points * 0.1
        max_hold = 24
        
        # Track stats
        long_trades = 0
        short_trades = 0
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            timestamp = timestamps[i]
            
            # Convert timestamp for proper display
            if hasattr(timestamp, 'isoformat'):
                ts_display = timestamp.isoformat()
            else:
                ts_display = str(timestamp)
            
            # Update dashboard with price
            if self.dashboard and i % 5 == 0:
                self.dashboard.update_price(timestamp, current_price)
                self.dashboard.update_equity(self.account_balance)
            
            # Check position for exit
            if position is not None:
                bars_held = i - position['entry_bar']
                is_long = position['is_long']
                
                # Check exit using GAP analysis
                tp_hit, sl_hit, _ = self.model.check_exit_gap_analysis(
                    position['entry'], opens[i], highs[i], lows[i], is_long=is_long
                )
                
                # Calculate exit price and PnL
                if tp_hit or sl_hit or bars_held >= max_hold:
                    current_trade_id = position['trade_id']
                    entry_ts = timestamps[position['entry_bar']]
                    entry_ts_str = entry_ts.isoformat() if hasattr(entry_ts, 'isoformat') else str(entry_ts)
                    
                    if tp_hit:
                        exit_price = self.model.get_exit_price(position['entry'], is_tp=True, is_long=is_long)
                        pnl = tp_move * position['size']
                        exit_reason = 'TAKE_PROFIT'
                        pnl_points = self.model.tp_points
                    elif sl_hit:
                        exit_price = self.model.get_exit_price(position['entry'], is_tp=False, is_long=is_long)
                        pnl = -sl_move * position['size']
                        exit_reason = 'STOP_LOSS'
                        pnl_points = -self.model.sl_points
                    else:  # TIMEOUT
                        exit_price = current_price
                        if is_long:
                            pnl = (current_price - position['entry']) * position['size']
                            pnl_points = (current_price - position['entry']) / 0.1
                        else:
                            pnl = (position['entry'] - current_price) * position['size']
                            pnl_points = (position['entry'] - current_price) / 0.1
                        exit_reason = 'TIMEOUT'
                    
                    self.account_balance += pnl
                    
                    trade = {
                        'id': current_trade_id,
                        'trade_id': current_trade_id,
                        'timestamp': ts_display,
                        'symbol': self.symbol,
                        'type': 'BUY' if is_long else 'SELL',
                        'entry_price': position['entry'],
                        'exit_price': exit_price,
                        'entry_time': entry_ts_str,
                        'exit_time': ts_display,
                        'pnl_points': pnl_points,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    }
                    self.trades.append(trade)
                    
                    if self.dashboard:
                        self.dashboard.add_trade(trade)
                        event_type = 'CLOSE_LONG' if is_long else 'CLOSE_SHORT'
                        self.dashboard.add_trade_event(
                            timestamp=timestamp,
                            event_type=event_type,
                            price=exit_price,
                            trade_id=str(current_trade_id),
                            pnl=pnl,
                            reason=exit_reason
                        )
                        self.dashboard.update_position(None)
                    
                    position = None
            
            # Check for entry signal (LONG=1, SHORT=-1)
            if position is None:
                signal = signals[i]
                
                if signal == 1:  # LONG signal
                    size = self.account_balance * 0.01 / sl_move
                    trade_id += 1
                    position = {
                        'entry': current_price,
                        'entry_bar': i,
                        'size': size,
                        'trade_id': trade_id,
                        'is_long': True
                    }
                    long_trades += 1
                    
                    if self.dashboard:
                        self.dashboard.add_signal('BUY', timestamp, current_price)
                        self.dashboard.add_trade_event(
                            timestamp=timestamp,
                            event_type='BUY',
                            price=current_price,
                            trade_id=str(trade_id),
                            tp_price=current_price + tp_move,
                            sl_price=current_price - sl_move
                        )
                        self.dashboard.update_position({
                            'symbol': self.symbol,
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'stop_loss': current_price - sl_move,
                            'take_profit': current_price + tp_move,
                            'lot_size': size
                        })
                
                elif signal == -1:  # SHORT signal
                    size = self.account_balance * 0.01 / sl_move
                    trade_id += 1
                    position = {
                        'entry': current_price,
                        'entry_bar': i,
                        'size': size,
                        'trade_id': trade_id,
                        'is_long': False
                    }
                    short_trades += 1
                    
                    if self.dashboard:
                        self.dashboard.add_signal('SELL', timestamp, current_price)
                        self.dashboard.add_trade_event(
                            timestamp=timestamp,
                            event_type='SELL',
                            price=current_price,
                            trade_id=str(trade_id),
                            tp_price=current_price - tp_move,
                            sl_price=current_price + sl_move
                        )
                        self.dashboard.update_position({
                            'symbol': self.symbol,
                            'direction': 'SELL',
                            'entry_price': current_price,
                            'stop_loss': current_price + sl_move,
                            'take_profit': current_price - tp_move,
                            'lot_size': size
                        })
            
            self.equity_curve.append(self.account_balance)
        
        # Calculate final metrics
        metrics = self._calculate_metrics()
        
        # Print results
        self.log("=" * 50, 'info')
        self.log("BACKTEST RESULTS", 'step')
        self.log("=" * 50, 'info')
        self.log(f"Configuration:", 'info')
        self.log(f"  LONG: RSI < {self.model.rsi_threshold_long} in UPTREND", 'info')
        self.log(f"  SHORT: RSI > {self.model.rsi_threshold_short} in DOWNTREND", 'info')
        self.log(f"  TP: {self.model.tp_points} pts | SL: {self.model.sl_points} pts", 'info')
        self.log("", 'info')
        self.log(f"Performance:", 'step')
        self.log(f"  Total Trades: {metrics['total_trades']}", 'info')
        self.log(f"  Wins: {metrics['wins']} | Losses: {metrics['losses']}", 'info')
        self.log(f"  Win Rate: {metrics['win_rate']:.2f}%", 'result')
        self.log(f"  Total P&L: ${metrics['total_pnl']:.2f}", 'result')
        self.log(f"  Return: {metrics['return_pct']:.1f}%", 'result')
        self.log(f"  Final Balance: ${self.account_balance:.2f}", 'result')
        
        if metrics['win_rate'] > 50 and metrics['total_pnl'] > 0:
            self.log("SUCCESS! Both criteria met!", 'success')
        
        self.log("=" * 50, 'info')
        
        # Save results if from_web
        if from_web and self.dashboard:
            import os
            from datetime import datetime as dt
            timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            
            # Save train history
            csv_path = self.dashboard.save_train_history(timestamp)
            
            # Emit training complete
            self.dashboard.server.socketio.emit('train_complete', {
                'csv_path': csv_path,
                'screenshot_path': None
            })
            return True
        
        if self.dashboard and not from_web:
            self.log("Dashboard running at http://localhost:5000", 'info')
            self.log("Press Ctrl+C to stop...", 'info')
            
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                self.log("Stopping dashboard...", 'info')
                self.dashboard.stop()
        
        return True
    
    def run_live_trading(self, check_interval_minutes: int = 1):
        """
        Run live trading with MTF RSI GAP strategy
        Continuously monitors market and executes trades in paper mode
        """
        import time
        
        print("\n" + "="*60)
        print(" MTF RSI GAP LIVE TRADING")
        print("="*60)
        
        # Start dashboard
        if self.dashboard:
            print("\nStarting web dashboard...")
            self.dashboard.start()
            print("   Dashboard available at: http://localhost:5000")
        
        tp_move = self.model.tp_points * 0.1
        sl_move = self.model.sl_points * 0.1
        position = None
        trade_id = 0
        
        print(f"\nChecking for signals every {check_interval_minutes} minute(s)...")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                current_time = datetime.now()
                
                # Fetch latest data
                df = self.data_fetcher.fetch_data(timeframe='5', n_bars=100)
                htf_df = self.data_fetcher.fetch_data(timeframe='60', n_bars=100)
                
                if df is None or len(df) < 50:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Failed to fetch data, retrying...")
                    time.sleep(60)
                    continue
                
                # Get latest candle
                prices = df['close'].values
                opens = df['open'].values
                highs = df['high'].values
                lows = df['low'].values
                current_price = prices[-1]
                timestamp = df.index[-1]
                
                # Determine HTF trend
                if htf_df is not None and len(htf_df) > 50:
                    htf_prices = htf_df['close'].values
                    htf_trends = self.model.determine_htf_trend(htf_prices)
                    current_trend = int(htf_trends[-1])
                else:
                    current_trend = 0  # SIDEWAY if no HTF data
                
                trend_name = "UPTREND" if current_trend == 1 else "DOWNTREND" if current_trend == -1 else "SIDEWAY"
                
                # Get signal for latest bar
                signals = self.model.predict(prices, htf_trend=current_trend)
                signal = signals[-1]
                
                # Update dashboard with price
                if self.dashboard:
                    self.dashboard.update_price(timestamp, current_price)
                    self.dashboard.update_equity(self.account_balance)
                
                # Check existing position for exit
                if position is not None:
                    is_long = position['is_long']
                    tp_hit, sl_hit, _ = self.model.check_exit_gap_analysis(
                        position['entry'], opens[-1], highs[-1], lows[-1], is_long=is_long
                    )
                    
                    if tp_hit:
                        exit_price = self.model.get_exit_price(position['entry'], True, is_long)
                        pnl = tp_move * position['size']
                        self.account_balance += pnl
                        direction = "LONG" if is_long else "SHORT"
                        print(f"[{current_time.strftime('%H:%M:%S')}] {direction} TP Hit! "
                              f"Entry: ${position['entry']:.2f} → Exit: ${exit_price:.2f} | "
                              f"P&L: +${pnl:.2f}")
                        position = None
                        
                    elif sl_hit:
                        exit_price = self.model.get_exit_price(position['entry'], False, is_long)
                        pnl = -sl_move * position['size']
                        self.account_balance += pnl
                        direction = "LONG" if is_long else "SHORT"
                        print(f"[{current_time.strftime('%H:%M:%S')}] {direction} SL Hit! "
                              f"Entry: ${position['entry']:.2f} → Exit: ${exit_price:.2f} | "
                              f"P&L: -${abs(pnl):.2f}")
                        position = None
                
                # Check for new entry signals
                if position is None:
                    if signal == 1:  # LONG signal
                        size = self.account_balance * 0.01 / sl_move
                        trade_id += 1
                        position = {
                            'entry': current_price,
                            'entry_time': timestamp,
                            'size': size,
                            'trade_id': trade_id,
                            'is_long': True
                        }
                        print(f"[{current_time.strftime('%H:%M:%S')}] LONG Entry @ ${current_price:.2f} "
                              f"(HTF: {trend_name}, RSI signal)")
                        
                        if self.dashboard:
                            self.dashboard.add_signal('BUY', timestamp, current_price)
                            
                    elif signal == -1:  # SHORT signal
                        size = self.account_balance * 0.01 / sl_move
                        trade_id += 1
                        position = {
                            'entry': current_price,
                            'entry_time': timestamp,
                            'size': size,
                            'trade_id': trade_id,
                            'is_long': False
                        }
                        print(f"[{current_time.strftime('%H:%M:%S')}] SHORT Entry @ ${current_price:.2f} "
                              f"(HTF: {trend_name}, RSI signal)")
                        
                        if self.dashboard:
                            self.dashboard.add_signal('SELL', timestamp, current_price)
                    else:
                        print(f"[{current_time.strftime('%H:%M:%S')}] Scanning... "
                              f"Price: ${current_price:.2f} | HTF: {trend_name} | "
                              f"Balance: ${self.account_balance:.2f}")
                else:
                    direction = "LONG" if position['is_long'] else "SHORT"
                    unrealized_pnl = (current_price - position['entry']) * position['size']
                    if not position['is_long']:
                        unrealized_pnl = -unrealized_pnl
                    print(f"[{current_time.strftime('%H:%M:%S')}] {direction} Position Open | "
                          f"Entry: ${position['entry']:.2f} | Current: ${current_price:.2f} | "
                          f"Unrealized: ${unrealized_pnl:+.2f}")
                
                # Wait for next check
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\nLive trading stopped by user")
            if self.dashboard:
                self.dashboard.stop()
    
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {'total_trades': 0, 'wins': 0, 'losses': 0, 
                    'win_rate': 0, 'total_pnl': 0, 'return_pct': 0}
        
        wins = [t for t in self.trades if t['pnl_points'] > 0]
        losses = [t for t in self.trades if t['pnl_points'] <= 0]
        
        total_pnl = sum(t['pnl_points'] * 0.1 for t in self.trades)
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_pnl': self.account_balance - self.starting_balance,
            'return_pct': (self.account_balance - self.starting_balance) / self.starting_balance * 100
        }


def main():
    """Main entry point"""
    use_web = '--web-dashboard' in sys.argv or '-w' in sys.argv or '--dashboard' in sys.argv
    
    if not use_web:
        use_web = True  # Default to web dashboard
    
    bot = RSIGapTradingBot(
        config_path='config.json',
        use_web_dashboard=use_web
    )
    
    bot.run_backtest_with_dashboard()


if __name__ == "__main__":
    main()
