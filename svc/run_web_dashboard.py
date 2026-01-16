"""
Web Dashboard Entry Point

This script starts the trading dashboard server and waits for training
to be triggered from the web interface.

Usage: python svc/run_web_dashboard.py
Then open http://localhost:5000 and click "Start Train/Test"
"""

import sys
import os
import time
import io
from contextlib import redirect_stdout

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.live_dashboard import LiveDashboard


class WebLogger:
    """Custom logger that captures print() output and sends to dashboard"""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.terminal = sys.__stdout__
        self.buffer = ""
    
    def write(self, message):
        # Write to terminal
        self.terminal.write(message)
        self.terminal.flush()
        
        # Send to dashboard (skip empty lines)
        message = message.strip()
        if message:
            # Determine log type based on content
            log_type = 'info'
            if 'Error' in message or 'error' in message or 'Failed' in message:
                log_type = 'error'
            elif 'Warning' in message or 'WARNING' in message:
                log_type = 'warning'
            elif 'success' in message.lower() or 'Fetched' in message or 'saved' in message.lower():
                log_type = 'success'
            elif 'Step' in message or '====' in message:
                log_type = 'step'
            elif 'initialized' in message.lower() or 'Model fitted' in message:
                log_type = 'result'
            
            self.dashboard.log(message, log_type)
    
    def flush(self):
        self.terminal.flush()


class WebDashboardApp:
    """Web-only dashboard that waits for training trigger from browser"""
    
    def __init__(self):
        print("="*60)
        print(" Trading Bot - Web Dashboard Mode")
        print("="*60)
        print()
        
        # Initialize dashboard
        self.dashboard = LiveDashboard(port=5000, max_bars=500)
        
        # Register training callback
        self.dashboard.set_training_callback(self._run_training)
        
        # Training state
        self.is_training = False
    
    def _run_training(self):
        """Called when user clicks 'Start Train/Test' on web"""
        if self.is_training:
            self.dashboard.log("Training already in progress", 'warning')
            return
        
        self.is_training = True
        
        # Redirect stdout to capture all print() calls
        old_stdout = sys.stdout
        sys.stdout = WebLogger(self.dashboard)
        
        try:
            # Reset all data for fresh training
            self.dashboard.reset_data()
            
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("Training request received from web client", 'step')
            
            import json
            import numpy as np
            from datetime import datetime
            
            # Load config
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Setup config
            config['rsi_gap'] = {
                'rsi_threshold_long': 45,
                'rsi_threshold_short': 55,
                'tp_points': 8,
                'sl_points': 8,
                'rsi_period': 14,
                'ema_fast': 20,
                'ema_slow': 50
            }
            
            # Import and initialize components (print statements will be captured)
            self.dashboard.log("Initializing components...", 'step')
            
            from src.core.data_fetcher import TradingViewDataFetcher
            from src.core.rsi_gap_model import RSIGapModel
            
            data_fetcher = TradingViewDataFetcher(config)
            model = RSIGapModel(config)
            
            # Fetch LTF data
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("Step 1: Fetching LTF (5min) data...", 'step')
            
            df = data_fetcher.fetch_data(timeframe='5', n_bars=5000)
            if df is None or len(df) < 1000:
                self.dashboard.log("Failed to fetch LTF data", 'error')
                self.is_training = False
                sys.stdout = old_stdout
                return
            
            # Fetch HTF data
            self.dashboard.log("Step 1b: Fetching HTF (1H) data...", 'step')
            htf_df = data_fetcher.fetch_data(timeframe='60', n_bars=500)
            
            prices = df['close'].values
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            timestamps = df.index.tolist()
            
            # Create HTF trend
            htf_trend_aligned = np.zeros(len(prices))
            
            if htf_df is not None and len(htf_df) > 100:
                htf_prices = htf_df['close'].values
                htf_timestamps = htf_df.index.tolist()
                htf_trends = model.determine_htf_trend(htf_prices)
                
                # Count trends
                uptrend = np.sum(htf_trends == 1)
                downtrend = np.sum(htf_trends == -1)
                sideway = np.sum(htf_trends == 0)
                self.dashboard.log(f"Trend distribution: UP={uptrend}, DOWN={downtrend}, SIDE={sideway}", 'info')
                
                htf_idx = 0
                for i in range(len(timestamps)):
                    ltf_time = timestamps[i]
                    while htf_idx < len(htf_timestamps) - 1 and htf_timestamps[htf_idx + 1] <= ltf_time:
                        htf_idx += 1
                    if htf_idx < len(htf_trends):
                        htf_trend_aligned[i] = htf_trends[htf_idx]
            
            # Fit model
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("Step 2: Fitting model...", 'step')
            model.fit(prices, highs, lows)
            
            # Get signals using actual HTF trend (LONG in UPTREND, SHORT in DOWNTREND)
            signals = model.predict_with_htf_array(prices, htf_trend_aligned)
            
            # Count signals
            long_signals = np.sum(signals == 1)
            short_signals = np.sum(signals == -1)
            self.dashboard.log(f"Generated {long_signals} LONG signals, {short_signals} SHORT signals", 'info')
            
            # Load OHLC to dashboard
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("Step 3: Loading chart data...", 'step')
            
            ohlc_list = []
            for i in range(len(df)):
                ts = timestamps[i]
                ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                ohlc_list.append({
                    'time': ts_str,
                    'open': float(opens[i]),
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'close': float(prices[i])
                })
            
            # Batch update OHLC
            self.dashboard.batch_update_ohlc(ohlc_list)
            self.dashboard.log(f"Loaded {len(ohlc_list)} candles to chart", 'success')
            
            # Emit initial state to update chart
            self.dashboard.server.socketio.emit('chart_data', {
                'ohlc': self.dashboard.server.ohlc_data
            })
            
            # Run backtest
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("Step 4: Running backtest...", 'step')
            
            account_balance = 10000.0
            starting_balance = account_balance
            trades = []
            trade_events = []
            equity_curve = [account_balance]
            position = None
            trade_id = 0
            tp_move = model.tp_points * 0.1
            sl_move = model.sl_points * 0.1
            max_hold = 24
            
            for i in range(1, len(prices)):
                current_price = prices[i]
                timestamp = timestamps[i]
                ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                
                # Check position exit
                if position is not None:
                    bars_held = i - position['entry_bar']
                    is_long = position['is_long']
                    
                    tp_hit, sl_hit, _ = model.check_exit_gap_analysis(
                        position['entry'], opens[i], highs[i], lows[i], is_long=is_long
                    )
                    
                    if tp_hit or sl_hit or bars_held >= max_hold:
                        if tp_hit:
                            exit_price = model.get_exit_price(position['entry'], is_tp=True, is_long=is_long)
                            reason = 'TAKE_PROFIT'
                        elif sl_hit:
                            exit_price = model.get_exit_price(position['entry'], is_tp=False, is_long=is_long)
                            reason = 'STOP_LOSS'
                        else:
                            exit_price = current_price
                            reason = 'TIMEOUT'
                        
                        # Calculate PnL with compound position sizing (matching terminal)
                        if reason == 'TAKE_PROFIT':
                            pnl_dollars = tp_move * position['size']
                            pnl_points = model.tp_points
                        elif reason == 'STOP_LOSS':
                            pnl_dollars = -sl_move * position['size']
                            pnl_points = -model.sl_points
                        else:  # TIMEOUT
                            if is_long:
                                pnl_dollars = (exit_price - position['entry']) * position['size']
                            else:
                                pnl_dollars = (position['entry'] - exit_price) * position['size']
                            pnl_points = pnl_dollars / 0.1
                        account_balance += pnl_dollars
                        equity_curve.append(account_balance)
                        
                        trade_record = {
                            'trade_id': position['trade_id'],
                            'type': 'LONG' if is_long else 'SHORT',
                            'direction': 'BUY' if is_long else 'SELL',
                            'entry_price': position['entry'],
                            'exit_price': exit_price,
                            'entry_time': position['entry_time'],
                            'exit_time': ts_str,
                            'pnl': pnl_dollars,
                            'pnl_points': pnl_points,
                            'exit_reason': reason,
                            'reason': reason
                        }
                        trades.append(trade_record)
                        
                        # Add exit event
                        exit_type = 'CLOSE_LONG' if is_long else 'CLOSE_SHORT'
                        trade_events.append({
                            'time': ts_str,
                            'type': exit_type,
                            'price': exit_price,
                            'trade_id': position['trade_id'],
                            'pnl': pnl_dollars,
                            'reason': reason
                        })
                        
                        position = None
                
                # Check for new entry
                if position is None:
                    signal = signals[i]
                    if signal == 1:  # BUY (LONG only with forced UPTREND)
                        trade_id += 1
                        # Compound position sizing: risk 1% of current balance
                        size = account_balance * 0.01 / sl_move
                        position = {
                            'trade_id': trade_id,
                            'entry': current_price,
                            'entry_bar': i,
                            'entry_time': ts_str,
                            'is_long': True,
                            'size': size,
                            'tp': current_price + tp_move,
                            'sl': current_price - sl_move
                        }
                        # Add entry event
                        trade_events.append({
                            'time': ts_str,
                            'type': 'BUY',
                            'price': current_price,
                            'trade_id': trade_id,
                            'tp_price': position['tp'],
                            'sl_price': position['sl']
                        })
                    elif signal == -1:  # SELL (should not happen with forced UPTREND)
                        trade_id += 1
                        size = account_balance * 0.01 / sl_move
                        position = {
                            'trade_id': trade_id,
                            'entry': current_price,
                            'entry_bar': i,
                            'entry_time': ts_str,
                            'is_long': False,
                            'size': size,
                            'tp': current_price - tp_move,
                            'sl': current_price + sl_move
                        }
                        # Add entry event
                        trade_events.append({
                            'time': ts_str,
                            'type': 'SELL',
                            'price': current_price,
                            'trade_id': trade_id,
                            'tp_price': position['tp'],
                            'sl_price': position['sl']
                        })
            
            # Calculate metrics
            if trades:
                wins = len([t for t in trades if t['pnl_points'] > 0])
                losses = len([t for t in trades if t['pnl_points'] <= 0])
                total_pnl = account_balance - starting_balance
                win_rate = wins / len(trades) * 100
                return_pct = total_pnl / starting_balance * 100
                
                # Calculate Max Drawdown from equity curve
                peak = starting_balance
                max_drawdown = 0
                for eq in equity_curve:
                    if eq > peak:
                        peak = eq
                    drawdown = (peak - eq) / peak * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                # Calculate Sharpe Ratio
                if len(trades) > 1:
                    returns = [t['pnl'] / starting_balance * 100 for t in trades]
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
                    
                self.dashboard.log(f"Max Drawdown: {max_drawdown:.2f}%", 'info')
                self.dashboard.log(f"Sharpe Ratio: {sharpe_ratio:.2f}", 'info')
            else:
                wins = losses = 0
                total_pnl = win_rate = return_pct = max_drawdown = sharpe_ratio = 0
            
            # Update dashboard with all data
            self.dashboard.log("Updating dashboard with results...", 'info')
            
            # Add trade events for chart markers
            for event in trade_events:
                self.dashboard.server.trade_events.append(event)
            
            # Add trades to dashboard
            for trade in trades:
                self.dashboard.add_trade(trade)
            
            # Update equity curve
            for eq in equity_curve:
                self.dashboard.update_equity(eq)
            
            # Update stats
            stats = {
                'total_trades': len(trades),
                'long_trades': len([t for t in trades if t['type'] == 'LONG']),
                'short_trades': len([t for t in trades if t['type'] == 'SHORT']),
                'winning_trades': wins,
                'losing_trades': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'current_balance': account_balance,
                'starting_balance': starting_balance,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            self.dashboard.update_stats(stats)
            
            # Emit full state update to connected clients
            self.dashboard.server.socketio.emit('initial_state', {
                'ohlc': self.dashboard.server.ohlc_data,
                'trade_events': trade_events,
                'trades': self.dashboard.server.trades,
                'equity': list(self.dashboard.server.equity_curve),
                'stats': stats
            })
            
            # Print results
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("BACKTEST RESULTS", 'step')
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log(f"Total Trades: {len(trades)}", 'info')
            self.dashboard.log(f"Wins: {wins} | Losses: {losses}", 'info')
            self.dashboard.log(f"Win Rate: {win_rate:.2f}%", 'result')
            self.dashboard.log(f"Total P&L: ${total_pnl:.2f}", 'result')
            self.dashboard.log(f"Return: {return_pct:.1f}%", 'result')
            self.dashboard.log(f"Final Balance: ${account_balance:.2f}", 'result')
            
            if win_rate > 50 and total_pnl > 0:
                self.dashboard.log("SUCCESS! Both criteria met!", 'success')
            
            # Save results
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = self.dashboard.save_train_history(timestamp_str)
            
            # Emit complete
            self.dashboard.server.socketio.emit('train_complete', {
                'csv_path': csv_path,
                'screenshot_path': None
            })
            
            # Mark session as having data (for reconnects)
            self.dashboard.server._session_has_data = True
            
            self.dashboard.log("=" * 50, 'info')
            self.dashboard.log("Training complete! Charts updated.", 'success')
            
        except Exception as e:
            self.dashboard.log(f"Error: {str(e)}", 'error')
            import traceback
            self.dashboard.log(traceback.format_exc(), 'error')
            self.dashboard.server.socketio.emit('train_error', {'message': str(e)})
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            self.is_training = False
    
    def start(self):
        """Start the dashboard and wait for web interaction"""
        print("Starting web dashboard...")
        print("Open http://localhost:5000 in your browser")
        print("Click 'Start Train/Test' to begin training")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        self.dashboard.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            self.dashboard.stop()
            print("Goodbye!")


def main():
    app = WebDashboardApp()
    app.start()


if __name__ == "__main__":
    main()
