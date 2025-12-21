"""
Main Trading Bot
Integrates all components and runs the complete trading system
"""

import json
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.svr_model import SVRModel
from src.core.data_processor import DataProcessor
from src.core.training_system import TrainingSystem
from src.core.data_fetcher import TradingViewDataFetcher
from src.core.signal_generator import SignalGenerator
from src.utils.multi_timeframe import MultiTimeframeAnalyzer
from src.core.risk_manager import RiskManager
from src.core.trading_filters import TradingFilters
from src.utils.dashboard import TradingDashboard


class TradingBot:
    """
    Main trading bot that orchestrates all components
    - Monitors H1 bars
    - Triggers retraining every 20 bars
    - Generates signals on new bars
    - Manages positions
    - Enforces risk and trading constraints
    """
    
    def __init__(self, config_path: str = 'config.json', use_dashboard: bool = False):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"{'='*60}")
        print(f"SVM Trading Bot - Initializing")
        print(f"{'='*60}")
        print(f"Symbol: {self.config['trading']['exchange']}:{self.config['trading']['symbol']}")
        svm_config = self.config.get('svm', {})
        print(f"Model: SVM with kernel={svm_config.get('kernel', 'rbf')}, C={svm_config.get('C', 10.0)}")
        print(f"Training: Every {self.config['training']['train_after_bars']} H1 bars")
        print(f"Risk: {self.config['risk_management']['risk_percentage']*100}% per trade")
        print(f"{'='*60}\n")
        
        # Initialize components
        self.symbol = self.config['trading']['symbol']
        self.account_balance = 10000.0  # Starting balance (configurable)
        self.starting_balance = self.account_balance
        
        self.data_fetcher = TradingViewDataFetcher(self.config)
        self.data_processor = DataProcessor(window_size=self.config['network']['input_size'])
        self.training_system = TrainingSystem(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.trading_filters = TradingFilters(self.config)
        
        # Dashboard (optional)
        self.use_dashboard = use_dashboard
        self.dashboard = TradingDashboard(max_bars=200) if use_dashboard else None
        
        # Initialize SVM model
        print("Initializing SVM model...")
        self.network = None
        self.last_bar_count = 0
        
        # Trading state
        self.is_running = False
        self.last_check_time = None
        self.iteration_count = 0
        self.last_dashboard_save = None
    
    def initialize(self):
        """Initialize the bot and load/train network"""
        print("\n Fetching initial data...")
        
        # Fetch sufficient H1 data for training and prediction
        # Use max available from TradingView (up to 5000 bars)
        required_bars = self.config['training']['training_bars']
        fetch_bars = min(5000, required_bars + 500)  # Fetch extra for safety
        
        h1_prices = self.data_fetcher.get_closing_prices(timeframe='60', n_bars=fetch_bars)
        
        if h1_prices is None:
            print(f" Failed to fetch H1 data")
            return False
        
        print(f" Fetched {len(h1_prices)} H1 bars")
        
        # Check if we have enough data
        if len(h1_prices) < required_bars:
            print(f"️ Warning: Only {len(h1_prices)} bars available, need {required_bars}")
            print(f"   Adjusting training_bars to {len(h1_prices) - 112}")
            # Temporarily adjust config
            self.config['training']['training_bars'] = max(340, len(h1_prices) - 112)
        
        # Initialize network with pre-trained weights or train from scratch
        self.network = self.training_system.initialize_network(self.symbol, h1_prices)
        
        self.last_bar_count = len(h1_prices)
        
        print("\n Bot initialized successfully!")
        return True
    
    def check_new_bars(self) -> int:
        """
        Check for new H1 bars
        
        Returns:
            Number of new bars since last check
        """
        h1_prices = self.data_fetcher.get_closing_prices(timeframe='60', n_bars=50)
        
        if h1_prices is None:
            return 0
        
        current_bar_count = len(h1_prices)
        new_bars = max(0, current_bar_count - self.last_bar_count)
        
        self.last_bar_count = current_bar_count
        
        return new_bars
    
    def process_signal(self, signal: Dict, mtf_decision: Dict) -> bool:
        """
        Process trading signal and execute if conditions met
        
        Args:
            signal: Signal from neural network
            mtf_decision: Multi-timeframe decision
        
        Returns:
            True if position opened, False otherwise
        """
        # Check if signal is actionable
        if signal['signal'] == 'HOLD':
            return False
        
        # Check multi-timeframe confirmation
        if mtf_decision['final_decision']['action'] != signal['signal']:
            print(f"️ MTF filter: {mtf_decision['final_decision']['action']} vs NN: {signal['signal']}")
            return False
        
        # Check trading filters
        filter_check = self.trading_filters.should_trade()
        if not filter_check['allowed']:
            print(f"️ Trading not allowed: {', '.join(filter_check['reasons'])}")
            return False
        
        # Check if position already exists
        if self.signal_generator.has_position(self.symbol):
            return False
        
        # Prepare position
        direction = signal['signal']
        entry_price = signal['current_price']
        
        position_info = self.risk_manager.get_position_info(
            self.symbol, direction, entry_price, self.account_balance
        )
        
        # Validate position
        if not self.risk_manager.validate_position(position_info):
            print(f" Position validation failed")
            return False
        
        # Print and register position
        self.risk_manager.print_position_info(position_info)
        
        self.signal_generator.register_position(
            self.symbol,
            direction,
            entry_price,
            position_info['lot_size'],
            position_info['stop_loss'],
            position_info['take_profit']
        )
        
        self.trading_filters.log_position(position_info)
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.add_signal(signal['signal'], signal['timestamp'], entry_price)
            self.dashboard.update_position(position_info)
        
        print(f" Position opened: {self.symbol} {direction} @ {entry_price:.2f}")
        
        return True
    
    def check_open_positions(self):
        """Check and manage open positions - now checks TP/SL on M5 data"""
        position = self.signal_generator.get_position(self.symbol)
        
        if position is None:
            return
        
        # Fetch latest M5 bar for precise TP/SL check
        latest_m5 = self.data_fetcher.get_latest_bar(timeframe='5')
        if latest_m5 is None:
            print("️ Could not fetch M5 data for position check")
            return
        
        current_price = latest_m5['close']
        high_price = latest_m5['high']
        low_price = latest_m5['low']
        
        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Check TP/SL hit on M5 candlestick
        position_closed = False
        close_price = None
        close_reason = None
        
        if direction == 'BUY':
            # Check if TP hit (high reached TP)
            if high_price >= take_profit:
                position_closed = True
                close_price = take_profit  # Assume filled at TP
                close_reason = 'TAKE_PROFIT'
                print(f" Take Profit HIT on M5! {self.symbol} @ {take_profit:.2f}")
            
            # Check if SL hit (low reached SL)
            elif low_price <= stop_loss:
                position_closed = True
                close_price = stop_loss  # Assume filled at SL
                close_reason = 'STOP_LOSS'
                print(f" Stop Loss HIT on M5! {self.symbol} @ {stop_loss:.2f}")
        
        elif direction == 'SELL':
            # Check if TP hit (low reached TP)
            if low_price <= take_profit:
                position_closed = True
                close_price = take_profit
                close_reason = 'TAKE_PROFIT'
                print(f" Take Profit HIT on M5! {self.symbol} @ {take_profit:.2f}")
            
            # Check if SL hit (high reached SL)
            elif high_price >= stop_loss:
                position_closed = True
                close_price = stop_loss
                close_reason = 'STOP_LOSS'
                print(f" Stop Loss HIT on M5! {self.symbol} @ {stop_loss:.2f}")
        
        # Close position if TP/SL hit
        if position_closed:
            close_result = self.signal_generator.close_position(
                self.symbol,
                close_price,
                reason=close_reason
            )
            if close_result:
                self.trading_filters.log_position_close(close_result)
                
                # Update balance
                pnl = close_result['pnl_points'] * 10  # Approximate P&L
                self.account_balance += pnl
                print(f" P&L: ${pnl:.2f} | Balance: ${self.account_balance:.2f}")
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.add_trade(close_result)
                    self.dashboard.update_position(None)
                    self.dashboard.update_equity(self.account_balance)
                    self._save_live_dashboard("position_closed")
                
                return True
        
        # Check timeout (4 hours max hold time)
        if self.risk_manager.check_position_timeout(position['entry_time']):
            print(f" Position timeout reached for {self.symbol}")
            close_result = self.signal_generator.close_position(
                self.symbol,
                current_price,
                reason='TIMEOUT'
            )
            if close_result:
                self.trading_filters.log_position_close(close_result)
                
                pnl = close_result['pnl_points'] * 10
                self.account_balance += pnl
                print(f" P&L: ${pnl:.2f} | Balance: ${self.account_balance:.2f}")
                
                if self.dashboard:
                    self.dashboard.add_trade(close_result)
                    self.dashboard.update_position(None)
                    self.dashboard.update_equity(self.account_balance)
                    self._save_live_dashboard("position_closed")
                
                return True
        
        return False
    
    def run_iteration(self):
        """Run one iteration of the trading loop"""
        current_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"Trading Loop - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Fetch latest data
        mtf_data = self.data_fetcher.get_multi_timeframe_data({
            '1D': 100,
            '60': 500,
            '5': 1000
        })
        
        if '60' not in mtf_data or len(mtf_data['60']) < self.config['network']['input_size']:
            print("️ Insufficient H1 data")
            return
        
        h1_prices = mtf_data['60']['close'].values
        current_price = h1_prices[-1]
        
        # Check for new bars
        new_bars = self.check_new_bars()
        if new_bars > 0:
            print(f" {new_bars} new H1 bar(s) detected")
            
            # Check if retraining needed
            if self.training_system.should_retrain(self.symbol, new_bars):
                print(f"\n Retraining triggered...")
                training_stats = self.training_system.train_network(self.network, h1_prices, self.symbol)
                
                # Update dashboard with training losses
                if self.dashboard and 'loss_history' in training_stats:
                    for epoch, loss in enumerate(training_stats['loss_history']):
                        self.dashboard.add_training_loss(epoch, loss)
        
        # Generate neural network signal
        normalized_input = self.data_processor.prepare_prediction_input(h1_prices, self.symbol)
        
        if normalized_input is None:
            print("️ Could not prepare input for prediction")
            return
        
        signal = self.signal_generator.generate_signal(
            self.network,
            normalized_input,
            self.symbol,
            current_price
        )
        
        # Update dashboard with price and NN output
        if self.dashboard:
            self.dashboard.update_price(current_time, current_price)
            self.dashboard.update_nn_output(signal['output_value'])
            self.dashboard.update_equity(self.account_balance)
        
        # Multi-timeframe analysis
        if all(tf in mtf_data for tf in ['1D', '60', '5']):
            mtf_decision = self.mtf_analyzer.get_multi_timeframe_decision(
                mtf_data['1D'],
                mtf_data['60'],
                mtf_data['5']
            )
            
            print(f"\n Multi-Timeframe Analysis:")
            print(f"  Daily Bias: {mtf_decision['daily']['bias']} "
                  f"(predicted {mtf_decision['daily']['predicted_day']} day)")
            print(f"  H1 Strength: {mtf_decision['h1']['strength']} "
                  f"(M5 duration: {mtf_decision['h1'].get('m5_recommended_duration', 'N/A')})")
            print(f"  M5 Entry: {mtf_decision['m5']['entry_signal']}")
            print(f"  Final Decision: {mtf_decision['final_decision']['action']} "
                  f"(confidence: {mtf_decision['final_decision']['confidence']:.2%})")
            
            # Process signal if actionable
            if signal['signal'] in ['BUY', 'SELL']:
                self.process_signal(signal, mtf_decision)
        
        # Check open positions
        self.check_open_positions()
        
        # Print current status
        bars_until_retrain = self.training_system.get_bars_until_retrain(self.symbol)
        print(f"\n Status: Bars until retrain: {bars_until_retrain}")
        
        if self.signal_generator.has_position(self.symbol):
            position = self.signal_generator.get_position(self.symbol)
            hold_time = datetime.now() - position['entry_time']
            print(f" Active position: {position['direction']} @ {position['entry_price']:.2f} "
                  f"(held {hold_time.total_seconds()/3600:.1f}h)")
        
        # Update dashboard display
        if self.dashboard:
            self.dashboard.update()
            plt.pause(0.01)  # Allow plot to update
            
            # Save dashboard periodically (every 12 hours = 12 iterations of 60min)
            if self.iteration_count % 12 == 0 and self.iteration_count > 0:
                self._save_live_dashboard("periodic")
    
    def run(self, check_interval_minutes: int = None):
        """
        Run the trading bot with hybrid checking strategy:
        - Check TP/SL every 1 minute when position is open (near real-time)
        - Check signals every N minutes for new trades (from config)
        
        Args:
            check_interval_minutes: How often to check signals (default from config or 60)
        """
        if not self.initialize():
            print(" Initialization failed")
            return
        
        # Get check interval from config (scalping uses shorter interval)
        if check_interval_minutes is None:
            scalping_config = self.config.get('scalping', {})
            if scalping_config.get('enabled', False):
                check_interval_minutes = scalping_config.get('check_interval_minutes', 5)
                timeframe_name = 'M5 Scalping'
            else:
                check_interval_minutes = 60  # Default H1
                timeframe_name = 'H1 Swing'
        else:
            timeframe_name = f'M{check_interval_minutes}'
        
        self.is_running = True
        print(f"\n Trading bot started")
        print(f" Strategy: {timeframe_name}")
        print(f"  - TP/SL check: Every 1 minute (when position open)")
        print(f"  - Signal check: Every {check_interval_minutes} minutes")
        print(f"Press Ctrl+C to stop\n")
        
        # Show dashboard if enabled
        if self.dashboard:
            plt.ion()  # Enable interactive mode
            self.dashboard.update()
            plt.show(block=False)
        
        last_h1_check = datetime.now()
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Check if we have an open position
                has_position = self.signal_generator.has_position(self.symbol)
                
                if has_position:
                    # FAST MODE: Check TP/SL every 1 minute
                    print(f"\n{'='*60}")
                    print(f" Position Monitoring - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}")
                    
                    position_closed = self.check_open_positions()
                    
                    if self.dashboard:
                        self.dashboard.update()
                        plt.pause(0.01)
                    
                    # If position still open, wait 1 minute
                    if not position_closed:
                        print(f"️ Position active - checking again in 1 minute...")
                        time.sleep(60)  # 1 minute
                    else:
                        print(f" Position closed - resuming normal checks")
                
                else:
                    # NO POSITION: Check less frequently
                    time_since_check = (current_time - last_h1_check).total_seconds() / 60
                    
                    if time_since_check >= check_interval_minutes:
                        # Time for signal check
                        self.iteration_count += 1
                        self.run_iteration()
                        last_h1_check = current_time
                        
                        # Save periodic dashboard (every 60 iterations for scalping)
                        save_interval = 60 if check_interval_minutes <= 5 else 12
                        if self.iteration_count % save_interval == 0 and self.iteration_count > 0:
                            self._save_live_dashboard("periodic")
                    
                    # Calculate wait time
                    remaining_seconds = max(0, (check_interval_minutes * 60) - (time_since_check * 60))
                    remaining_minutes = remaining_seconds / 60
                    print(f"\n️ No position - next signal check in {remaining_minutes:.1f} min...")
                    time.sleep(max(check_interval_minutes * 60 // 10, 30))  # Check periodically
        
        except KeyboardInterrupt:
            print("\n\n️ Stopping bot...")
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        
        # Save final dashboard state
        if self.dashboard:
            self._save_live_dashboard("final_state")
        
        # Print final statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        
        self.trading_filters.print_statistics()
        
        print(" Bot stopped successfully")
    
    def backtest(self, start_date: str, end_date: str):
        """
        Run backtest on historical data (simplified version)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING MODE")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*60}\n")
        
        # This is a placeholder for backtesting functionality
        # Full implementation would:
        # 1. Fetch historical data for the period
        # 2. Simulate bar-by-bar execution
        # 3. Track all trades and performance metrics
        # 4. Generate comprehensive report
        
        print("️ Backtesting not fully implemented yet")
        print("For backtesting, run bot in simulation mode with historical data")
    
    def _save_live_dashboard(self, reason: str):
        """Save live dashboard snapshot to outputs/live folder"""
        import os
        from datetime import datetime
        
        # Create outputs/live directory if not exists
        os.makedirs("outputs/live", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/live/live_trading_{reason}_{timestamp}.png"
        
        try:
            self.dashboard.save_figure(filename)
            print(f" Live dashboard saved: {filename}")
            self.last_dashboard_save = datetime.now()
        except Exception as e:
            print(f"️ Failed to save dashboard: {e}")


def main():
    """Main entry point"""
    import sys
    
    # Check for dashboard flag
    use_dashboard = '--dashboard' in sys.argv or '-d' in sys.argv
    
    bot = TradingBot(config_path='config.json', use_dashboard=use_dashboard)
    
    if use_dashboard:
        print("\n Dashboard mode enabled - live visualization active")
    
    # Run in live mode (checks every hour)
    bot.run(check_interval_minutes=60)
    
    # For backtesting:
    # bot.backtest('2024-01-01', '2024-12-01')


if __name__ == "__main__":
    main()
