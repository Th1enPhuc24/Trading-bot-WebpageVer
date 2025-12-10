"""
Complete Trading Bot Pipeline
Quy trình: Data Collection → Processing → Training → Testing → Live Trading

Author: Trading Bot Team
Date: December 2, 2025
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import (
    TradingViewDataFetcher, DataProcessor, NeuralNetwork,
    TrainingSystem, BacktestEngine, SignalGenerator, RiskManager
)
from src.utils import TradingDashboard


class TradingPipeline:
    """Main trading pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize pipeline with configuration"""
        print("="*70)
        print("🚀 NEURAL NETWORK TRADING BOT - COMPLETE PIPELINE")
        print("="*70)
        print()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.symbol = self.config['trading']['symbol']
        self.timeframe = self.config['trading']['timeframes'][1]  # Use H1 (60 minutes)
        
        # Initialize components
        self.data_fetcher = TradingViewDataFetcher(self.config)
        self.data_processor = DataProcessor(self.config)
        self.network = NeuralNetwork(
            input_size=self.config['network']['input_size'],
            hidden_size=self.config['network']['hidden_size'],
            output_size=self.config['network']['output_size']
        )
        self.training_system = TrainingSystem(self.config)
        self.backtest_system = BacktestEngine(self.config)
        self.signal_generator = None  # Will be initialized after training
        self.risk_manager = None  # Will be initialized after training
        
        # Data storage
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.backtest_results = None
        
        print(f"✓ Configuration loaded")
        print(f"  Symbol: {self.symbol}")
        print(f"  Timeframe: {self.timeframe}")
        print(f"  Training epochs: {self.config['training']['epochs']}")
        print()
    
    def step1_collect_data(self, num_bars: int = 5000):
        """Step 1: Thu thập dữ liệu từ TradingView"""
        print("="*70)
        print("📊 STEP 1: DATA COLLECTION")
        print("="*70)
        
        print(f"Fetching {num_bars} bars of {self.symbol}...")
        df = self.data_fetcher.fetch_data(
            timeframe=self.timeframe,
            n_bars=num_bars
        )
        
        if df is None or len(df) == 0:
            print("❌ Failed to fetch data")
            return False
        
        self.raw_data = df
        print(f"✓ Collected {len(df)} bars")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print()
        
        return True
    
    def step2_process_data(self, train_split: float = 0.7):
        """Step 2: Xử lý và chuẩn hóa dữ liệu, chia train/test"""
        print("="*70)
        print("🔧 STEP 2: DATA PROCESSING & NORMALIZATION")
        print("="*70)
        
        if self.raw_data is None:
            print("❌ No data to process. Run step1_collect_data first.")
            return False
        
        # Extract prices
        prices = self.raw_data['close'].values
        
        # Split into train/test
        split_idx = int(len(prices) * train_split)
        self.train_data = prices[:split_idx]
        self.test_data = prices[split_idx:]
        
        print(f"✓ Data processed and normalized")
        print(f"  Total bars: {len(prices)}")
        print(f"  Training set: {len(self.train_data)} bars ({train_split*100:.0f}%)")
        print(f"  Test set: {len(self.test_data)} bars ({(1-train_split)*100:.0f}%)")
        print()
        
        return True
    
    def step3_train_model(self):
        """Step 3: Train neural network trên training data"""
        print("="*70)
        print("🧠 STEP 3: NEURAL NETWORK TRAINING")
        print("="*70)
        
        if self.train_data is None:
            print("❌ No training data. Run step2_process_data first.")
            return False
        
        print(f"Training on {len(self.train_data)} bars...")
        print(f"Network: {self.config['network']['input_size']} → "
              f"{self.config['network']['hidden_size']} → 1")
        print()
        
        # Train the network
        history = self.training_system.train(
            network=self.network,
            prices=self.train_data,
            symbol=self.symbol
        )
        
        if history is None:
            print("❌ Training failed")
            return False
        
        print(f"✓ Training completed successfully")
        print(f"  Initial loss: {history['initial_loss']:.6f}")
        print(f"  Final loss: {history['final_loss']:.6f}")
        print(f"  Loss reduction: {history['loss_reduction']:.6f}")
        print(f"  Model saved to: {history['weights_path']}")
        print()
        
        return True
    
    def step4_test_model(self):
        """Step 4: Test model trên test data với backtest"""
        print("="*70)
        print("🧪 STEP 4: MODEL TESTING (BACKTEST)")
        print("="*70)
        
        if self.test_data is None:
            print("❌ No test data. Run step2_process_data first.")
            return False
        
        print(f"Running backtest on {len(self.test_data)} bars...")
        print("This simulates real trading on unseen data...")
        print()
        
        # Run backtest
        self.backtest_results = self.backtest_system.run_backtest(
            network=self.network,
            prices=self.test_data,
            symbol=self.symbol,
            initial_balance=10000.0,
            verbose=True
        )
        
        if self.backtest_results is None:
            print("❌ Backtest failed")
            return False
        
        print(f"✓ Backtest completed")
        print()
        
        return True
    
    def step5_generate_dashboard(self):
        """Step 5: Tạo và xuất dashboard với kết quả test"""
        print("="*70)
        print("📈 STEP 5: DASHBOARD GENERATION")
        print("="*70)
        
        if self.backtest_results is None:
            print("❌ No backtest results. Run step4_test_model first.")
            return False
        
        print("Creating dashboard visualization...")
        
        # Initialize dashboard
        dashboard = TradingDashboard(symbol=self.symbol)
        
        # Populate dashboard with backtest data
        print("Populating dashboard with test results...")
        
        # Calculate base time for timestamps
        if self.raw_data is not None:
            base_time = self.raw_data.index[-len(self.test_data)]
        else:
            base_time = datetime.now()
        
        # Add price data
        for i, price in enumerate(self.test_data):
            timestamp = base_time + pd.Timedelta(hours=i)
            dashboard.update_price(timestamp, price)
        
        # Add signals from backtest
        for trade in self.backtest_system.trades:
            entry_time = base_time + pd.Timedelta(hours=trade['entry_time'])
            exit_time = base_time + pd.Timedelta(hours=trade['exit_time'])
            
            # Add entry signal
            signal_type = 'BUY' if trade['type'] == 'BUY' else 'SELL'
            dashboard.add_signal(signal_type, entry_time, trade['entry_price'])
            
            # Add exit signal
            dashboard.add_signal('CLOSE', exit_time, trade['exit_price'])
            
            # Add trade
            dashboard.add_trade(trade)
        
        # Add equity curve
        for i, equity in enumerate(self.backtest_system.equity_curve):
            timestamp = base_time + pd.Timedelta(hours=i)
            dashboard.update_equity(equity)
        
        # Update metrics
        stats = self.backtest_results
        dashboard.metrics.update({
            'total_trades': stats['total_trades'],
            'long_trades': stats['total_trades'],  # All trades are long in this system
            'short_trades': 0,
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'win_rate': stats['win_rate'],
            'total_pnl': stats['total_pnl'],
            'total_reward': stats['total_pnl'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown_pct'],
            'current_balance': stats['final_balance'],
            'starting_balance': 10000.0
        })
        
        # Update and save dashboard
        dashboard.update()
        
        # Save to outputs folder
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/backtests/backtest_{self.symbol}_{timestamp_str}.png"
        dashboard.save_figure(output_path)
        
        print(f"✓ Dashboard generated and saved")
        print(f"  Output: {output_path}")
        print()
        
        # Display dashboard
        print("Displaying dashboard...")
        print("Close the window to continue...")
        plt.show()
        
        return True
    
    def step6_evaluate_results(self):
        """Step 6: Đánh giá kết quả và quyết định live trading"""
        print("="*70)
        print("🎯 STEP 6: RESULTS EVALUATION")
        print("="*70)
        
        if self.backtest_results is None:
            print("❌ No results to evaluate. Run step4_test_model first.")
            return False
        
        stats = self.backtest_results
        
        print("\n📊 BACKTEST SUMMARY:")
        print(f"  Win Rate: {stats['win_rate']:.2f}%")
        print(f"  Total Return: {stats['return_pct']:.2f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print()
        
        # Performance criteria
        is_profitable = stats['return_pct'] > 0
        good_winrate = stats['win_rate'] >= 50
        good_profit_factor = stats['profit_factor'] >= 1.5
        acceptable_drawdown = stats['max_drawdown_pct'] <= 20
        
        print("🔍 PERFORMANCE EVALUATION:")
        print(f"  {'✓' if is_profitable else '✗'} Profitable: {stats['return_pct']:.2f}%")
        print(f"  {'✓' if good_winrate else '✗'} Win Rate: {stats['win_rate']:.2f}%")
        print(f"  {'✓' if good_profit_factor else '✗'} Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  {'✓' if acceptable_drawdown else '✗'} Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print()
        
        all_good = is_profitable and good_winrate and good_profit_factor and acceptable_drawdown
        
        if all_good:
            print("✅ Model performance looks GOOD!")
            print("💡 Recommendation: RECOMMENDED for live trading")
        else:
            print("⚠️ Model performance needs improvement")
            print("💡 Recommendation: Adjust parameters or retrain before going live")
        
        print()
        return all_good
    
    def step7_start_live_trading(self):
        """Step 7: Bắt đầu live trading"""
        print("="*70)
        print("🚀 STEP 7: LIVE TRADING")
        print("="*70)
        print()
        
        response = input("🚀 Do you want to proceed with LIVE TRADING? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print()
            print("="*70)
            print("⏹️ LIVE TRADING CANCELLED")
            print("="*70)
            print()
            print("Model has been trained and tested.")
            print("You can:")
            print("  1. Review the backtest results")
            print("  2. Adjust parameters in config.json")
            print("  3. Re-run this pipeline")
            print("  4. Start live trading later with: python run_live.py")
            print("="*70)
            return False
        
        print()
        print("="*70)
        print("🎯 STARTING LIVE TRADING")
        print("="*70)
        print()
        
        # Initialize live trading components
        self.signal_generator = SignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Import and start trading bot
        from trading_bot import TradingBot
        
        bot = TradingBot(
            config=self.config,
            network=self.network,
            use_dashboard=True
        )
        
        if bot.initialize():
            bot.run(check_interval_minutes=60)
        else:
            print("❌ Failed to initialize live trading bot")
            return False
        
        return True
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from data collection to live trading"""
        try:
            # Step 1: Collect data
            if not self.step1_collect_data(num_bars=5000):
                return False
            
            # Step 2: Process data
            if not self.step2_process_data(train_split=0.7):
                return False
            
            # Step 3: Train model
            if not self.step3_train_model():
                return False
            
            # Step 4: Test model
            if not self.step4_test_model():
                return False
            
            # Step 5: Generate dashboard
            if not self.step5_generate_dashboard():
                return False
            
            # Step 6: Evaluate results
            if not self.step6_evaluate_results():
                return False
            
            # Step 7: Start live trading (optional)
            self.step7_start_live_trading()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Pipeline interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    
    pipeline = TradingPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline completed with errors")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
