"""
Complete Training & Backtesting Pipeline
1. Fetch historical data
2. Split into train/test sets
3. Train model on training set
4. Backtest on test set
5. Show results in dashboard
6. Ask user before going live
"""

import json
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_fetcher import TradingViewDataFetcher
from src.core.neural_network import NeuralNetwork
from src.core.data_processor import DataProcessor
from src.core.training_system import TrainingSystem
from src.core.backtest_system import BacktestEngine
from src.utils.dashboard import TradingDashboard
import matplotlib.pyplot as plt


def main():
    print(f"\n{'='*70}")
    print(f"🚀 COMPLETE TRADING BOT PIPELINE")
    print(f"{'='*70}")
    print(f"Pipeline steps:")
    print(f"  1. ✓ Fetch historical data")
    print(f"  2. ✓ Split into train/test (70%/30%)")
    print(f"  3. ✓ Train neural network on training set")
    print(f"  4. ✓ Backtest on test set")
    print(f"  5. ✓ Display results in dashboard")
    print(f"  6. ✓ Ask user before going live")
    print(f"{'='*70}\n")
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    symbol = config['trading']['symbol']
    initial_balance = 10000.0
    
    try:
        # ============================================================
        # STEP 1: Fetch Historical Data
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📊 STEP 1: FETCHING HISTORICAL DATA")
        print(f"{'='*70}")
        
        fetcher = TradingViewDataFetcher(config)
        
        # Fetch maximum available data
        print(f"Fetching maximum available bars...")
        prices = fetcher.get_closing_prices('60', n_bars=5000)
        
        if prices is None or len(prices) < 1000:
            print(f"❌ Insufficient data. Need at least 1000 bars.")
            return
        
        print(f"✓ Fetched {len(prices)} bars")
        print(f"  Date range: ~{len(prices)/24:.0f} days")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        # ============================================================
        # STEP 2: Split Data
        # ============================================================
        print(f"\n{'='*70}")
        print(f"✂️ STEP 2: SPLITTING DATA")
        print(f"{'='*70}")
        
        # Split 70% training, 30% testing
        split_idx = int(len(prices) * 0.7)
        
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]
        
        print(f"Total bars: {len(prices)}")
        print(f"Training set: {len(train_prices)} bars (70%)")
        print(f"Test set: {len(test_prices)} bars (30%)")
        print(f"Training period: ~{len(train_prices)/24:.0f} days")
        print(f"Test period: ~{len(test_prices)/24:.0f} days")
        
        # ============================================================
        # STEP 3: Train Model
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🧠 STEP 3: TRAINING NEURAL NETWORK")
        print(f"{'='*70}")
        
        # Adjust training_bars to available data
        max_training_bars = len(train_prices) - config['network']['input_size']
        training_bars = min(config['training']['training_bars'], max_training_bars)
        
        print(f"Training configuration:")
        print(f"  Training bars: {training_bars}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        
        # Initialize and train
        trainer = TrainingSystem(config)
        
        # Temporarily adjust config
        original_training_bars = config['training']['training_bars']
        config['training']['training_bars'] = training_bars
        
        network = NeuralNetwork(
            input_size=config['network']['input_size'],
            hidden_size=config['network']['hidden_size'],
            output_size=config['network']['output_size']
        )
        
        print(f"\nStarting training...")
        start_time = time.time()
        
        stats = trainer.train_network(network, train_prices, symbol, verbose=True)
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        print(f"  Initial loss: {stats['initial_loss']:.6f}")
        print(f"  Final loss: {stats['final_loss']:.6f}")
        print(f"  Loss reduction: {stats['loss_reduction']:.6f}")
        
        # Restore config
        config['training']['training_bars'] = original_training_bars
        
        # ============================================================
        # STEP 4: Backtest on Test Set
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🧪 STEP 4: BACKTESTING ON TEST SET")
        print(f"{'='*70}")
        
        backtest = BacktestEngine(config)
        
        print(f"Running backtest on {len(test_prices)} bars...")
        print(f"This simulates real trading on unseen data...")
        
        backtest_stats = backtest.run_backtest(
            network=network,
            prices=test_prices,
            symbol=symbol,
            initial_balance=initial_balance,
            verbose=True
        )
        
        # ============================================================
        # STEP 5: Visualize Results
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📊 STEP 5: VISUALIZING RESULTS")
        print(f"{'='*70}")
        
        print(f"Creating dashboard with backtest results...")
        
        # Create dashboard
        dashboard = TradingDashboard(max_bars=len(test_prices))
        
        # Populate dashboard with backtest data
        base_time = datetime.now() - timedelta(hours=len(test_prices))
        
        # Add all price points and equity
        for i, equity in enumerate(backtest.equity_curve):
            if i < len(test_prices):
                timestamp = base_time + timedelta(hours=i)
                dashboard.update_price(timestamp, test_prices[i])
                dashboard.update_equity(equity)
        
        # Add all trading signals with correct signature: (signal_type, timestamp, price)
        for signal_data in backtest.signals_history:
            if signal_data['signal'] in ['BUY', 'SELL']:
                # Find timestamp for this signal
                signal_idx = signal_data.get('bar_idx', 0)
                if signal_idx < len(test_prices):
                    timestamp = base_time + timedelta(hours=signal_idx)
                    dashboard.add_signal(
                        signal_data['signal'],  # 'BUY' or 'SELL'
                        timestamp,
                        signal_data['price']
                    )
        
        # Add close signals from trades
        for trade in backtest.trades:
            # Find exit timestamp
            exit_idx = trade.get('exit_bar', trade.get('entry_bar', 0) + 1)
            if exit_idx < len(test_prices):
                exit_time = base_time + timedelta(hours=exit_idx)
                dashboard.add_signal('CLOSE', exit_time, trade['exit_price'])
            
            # Add trade to dashboard
            dashboard.add_trade({
                'type': trade['type'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl': trade['pnl'],
                'exit_reason': trade['reason']
            })
        
        # Add training loss
        if 'loss_history' in stats:
            for epoch, loss in enumerate(stats['loss_history']):
                if epoch % 10 == 0:  # Every 10 epochs
                    dashboard.add_training_loss(epoch, loss)
        
        # Update metrics
        dashboard.metrics['episode'] = 1
        dashboard.metrics['step'] = len(test_prices)
        dashboard.metrics['total_trades'] = backtest_stats['total_trades']
        dashboard.metrics['long_trades'] = sum(1 for t in backtest.trades if t['type'] == 'BUY')
        dashboard.metrics['short_trades'] = sum(1 for t in backtest.trades if t['type'] == 'SELL')
        dashboard.metrics['winning_trades'] = backtest_stats['winning_trades']
        dashboard.metrics['losing_trades'] = backtest_stats['losing_trades']
        dashboard.metrics['win_rate'] = backtest_stats['win_rate']
        dashboard.metrics['total_pnl'] = backtest_stats['total_pnl']
        dashboard.metrics['total_reward'] = backtest_stats['total_pnl']
        dashboard.metrics['sharpe_ratio'] = backtest_stats['sharpe_ratio']
        dashboard.metrics['max_drawdown'] = backtest_stats['max_drawdown_pct']
        dashboard.metrics['current_balance'] = backtest_stats['final_balance']
        dashboard.metrics['starting_balance'] = 10000.0  # Initial balance is hardcoded
        
        # Update metrics (skip if dashboard doesn't have this method)
        # dashboard.update_metrics(
        #     win_rate=backtest_stats['win_rate'],
        #     sharpe=backtest_stats['sharpe_ratio'],
        #     max_dd=backtest_stats['max_drawdown_pct'],
        #     total_pnl=backtest_stats['total_pnl']
        # )
        
        # Show dashboard
        print(f"✓ Dashboard created with backtest results")
        
        # Save dashboard image with timestamp to outputs folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"outputs/backtests/backtest_results_{timestamp}.png"
        dashboard.save_figure(image_filename)
        print(f"✓ Dashboard saved to: {image_filename}")
        
        print(f"\nDisplaying dashboard...")
        print(f"Close the dashboard window to continue...\n")
        
        dashboard.update()
        plt.show()
        
        # ============================================================
        # STEP 6: User Decision
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🎯 STEP 6: DECISION TIME")
        print(f"{'='*70}")
        
        print(f"\n📊 BACKTEST SUMMARY:")
        print(f"  Win Rate: {backtest_stats['win_rate']:.2f}%")
        print(f"  Total Return: {backtest_stats['return_pct']:.2f}%")
        print(f"  Profit Factor: {backtest_stats['profit_factor']:.2f}")
        print(f"  Max Drawdown: {backtest_stats['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {backtest_stats['sharpe_ratio']:.2f}")
        
        # Evaluate performance
        print(f"\n🔍 PERFORMANCE EVALUATION:")
        
        is_profitable = backtest_stats['return_pct'] > 0
        good_win_rate = backtest_stats['win_rate'] >= 50
        good_profit_factor = backtest_stats['profit_factor'] >= 1.5
        acceptable_dd = backtest_stats['max_drawdown_pct'] <= 20
        
        print(f"  {'✓' if is_profitable else '✗'} Profitable: {backtest_stats['return_pct']:.2f}%")
        print(f"  {'✓' if good_win_rate else '✗'} Win Rate: {backtest_stats['win_rate']:.2f}%")
        print(f"  {'✓' if good_profit_factor else '✗'} Profit Factor: {backtest_stats['profit_factor']:.2f}")
        print(f"  {'✓' if acceptable_dd else '✗'} Max Drawdown: {backtest_stats['max_drawdown_pct']:.2f}%")
        
        all_good = is_profitable and good_win_rate and good_profit_factor and acceptable_dd
        
        if all_good:
            print(f"\n✅ Model performance looks GOOD!")
            recommendation = "RECOMMENDED for live trading"
        elif is_profitable:
            print(f"\n⚠️ Model performance is ACCEPTABLE but could be better")
            recommendation = "Consider live trading with caution"
        else:
            print(f"\n❌ Model performance is POOR")
            recommendation = "NOT RECOMMENDED for live trading"
        
        print(f"\n💡 Recommendation: {recommendation}")
        
        # Ask user
        print(f"\n{'='*70}")
        response = input(f"\n🚀 Do you want to proceed with LIVE TRADING? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print(f"\n{'='*70}")
            print(f"🚀 STARTING LIVE TRADING")
            print(f"{'='*70}")
            print(f"\nLaunching bot with trained model...")
            print(f"Run: python run_with_dashboard.py")
            print(f"\nModel saved to: weights/weights_{symbol}.bin")
            print(f"You can start live trading anytime!")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"⏹️ LIVE TRADING CANCELLED")
            print(f"{'='*70}")
            print(f"\nModel has been trained and saved.")
            print(f"You can:")
            print(f"  1. Review the backtest results")
            print(f"  2. Adjust parameters in config.json")
            print(f"  3. Re-run this pipeline")
            print(f"  4. Start live trading later with: python run_with_dashboard.py")
            print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
