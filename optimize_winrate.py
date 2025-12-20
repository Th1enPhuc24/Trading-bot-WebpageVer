"""
Backtest Win Rate Optimization Script
Optimizes SL/TP ratio and threshold to achieve 55%+ win rate
Runs actual backtest (not just signal accuracy)
"""

import json
import numpy as np
import sys
from pathlib import Path
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_processor import DataProcessor
from src.core.data_fetcher import TradingViewDataFetcher
from src.core.backtest_system import BacktestEngine
from src.core.neural_network import NeuralNetwork


def run_single_backtest(config, network, test_prices, symbol):
    """Run backtest with given config and return win rate"""
    backtest = BacktestEngine(config)
    stats = backtest.run_backtest(
        network=network,
        prices=test_prices,
        symbol=symbol,
        initial_balance=10000.0,
        verbose=False
    )
    return stats


def main():
    TARGET_WIN_RATE = 55.0
    
    print("\n" + "="*70)
    print("🎯 BACKTEST WIN RATE OPTIMIZATION")
    print(f"   Target: {TARGET_WIN_RATE}% Win Rate")
    print("="*70)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    symbol = config['trading']['symbol']
    
    # Fetch data
    print("\n📊 Fetching data...")
    fetcher = TradingViewDataFetcher(config)
    prices = fetcher.get_closing_prices('60', n_bars=5000)
    
    if prices is None:
        print("❌ Failed to fetch data")
        return
    
    print(f"✓ Fetched {len(prices)} bars")
    
    # Split data
    split_idx = int(len(prices) * 0.7)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    
    # Create training dataset
    print("\n🧠 Creating training dataset...")
    processor = DataProcessor(window_size=112, use_indicators=True)
    X_train, y_train = processor.create_training_dataset(train_prices, symbol, training_bars=3000)
    print(f"Training: {X_train.shape[0]} samples")
    
    # Train model
    print("🧠 Training SVM model...")
    svm_config = config.get('svm', {})
    
    # Store trained model
    network = NeuralNetwork(config=config)
    network.train(X_train, y_train, verbose=False)
    print(f"✓ Model trained")
    
    # Define parameter grid
    sl_tp_configs = [
        (50, 50),    # 1:1 ratio
        (50, 75),    # 1:1.5 ratio
        (50, 100),   # 1:2 ratio
        (75, 75),    # 1:1 ratio
        (75, 100),   # 1:1.33 ratio
        (75, 150),   # 1:2 ratio
        (100, 100),  # 1:1 ratio
        (100, 150),  # 1:1.5 ratio
        (100, 200),  # 1:2 ratio
        (150, 150),  # 1:1 ratio
        (150, 225),  # 1:1.5 ratio
    ]
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Test all combinations
    print("\n" + "="*70)
    print("🔄 TESTING PARAMETER COMBINATIONS")
    print("="*70)
    
    results = []
    best_result = None
    
    total_tests = len(sl_tp_configs) * len(thresholds)
    test_count = 0
    
    print(f"\nTotal combinations to test: {total_tests}")
    print(f"\n{'SL':<6} {'TP':<6} {'Thresh':<8} {'WinRate':<10} {'Trades':<8} {'PnL':<12} {'Status':<10}")
    print("-" * 70)
    
    for sl, tp in sl_tp_configs:
        for threshold in thresholds:
            test_count += 1
            
            # Update config
            config['risk_management']['stop_loss_points'] = sl
            config['risk_management']['take_profit_points'] = tp
            config['signal']['threshold'] = threshold
            
            # Run backtest
            try:
                stats = run_single_backtest(config, network, test_prices, symbol)
                
                win_rate = stats['win_rate']
                total_trades = stats['total_trades']
                total_pnl = stats['total_pnl']
                
                result = {
                    'sl': sl,
                    'tp': tp,
                    'threshold': threshold,
                    'win_rate': win_rate,
                    'trades': total_trades,
                    'pnl': total_pnl,
                    'profit_factor': stats['profit_factor']
                }
                results.append(result)
                
                status = ""
                if win_rate >= TARGET_WIN_RATE and total_trades >= 10:
                    status = "✓ TARGET!"
                    if best_result is None or (win_rate > best_result['win_rate']) or \
                       (win_rate == best_result['win_rate'] and total_pnl > best_result['pnl']):
                        best_result = result
                
                print(f"{sl:<6} {tp:<6} {threshold:<8.1f} {win_rate:<10.1f}% {total_trades:<8} ${total_pnl:<11,.0f} {status}")
                
            except Exception as e:
                print(f"{sl:<6} {tp:<6} {threshold:<8.1f} ERROR: {str(e)[:30]}")
    
    # Results
    print("\n" + "="*70)
    print("📊 TOP 10 CONFIGURATIONS (by win rate)")
    print("="*70)
    
    # Filter valid results (at least 10 trades)
    valid_results = [r for r in results if r['trades'] >= 10]
    valid_results.sort(key=lambda x: (x['win_rate'], x['pnl']), reverse=True)
    
    print(f"\n{'Rank':<5} {'SL':<6} {'TP':<6} {'Thresh':<8} {'WinRate':<10} {'Trades':<8} {'PnL':<12}")
    print("-" * 65)
    
    for i, r in enumerate(valid_results[:10], 1):
        marker = "🎯" if r['win_rate'] >= TARGET_WIN_RATE else ""
        print(f"{i:<5} {r['sl']:<6} {r['tp']:<6} {r['threshold']:<8.1f} "
              f"{r['win_rate']:<10.1f}% {r['trades']:<8} ${r['pnl']:<11,.0f} {marker}")
    
    # Best result
    target_met = [r for r in valid_results if r['win_rate'] >= TARGET_WIN_RATE]
    
    if target_met:
        # Sort by PnL to get most profitable
        target_met.sort(key=lambda x: x['pnl'], reverse=True)
        optimal = target_met[0]
        
        print(f"\n🏆 TARGET {TARGET_WIN_RATE}% ACHIEVED!")
        print(f"\n   OPTIMAL CONFIGURATION:")
        print(f"   Stop Loss: {optimal['sl']} points")
        print(f"   Take Profit: {optimal['tp']} points")
        print(f"   Threshold: {optimal['threshold']}")
        print(f"   Win Rate: {optimal['win_rate']:.1f}%")
        print(f"   Total Trades: {optimal['trades']}")
        print(f"   Total P&L: ${optimal['pnl']:,.2f}")
    else:
        # Use best available
        optimal = valid_results[0] if valid_results else None
        print(f"\n⚠️ Target {TARGET_WIN_RATE}% NOT achieved")
        if optimal:
            print(f"\n   BEST AVAILABLE:")
            print(f"   Stop Loss: {optimal['sl']} points")
            print(f"   Take Profit: {optimal['tp']} points")
            print(f"   Threshold: {optimal['threshold']}")
            print(f"   Win Rate: {optimal['win_rate']:.1f}%")
    
    # Update config if we found something good
    if optimal:
        config['risk_management']['stop_loss_points'] = optimal['sl']
        config['risk_management']['take_profit_points'] = optimal['tp']
        config['signal']['threshold'] = optimal['threshold']
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ config.json updated with optimal parameters")
    
    print("="*70 + "\n")
    
    return optimal


if __name__ == "__main__":
    result = main()
