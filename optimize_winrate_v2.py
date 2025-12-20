"""
Extended Optimization - Smaller SL and Higher Thresholds
Trying to reach 55%+ win rate
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
    print("🎯 EXTENDED OPTIMIZATION - SMALLER SL VALUES")
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
    
    # Split data
    split_idx = int(len(prices) * 0.7)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    
    # Create training dataset
    print("🧠 Training model...")
    processor = DataProcessor(window_size=112, use_indicators=True)
    X_train, y_train = processor.create_training_dataset(train_prices, symbol, training_bars=3000)
    
    network = NeuralNetwork(config=config)
    network.train(X_train, y_train, verbose=False)
    print(f"✓ Model trained")
    
    # Extended parameter grid - smaller SL values
    sl_tp_configs = [
        # Very tight stops
        (25, 25),
        (25, 30),
        (25, 35),
        (30, 30),
        (30, 35),
        (30, 40),
        (35, 35),
        (35, 40),
        (35, 45),
        (40, 40),
        (40, 45),
        (40, 50),
        (45, 45),
        (45, 50),
        # Also test the working config with higher thresholds
        (75, 75),
        (50, 50),
    ]
    
    # Higher thresholds
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    results = []
    
    print(f"\n{'SL':<6} {'TP':<6} {'Thresh':<8} {'WinRate':<10} {'Trades':<8} {'PnL':<12} {'Status':<10}")
    print("-" * 70)
    
    for sl, tp in sl_tp_configs:
        for threshold in thresholds:
            config['risk_management']['stop_loss_points'] = sl
            config['risk_management']['take_profit_points'] = tp
            config['signal']['threshold'] = threshold
            
            try:
                stats = run_single_backtest(config, network, test_prices, symbol)
                
                win_rate = stats['win_rate']
                total_trades = stats['total_trades']
                total_pnl = stats['total_pnl']
                
                result = {
                    'sl': sl, 'tp': tp, 'threshold': threshold,
                    'win_rate': win_rate, 'trades': total_trades, 'pnl': total_pnl
                }
                results.append(result)
                
                status = "✓ TARGET!" if win_rate >= TARGET_WIN_RATE and total_trades >= 5 else ""
                
                if total_trades > 0:
                    print(f"{sl:<6} {tp:<6} {threshold:<8.1f} {win_rate:<10.1f}% {total_trades:<8} ${total_pnl:<11,.0f} {status}")
                    
            except Exception as e:
                pass
    
    # Results
    print("\n" + "="*70)
    print("📊 TOP CONFIGURATIONS (by win rate, min 5 trades)")
    print("="*70)
    
    valid = [r for r in results if r['trades'] >= 5]
    valid.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print(f"\n{'Rank':<5} {'SL':<6} {'TP':<6} {'Thresh':<8} {'WinRate':<10} {'Trades':<8} {'PnL':<12}")
    print("-" * 65)
    
    for i, r in enumerate(valid[:15], 1):
        marker = "🎯" if r['win_rate'] >= TARGET_WIN_RATE else ""
        print(f"{i:<5} {r['sl']:<6} {r['tp']:<6} {r['threshold']:<8.1f} "
              f"{r['win_rate']:<10.1f}% {r['trades']:<8} ${r['pnl']:<11,.0f} {marker}")
    
    # Find target
    target_met = [r for r in valid if r['win_rate'] >= TARGET_WIN_RATE]
    
    if target_met:
        target_met.sort(key=lambda x: x['trades'], reverse=True)
        optimal = target_met[0]
        
        print(f"\n🏆 TARGET {TARGET_WIN_RATE}% ACHIEVED!")
        print(f"   SL: {optimal['sl']}, TP: {optimal['tp']}, Threshold: {optimal['threshold']}")
        print(f"   Win Rate: {optimal['win_rate']:.1f}%, Trades: {optimal['trades']}")
        
        config['risk_management']['stop_loss_points'] = optimal['sl']
        config['risk_management']['take_profit_points'] = optimal['tp']
        config['signal']['threshold'] = optimal['threshold']
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ config.json updated")
    else:
        print(f"\n⚠️ Target {TARGET_WIN_RATE}% not reached. Best: {valid[0]['win_rate']:.1f}%")
        # Use best anyway
        optimal = valid[0]
        config['risk_management']['stop_loss_points'] = optimal['sl']
        config['risk_management']['take_profit_points'] = optimal['tp']
        config['signal']['threshold'] = optimal['threshold']
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    print("="*70)
    return optimal


if __name__ == "__main__":
    main()
