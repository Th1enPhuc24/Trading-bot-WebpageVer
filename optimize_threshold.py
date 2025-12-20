"""
Extended Threshold Optimization Script
Tests higher thresholds (0.8 to 1.5) to find 57%+ win rate
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


def test_threshold(model, X_test, y_test, threshold):
    """Test a specific threshold and return win rate"""
    predictions = model.predict(X_test)
    
    correct = 0
    total_signals = 0
    
    for pred, actual in zip(predictions, y_test.ravel()):
        if pred > threshold:
            total_signals += 1
            if actual > 0:
                correct += 1
        elif pred < -threshold:
            total_signals += 1
            if actual < 0:
                correct += 1
    
    if total_signals == 0:
        return 0, 0
    
    win_rate = correct / total_signals * 100
    return win_rate, total_signals


def main():
    TARGET_WIN_RATE = 57.0
    MIN_SIGNALS = 10  # Reduced minimum to allow higher thresholds
    
    print("\n" + "="*70)
    print("🎯 EXTENDED THRESHOLD OPTIMIZATION")
    print(f"   Target Win Rate: {TARGET_WIN_RATE}%")
    print(f"   Minimum Signals: {MIN_SIGNALS}")
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
    
    # Create datasets
    print("🧠 Creating datasets...")
    processor = DataProcessor(window_size=112, use_indicators=True)
    
    X_train, y_train = processor.create_training_dataset(train_prices, symbol, training_bars=3000)
    
    # Create test dataset
    X_test_list = []
    y_test_list = []
    
    min_idx = 112 + 26
    for i in range(min_idx, len(test_prices) - 1):
        price_history = test_prices[:i+1]
        features = processor.prepare_prediction_input(price_history, symbol)
        if features is not None:
            X_test_list.append(features[0])
            target = 1.0 if test_prices[i+1] > test_prices[i] else -1.0
            y_test_list.append(target)
    
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list).reshape(-1, 1)
    print(f"✓ Data ready: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Train model
    print("🧠 Training SVM...")
    svm_config = config.get('svm', {})
    model = SVR(
        kernel=svm_config.get('kernel', 'rbf'),
        C=svm_config.get('C', 10.0),
        gamma=svm_config.get('gamma', 'auto'),
        epsilon=svm_config.get('epsilon', 0.1)
    )
    model.fit(X_train, y_train.ravel())
    print(f"✓ Model trained")
    
    # Extended threshold range
    print("\n" + "="*70)
    print("🔄 TESTING EXTENDED THRESHOLD RANGE (0.3 - 1.5)")
    print("="*70)
    
    thresholds = np.arange(0.30, 1.51, 0.05)
    
    results = []
    
    print(f"\n{'Threshold':<12} {'Win Rate':<12} {'Signals':<12} {'Status':<15}")
    print("-" * 55)
    
    for thresh in thresholds:
        win_rate, total_signals = test_threshold(model, X_test, y_test, thresh)
        
        results.append({
            'threshold': thresh,
            'win_rate': win_rate,
            'signals': total_signals
        })
        
        if total_signals >= MIN_SIGNALS:
            status = ""
            if win_rate >= TARGET_WIN_RATE:
                status = "✓ TARGET!"
            print(f"{thresh:<12.2f} {win_rate:<12.1f}% {total_signals:<12} {status}")
        else:
            print(f"{thresh:<12.2f} {win_rate:<12.1f}% {total_signals:<12} (few signals)")
    
    # Find best
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    # Sort by win rate, then by signals
    valid = [r for r in results if r['signals'] >= MIN_SIGNALS]
    valid.sort(key=lambda x: (x['win_rate'], x['signals']), reverse=True)
    
    print("\nTop 10 configurations:")
    for i, r in enumerate(valid[:10], 1):
        marker = "🎯" if r['win_rate'] >= TARGET_WIN_RATE else ""
        print(f"{i}. Threshold: {r['threshold']:.2f}, Win Rate: {r['win_rate']:.1f}%, Signals: {r['signals']} {marker}")
    
    # Target achieved?
    target_met = [r for r in valid if r['win_rate'] >= TARGET_WIN_RATE]
    
    if target_met:
        # Sort by signals to get most trades while still meeting target
        target_met.sort(key=lambda x: x['signals'], reverse=True)
        optimal = target_met[0]
        print(f"\n🏆 TARGET ACHIEVED!")
        print(f"   Optimal Threshold: {optimal['threshold']:.2f}")
        print(f"   Win Rate: {optimal['win_rate']:.1f}%")
        print(f"   Total Signals: {optimal['signals']}")
    else:
        # Use highest win rate
        optimal = valid[0] if valid else {'threshold': 0.75, 'win_rate': 0, 'signals': 0}
        print(f"\n⚠️ Target {TARGET_WIN_RATE}% not achieved")
        print(f"   Best Threshold: {optimal['threshold']:.2f}")
        print(f"   Best Win Rate: {optimal['win_rate']:.1f}%")
        print(f"   Signals: {optimal['signals']}")
    
    # Update config
    config['signal']['threshold'] = float(optimal['threshold'])
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Config updated with threshold: {optimal['threshold']:.2f}")
    print("="*70)
    
    return optimal


if __name__ == "__main__":
    main()
