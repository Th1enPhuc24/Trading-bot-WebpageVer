"""
Diagnostic Script for SVM Model Performance Analysis
Analyzes why the model has high loss rate
Updated: Uses proper data processor with indicators
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.neural_network import NeuralNetwork
from src.core.data_processor import DataProcessor
from src.core.data_fetcher import TradingViewDataFetcher


def main():
    print("\n" + "="*70)
    print("🔍 SVM MODEL DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    symbol = config['trading']['symbol']
    threshold = config['signal']['threshold']
    
    # Step 1: Fetch data
    print("\n📊 Step 1: Fetching data...")
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
    
    print(f"Training: {len(train_prices)} bars, Test: {len(test_prices)} bars")
    
    # Step 2: Create and train model
    print("\n🧠 Step 2: Training SVM...")
    processor = DataProcessor(window_size=112, use_indicators=True)
    
    X_train, y_train = processor.create_training_dataset(train_prices, symbol, training_bars=3000)
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"y distribution: +1={np.sum(y_train > 0)}, -1={np.sum(y_train < 0)}")
    
    model = NeuralNetwork(config=config)
    model.train(X_train, y_train, verbose=True)
    
    # Step 3: Analyze predictions on test data
    print("\n📈 Step 3: Analyzing predictions on test data...")
    
    predictions = []
    actual_directions = []
    correct_predictions = 0
    
    min_idx = 112 + 26  # Need enough data for indicators
    
    for i in range(min_idx, len(test_prices) - 1):
        # Get all prices up to current point
        price_history = test_prices[:i+1]
        
        # Use prepare_prediction_input (returns properly shaped input with indicators)
        normalized = processor.prepare_prediction_input(price_history, symbol)
        
        if normalized is None:
            continue
        
        # Predict
        pred = model.predict(normalized)[0, 0]
        predictions.append(pred)
        
        # Actual direction
        current_price = test_prices[i]
        next_price = test_prices[i+1]
        actual_dir = 1.0 if next_price > current_price else -1.0
        actual_directions.append(actual_dir)
        
        # Check if prediction matches
        if pred > threshold and actual_dir > 0:
            correct_predictions += 1
        elif pred < -threshold and actual_dir < 0:
            correct_predictions += 1
    
    predictions = np.array(predictions)
    actual_directions = np.array(actual_directions)
    
    # Statistics
    print("\n" + "="*70)
    print("📊 PREDICTION STATISTICS")
    print("="*70)
    
    print(f"\nPrediction Output Distribution:")
    print(f"  Mean: {np.mean(predictions):.4f}")
    print(f"  Std:  {np.std(predictions):.4f}")
    print(f"  Min:  {np.min(predictions):.4f}")
    print(f"  Max:  {np.max(predictions):.4f}")
    
    buy_signals = np.sum(predictions > threshold)
    sell_signals = np.sum(predictions < -threshold)
    hold_signals = len(predictions) - buy_signals - sell_signals
    
    print(f"\nSignal Distribution (threshold={threshold}):")
    print(f"  BUY:  {buy_signals} ({buy_signals/len(predictions)*100:.1f}%)")
    print(f"  SELL: {sell_signals} ({sell_signals/len(predictions)*100:.1f}%)")
    print(f"  HOLD: {hold_signals} ({hold_signals/len(predictions)*100:.1f}%)")
    
    # Accuracy only for signals (not HOLD)
    total_signals = buy_signals + sell_signals
    if total_signals > 0:
        signal_accuracy = correct_predictions / total_signals * 100
    else:
        signal_accuracy = 0
    
    print(f"\nSignal Accuracy:")
    print(f"  Total signals: {total_signals}")
    print(f"  Correct: {correct_predictions}")
    print(f"  Accuracy: {signal_accuracy:.2f}%")
    
    # Actual market movements
    up_moves = np.sum(actual_directions > 0)
    down_moves = np.sum(actual_directions < 0)
    print(f"\nActual Market Movements:")
    print(f"  UP:   {up_moves} ({up_moves/len(actual_directions)*100:.1f}%)")
    print(f"  DOWN: {down_moves} ({down_moves/len(actual_directions)*100:.1f}%)")
    
    # Correlation check
    correlation = np.corrcoef(predictions, actual_directions)[0, 1]
    print(f"\nPrediction vs Actual Correlation: {correlation:.4f}")
    
    # Issue detection
    print("\n" + "="*70)
    print("🔍 ISSUE DETECTION")
    print("="*70)
    
    issues = []
    
    if np.std(predictions) < 0.1:
        issues.append("⚠️ LOW VARIANCE: Model predictions have very low variance")
    
    if abs(np.mean(predictions)) > 0.3:
        issues.append(f"⚠️ BIASED OUTPUT: Mean prediction is {np.mean(predictions):.4f}")
    
    if abs(correlation) < 0.1:
        issues.append(f"⚠️ NO CORRELATION: Predictions have {correlation:.4f} correlation")
    
    if hold_signals < len(predictions) * 0.3:
        issues.append(f"⚠️ OVER-TRADING: Only {hold_signals/len(predictions)*100:.0f}% HOLD signals")
    
    if signal_accuracy < 50 and total_signals > 10:
        issues.append(f"⚠️ LOW ACCURACY: Signal accuracy is {signal_accuracy:.1f}%")
    
    if len(issues) == 0:
        print("✓ No obvious issues detected")
    else:
        for issue in issues:
            print(issue)
    
    print("\n" + "="*70)
    print("📊 SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<25} {'Value':<15} {'Target':<15}")
    print("-" * 55)
    print(f"{'Signal Accuracy':<25} {signal_accuracy:.1f}%{'':<10} >55%")
    print(f"{'Correlation':<25} {correlation:.4f}{'':<10} >0.1")
    print(f"{'HOLD Signals':<25} {hold_signals/len(predictions)*100:.1f}%{'':<10} >30%")
    print(f"{'Total Signals':<25} {total_signals}{'':<10} <500")
    print("="*70 + "\n")
    
    return {
        'signal_accuracy': signal_accuracy,
        'correlation': correlation,
        'hold_pct': hold_signals/len(predictions)*100,
        'total_signals': total_signals
    }


if __name__ == "__main__":
    main()
