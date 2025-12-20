"""
Kernel Comparison Script for SVM Trading Model
Tests different kernels and hyperparameters to find the best configuration
"""

import json
import numpy as np
import sys
from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_processor import DataProcessor
from src.core.data_fetcher import TradingViewDataFetcher


def test_kernel_config(X_train, y_train, X_test, y_test, kernel, C, gamma='scale', degree=3):
    """Test a specific kernel configuration and return metrics"""
    
    try:
        model = SVR(kernel=kernel, C=C, gamma=gamma, degree=degree, epsilon=0.1)
        model.fit(X_train, y_train.ravel())
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # MSE
        train_mse = np.mean((train_pred - y_train.ravel()) ** 2)
        test_mse = np.mean((test_pred - y_test.ravel()) ** 2)
        
        # Correlation with actual
        correlation = np.corrcoef(test_pred, y_test.ravel())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Direction accuracy
        threshold = 0.15
        correct = 0
        total_signals = 0
        
        for pred, actual in zip(test_pred, y_test.ravel()):
            if pred > threshold:
                total_signals += 1
                if actual > 0:
                    correct += 1
            elif pred < -threshold:
                total_signals += 1
                if actual < 0:
                    correct += 1
        
        accuracy = (correct / total_signals * 100) if total_signals > 0 else 0
        
        return {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'correlation': correlation,
            'accuracy': accuracy,
            'total_signals': total_signals,
            'n_support': len(model.support_)
        }
    except Exception as e:
        return {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'error': str(e)
        }


def main():
    print("\n" + "="*70)
    print("🔬 SVM KERNEL COMPARISON TEST")
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
    
    # Create datasets
    print("\n🧠 Creating training dataset...")
    processor = DataProcessor(window_size=112, use_indicators=True)
    
    X_train, y_train = processor.create_training_dataset(train_prices, symbol, training_bars=3000)
    print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Create test dataset
    print("Creating test dataset...")
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
    print(f"Test: {X_test.shape[0]} samples")
    
    # Test configurations
    print("\n" + "="*70)
    print("🧪 TESTING DIFFERENT KERNEL CONFIGURATIONS")
    print("="*70)
    
    configs = [
        # Linear kernel - different C values
        ('linear', 0.01, 'scale'),
        ('linear', 0.1, 'scale'),
        ('linear', 1.0, 'scale'),
        ('linear', 10.0, 'scale'),
        
        # RBF kernel - different C and gamma values
        ('rbf', 0.1, 'scale'),
        ('rbf', 1.0, 'scale'),
        ('rbf', 10.0, 'scale'),
        ('rbf', 1.0, 'auto'),
        ('rbf', 10.0, 'auto'),
        
        # Polynomial kernel
        ('poly', 1.0, 'scale'),
        ('poly', 10.0, 'scale'),
        
        # Sigmoid kernel
        ('sigmoid', 1.0, 'scale'),
        ('sigmoid', 10.0, 'scale'),
    ]
    
    results = []
    
    for kernel, C, gamma in configs:
        print(f"\nTesting: kernel={kernel}, C={C}, gamma={gamma}...")
        result = test_kernel_config(X_train, y_train, X_test, y_test, kernel, C, gamma)
        results.append(result)
        
        if 'error' not in result:
            print(f"  Train MSE: {result['train_mse']:.4f}, Test MSE: {result['test_mse']:.4f}")
            print(f"  Correlation: {result['correlation']:.4f}, Accuracy: {result['accuracy']:.1f}%")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    # Sort by accuracy
    valid_results = [r for r in results if 'error' not in r and r['total_signals'] > 10]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print ranking
    print("\n" + "="*70)
    print("📊 RESULTS RANKING (by accuracy)")
    print("="*70)
    print(f"\n{'Rank':<5} {'Kernel':<10} {'C':<8} {'Gamma':<8} {'Accuracy':<10} {'Correlation':<12} {'Signals':<10}")
    print("-" * 75)
    
    for i, r in enumerate(valid_results[:10], 1):
        print(f"{i:<5} {r['kernel']:<10} {r['C']:<8} {r['gamma']:<8} "
              f"{r['accuracy']:.1f}%{'':<5} {r['correlation']:.4f}{'':<6} {r['total_signals']}")
    
    # Best configuration
    if valid_results:
        best = valid_results[0]
        print("\n" + "="*70)
        print("🏆 BEST CONFIGURATION")
        print("="*70)
        print(f"  Kernel: {best['kernel']}")
        print(f"  C: {best['C']}")
        print(f"  Gamma: {best['gamma']}")
        print(f"  Accuracy: {best['accuracy']:.2f}%")
        print(f"  Correlation: {best['correlation']:.4f}")
        print(f"  Test MSE: {best['test_mse']:.4f}")
        print(f"  Support Vectors: {best['n_support']}")
        print("="*70 + "\n")
        
        # Update config recommendation
        print("💡 To update config.json with best kernel:")
        print(f'   "svm": {{"kernel": "{best["kernel"]}", "C": {best["C"]}, "gamma": "{best["gamma"]}", ...}}')
    
    return results


if __name__ == "__main__":
    main()
