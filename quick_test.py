"""
Quick Test Script for SVM Trading Bot
Tests the SVM model, data processor, and training system integration
"""

import json
import numpy as np
import sys
sys.path.insert(0, '.')

from src.core.neural_network import NeuralNetwork
from src.core.data_processor import DataProcessor
from src.core.training_system import TrainingSystem


def test_svm_model():
    """Test basic SVM model operations"""
    print("\n" + "="*60)
    print("TEST 1: SVM Model Basic Operations")
    print("="*60)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create SVM model
    model = NeuralNetwork(config=config)
    print(f"✓ Model created successfully")
    print(f"   Kernel: {model.model.kernel}")
    print(f"   C: {model.model.C}")
    print(f"   Gamma: {model.model.gamma}")
    
    # Test prediction before training (should return zeros)
    test_input = np.random.randn(1, 112)
    output = model.predict(test_input)
    print(f"✓ Prediction before training (should be 0): {output[0,0]}")
    assert output[0,0] == 0.0, "Untrained model should output 0"
    
    # Create synthetic training data
    X_train = np.random.randn(500, 112)
    y_train = np.sign(np.random.randn(500, 1))  # +1 or -1
    
    # Train model
    losses = model.train(X_train, y_train, verbose=True)
    print(f"✓ Training completed, MSE: {losses[0]:.6f}")
    
    # Test prediction after training
    output = model.predict(test_input)
    print(f"✓ Prediction after training: {output[0,0]:.6f}")
    
    # Test save and load
    model.save_weights("models/weights/test_weights.joblib", "TEST")
    
    # Create new model and load weights
    model2 = NeuralNetwork(config=config)
    loaded = model2.load_weights("models/weights/test_weights.joblib")
    print(f"✓ Model save/load: {'Success' if loaded else 'Failed'}")
    
    # Verify predictions match
    output2 = model2.predict(test_input)
    print(f"✓ Loaded model prediction: {output2[0,0]:.6f}")
    assert np.isclose(output[0,0], output2[0,0]), "Predictions should match after load"
    
    print("\n✓ TEST 1 PASSED: SVM Model Basic Operations\n")
    return True


def test_data_processor():
    """Test data processor with Z-score normalization"""
    print("\n" + "="*60)
    print("TEST 2: Data Processor (Z-score Normalization)")
    print("="*60)
    
    # Create processor
    processor = DataProcessor(window_size=112)
    
    # Create synthetic price data
    np.random.seed(42)
    prices = 2000 + np.cumsum(np.random.randn(3500))  # Random walk around 2000
    
    print(f"✓ Price data created: {len(prices)} bars")
    print(f"   Min price: {prices.min():.2f}")
    print(f"   Max price: {prices.max():.2f}")
    
    # Test normalization
    normalized = processor.normalize_prices(prices[-112:], "TEST")
    print(f"✓ Normalized prices stats:")
    print(f"   Mean: {np.mean(normalized):.6f} (should be ~0)")
    print(f"   Std: {np.std(normalized):.6f} (should be ~1)")
    
    # Test input window creation
    window = processor.create_input_window(prices, "TEST")
    print(f"✓ Input window created: shape {window.shape}")
    
    # Test training dataset creation
    X, y = processor.create_training_dataset(prices, "TEST", training_bars=3000)
    print(f"✓ Training dataset created:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y values: +1 count={np.sum(y > 0)}, -1 count={np.sum(y < 0)}")
    
    print("\n✓ TEST 2 PASSED: Data Processor\n")
    return True


def test_training_system():
    """Test training system integration"""
    print("\n" + "="*60)
    print("TEST 3: Training System Integration")
    print("="*60)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create training system
    training_system = TrainingSystem(config)
    print(f"✓ Training system created")
    print(f"   Train after bars: {training_system.train_after_bars}")
    print(f"   Training bars: {training_system.training_bars}")
    
    # Create synthetic price data
    np.random.seed(42)
    prices = 2000 + np.cumsum(np.random.randn(3500))
    
    # Initialize network (should train from scratch)
    network = training_system.initialize_network("TESTSYM", prices)
    print(f"✓ Network initialized and trained")
    
    # Get training stats
    stats = training_system.get_training_stats("TESTSYM")
    if stats:
        print(f"✓ Training stats:")
        print(f"   MSE Loss: {stats['mse_loss']:.6f}")
        print(f"   Training time: {stats['training_time_seconds']:.2f} seconds")
        print(f"   Training samples: {stats['training_samples']}")
    
    # Test should_retrain logic
    bars_until = training_system.get_bars_until_retrain("TESTSYM")
    print(f"✓ Bars until retrain: {bars_until}")
    
    print("\n✓ TEST 3 PASSED: Training System Integration\n")
    return True


def main():
    print("\n" + "="*60)
    print("SVM TRADING BOT - QUICK TEST")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_svm_model()
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_data_processor()
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_training_system()
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
