"""
Example script demonstrating how to use the trading bot components
"""

import json
import numpy as np
from neural_network import NeuralNetwork
from data_processor import DataProcessor
from training_system import TrainingSystem
from data_fetcher import TradingViewDataFetcher
from signal_generator import SignalGenerator
from multi_timeframe import MultiTimeframeAnalyzer
from risk_manager import RiskManager
from trading_filters import TradingFilters


def example_neural_network():
    """Example: Using the neural network directly"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Neural Network")
    print("="*60)
    
    # Create network
    network = NeuralNetwork(input_size=112, hidden_size=7, output_size=1)
    
    # Generate random input (normally this would be normalized prices)
    input_data = np.random.randn(1, 112)
    
    # Make prediction
    output = network.predict(input_data)
    print(f"Network output: {output[0, 0]:.6f}")
    
    # Training example
    X_train = np.random.randn(100, 112)
    y_train = np.random.choice([-1, 1], size=(100, 1)).astype(float)
    
    print("\nTraining network (5 epochs for demo)...")
    losses = network.train(X_train, y_train, epochs=5, verbose=True)
    
    # Save weights
    network.save_weights('weights/demo_weights.bin', 'DEMO')


def example_data_processing():
    """Example: Data normalization and preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Data Processing")
    print("="*60)
    
    processor = DataProcessor(window_size=112)
    
    # Simulate price data
    prices = np.array([2600 + i*0.1 + np.random.randn()*2 for i in range(400)])
    
    print(f"Raw prices: min={prices.min():.2f}, max={prices.max():.2f}")
    
    # Normalize
    normalized = processor.normalize_prices(prices, 'GC1!')
    print(f"Normalized: min={normalized.min():.4f}, max={normalized.max():.4f}")
    
    # Create training dataset
    X, y = processor.create_training_dataset(prices, 'GC1!', training_bars=340)
    print(f"\nTraining dataset created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Targets: {np.sum(y == 1)} ups, {np.sum(y == -1)} downs")


def example_signal_generation():
    """Example: Generating trading signals"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Signal Generation")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    network = NeuralNetwork(input_size=112, hidden_size=7, output_size=1)
    signal_gen = SignalGenerator(config)
    
    # Simulate normalized input
    normalized_input = np.random.randn(1, 112) * 0.5
    
    # Generate signal
    signal = signal_gen.generate_signal(
        network,
        normalized_input,
        'GC1!',
        current_price=2650.5
    )
    
    print(f"\nSignal: {signal['signal']}")
    print(f"Output: {signal['output_value']:.6f}")
    print(f"Threshold: ±{signal['threshold']:.6f}")


def example_multi_timeframe():
    """Example: Multi-timeframe analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multi-Timeframe Analysis")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    mtf_analyzer = MultiTimeframeAnalyzer(config)
    
    # Note: This example requires actual TradingView data
    # For demo, we'll show the structure
    print("\nMulti-timeframe analysis requires:")
    print("  1. Daily data → determine buy/sell bias")
    print("  2. H1 data → assess strength (strong/moderate/weak)")
    print("  3. M5 data → precise entry timing")
    print("\nSee trading_bot.py for full implementation")


def example_risk_management():
    """Example: Risk and money management"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Risk Management")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    risk_mgr = RiskManager(config)
    
    # Calculate position info
    position_info = risk_mgr.get_position_info(
        symbol='GC1!',
        direction='BUY',
        entry_price=2650.0,
        account_balance=10000.0
    )
    
    risk_mgr.print_position_info(position_info)


def example_trading_filters():
    """Example: Trading filters and constraints"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Trading Filters")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    filters = TradingFilters(config)
    
    # Check trading hours
    in_hours = filters.is_trading_hours()
    print(f"In trading hours: {in_hours}")
    
    # Check if should trade
    should_trade = filters.should_trade(volume=5000)
    print(f"\nShould trade: {should_trade['allowed']}")
    if not should_trade['allowed']:
        print(f"Reasons: {', '.join(should_trade['reasons'])}")


def example_full_workflow():
    """Example: Complete trading workflow"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Complete Workflow")
    print("="*60)
    
    print("\nFor complete workflow, run:")
    print("  python trading_bot.py")
    print("\nThis will:")
    print("  1. Initialize all components")
    print("  2. Fetch TradingView data")
    print("  3. Load/train neural network")
    print("  4. Monitor H1 bars")
    print("  5. Generate signals")
    print("  6. Execute trades (simulation)")
    print("  7. Auto-retrain every 20 bars")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("TRADING BOT EXAMPLES")
    print("="*60)
    
    try:
        example_neural_network()
        example_data_processing()
        example_signal_generation()
        example_multi_timeframe()
        example_risk_management()
        example_trading_filters()
        example_full_workflow()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
