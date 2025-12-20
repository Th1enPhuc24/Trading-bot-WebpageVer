"""
Large-Scale Training Script
Trains neural network with 10,000+ bars of historical data
"""

import json
import numpy as np
from datetime import datetime
import time

from data_fetcher import TradingViewDataFetcher
from neural_network import NeuralNetwork
from data_processor import DataProcessor
from training_system import TrainingSystem


def fetch_large_dataset(config, n_bars=10000):
    """
    Fetch large dataset from TradingView
    
    Args:
        config: Configuration dictionary
        n_bars: Number of bars to fetch (default 10000)
    
    Returns:
        prices: Numpy array of closing prices
    """
    print(f"\n{'='*70}")
    print(f" FETCHING LARGE DATASET")
    print(f"{'='*70}")
    print(f"Symbol: {config['trading']['symbol']}")
    print(f"Timeframe: H1 (Hourly)")
    print(f"Target bars: {n_bars}")
    print(f"Equivalent time: ~{n_bars/24:.0f} days (≈{n_bars/24/30:.1f} months)")
    print(f"{'='*70}\n")
    
    fetcher = TradingViewDataFetcher(config)
    
    # Fetch data in batches if needed (tvdatafeed-enhanced supports up to 5000 per request)
    if n_bars <= 5000:
        print(f"Fetching {n_bars} bars in single request...")
        prices = fetcher.get_closing_prices('60', n_bars)
    else:
        print(f"Fetching {n_bars} bars in multiple batches...")
        all_prices = []
        remaining = n_bars
        batch_num = 1
        
        while remaining > 0:
            batch_size = min(5000, remaining)
            print(f"  Batch {batch_num}: Fetching {batch_size} bars...")
            
            batch_prices = fetcher.get_closing_prices('60', batch_size)
            
            if batch_prices is not None:
                all_prices.append(batch_prices)
                remaining -= len(batch_prices)
                print(f"     Got {len(batch_prices)} bars")
            else:
                print(f"     Failed to fetch batch {batch_num}")
                break
            
            batch_num += 1
            
            # Small delay to avoid rate limiting
            if remaining > 0:
                time.sleep(2)
        
        if all_prices:
            # Concatenate and remove duplicates
            prices = np.concatenate(all_prices)
            prices = np.unique(prices)
            print(f"\n Total fetched: {len(prices)} unique bars")
        else:
            prices = None
    
    if prices is not None:
        print(f"\n{'='*70}")
        print(f" DATA FETCH SUCCESSFUL")
        print(f"{'='*70}")
        print(f"Total bars: {len(prices)}")
        print(f"Date range: ~{len(prices)/24:.0f} days")
        print(f"Min price: ${prices.min():.2f}")
        print(f"Max price: ${prices.max():.2f}")
        print(f"Mean price: ${prices.mean():.2f}")
        print(f"Latest price: ${prices[-1]:.2f}")
        print(f"{'='*70}\n")
    else:
        print(f"\n DATA FETCH FAILED\n")
    
    return prices


def train_with_large_dataset(config, prices):
    """
    Train neural network with large dataset
    
    Args:
        config: Configuration dictionary
        prices: Historical price data
    """
    symbol = config['trading']['symbol']
    training_bars = config['training']['training_bars']
    
    print(f"\n{'='*70}")
    print(f"🧠 TRAINING NEURAL NETWORK")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"Training bars: {training_bars}")
    print(f"Available data: {len(prices)} bars")
    print(f"Training samples: {len(prices) - config['network']['input_size']} samples")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"{'='*70}\n")
    
    # Initialize training system
    trainer = TrainingSystem(config)
    
    # Initialize network
    network = NeuralNetwork(
        input_size=config['network']['input_size'],
        hidden_size=config['network']['hidden_size'],
        output_size=config['network']['output_size']
    )
    
    # Train network
    start_time = time.time()
    
    stats = trainer.train_network(network, prices, symbol, verbose=True)
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Initial loss: {stats['initial_loss']:.6f}")
    print(f"Final loss: {stats['final_loss']:.6f}")
    print(f"Loss reduction: {stats['loss_reduction']:.6f} ({stats['loss_reduction']/stats['initial_loss']*100:.2f}%)")
    print(f"Weights saved: weights/weights_{symbol}.bin")
    print(f"{'='*70}\n")
    
    return network, stats


def test_predictions(network, prices, config):
    """
    Test network predictions on recent data
    
    Args:
        network: Trained neural network
        prices: Historical price data
        config: Configuration dictionary
    """
    print(f"\n{'='*70}")
    print(f" TESTING PREDICTIONS")
    print(f"{'='*70}\n")
    
    processor = DataProcessor(window_size=config['network']['input_size'])
    symbol = config['trading']['symbol']
    
    # Test on last 10 periods
    test_samples = 10
    
    print(f"Testing on last {test_samples} bars:\n")
    print(f"{'Bar':<6} {'Actual Price':<15} {'Prediction':<12} {'Signal':<10}")
    print(f"{'-'*50}")
    
    correct_direction = 0
    
    for i in range(test_samples):
        idx = -(test_samples - i)
        
        # Get input window
        window_end = len(prices) + idx
        window_start = window_end - config['network']['input_size']
        
        if window_start < 0:
            continue
        
        window_prices = prices[window_start:window_end]
        
        # Normalize
        normalized = processor.normalize_prices(window_prices, symbol)
        
        # Predict
        prediction = network.predict(normalized)
        pred_value = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
        
        # Get actual price
        actual_price = prices[window_end] if window_end < len(prices) else prices[-1]
        
        # Determine signal
        threshold = config['signal']['threshold']
        if pred_value > threshold:
            signal = "BUY ↗"
        elif pred_value < -threshold:
            signal = "SELL ↘"
        else:
            signal = "HOLD —"
        
        # Check direction accuracy (if we have next bar)
        if window_end < len(prices) - 1:
            next_price = prices[window_end + 1]
            actual_direction = "UP" if next_price > actual_price else "DOWN"
            pred_direction = "UP" if pred_value > 0 else "DOWN"
            
            if actual_direction == pred_direction:
                correct_direction += 1
        
        print(f"{i+1:<6} ${actual_price:<14.2f} {pred_value:>11.6f} {signal:<10}")
    
    if test_samples > 1:
        accuracy = (correct_direction / (test_samples - 1)) * 100
        print(f"\n{'='*70}")
        print(f"Direction Accuracy: {correct_direction}/{test_samples-1} = {accuracy:.2f}%")
        print(f"{'='*70}\n")


def main():
    """Main training script"""
    print(f"\n{'='*70}")
    print(f" LARGE-SCALE NEURAL NETWORK TRAINING")
    print(f"{'='*70}")
    print(f"Training Gold Futures (COMEX:GC1!) with 10,000+ bars")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    try:
        # Step 1: Fetch large dataset
        desired_bars = config['training'].get('min_bars_required', 10000)
        prices = fetch_large_dataset(config, desired_bars)
        
        if prices is None:
            print(f" No data fetched. Cannot proceed with training.")
            return
        
        # Use maximum available data
        actual_bars = len(prices)
        training_bars = min(config['training']['training_bars'], actual_bars - config['network']['input_size'])
        
        print(f"\n DATA SUMMARY:")
        print(f"  Requested: {desired_bars} bars")
        print(f"  Received: {actual_bars} bars")
        print(f"  Will use: {training_bars} bars for training")
        print(f"  Training samples: {actual_bars - config['network']['input_size']}")
        
        if actual_bars < 1000:
            print(f"\n️ WARNING: Very limited data ({actual_bars} bars)")
            print(f"Results may not be reliable. Recommended minimum: 1000 bars")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
        
        # Update config temporarily for this training session
        config['training']['training_bars'] = training_bars
        
        # Step 2: Train network
        network, stats = train_with_large_dataset(config, prices)
        
        # Step 3: Test predictions
        test_predictions(network, prices, config)
        
        print(f"\n{'='*70}")
        print(f" ALL STEPS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nYou can now use the trained model for live trading:")
        print(f"  python run_with_dashboard.py")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
