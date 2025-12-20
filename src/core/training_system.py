"""
Training System for SVM Model
Handles training and retraining of SVM model
Updated: Removed epochs loop, SVM uses batch learning
"""

import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import os
from .neural_network import NeuralNetwork
from .data_processor import DataProcessor


class TrainingSystem:
    """
    Manages SVM model training and retraining
    - Retrains every N new bars (TrainAfterBars from config)
    - Uses sliding window of most recent bars for training
    - SVM trains in one pass (no epochs)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.train_after_bars = config['training']['train_after_bars']
        self.training_bars = config['training']['training_bars']
        
        # Track bars since last training per symbol
        self.bars_since_training = {}
        self.last_training_time = {}
        self.training_history = {}
        
        # Initialize processors
        self.data_processor = DataProcessor(window_size=config['network']['input_size'])
        
        # Weights directory - updated to models folder
        self.weights_dir = "models/weights"
        os.makedirs(self.weights_dir, exist_ok=True)
    
    def should_retrain(self, symbol: str, new_bars_count: int = 1) -> bool:
        """
        Check if retraining should be triggered
        
        Args:
            symbol: Trading symbol
            new_bars_count: Number of new bars since last check
        
        Returns:
            True if should retrain (every train_after_bars), False otherwise
        """
        if symbol not in self.bars_since_training:
            self.bars_since_training[symbol] = 0
        
        self.bars_since_training[symbol] += new_bars_count
        
        if self.bars_since_training[symbol] >= self.train_after_bars:
            return True
        
        return False
    
    def train_network(self, network: NeuralNetwork, prices: np.ndarray, 
                     symbol: str, verbose: bool = True) -> Dict:
        """
        Train SVM model on historical data
        
        Args:
            network: SVM model instance (NeuralNetwork wrapper)
            prices: Historical closing prices (must have >= training_bars)
            symbol: Trading symbol
            verbose: Print training progress
        
        Returns:
            Dictionary with training statistics
        """
        if len(prices) < self.training_bars:
            raise ValueError(f"Insufficient data for training: need {self.training_bars}, got {len(prices)}")
        
        # Create training dataset using sliding window
        X, y = self.data_processor.create_training_dataset(
            prices, symbol, self.training_bars
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training SVM for {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Training samples: {len(X)}")
            print(f"Features per sample: {X.shape[1]}")
            print(f"{'='*60}")
        
        # Train SVM model (single pass, no epochs)
        start_time = datetime.now()
        losses = network.train(X, y, verbose=verbose)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save updated model
        weights_path = os.path.join(self.weights_dir, f"weights_{symbol}.joblib")
        network.save_weights(weights_path, symbol)
        
        # Update tracking
        self.bars_since_training[symbol] = 0
        self.last_training_time[symbol] = datetime.now()
        
        # Store training history
        if symbol not in self.training_history:
            self.training_history[symbol] = []
        
        training_stats = {
            'timestamp': datetime.now(),
            'training_bars': self.training_bars,
            'training_samples': len(X),
            'training_time_seconds': training_time,
            'mse_loss': losses[0],
            'initial_loss': losses[0],  # For compatibility with NN interface
            'final_loss': losses[0],    # Same as initial for SVM (single pass)
            'loss_reduction': 0.0,       # No reduction for SVM
            'loss_history': losses
        }
        
        self.training_history[symbol].append(training_stats)
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds!")
            print(f"MSE Loss: {losses[0]:.6f}")
            print(f"Model saved to: {weights_path}")
            print(f"{'='*60}\n")
        
        return training_stats
    
    def initialize_network(self, symbol: str, prices: Optional[np.ndarray] = None) -> NeuralNetwork:
        """
        Initialize SVM model with pre-trained weights or train from scratch
        
        Args:
            symbol: Trading symbol
            prices: Historical prices for training if no pre-trained weights exist
        
        Returns:
            Initialized SVM model
        """
        # Create SVM model with config
        network = NeuralNetwork(
            config=self.config,
            input_size=self.config['network']['input_size'],
            output_size=self.config['network']['output_size']
        )
        
        # Try to load pre-trained weights
        weights_path = os.path.join(self.weights_dir, f"weights_{symbol}.joblib")
        
        if network.load_weights(weights_path):
            print(f"Loaded pre-trained SVM model for {symbol}")
            self.bars_since_training[symbol] = 0
        else:
            print(f"No pre-trained model found for {symbol}")
            
            # If prices provided, train from scratch
            if prices is not None and len(prices) >= self.training_bars:
                print(f"Training SVM for {symbol} from scratch...")
                self.train_network(network, prices, symbol, verbose=True)
            else:
                print(f"Starting with untrained SVM model for {symbol}")
                self.bars_since_training[symbol] = 0
        
        return network
    
    def check_and_retrain(self, network: NeuralNetwork, prices: np.ndarray, 
                         symbol: str, new_bars: int = 1) -> bool:
        """
        Check if retraining is needed and execute if so
        
        Args:
            network: SVM model instance
            prices: Current historical prices
            symbol: Trading symbol
            new_bars: Number of new bars since last check
        
        Returns:
            True if retraining was performed, False otherwise
        """
        if self.should_retrain(symbol, new_bars):
            if len(prices) >= self.training_bars:
                print(f"\n🔄 Retraining triggered for {symbol} (after {self.train_after_bars} bars)")
                self.train_network(network, prices, symbol, verbose=True)
                return True
            else:
                print(f"⚠️ Cannot retrain {symbol}: insufficient data ({len(prices)}/{self.training_bars})")
                return False
        
        return False
    
    def get_training_stats(self, symbol: str) -> Optional[Dict]:
        """Get latest training statistics for a symbol"""
        if symbol in self.training_history and len(self.training_history[symbol]) > 0:
            return self.training_history[symbol][-1]
        return None
    
    def get_bars_until_retrain(self, symbol: str) -> int:
        """Get number of bars until next retraining"""
        if symbol not in self.bars_since_training:
            return self.train_after_bars
        return self.train_after_bars - self.bars_since_training[symbol]
