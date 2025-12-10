"""
Training System with Exact Specifications
Handles retraining every 20 H1 bars with 340 bars, 270 epochs, learning rate 0.0155
"""

import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import os
from .neural_network import NeuralNetwork
from .data_processor import DataProcessor


class TrainingSystem:
    """
    Manages neural network training and retraining
    - Retrains every 20 new H1 bars (TrainAfterBars=20)
    - Uses 340 most recent bars for training (TrainingBars=340)
    - Runs 270 epochs per training session (Epochs=270)
    - Fixed learning rate 0.0155
    - Binary target: (Close[i-1] > Close[i]) ? +1 : -1
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.train_after_bars = config['training']['train_after_bars']
        self.training_bars = config['training']['training_bars']
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        
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
            True if should retrain (every 20 bars), False otherwise
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
        Train network with exact specifications
        
        Args:
            network: Neural network instance
            prices: Historical closing prices (must have >= training_bars)
            symbol: Trading symbol
            verbose: Print training progress
        
        Returns:
            Dictionary with training statistics
        """
        if len(prices) < self.training_bars:
            raise ValueError(f"Insufficient data for training: need {self.training_bars}, got {len(prices)}")
        
        # Create training dataset using 340 most recent bars
        X, y = self.data_processor.create_training_dataset(
            prices, symbol, self.training_bars
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Training samples: {len(X)}")
            print(f"Epochs: {self.epochs}, Learning rate: {self.learning_rate}")
            print(f"{'='*60}")
        
        # Train network: 270 epochs, learning rate 0.0155
        losses = network.train(
            X, y, 
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            verbose=verbose
        )
        
        # Save updated weights
        weights_path = os.path.join(self.weights_dir, f"weights_{symbol}.bin")
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
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'loss_reduction': losses[0] - losses[-1],
            'loss_history': losses
        }
        
        self.training_history[symbol].append(training_stats)
        
        if verbose:
            print(f"Training completed!")
            print(f"Initial loss: {losses[0]:.6f}")
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Loss reduction: {losses[0] - losses[-1]:.6f}")
            print(f"Weights saved to: {weights_path}")
            print(f"{'='*60}\n")
        
        return training_stats
    
    def initialize_network(self, symbol: str, prices: Optional[np.ndarray] = None) -> NeuralNetwork:
        """
        Initialize neural network with pre-trained weights or train from scratch
        
        Args:
            symbol: Trading symbol
            prices: Historical prices for training if no pre-trained weights exist
        
        Returns:
            Initialized neural network
        """
        network = NeuralNetwork(
            input_size=self.config['network']['input_size'],
            hidden_size=self.config['network']['hidden_size'],
            output_size=self.config['network']['output_size']
        )
        
        # Try to load pre-trained weights
        weights_path = os.path.join(self.weights_dir, f"weights_{symbol}.bin")
        
        if network.load_weights(weights_path):
            print(f"Loaded pre-trained weights for {symbol}")
            self.bars_since_training[symbol] = 0
        else:
            print(f"No pre-trained weights found for {symbol}")
            
            # If prices provided, train from scratch
            if prices is not None and len(prices) >= self.training_bars:
                print(f"Training {symbol} from scratch...")
                self.train_network(network, prices, symbol, verbose=True)
            else:
                print(f"Starting with random weights for {symbol}")
                self.bars_since_training[symbol] = 0
        
        return network
    
    def check_and_retrain(self, network: NeuralNetwork, prices: np.ndarray, 
                         symbol: str, new_bars: int = 1) -> bool:
        """
        Check if retraining is needed and execute if so
        
        Args:
            network: Neural network instance
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
