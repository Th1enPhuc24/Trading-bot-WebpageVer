"""
Neural Network Core: 112 → 7 → 1 Feed-Forward Network
Implements exact specifications with tanh activation and backpropagation
"""

import numpy as np
import pickle
import os
from typing import Tuple, Optional


class NeuralNetwork:
    """
    Fully connected feed-forward network: 112 input → 7 hidden → 1 output
    Activation: hyperbolic tangent (tanh) for both hidden and output layers
    Derivative: 1 - x² (used during backpropagation)
    """
    
    def __init__(self, input_size: int = 112, hidden_size: int = 7, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        
        # Store activations for backpropagation
        self.hidden_activation = None
        self.output_activation = None
        self.input_data = None
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh: 1 - x²"""
        return 1 - x ** 2
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            X: Input data, shape (batch_size, 112) or (112,)
        
        Returns:
            Output predictions, shape (batch_size, 1) or (1,)
        """
        # Ensure input is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        self.input_data = X
        
        # Hidden layer: input → hidden with tanh activation
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = self.tanh(hidden_input)
        
        # Output layer: hidden → output with tanh activation
        output_input = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output
        self.output_activation = self.tanh(output_input)
        
        return self.output_activation
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """
        Backpropagation with stochastic gradient descent
        
        Args:
            X: Input data, shape (batch_size, 112)
            y: Target values, shape (batch_size, 1)
            learning_rate: Learning rate (default 0.0155)
        
        Returns:
            Mean squared error loss
        """
        batch_size = X.shape[0]
        
        # Forward pass
        output = self.forward(X)
        
        # Calculate loss (MSE)
        loss = np.mean((output - y) ** 2)
        
        # Output layer gradients
        output_error = output - y
        output_delta = output_error * self.tanh_derivative(self.output_activation)
        
        # Hidden layer gradients
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.tanh_derivative(self.hidden_activation)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_activation.T, output_delta) / batch_size
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / batch_size
        
        self.weights_input_hidden -= learning_rate * np.dot(self.input_data.T, hidden_delta) / batch_size
        self.bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / batch_size
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 270, 
              learning_rate: float = 0.0155, verbose: bool = True) -> list:
        """
        Train the network using stochastic gradient descent
        
        Args:
            X: Training data, shape (training_bars, 112)
            y: Target values, shape (training_bars, 1)
            epochs: Number of training epochs (default 270)
            learning_rate: Learning rate (default 0.0155)
            verbose: Print training progress
        
        Returns:
            List of losses per epoch
        """
        losses = []
        
        for epoch in range(epochs):
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network
        
        Args:
            X: Input data, shape (batch_size, 112) or (112,)
        
        Returns:
            Predictions, shape (batch_size, 1) or (1,)
        """
        return self.forward(X)
    
    def save_weights(self, filepath: str, symbol: str):
        """
        Save network weights and biases to file
        
        Args:
            filepath: Path to save weights
            symbol: Trading symbol for identification
        """
        weights_data = {
            'symbol': symbol,
            'weights_input_hidden': self.weights_input_hidden,
            'bias_hidden': self.bias_hidden,
            'weights_hidden_output': self.weights_hidden_output,
            'bias_output': self.bias_output
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(weights_data, f)
        
        print(f"Weights saved for {symbol} to {filepath}")
    
    def load_weights(self, filepath: str) -> bool:
        """
        Load pre-trained weights from file
        
        Args:
            filepath: Path to weights file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Weights file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)
            
            self.weights_input_hidden = weights_data['weights_input_hidden']
            self.bias_hidden = weights_data['bias_hidden']
            self.weights_hidden_output = weights_data['weights_hidden_output']
            self.bias_output = weights_data['bias_output']
            
            print(f"Weights loaded for {weights_data['symbol']} from {filepath}")
            return True
        
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def get_weights_info(self) -> dict:
        """Get information about current network weights"""
        return {
            'input_hidden_shape': self.weights_input_hidden.shape,
            'input_hidden_mean': np.mean(self.weights_input_hidden),
            'input_hidden_std': np.std(self.weights_input_hidden),
            'hidden_output_shape': self.weights_hidden_output.shape,
            'hidden_output_mean': np.mean(self.weights_hidden_output),
            'hidden_output_std': np.std(self.weights_hidden_output)
        }
