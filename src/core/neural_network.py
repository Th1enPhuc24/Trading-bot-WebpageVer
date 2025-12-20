"""
SVM Model Core: Support Vector Regression for Trading Signal Prediction
Replaces Neural Network with sklearn's SVR for deterministic, batch-learning approach
"""

import numpy as np
import os
import joblib
from typing import Optional


class NeuralNetwork:  # Keep class name for backward compatibility
    """
    SVM-based model using Support Vector Regression (SVR)
    - Kernel: RBF (Radial Basis Function)
    - Training: Batch learning (no epochs, train once on the whole dataset)
    - Output: Continuous value (will be used with threshold for signals)
    """
    
    def __init__(self, config: dict = None, input_size: int = 112, 
                 hidden_size: int = 7, output_size: int = 1):
        """
        Initialize SVM model with configuration
        
        Args:
            config: Configuration dictionary with 'svm' section
            input_size: Number of input features (kept for compatibility)
            hidden_size: Ignored (NN parameter, kept for compatibility)
            output_size: Number of outputs (kept for compatibility)
        """
        from sklearn.svm import SVR
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Get SVM configuration
        if config is not None:
            svm_conf = config.get('svm', {})
        else:
            svm_conf = {}
        
        # Initialize SVR model with RBF kernel
        self.model = SVR(
            kernel=svm_conf.get('kernel', 'rbf'),
            C=svm_conf.get('C', 10.0),
            gamma=svm_conf.get('gamma', 'scale'),
            epsilon=svm_conf.get('epsilon', 0.01)
        )
        
        self.is_trained = False
        self._config = svm_conf
        
        print(f"SVM Model initialized with kernel={self.model.kernel}, C={self.model.C}, gamma={self.model.gamma}")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = None,
              learning_rate: float = None, verbose: bool = True) -> list:
        """
        Train SVM model on the provided data
        
        Args:
            X: Training data, shape (n_samples, n_features)
            y: Target values, shape (n_samples, 1) or (n_samples,)
            epochs: Ignored (SVM trains in one pass)
            learning_rate: Ignored (SVM doesn't use learning rate like NN)
            verbose: Print training progress
            
        Returns:
            List containing single MSE value (for compatibility with NN interface)
        """
        # Flatten y to 1D as required by sklearn
        y_flat = y.ravel()
        
        if verbose:
            print(f"Training SVM on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Fit the SVM model
        self.model.fit(X, y_flat)
        self.is_trained = True
        
        # Calculate training MSE for reporting
        predictions = self.model.predict(X)
        mse = np.mean((predictions - y_flat) ** 2)
        
        if verbose:
            print(f"Training completed! MSE: {mse:.6f}")
            print(f"Number of support vectors: {len(self.model.support_)}")
        
        # Return as list for compatibility with NN's loss history
        return [mse]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass - same as predict for SVM
        Kept for compatibility with NN interface
        """
        return self.predict(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained SVM model
        
        Args:
            X: Input data, shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Predictions, shape (n_samples, 1)
        """
        if not self.is_trained:
            # Return 0 (HOLD signal) if model not trained yet
            if X.ndim == 1:
                return np.zeros((1, 1))
            return np.zeros((X.shape[0], 1))
        
        # Ensure input is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from SVM
        pred = self.model.predict(X)
        
        # Reshape to (n_samples, 1) for compatibility with NN output format
        return pred.reshape(-1, 1)
    
    def save_weights(self, filepath: str, symbol: str = ""):
        """
        Save SVM model to file using joblib
        
        Args:
            filepath: Path to save model (will be saved as .joblib)
            symbol: Trading symbol for identification
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Change extension to .joblib for clarity
        if filepath.endswith('.bin'):
            filepath = filepath.replace('.bin', '.joblib')
        elif not filepath.endswith('.joblib'):
            filepath = filepath + '.joblib'
        
        # Save model data
        model_data = {
            'model': self.model,
            'symbol': symbol,
            'is_trained': self.is_trained,
            'config': self._config
        }
        
        joblib.dump(model_data, filepath)
        print(f"SVM Model saved to {filepath}")
    
    def load_weights(self, filepath: str) -> bool:
        """
        Load pre-trained SVM model from file
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # Try both .bin and .joblib extensions
        paths_to_try = [filepath]
        if filepath.endswith('.bin'):
            paths_to_try.append(filepath.replace('.bin', '.joblib'))
        elif not filepath.endswith('.joblib'):
            paths_to_try.append(filepath + '.joblib')
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    model_data = joblib.load(path)
                    self.model = model_data['model']
                    self.is_trained = model_data.get('is_trained', True)
                    self._config = model_data.get('config', {})
                    
                    symbol = model_data.get('symbol', 'unknown')
                    print(f"SVM Model loaded for {symbol} from {path}")
                    print(f"Model has {len(self.model.support_)} support vectors")
                    return True
                except Exception as e:
                    print(f"Error loading SVM from {path}: {e}")
                    continue
        
        print(f"Model file not found: {filepath}")
        return False
    
    def get_weights_info(self) -> dict:
        """
        Get information about current SVM model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {
                'model_type': 'SVR',
                'is_trained': False,
                'message': 'Model not yet trained'
            }
        
        return {
            'model_type': 'SVR',
            'is_trained': True,
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma,
            'epsilon': self.model.epsilon,
            'n_support_vectors': len(self.model.support_),
            'support_vector_indices': self.model.support_.tolist()[:10]  # First 10 for info
        }
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """
        Backward pass - for compatibility only
        SVM doesn't use backpropagation, this just returns the MSE
        """
        if not self.is_trained:
            return 1.0  # Return high loss if not trained
        
        predictions = self.predict(X)
        y_flat = y.ravel()
        mse = np.mean((predictions.ravel() - y_flat) ** 2)
        return mse
