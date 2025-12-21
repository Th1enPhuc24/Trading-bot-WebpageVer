"""
SVR Model Core: Support Vector Regression for Trading Signal Prediction
Uses sklearn's SVR for deterministic, batch-learning approach
"""

import numpy as np
import os
import joblib
from typing import Optional


class SVRModel:
    """
    Support Vector Regression (SVR) Model for Trading Signal Prediction
    
    Features:
    - Kernel: RBF (Radial Basis Function)
    - Training: Batch learning (train once on the whole dataset)
    - Output: Continuous value (used with threshold for BUY/SELL/HOLD signals)
    """
    
    def __init__(self, config: dict = None, input_size: int = 112, output_size: int = 1):
        """
        Initialize SVR model with configuration
        
        Args:
            config: Configuration dictionary with 'svm' section
            input_size: Number of input features
            output_size: Number of outputs
        """
        from sklearn.svm import SVR
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Get SVR configuration
        if config is not None:
            svr_conf = config.get('svm', {})
        else:
            svr_conf = {}
        
        # Initialize SVR model with RBF kernel
        self.model = SVR(
            kernel=svr_conf.get('kernel', 'rbf'),
            C=svr_conf.get('C', 10.0),
            gamma=svr_conf.get('gamma', 'scale'),
            epsilon=svr_conf.get('epsilon', 0.01)
        )
        
        self.is_trained = False
        self._config = svr_conf
        
        print(f"SVR Model initialized with kernel={self.model.kernel}, C={self.model.C}, gamma={self.model.gamma}")
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> list:
        """
        Train SVR model on the provided data
        
        Args:
            X: Training data, shape (n_samples, n_features)
            y: Target values, shape (n_samples, 1) or (n_samples,)
            verbose: Print training progress
            
        Returns:
            List containing single MSE value
        """
        # Flatten y to 1D as required by sklearn
        y_flat = y.ravel()
        
        if verbose:
            print(f"Training SVR on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Fit the SVR model
        self.model.fit(X, y_flat)
        self.is_trained = True
        
        # Calculate training MSE for reporting
        predictions = self.model.predict(X)
        mse = np.mean((predictions - y_flat) ** 2)
        
        if verbose:
            print(f"Training completed! MSE: {mse:.6f}")
            print(f"Number of support vectors: {len(self.model.support_)}")
        
        return [mse]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass - alias for predict method
        """
        return self.predict(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained SVR model
        
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
        
        # Get predictions from SVR
        pred = self.model.predict(X)
        
        # Reshape to (n_samples, 1)
        return pred.reshape(-1, 1)
    
    def save_model(self, filepath: str, symbol: str = ""):
        """
        Save SVR model to file using joblib
        
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
        print(f"SVR Model saved to {filepath}")
    
    # Alias for backward compatibility
    def save_weights(self, filepath: str, symbol: str = ""):
        """Alias for save_model (backward compatibility)"""
        self.save_model(filepath, symbol)
    
    def load_model(self, filepath: str) -> bool:
        """
        Load pre-trained SVR model from file
        
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
                    print(f"SVR Model loaded for {symbol} from {path}")
                    print(f"Model has {len(self.model.support_)} support vectors")
                    return True
                except Exception as e:
                    print(f"Error loading SVR from {path}: {e}")
                    continue
        
        print(f"Model file not found: {filepath}")
        return False
    
    # Alias for backward compatibility
    def load_weights(self, filepath: str) -> bool:
        """Alias for load_model (backward compatibility)"""
        return self.load_model(filepath)
    
    def get_model_info(self) -> dict:
        """
        Get information about current SVR model
        
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
    
    # Alias for backward compatibility
    def get_weights_info(self) -> dict:
        """Alias for get_model_info (backward compatibility)"""
        return self.get_model_info()
