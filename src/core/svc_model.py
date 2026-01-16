"""
Support Vector Classification (SVC) Model for Trading Signal Classification
Predicts BUY (1), SELL (-1), or HOLD (0) signals directly
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import joblib
import os
from typing import Optional, Tuple


class SVCModel:
    """
    Support Vector Classification model for trading signal prediction
    Classifies market direction as BUY (1), SELL (-1), or HOLD (0)
    """
    
    def __init__(self, config: dict, input_size: int = 112, **kwargs):
        self.config = config
        self.input_size = input_size
        
        # Get SVC parameters from config
        svc_config = config.get('svc', config.get('svm', {}))
        
        self.kernel = svc_config.get('kernel', 'rbf')
        self.C = svc_config.get('C', 1.0)
        self.gamma = svc_config.get('gamma', 'scale')
        self.auto_tune = svc_config.get('auto_tune', True)
        
        # Initialize model
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # Enable probability estimates
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        print(f"SVC Classification Model initialized: kernel={self.kernel}, Auto-Tune={'ON' if self.auto_tune else 'OFF'}")
    
    def create_classification_labels(
        self, 
        prices: np.ndarray, 
        highs: np.ndarray,
        lows: np.ndarray,
        tp_points: float,
        sl_points: float,
        lookahead: int = 24
    ) -> np.ndarray:
        """
        Create classification labels based on which hits first: TP or SL
        
        Args:
            prices: Close prices
            highs: High prices
            lows: Low prices
            tp_points: Take profit in points
            sl_points: Stop loss in points
            lookahead: Number of bars to look ahead
            
        Returns:
            labels: 1 (BUY profitable), -1 (SELL profitable), 0 (HOLD/undetermined)
        """
        n = len(prices)
        labels = np.zeros(n)
        
        # Convert points to price movement (1 point = $0.1 for Gold)
        tp_move = tp_points * 0.1
        sl_move = sl_points * 0.1
        
        for i in range(n - lookahead):
            entry_price = prices[i]
            
            # Check BUY scenario: Entry at close, check if TP or SL hit first
            buy_tp = entry_price + tp_move
            buy_sl = entry_price - sl_move
            
            # Check SELL scenario
            sell_tp = entry_price - tp_move
            sell_sl = entry_price + sl_move
            
            buy_hit_tp = False
            buy_hit_sl = False
            sell_hit_tp = False
            sell_hit_sl = False
            
            buy_tp_bar = lookahead + 1
            buy_sl_bar = lookahead + 1
            sell_tp_bar = lookahead + 1
            sell_sl_bar = lookahead + 1
            
            # Look ahead to find which level is hit first
            for j in range(1, min(lookahead + 1, n - i)):
                future_high = highs[i + j]
                future_low = lows[i + j]
                
                # BUY: Check if high reaches TP or low reaches SL
                if not buy_hit_tp and future_high >= buy_tp:
                    buy_hit_tp = True
                    buy_tp_bar = j
                if not buy_hit_sl and future_low <= buy_sl:
                    buy_hit_sl = True
                    buy_sl_bar = j
                
                # SELL: Check if low reaches TP or high reaches SL
                if not sell_hit_tp and future_low <= sell_tp:
                    sell_hit_tp = True
                    sell_tp_bar = j
                if not sell_hit_sl and future_high >= sell_sl:
                    sell_hit_sl = True
                    sell_sl_bar = j
                
                # Early exit if both scenarios resolved
                if (buy_hit_tp or buy_hit_sl) and (sell_hit_tp or sell_hit_sl):
                    break
            
            # Determine label based on which scenario is more profitable
            buy_profitable = buy_hit_tp and (not buy_hit_sl or buy_tp_bar < buy_sl_bar)
            sell_profitable = sell_hit_tp and (not sell_hit_sl or sell_tp_bar < sell_sl_bar)
            
            if buy_profitable and not sell_profitable:
                labels[i] = 1  # BUY signal
            elif sell_profitable and not buy_profitable:
                labels[i] = -1  # SELL signal
            elif buy_profitable and sell_profitable:
                # Both profitable - choose the one that hits TP first
                if buy_tp_bar < sell_tp_bar:
                    labels[i] = 1
                elif sell_tp_bar < buy_tp_bar:
                    labels[i] = -1
                else:
                    labels[i] = 0  # HOLD - equal
            else:
                labels[i] = 0  # HOLD - neither profitable
        
        return labels
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        """
        Train the SVC model
        
        Args:
            X: Feature matrix
            y: Classification labels (1, 0, -1)
            verbose: Print training info
            
        Returns:
            Training statistics
        """
        if verbose:
            print(f"Training SVC on {len(X)} samples with {X.shape[1]} features...")
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Class distribution: {dict(zip(['SELL', 'HOLD', 'BUY'], counts))}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.auto_tune:
            if verbose:
                print("\nAuto-tuning with RandomizedSearchCV...")
            
            param_distributions = {
                'C': uniform(0.1, 10),
                'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(5)),
                'kernel': ['rbf', 'poly'],
                'degree': randint(2, 5)  # For poly kernel
            }
            
            tscv = TimeSeriesSplit(n_splits=5)
            
            search = RandomizedSearchCV(
                SVC(probability=True, class_weight='balanced', random_state=42),
                param_distributions,
                n_iter=15,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1 if verbose else 0,
                random_state=42
            )
            
            search.fit(X_scaled, y)
            self.model = search.best_estimator_
            
            if verbose:
                print(f"\nBest parameters: {search.best_params_}")
                print(f"   Best CV accuracy: {search.best_score_:.2%}")
        else:
            self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_scaled)
        train_accuracy = np.mean(train_pred == y)
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"   Training accuracy: {train_accuracy:.2%}")
            print(f"   Support vectors: {len(self.model.support_vectors_)}")
        
        return {
            'train_accuracy': train_accuracy,
            'n_support_vectors': len(self.model.support_vectors_),
            'classes': self.model.classes_.tolist()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict trading signal
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability for each class [SELL, HOLD, BUY]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, filepath: str):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'input_size': self.input_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and scaler"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.config = data.get('config', self.config)
        self.input_size = data.get('input_size', self.input_size)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
