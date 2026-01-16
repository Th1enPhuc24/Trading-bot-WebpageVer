"""
Trading Optimizer: Multi-Objective Hyperparameter Optimization
Optimizes models for both Win Rate AND Return using Optuna
"""

import numpy as np
import optuna
from typing import Dict, Tuple, Optional, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TradingOptimizer:
    """
    Multi-objective hyperparameter optimizer for trading models.
    
    Optimizes for:
    - Win Rate (percentage of winning trades)
    - Total Return (profit percentage)
    - Profit Factor (gross profit / gross loss)
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Optimization settings
        opt_config = config.get('optimization', {})
        self.n_trials = opt_config.get('n_trials', 30)
        self.min_trades = opt_config.get('min_trades', 5)
        
        # Weights for combined objective
        self.win_rate_weight = opt_config.get('win_rate_weight', 0.4)
        self.return_weight = opt_config.get('return_weight', 0.4)
        self.profit_factor_weight = opt_config.get('profit_factor_weight', 0.2)
        
        # Signal threshold for backtesting
        self.signal_threshold = config.get('signal', {}).get('threshold', 0.1)
        
        # Best results storage
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
    
    def quick_backtest(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                       prices: np.ndarray) -> Dict:
        """
        Run quick backtest to evaluate model performance.
        
        Args:
            model: Trained model with predict() method
            X_test: Test features
            y_test: Test targets (not used, but kept for interface)
            prices: Price series for P&L calculation
            
        Returns:
            Dict with backtest metrics
        """
        predictions = model.predict(X_test)
        
        # Generate signals based on threshold
        signals = []
        for pred in predictions:
            val = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            if val > self.signal_threshold:
                signals.append(1)  # BUY
            elif val < -self.signal_threshold:
                signals.append(-1)  # SELL
            else:
                signals.append(0)  # HOLD
        
        # Simulate trades
        trades = []
        position = None
        entry_price = None
        
        for i, signal in enumerate(signals):
            current_price = prices[i] if i < len(prices) else prices[-1]
            
            # Open position
            if position is None and signal != 0:
                position = signal
                entry_price = current_price
            
            # Close position (opposite signal or end of data)
            elif position is not None:
                # Close on opposite signal or timeout (10 bars)
                bars_held = len([t for t in trades]) if trades else 0
                should_close = (signal == -position) or (i == len(signals) - 1)
                
                if should_close:
                    exit_price = current_price
                    if position == 1:  # Long
                        pnl = exit_price - entry_price
                    else:  # Short
                        pnl = entry_price - exit_price
                    
                    trades.append({
                        'direction': 'BUY' if position == 1 else 'SELL',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl
                    })
                    position = None
                    entry_price = None
        
        # Calculate metrics
        if len(trades) < self.min_trades:
            return {
                'win_rate': 0,
                'total_return': 0,
                'profit_factor': 0,
                'n_trades': len(trades),
                'valid': False
            }
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        total_pnl = sum(t['pnl'] for t in trades)
        total_return = total_pnl / prices[0] * 100 if prices[0] > 0 else 0
        
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'n_trades': len(trades),
            'valid': True
        }
    
    def combined_score(self, metrics: Dict) -> float:
        """
        Calculate combined score from multiple objectives.
        
        Higher score = better model
        """
        if not metrics.get('valid', False):
            return -1000  # Penalty for invalid results
        
        # Normalize metrics
        win_rate_score = metrics['win_rate'] / 100  # 0-1
        return_score = min(max(metrics['total_return'] / 50, -1), 1)  # Capped at 50%
        pf_score = min(metrics['profit_factor'] / 3, 1)  # Capped at 3
        
        # Weighted combination
        score = (
            self.win_rate_weight * win_rate_score +
            self.return_weight * return_score +
            self.profit_factor_weight * pf_score
        )
        
        return score
    
    def optimize_svr(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     prices_val: np.ndarray, verbose: bool = True) -> Dict:
        """
        Optimize SVR hyperparameters for trading performance.
        
        Returns:
            Dict with best parameters and performance metrics
        """
        from sklearn.svm import SVR
        
        def objective(trial):
            # Sample hyperparameters
            C = trial.suggest_float('C', 1, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            epsilon = trial.suggest_float('epsilon', 0.01, 0.3)
            
            # Train model
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train, y_train.ravel())
            
            # Create wrapper for prediction
            class ModelWrapper:
                def __init__(self, m):
                    self.model = m
                def predict(self, X):
                    return self.model.predict(X).reshape(-1, 1)
            
            wrapper = ModelWrapper(model)
            
            # Evaluate
            metrics = self.quick_backtest(wrapper, X_val, y_val, prices_val)
            score = self.combined_score(metrics)
            
            # Store trial info
            trial.set_user_attr('win_rate', metrics['win_rate'])
            trial.set_user_attr('total_return', metrics['total_return'])
            trial.set_user_attr('profit_factor', metrics['profit_factor'])
            trial.set_user_attr('n_trades', metrics['n_trades'])
            
            return score
        
        # Create study
        if verbose:
            print(f"\n Optimizing SVR for Win Rate + Return...")
            print(f"   Trials: {self.n_trials}")
            print(f"   Weights: WR={self.win_rate_weight}, Return={self.return_weight}, PF={self.profit_factor_weight}")
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=verbose)
        
        # Get best result
        best_trial = study.best_trial
        self.best_params = best_trial.params
        self.best_score = best_trial.value
        
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'win_rate': best_trial.user_attrs.get('win_rate', 0),
            'total_return': best_trial.user_attrs.get('total_return', 0),
            'profit_factor': best_trial.user_attrs.get('profit_factor', 0),
            'n_trades': best_trial.user_attrs.get('n_trades', 0)
        }
        
        if verbose:
            print(f"\nBest SVR parameters found:")
            print(f"   C: {self.best_params['C']:.4f}")
            print(f"   gamma: {self.best_params['gamma']}")
            print(f"   epsilon: {self.best_params['epsilon']:.4f}")
            print(f"\n Expected Performance:")
            print(f"   Win Rate: {result['win_rate']:.2f}%")
            print(f"   Return: {result['total_return']:.2f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Trades: {result['n_trades']}")
        
        return result
    

