"""
Backtesting System
Tests SVM model performance on historical data before live trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from .svr_model import SVRModel
from .data_processor import DataProcessor
from .signal_generator import SignalGenerator
from ..utils.multi_timeframe import MultiTimeframeAnalyzer
from .risk_manager import RiskManager
from .trading_filters import TradingFilters


class BacktestEngine:
    """
    Backtest engine for testing trading strategy on historical data
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.data_processor = DataProcessor(window_size=config['network']['input_size'])
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.trading_filters = TradingFilters(config)
        
        # Backtest results
        self.trades = []
        self.equity_curve = []
        self.signals_history = []
        
    def run_backtest(
        self, 
        model: SVRModel,
        prices: np.ndarray,
        symbol: str,
        initial_balance: float = 10000.0,
        verbose: bool = True
    ) -> Dict:
        """
        Run backtest on historical data
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🧪 BACKTESTING ON HISTORICAL DATA")
            print(f"{'='*70}")
            print(f"Symbol: {symbol}")
            print(f"Data points: {len(prices)}")
            print(f"Initial balance: ${initial_balance:,.2f}")
            print(f"Signal threshold: {self.config['signal']['threshold']}")
            print(f"{'='*70}\n")
        
        balance = initial_balance
        position = None
        trades = []
        equity_curve = [initial_balance]
        signals_history = []
        
        input_size = self.config['network']['input_size']
        max_hold_bars = self.config['risk_management']['max_hold_hours']
        threshold = self.config['signal']['threshold']
        
        # Start backtesting - need extra bars for indicators
        start_bar = input_size + 50
        
        for i in range(start_bar, len(prices)):
            current_price = prices[i]
            
            # Get all prices up to current point and create proper input
            price_history = prices[:i+1]
            normalized = self.data_processor.prepare_prediction_input(price_history, symbol)
            
            if normalized is None:
                continue
            
            # Check if we have open position
            if position is not None:
                pos_type = position['type']
                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                lot_size = position['lot_size']
                
                should_close = False
                close_price = current_price
                close_reason = None
                
                if pos_type == 'BUY':
                    # BUY position: SL is below entry, TP is above entry
                    if current_price <= stop_loss:
                        should_close = True
                        close_price = stop_loss
                        close_reason = 'STOP_LOSS'
                    elif current_price >= take_profit:
                        should_close = True
                        close_price = take_profit
                        close_reason = 'TAKE_PROFIT'
                else:  # SELL
                    # SELL position: SL is above entry, TP is below entry
                    if current_price >= stop_loss:
                        should_close = True
                        close_price = stop_loss
                        close_reason = 'STOP_LOSS'
                    elif current_price <= take_profit:
                        should_close = True
                        close_price = take_profit
                        close_reason = 'TAKE_PROFIT'
                
                # Check timeout
                if not should_close and (i - position['entry_bar'] >= max_hold_bars):
                    should_close = True
                    close_price = current_price
                    close_reason = 'TIMEOUT'
                
                if should_close:
                    # Calculate P&L
                    if pos_type == 'BUY':
                        pnl = (close_price - entry_price) * lot_size
                    else:  # SELL
                        pnl = (entry_price - close_price) * lot_size
                    
                    balance += pnl
                    
                    trade = {
                        'entry_time': position['entry_bar'],
                        'exit_time': i,
                        'entry_bar': position['entry_bar'],
                        'exit_bar': i,
                        'type': pos_type,
                        'entry_price': entry_price,
                        'exit_price': close_price,
                        'pnl': pnl,
                        'reason': close_reason,
                        'hold_bars': i - position['entry_bar']
                    }
                    trades.append(trade)
                    position = None
            
            # If no position, check for new signal
            if position is None:
                # Get prediction
                output = network.predict(normalized)
                output_value = float(output[0, 0])
                
                # Clip output
                output_value = np.clip(output_value, -1.0, 1.0)
                
                # Determine signal
                if output_value > threshold:
                    signal_type = 'BUY'
                elif output_value < -threshold:
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'
                
                signals_history.append({
                    'bar': i,
                    'bar_idx': i,
                    'price': current_price,
                    'signal': signal_type,
                    'output': output_value
                })
                
                # Open new position if signal
                if signal_type in ['BUY', 'SELL']:
                    # Calculate SL/TP
                    sl_tp = self.risk_manager.calculate_sl_tp_prices(
                        symbol, signal_type, current_price
                    )
                    
                    # Calculate position size
                    lot_size = self.risk_manager.calculate_lot_size(
                        symbol, balance, current_price, sl_tp['stop_loss']
                    )
                    
                    position = {
                        'type': signal_type,
                        'entry_bar': i,
                        'entry_price': current_price,
                        'stop_loss': sl_tp['stop_loss'],
                        'take_profit': sl_tp['take_profit'],
                        'lot_size': lot_size
                    }
            
            # Record equity
            current_equity = balance
            if position is not None:
                # Calculate unrealized P&L
                if position['type'] == 'BUY':
                    unrealized_pnl = (current_price - position['entry_price']) * position['lot_size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['lot_size']
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
        
        # Close any remaining position
        if position is not None:
            current_price = prices[-1]
            if position['type'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['lot_size']
            else:
                pnl = (position['entry_price'] - current_price) * position['lot_size']
            
            balance += pnl
            
            trade = {
                'entry_time': position['entry_bar'],
                'exit_time': len(prices) - 1,
                'entry_bar': position['entry_bar'],
                'exit_bar': len(prices) - 1,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'reason': 'END_OF_DATA',
                'hold_bars': len(prices) - 1 - position['entry_bar']
            }
            trades.append(trade)
        
        # Calculate statistics
        stats = self._calculate_statistics(
            trades, equity_curve, initial_balance, balance
        )
        
        if verbose:
            self._print_results(stats, trades)
        
        # Store results
        self.trades = trades
        self.equity_curve = equity_curve
        self.signals_history = signals_history
        
        return stats
    
    def _calculate_statistics(
        self, 
        trades: List[Dict],
        equity_curve: List[float],
        initial_balance: float,
        final_balance: float
    ) -> Dict:
        """Calculate backtest statistics"""
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'final_balance': final_balance,
                'return_pct': 0.0
            }
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in trades)
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        # Win rate
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # Average win/loss
        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        max_dd_pct = (max_dd / initial_balance * 100) if initial_balance > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'sharpe_ratio': sharpe,
            'final_balance': final_balance,
            'return_pct': (final_balance - initial_balance) / initial_balance * 100
        }
    
    def _print_results(self, stats: Dict, trades: List[Dict]):
        """Print backtest results"""
        print(f"\n{'='*70}")
        print(f" BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"Total Trades:      {stats['total_trades']}")
        print(f"Winning Trades:    {stats['winning_trades']} ({stats['win_rate']:.2f}%)")
        print(f"Losing Trades:     {stats['losing_trades']}")
        print(f"")
        print(f"Total P&L:         ${stats['total_pnl']:,.2f}")
        print(f"Average Win:       ${stats['avg_win']:,.2f}")
        print(f"Average Loss:      ${stats['avg_loss']:,.2f}")
        print(f"Profit Factor:     {stats['profit_factor']:.2f}")
        print(f"")
        print(f"Max Drawdown:      ${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.2f}%)")
        print(f"Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
        print(f"")
        print(f"Initial Balance:   ${stats['final_balance'] - stats['total_pnl']:,.2f}")
        print(f"Final Balance:     ${stats['final_balance']:,.2f}")
        print(f"Return:            {stats['return_pct']:,.2f}%")
        print(f"{'='*70}")
        
        # Print recent trades
        if trades:
            print(f"\n Last 10 Trades:")
            print(f"{'Type':<6} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Reason':<12}")
            print(f"{'-'*60}")
            
            for trade in trades[-10:]:
                pnl_str = f"${trade['pnl']:>10,.2f}"
                print(f"{trade['type']:<6} "
                      f"${trade['entry_price']:<9.2f} "
                      f"${trade['exit_price']:<9.2f} "
                      f"{pnl_str:<12} "
                      f"{trade['reason']:<12}")
        
        print()
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to pandas DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to pandas DataFrame"""
        return pd.DataFrame({
            'bar': range(len(self.equity_curve)),
            'equity': self.equity_curve
        })
