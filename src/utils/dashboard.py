"""
Real-time Trading Dashboard
Displays bot performance, signals, training metrics, and positions
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import pandas as pd


class TradingDashboard:
    """
    Real-time dashboard for monitoring trading bot
    - Price chart with buy/sell/close signals
    - Equity curve & drawdown
    - Training gain/loss history
    - Trading statistics
    """
    
    def __init__(self, max_bars: int = 100):
        self.max_bars = max_bars
        
        # Data storage
        self.timestamps = deque(maxlen=max_bars)
        self.prices = deque(maxlen=max_bars)
        self.nn_outputs = deque(maxlen=max_bars)
        self.equity = deque(maxlen=max_bars)
        
        self.buy_signals = {'times': [], 'prices': []}
        self.sell_signals = {'times': [], 'prices': []}
        self.close_signals = {'times': [], 'prices': []}
        
        self.training_epochs = []
        self.training_losses = []
        
        self.positions = []
        self.trades = []
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_reward': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_balance': 10000.0,
            'starting_balance': 10000.0,
            'episode': 1,
            'step': 0
        }
        
        # Initialize plot
        self.fig = None
        self.axes = {}
        self._setup_plot()
    
    def _setup_plot(self):
        """Setup the dashboard layout"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.suptitle('Price Chart with Best Agent Trades', 
                         fontsize=18, fontweight='bold')
        
        # Create grid layout: 2 rows
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.4,
                     height_ratios=[1.2, 1])
        
        # 1. Price chart with buy/sell/close signals (top, full width)
        self.axes['price'] = self.fig.add_subplot(gs[0, :])
        self.axes['price'].set_title('Price Chart with Best Agent Trades', fontsize=14, pad=10)
        self.axes['price'].set_ylabel('Price', fontsize=11)
        self.axes['price'].set_xlabel('Step', fontsize=11)
        self.axes['price'].grid(True, alpha=0.3, linestyle='--')
        
        # 2. Equity Curve & Drawdown (bottom left)
        self.axes['equity'] = self.fig.add_subplot(gs[1, 0])
        self.axes['equity'].set_title('Equity Curve & Drawdown (Best Agent)', fontsize=12, pad=10)
        self.axes['equity'].set_ylabel('Value', fontsize=10)
        self.axes['equity'].set_xlabel('Step', fontsize=10)
        self.axes['equity'].grid(True, alpha=0.3, linestyle='--')
        
        # 3. Cumulative Trade P&L (bottom center) - was Training Gain/Loss History
        self.axes['training'] = self.fig.add_subplot(gs[1, 1])
        self.axes['training'].set_title('Cumulative Trade P&L', fontsize=12, pad=10)
        self.axes['training'].set_ylabel('P&L ($)', fontsize=10)
        self.axes['training'].set_xlabel('Trade #', fontsize=10)
        self.axes['training'].grid(True, alpha=0.3, linestyle='--')
        
        # 4. Trading Statistics (bottom right)
        self.axes['stats'] = self.fig.add_subplot(gs[1, 2])
        self.axes['stats'].set_title('Trading Statistics', fontsize=12, pad=10)
        self.axes['stats'].axis('off')
    
    def update_price(self, timestamp: datetime, price: float):
        """Update price data"""
        self.timestamps.append(timestamp)
        self.prices.append(price)
    
    def update_nn_output(self, output: float):
        """Update neural network output"""
        self.nn_outputs.append(output)
    
    def update_equity(self, balance: float):
        """Update equity curve"""
        # Only append equity if we have corresponding timestamp
        if len(self.timestamps) > 0:
            # Sync equity with timestamps - don't add more equity than timestamps
            if len(self.equity) < len(self.timestamps):
                self.equity.append(balance)
            elif len(self.equity) == len(self.timestamps):
                # Replace last equity value if same length
                if len(self.equity) > 0:
                    self.equity[-1] = balance
        self.metrics['current_balance'] = balance
    
    def add_signal(self, signal_type: str, timestamp: datetime, price: float):
        """Record a trading signal"""
        if signal_type == 'BUY':
            self.buy_signals['times'].append(timestamp)
            self.buy_signals['prices'].append(price)
        elif signal_type == 'SELL':
            self.sell_signals['times'].append(timestamp)
            self.sell_signals['prices'].append(price)
        elif signal_type == 'CLOSE':
            self.close_signals['times'].append(timestamp)
            self.close_signals['prices'].append(price)
    
    def add_training_loss(self, epoch: int, loss: float):
        """Record training loss"""
        self.training_epochs.append(epoch)
        self.training_losses.append(loss)
    
    def add_trade(self, trade: Dict):
        """Record completed trade"""
        self.trades.append(trade)
        
        # Update metrics
        self.metrics['total_trades'] += 1
        
        # Track Long/Short
        trade_type = trade.get('type', 'BUY')
        if trade_type == 'BUY':
            self.metrics['long_trades'] += 1
        else:
            self.metrics['short_trades'] += 1
        
        # Track wins/losses
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        self.metrics['total_pnl'] += pnl
        self.metrics['total_reward'] += pnl
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / 
                                       self.metrics['total_trades'] * 100)
    
    def update_position(self, position: Optional[Dict]):
        """Update current position"""
        if position:
            self.positions.append(position)
    
    def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        self.metrics.update(metrics)
        
        # Calculate additional metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate derived metrics"""
        # Sharpe ratio (simplified)
        if len(self.equity) > 1:
            returns = np.diff(list(self.equity)) / list(self.equity)[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Maximum drawdown
        if len(self.equity) > 0:
            equity_array = np.array(list(self.equity))
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            self.metrics['max_drawdown'] = np.min(drawdown) * 100
    
    def update(self):
        """Update all plots"""
        for ax in self.axes.values():
            ax.clear()
        
        self._setup_axes_properties()
        
        # Update price chart
        self._plot_price_chart()
        
        # Update equity curve & drawdown
        self._plot_equity_curve()
        
        # Update training loss
        self._plot_training_loss()
        
        # Update trading statistics
        self._plot_trading_statistics()
        
        plt.tight_layout()
    
    def _setup_axes_properties(self):
        """Re-setup axes properties after clearing"""
        self.axes['price'].set_title('Price Chart with Best Agent Trades', fontsize=14, pad=10)
        self.axes['price'].set_ylabel('Price', fontsize=11)
        self.axes['price'].set_xlabel('Step', fontsize=11)
        self.axes['price'].grid(True, alpha=0.3, linestyle='--')
        
        self.axes['equity'].set_title('Equity Curve & Drawdown (Best Agent)', fontsize=12, pad=10)
        self.axes['equity'].set_ylabel('Value', fontsize=10)
        self.axes['equity'].set_xlabel('Step', fontsize=10)
        self.axes['equity'].grid(True, alpha=0.3, linestyle='--')
        
        self.axes['training'].set_title('Cumulative Trade P&L', fontsize=12, pad=10)
        self.axes['training'].set_ylabel('P&L ($)', fontsize=10)
        self.axes['training'].set_xlabel('Trade #', fontsize=10)
        self.axes['training'].grid(True, alpha=0.3, linestyle='--')
        
        self.axes['stats'].axis('off')
    
    def _plot_price_chart(self):
        """Plot price with buy/sell/close signals"""
        times = list(self.timestamps)
        prices = list(self.prices)
        
        if len(times) == 0:
            return
        
        # Plot price line (yellow)
        self.axes['price'].plot(times, prices, color='yellow', linewidth=2, label='Price', zorder=1)
        
        # Plot buy signals (red dots - mua vào)
        if len(self.buy_signals['times']) > 0:
            buy_times = [t for t in self.buy_signals['times'] if times[0] <= t <= times[-1]]
            buy_prices = [p for t, p in zip(self.buy_signals['times'], self.buy_signals['prices']) 
                         if times[0] <= t <= times[-1]]
            
            if len(buy_times) > 0:
                self.axes['price'].scatter(buy_times, buy_prices, color='red', 
                                         marker='o', s=50, label='Buy', zorder=5, edgecolors='darkred', linewidths=1)
        
        # Plot sell signals (không hiển thị - theo ảnh không có)
        
        # Plot close signals (yellow dots - đóng lệnh)
        if len(self.close_signals['times']) > 0:
            close_times = [t for t in self.close_signals['times'] if times[0] <= t <= times[-1]]
            close_prices = [p for t, p in zip(self.close_signals['times'], self.close_signals['prices']) 
                           if times[0] <= t <= times[-1]]
            
            if len(close_times) > 0:
                self.axes['price'].scatter(close_times, close_prices, color='yellow', 
                                         marker='o', s=50, label='Close', zorder=6, edgecolors='gold', linewidths=1)
        
        self.axes['price'].legend(loc='upper left', fontsize=10, framealpha=0.8)
        self.axes['price'].tick_params(labelsize=9)
    
    def _plot_equity_curve(self):
        """Plot equity curve and drawdown"""
        if len(self.equity) == 0:
            return
        
        # Ensure times and equity have same length
        equity_values = list(self.equity)
        times = list(self.timestamps)[-len(equity_values):]
        
        # If mismatch, adjust
        if len(times) != len(equity_values):
            min_len = min(len(times), len(equity_values))
            times = times[-min_len:]
            equity_values = equity_values[-min_len:]
        
        if len(times) == 0 or len(equity_values) == 0:
            return
        
        # Create twin axis for drawdown
        ax2 = self.axes['equity'].twinx()
        
        # Plot equity curve (green line)
        self.axes['equity'].plot(times, equity_values, color='#00ff00', linewidth=2.5, label='Equity')
        self.axes['equity'].axhline(y=self.metrics['starting_balance'], 
                                   color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Calculate drawdown
        if len(equity_values) > 0:
            running_max = [max(equity_values[:i+1]) for i in range(len(equity_values))]
            drawdown = [(equity_values[i] - running_max[i]) / running_max[i] * 100 
                       for i in range(len(equity_values))]
            
            # Plot drawdown (red area)
            ax2.fill_between(times, drawdown, 0, color='red', alpha=0.3, label='Drawdown %')
            ax2.set_ylabel('Drawdown (%)', fontsize=10, color='red')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=9)
            ax2.set_ylim(min(drawdown) * 1.2 if min(drawdown) < 0 else -5, 5)
        
        self.axes['equity'].set_ylabel('Equity Value', fontsize=10, color='#00ff00')
        self.axes['equity'].tick_params(axis='y', labelcolor='#00ff00', labelsize=9)
        self.axes['equity'].tick_params(axis='x', labelsize=9, rotation=30)
        
        # Combine legends
        lines1, labels1 = self.axes['equity'].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.axes['equity'].legend(lines1 + lines2, labels1 + labels2, 
                                   loc='upper left', fontsize=9, framealpha=0.8)
    
    def _plot_training_loss(self):
        """Plot training loss history (gain/loss)"""
        if len(self.training_losses) == 0:
            self.axes['training'].text(0.5, 0.5, 'No trades yet', 
                                      ha='center', va='center', fontsize=10)
            return
        
        epochs = list(range(1, len(self.training_losses) + 1))
        losses = self.training_losses
        
        # Plot loss curve
        self.axes['training'].plot(epochs, losses, color='cyan', linewidth=2, marker='o', markersize=3)
        
        # Fill areas based on loss value (gain=green, loss=red)
        self.axes['training'].fill_between(epochs, losses, 0,
                                          where=[l >= 0 for l in losses],
                                          color='green', alpha=0.3, interpolate=True, label='Gain')
        self.axes['training'].fill_between(epochs, losses, 0,
                                          where=[l < 0 for l in losses],
                                          color='red', alpha=0.3, interpolate=True, label='Loss')
        
        self.axes['training'].axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        self.axes['training'].legend(loc='upper right', fontsize=9, framealpha=0.8)
        self.axes['training'].tick_params(labelsize=9)
    
    def _plot_trading_statistics(self):
        """Plot trading statistics panel"""
        stats_text = []
        
        # Episode and Step info
        stats_text.append(f"Episode: {self.metrics.get('episode', 1)}")
        stats_text.append(f"Step: {self.metrics.get('step', len(self.prices))}")
        stats_text.append("")
        
        # Trading Stats section
        stats_text.append("Trading Stats:")
        stats_text.append(f"  Total: {self.metrics['total_trades']}")
        stats_text.append(f"  Long (Buy) Position: {self.metrics.get('long_trades', 0)}")
        stats_text.append(f"  Short (Sell) Position: {self.metrics.get('short_trades', 0)}")
        stats_text.append(f"  Win Rate: {self.metrics['win_rate']:.2f}%")
        stats_text.append("")
        
        # Performance section
        stats_text.append("Performance:")
        reward = self.metrics.get('total_reward', self.metrics['total_pnl'])
        stats_text.append(f"  Reward: ${reward:.2f}")
        stats_text.append("")
        
        # Additional metrics
        stats_text.append("Additional Metrics:")
        stats_text.append(f"  Current Balance: ${self.metrics['current_balance']:.2f}")
        stats_text.append(f"  Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        stats_text.append(f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        
        # Display text
        text_str = '\n'.join(stats_text)
        self.axes['stats'].text(0.05, 0.95, text_str,
                               transform=self.axes['stats'].transAxes,
                               fontsize=11,
                               verticalalignment='top',
                               fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='gray'))
    
    def save_figure(self, filepath: str):
        """Save dashboard figure to file"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Dashboard saved to: {filepath}")
    
    def show(self):
        """Display the dashboard"""
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
