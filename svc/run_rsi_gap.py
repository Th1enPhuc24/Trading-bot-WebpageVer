"""
RSI GAP Trading Pipeline
Complete workflow: Fetch Data -> Train Model -> Backtest -> Save Results

Best Model Configuration:
- RSI Threshold: 45
- TP: 8 points, SL: 8 points
- Win Rate: 53.03%
- Return: +157.3%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.core.data_fetcher import TradingViewDataFetcher
from src.core.rsi_gap_model import RSIGapModel


def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def run_backtest(df, model):
    """
    Run backtest with GAP analysis
    """
    balance = 10000.0
    position = None
    trades = []
    equity_curve = [balance]
    
    prices = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Get signals using HTF trend (LONG in UPTREND, SHORT in DOWNTREND)
    # Use determine_htf_trend on LTF prices to get trend array
    htf_trends = model.determine_htf_trend(prices)
    signals = model.predict_with_htf_array(prices, htf_trends)
    
    # Count signals
    long_signals = np.sum(signals == 1)
    short_signals = np.sum(signals == -1)
    print(f"Generated {long_signals} LONG signals, {short_signals} SHORT signals")
    
    tp_move = model.tp_points * 0.1
    sl_move = model.sl_points * 0.1
    max_hold = 24
    
    for i in range(1, len(prices)):
        # Check position
        if position is not None:
            bars_held = i - position['entry_bar']
            is_long = position.get('is_long', True)
            
            tp_hit, sl_hit, ambiguous = model.check_exit_gap_analysis(
                position['entry'], opens[i], highs[i], lows[i], is_long=is_long
            )
            
            if tp_hit:
                pnl = tp_move * position['size']
                balance += pnl
                if is_long:
                    exit_price = position['entry'] + tp_move
                else:
                    exit_price = position['entry'] - tp_move
                trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'win': True,
                    'exit_type': 'TP',
                    'type': 'LONG' if is_long else 'SHORT'
                })
                position = None
            elif sl_hit:
                pnl = -sl_move * position['size']
                balance += pnl
                if is_long:
                    exit_price = position['entry'] - sl_move
                else:
                    exit_price = position['entry'] + sl_move
                trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'win': False,
                    'exit_type': 'SL',
                    'type': 'LONG' if is_long else 'SHORT'
                })
                position = None
            elif bars_held >= max_hold:
                if is_long:
                    pnl = (prices[i] - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - prices[i]) * position['size']
                balance += pnl
                trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'entry_price': position['entry'],
                    'exit_price': prices[i],
                    'pnl': pnl,
                    'win': pnl > 0,
                    'exit_type': 'TIMEOUT',
                    'type': 'LONG' if is_long else 'SHORT'
                })
                position = None
        
        # Check for entry signal (LONG=1, SHORT=-1)
        if position is None:
            if signals[i] == 1:  # LONG signal
                size = balance * 0.01 / sl_move  # Risk 1%
                position = {
                    'entry': prices[i],
                    'entry_bar': i,
                    'size': size,
                    'is_long': True
                }
            elif signals[i] == -1:  # SHORT signal
                size = balance * 0.01 / sl_move  # Risk 1%
                position = {
                    'entry': prices[i],
                    'entry_bar': i,
                    'size': size,
                    'is_long': False
                }
        
        equity_curve.append(balance)
    
    return trades, equity_curve, balance


def save_trades_to_csv(trades, filepath):
    """Save trades to CSV file"""
    if not trades:
        return
    
    df = pd.DataFrame(trades)
    df.to_csv(filepath, index=False)
    print(f"Trades saved to: {filepath}")


def calculate_metrics(trades, initial_balance=10000):
    """Calculate performance metrics"""
    if not trades:
        return {}
    
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    
    total_pnl = sum(t['pnl'] for t in trades)
    
    metrics = {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'total_pnl': total_pnl,
        'return_pct': total_pnl / initial_balance * 100,
        'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
        'profit_factor': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else float('inf'),
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0
    }
    
    # Calculate max consecutive wins/losses
    current_streak = 0
    max_wins = 0
    max_losses = 0
    
    for t in trades:
        if t['win']:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_wins = max(max_wins, current_streak)
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_losses = max(max_losses, abs(current_streak))
    
    metrics['max_consecutive_wins'] = max_wins
    metrics['max_consecutive_losses'] = max_losses
    
    return metrics


def main():
    print("\n" + "="*70)
    print(" RSI GAP TRADING PIPELINE")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Add RSI GAP config to config
    config['rsi_gap'] = {
        'rsi_threshold': 45,
        'tp_points': 8,
        'sl_points': 8,
        'rsi_period': 14
    }
    
    # Step 1: Fetch Data
    print("\nStep 1: Fetching data...")
    fetcher = TradingViewDataFetcher(config)
    df = fetcher.fetch_data(timeframe='5', n_bars=5000)
    
    if df is None or len(df) < 1000:
        print("Failed to fetch data")
        return
    
    print(f"Fetched {len(df)} bars")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Initialize Model
    print("\nStep 2: Initializing RSI GAP Model...")
    model = RSIGapModel(config)
    
    # Step 3: Fit Model
    print("\nStep 3: Fitting model...")
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    fit_stats = model.fit(prices, highs, lows)
    
    # Step 4: Run Backtest
    print("\nStep 4: Running backtest...")
    trades, equity_curve, final_balance = run_backtest(df, model)
    
    # Step 5: Calculate Metrics
    print("\nStep 5: Calculating metrics...")
    metrics = calculate_metrics(trades)
    
    # Step 6: Display Results
    print("\n" + "="*70)
    print(" BACKTEST RESULTS")
    print("="*70)
    
    print(f"""
   Configuration:
      • RSI Threshold: < {model.rsi_threshold_long}
      • Take Profit: {model.tp_points} points (${model.tp_points * 0.1:.2f})
      • Stop Loss: {model.sl_points} points (${model.sl_points * 0.1:.2f})
      
   Performance:
      • Total Trades: {metrics['total_trades']}
      • Wins: {metrics['wins']}
      • Losses: {metrics['losses']}
      • Win Rate: {metrics['win_rate']:.2f}%
      • Total P&L: ${metrics['total_pnl']:.2f}
      • Return: {metrics['return_pct']:.1f}%
      • Final Balance: ${final_balance:.2f}
      
   Trade Statistics:
      • Avg Win: ${metrics['avg_win']:.2f}
      • Avg Loss: ${metrics['avg_loss']:.2f}
      • Profit Factor: {metrics['profit_factor']:.2f}
      • Max Consecutive Wins: {metrics['max_consecutive_wins']}
      • Max Consecutive Losses: {metrics['max_consecutive_losses']}
    """)
    
    # Step 7: Save Results
    print("\nStep 7: Saving results...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'rsi_gap')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save trades
    trades_path = os.path.join(output_dir, f'trades_{timestamp}.csv')
    save_trades_to_csv(trades, trades_path)
    
    # Save model
    model_path = os.path.join(output_dir, f'model_{timestamp}.joblib')
    model.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Check if meets criteria
    if metrics['win_rate'] > 50 and metrics['total_pnl'] > 0:
        print("\n" + "="*70)
        print(" SUCCESS! Both criteria met:")
        print(f"    Win Rate: {metrics['win_rate']:.2f}% > 50%")
        print(f"    P&L: ${metrics['total_pnl']:.2f} > $0")
        print("="*70)
    else:
        print("\nCriteria not fully met")
    
    print("\nPipeline completed!")
    print("="*70)


if __name__ == "__main__":
    main()
