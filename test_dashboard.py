"""
Test Dashboard Standalone
Demonstrates dashboard functionality with simulated data
"""

import numpy as np
from datetime import datetime, timedelta
from dashboard import TradingDashboard


def generate_realistic_price_data(n_bars: int = 100, base_price: float = 2650.0):
    """Generate realistic gold price movements"""
    prices = [base_price]
    
    for _ in range(n_bars - 1):
        # Random walk with slight upward bias
        change = np.random.randn() * 2.0 + 0.05
        new_price = prices[-1] + change
        prices.append(new_price)
    
    return np.array(prices)


def simulate_trading_session():
    """Simulate a trading session with realistic data"""
    print("="*60)
    print("🎬 Dashboard Demo - Simulated Trading Session")
    print("="*60)
    print()
    print("Generating simulated data...")
    print("  - 100 H1 bars of price data")
    print("  - Neural network predictions")
    print("  - Trading signals and positions")
    print("  - Training sessions")
    print("  - Performance metrics")
    print()
    
    # Create dashboard
    dashboard = TradingDashboard(max_bars=100)
    
    # Generate base data
    n_bars = 100
    base_time = datetime.now() - timedelta(hours=n_bars)
    prices = generate_realistic_price_data(n_bars, base_price=2650.0)
    
    # Simulate neural network outputs
    nn_outputs = np.random.randn(n_bars) * 0.002
    
    # Starting balance
    balance = 10000.0
    position_open = False
    entry_price = 0
    entry_direction = None
    trades_count = 0
    
    print("Simulating trading activity...")
    
    for i in range(n_bars):
        timestamp = base_time + timedelta(hours=i)
        price = prices[i]
        nn_output = nn_outputs[i]
        
        # Update dashboard data
        dashboard.update_price(timestamp, price)
        dashboard.update_nn_output(nn_output)
        
        # Generate signals based on NN output
        if not position_open:
            if nn_output > 0.0005:
                # Buy signal
                dashboard.add_signal('BUY', timestamp, price)
                position_open = True
                entry_price = price
                entry_direction = 'BUY'
                
                dashboard.update_position({
                    'symbol': 'GC1!',
                    'direction': 'BUY',
                    'entry_price': entry_price,
                    'lot_size': 0.1,
                    'stop_loss': entry_price - 50,
                    'take_profit': entry_price + 7,
                    'entry_time': timestamp
                })
                
            elif nn_output < -0.0005:
                # Sell signal
                dashboard.add_signal('SELL', timestamp, price)
                position_open = True
                entry_price = price
                entry_direction = 'SELL'
                
                dashboard.update_position({
                    'symbol': 'GC1!',
                    'direction': 'SELL',
                    'entry_price': entry_price,
                    'lot_size': 0.1,
                    'stop_loss': entry_price + 50,
                    'take_profit': entry_price - 7,
                    'entry_time': timestamp
                })
        
        else:
            # Check for exit conditions
            should_close = False
            close_reason = ''
            
            if entry_direction == 'BUY':
                # Check TP/SL
                if price >= entry_price + 7:
                    should_close = True
                    close_reason = 'TP'
                elif price <= entry_price - 50:
                    should_close = True
                    close_reason = 'SL'
                elif i - trades_count * 10 > 8:  # Simulate timeout
                    should_close = True
                    close_reason = 'timeout'
            
            else:  # SELL
                if price <= entry_price - 7:
                    should_close = True
                    close_reason = 'TP'
                elif price >= entry_price + 50:
                    should_close = True
                    close_reason = 'SL'
                elif i - trades_count * 10 > 8:
                    should_close = True
                    close_reason = 'timeout'
            
            if should_close:
                # Calculate P&L
                if entry_direction == 'BUY':
                    pnl_points = price - entry_price
                else:
                    pnl_points = entry_price - price
                
                # Update balance (simplified)
                balance += pnl_points * 10
                
                # Close position
                dashboard.add_trade({
                    'direction': entry_direction,
                    'pnl_points': pnl_points,
                    'reason': close_reason
                })
                
                dashboard.update_position(None)
                position_open = False
                trades_count += 1
        
        # Update equity
        dashboard.update_equity(balance)
    
    # Add training data (simulate 270 epochs)
    print("Adding training loss history...")
    for epoch in range(270):
        # Exponential decay with noise
        loss = 0.5 * np.exp(-epoch / 50) + np.random.rand() * 0.05
        dashboard.add_training_loss(epoch, loss)
    
    # Final stats
    print()
    print(f"✓ Simulation complete!")
    print(f"  - Generated {n_bars} price bars")
    print(f"  - Executed {trades_count} trades")
    print(f"  - Final balance: ${balance:.2f}")
    print(f"  - P&L: ${balance - 10000:+.2f}")
    print()
    print("Displaying dashboard...")
    print("Close the window to exit.")
    print("="*60)
    
    # Display dashboard
    dashboard.update_plot()
    dashboard.show()


def main():
    """Main entry point"""
    try:
        simulate_trading_session()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
