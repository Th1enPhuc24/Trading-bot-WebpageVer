"""
Test Dashboard Save Functionality
"""

from src.utils.dashboard import TradingDashboard
from datetime import datetime, timedelta
import os

print("Testing Dashboard Save Functionality...")
print("="*60)

# Create dashboard
dashboard = TradingDashboard(max_bars=100)

# Add some test data
base_time = datetime.now()
for i in range(50):
    timestamp = base_time + timedelta(hours=i)
    price = 4000 + i * 2
    dashboard.update_price(timestamp, price)
    dashboard.update_nn_output(0.5 + i * 0.01)
    dashboard.update_equity(10000 + i * 50)

# Update dashboard
dashboard.update()

# Test saving to backtests folder
os.makedirs("outputs/backtests", exist_ok=True)
backtest_file = "outputs/backtests/test_backtest_dashboard.png"
dashboard.save_figure(backtest_file)
print(f"✓ Backtest dashboard saved: {backtest_file}")

# Test saving to live folder
os.makedirs("outputs/live", exist_ok=True)
live_file = "outputs/live/test_live_dashboard.png"
dashboard.save_figure(live_file)
print(f"✓ Live dashboard saved: {live_file}")

print("\n" + "="*60)
print("✅ Dashboard save test completed!")
print("\nCheck these files:")
print(f"  - {backtest_file}")
print(f"  - {live_file}")
