"""
Run Live Trading Bot
Starts the trading bot with trained model for live trading
"""

import sys
from pathlib import Path

# Check if model exists
models_path = Path("models/weights")
if not models_path.exists() or not list(models_path.glob("*.bin")):
    print("="*70)
    print(" ERROR: No trained model found!")
    print("="*70)
    print()
    print("You need to train a model first before running live trading.")
    print()
    print("Please run one of these commands:")
    print("  1. python main_pipeline.py   (Complete pipeline: train → test → live)")
    print("  2. python run_backtest_pipeline.py   (Train and backtest only)")
    print()
    print("="*70)
    sys.exit(1)

# Import and run trading bot
from trading_bot import TradingBot

def main():
    """Main entry point for live trading"""
    print("="*70)
    print(" STARTING LIVE TRADING BOT")
    print("="*70)
    print()
    print("️  LIVE TRADING MODE - Using real market data")
    print()
    
    response = input("Are you sure you want to start live trading? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n️ Live trading cancelled")
        return 0
    
    print()
    print("Starting bot with dashboard...")
    print()
    
    # Initialize and run bot
    bot = TradingBot(use_dashboard=True)
    
    if bot.initialize():
        bot.run(check_interval_minutes=60)
    else:
        print(" Failed to initialize trading bot")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
