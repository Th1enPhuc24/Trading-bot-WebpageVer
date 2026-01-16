"""
MTF RSI GAP Live Trading Bot Launcher
Starts the RSI GAP trading bot for paper/live trading with dashboard
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from svc.trading_bot import RSIGapTradingBot


def main():
    """Main entry point for MTF RSI GAP live trading"""
    print("="*70)
    print(" MTF RSI GAP LIVE TRADING BOT")
    print("="*70)
    print()
    print("PAPER TRADING MODE - Using real market data with MTF RSI GAP model")
    print()
    print("Strategy:")
    print("  • LONG: RSI < 45 when 1H trend is UPTREND")
    print("  • SHORT: RSI > 55 when 1H trend is DOWNTREND")
    print("  • TP/SL: 8 points ($0.80) each")
    print()
    
    response = input("Are you sure you want to start live trading? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nLive trading cancelled")
        return 0
    
    print()
    print("Starting MTF RSI GAP bot with dashboard...")
    print()
    
    # Initialize and run bot
    bot = RSIGapTradingBot(use_web_dashboard=True)
    
    # Check for check interval argument
    check_interval = 1  # Default: check every 1 minute
    for arg in sys.argv:
        if arg.startswith('--interval='):
            try:
                check_interval = int(arg.split('=')[1])
            except:
                pass
    
    # Run live trading
    try:
        bot.run_live_trading(check_interval_minutes=check_interval)
    except KeyboardInterrupt:
        print("\n\nLive trading stopped by user")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
