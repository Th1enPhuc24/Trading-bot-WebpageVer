"""
Run RSI GAP Trading Bot with Live Dashboard
Quick launcher script for dashboard visualization
"""

import sys
import subprocess
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Launch RSI GAP trading bot with dashboard enabled"""
    print("="*60)
    print(" Starting RSI GAP Trading Bot with Live Dashboard")
    print("="*60)
    print()
    print("Dashboard features:")
    print("   Real-time price chart with buy/sell signals")
    print("   Equity curve tracking")
    print("   Performance statistics")
    print("   Trade history table")
    print()
    print("Best Model Configuration:")
    print("   • RSI Threshold: < 45")
    print("   • TP: 8 points ($0.80)")
    print("   • SL: 8 points ($0.80)")
    print("   • Expected WR: ~53%")
    print("   • Expected Return: ~150%")
    print()
    print("Dashboard URL: http://localhost:5000")
    print("Press Ctrl+C to stop the bot")
    print("="*60)
    print()
    
    # Check if config exists
    config_path = Path("config.json")
    if not config_path.exists():
        print("Error: config.json not found!")
        print("Please ensure config.json exists in the current directory.")
        return 1
    
    # Run RSI GAP trading bot with dashboard flag
    try:
        result = subprocess.run(
            [sys.executable, "svc/trading_bot.py", "--dashboard"],
            check=False
        )
        return result.returncode
    
    except KeyboardInterrupt:
        print("\n\nRSI GAP Bot stopped by user")
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
