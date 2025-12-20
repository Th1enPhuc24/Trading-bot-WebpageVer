"""
Run Trading Bot with Live Dashboard
Quick launcher script for dashboard visualization
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch trading bot with dashboard enabled"""
    print("="*60)
    print(" Starting Neural Network Trading Bot with Live Dashboard")
    print("="*60)
    print()
    print("Dashboard features:")
    print("   Real-time price chart with buy/sell signals")
    print("   Neural network output visualization")
    print("   Equity curve tracking")
    print("   Training loss history")
    print("   Current position info")
    print("   Performance metrics (win rate, Sharpe, drawdown)")
    print("   Recent trade history")
    print()
    print("Press Ctrl+C to stop the bot")
    print("="*60)
    print()
    
    # Check if config exists
    config_path = Path("config.json")
    if not config_path.exists():
        print(" Error: config.json not found!")
        print("Please ensure config.json exists in the current directory.")
        return 1
    
    # Check if models directory exists
    models_path = Path("models/weights")
    if not models_path.exists():
        print("️ Warning: models/weights directory not found")
        print("Creating models directory...")
        models_path.mkdir(parents=True, exist_ok=True)
    
    # Run trading bot with dashboard flag
    try:
        result = subprocess.run(
            [sys.executable, "trading_bot.py", "--dashboard"],
            check=False
        )
        return result.returncode
    
    except KeyboardInterrupt:
        print("\n\n️ Bot stopped by user")
        return 0
    
    except Exception as e:
        print(f"\n Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
