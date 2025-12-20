# SVM Trading Bot

A machine learning trading bot using Support Vector Machine (SVR) for Gold Futures (GC1!) signal prediction.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python run_backtest_pipeline.py

# Run live trading with dashboard
python run_with_dashboard.py
```

## Project Structure

```
Trading-bot-V2-SVM/
├── config.json              # Configuration (SVM params, trading settings)
├── requirements.txt         # Python dependencies
├── trading_bot.py          # Main trading bot
├── run_backtest_pipeline.py # Backtest with train/test split
├── run_with_dashboard.py   # Live trading launcher
├── run_live.py             # Alternative live launcher
├── src/
│   ├── core/
│   │   ├── neural_network.py   # SVM model wrapper
│   │   ├── data_fetcher.py     # TradingView data fetcher
│   │   ├── data_processor.py   # Feature engineering
│   │   ├── signal_generator.py # Buy/Sell signal logic
│   │   ├── training_system.py  # Model training
│   │   ├── backtest_system.py  # Backtesting engine
│   │   └── risk_manager.py     # Risk management
│   └── utils/
│       ├── dashboard.py        # Live visualization
│       └── multi_timeframe.py  # Multi-TF analysis
├── models/
│   └── weights/               # Trained model files
├── outputs/                   # Generated outputs (gitignored)
│   ├── backtests/
│   ├── live/
│   └── data/
└── docs/                      # Documentation
```

## Configuration

Edit `config.json`:

```json
{
  "svm": {
    "kernel": "rbf",
    "C": 10.0,
    "gamma": "auto"
  },
  "signal": {
    "threshold": 0.7
  },
  "risk_management": {
    "stop_loss_points": 5,
    "take_profit_points": 10
  }
}
```

## Features

- SVM (SVR) with RBF kernel for price prediction
- Technical indicators: RSI, MACD, Bollinger Bands, Momentum, Volatility
- Real-time data from TradingView
- Live dashboard visualization
- Walk-forward backtesting

## License

MIT
