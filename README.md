# RSI GAP Trading Bot with Web Dashboard

A machine learning trading bot using RSI GAP Analysis with Multi-Timeframe (MTF) strategy for Gold Futures (GC1!) signal prediction. Features a real-time web dashboard for visualization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web dashboard (recommended)
python svc/run_web_dashboard.py
# Then open http://localhost:5000 and click "Start Train/Test"

# Or run terminal backtest
python svc/run_rsi_gap.py

# Run live trading
python svc/run_live.py
```

## Project Structure

```
Trading-bot-WebpageVer/
├── config.json              # Configuration (model params, trading settings)
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── svc/                    # Entry point scripts
│   ├── run_web_dashboard.py    # Web dashboard with backtest
│   ├── run_rsi_gap.py          # Terminal backtest pipeline
│   ├── run_live.py             # Live trading launcher
│   ├── run_with_dashboard.py   # Alternative launcher
│   ├── trading_bot.py          # Main trading bot logic
│   └── best_model_config.json  # Best model configuration
│
├── src/
│   ├── core/                   # Core trading components
│   │   ├── rsi_gap_model.py    # MTF RSI GAP model
│   │   ├── svc_model.py        # SVC classification model
│   │   ├── data_fetcher.py     # TradingView data fetcher
│   │   ├── data_processor.py   # Feature engineering
│   │   ├── signal_generator.py # Buy/Sell signal logic
│   │   ├── backtest_system.py  # Backtesting engine
│   │   ├── risk_manager.py     # Risk management
│   │   ├── trading_filters.py  # Trading constraints
│   │   └── optimizer.py        # Hyperparameter optimization
│   │
│   └── utils/
│       ├── live_dashboard.py       # Dashboard integration
│       ├── multi_timeframe.py      # Multi-TF analysis
│       └── web_dashboard/          # Web dashboard module
│           ├── server.py           # Flask-SocketIO server
│           ├── static/             # CSS, JavaScript
│           └── templates/          # HTML templates
│
├── models/                 # Trained model files
└── outputs/               # Generated outputs (gitignored)
    ├── rsi_gap/           # RSI GAP backtest results
    └── train-history/     # Training history
```

## Configuration

Edit `config.json`:

```json
{
  "rsi_gap": {
    "rsi_threshold_long": 45,
    "rsi_threshold_short": 55,
    "tp_points": 8,
    "sl_points": 8
  },
  "trading": {
    "symbol": "GC1!",
    "primary_timeframe": "5"
  }
}
```

## Trading Strategy

**MTF RSI GAP Analysis:**
- **Higher Timeframe (1H)**: EMA20 vs EMA50 crossover determines trend direction
- **Lower Timeframe (5min)**: RSI thresholds for entry signals
  - LONG: RSI < 45 when HTF is UPTREND
  - SHORT: RSI > 55 when HTF is DOWNTREND
- **Exit**: TP/SL = 8 points each using GAP Analysis

**Performance (Best Config):**
- Win Rate: ~53%
- Return: ~150%+
- TP: 8 points ($0.80)
- SL: 8 points ($0.80)

## Features

- MTF RSI-based trading strategy with GAP analysis
- Real-time web dashboard with TradingView-style charts
- WebSocket-based live updates
- Compound position sizing
- Support for both LONG and SHORT positions
- Technical indicators: RSI, EMA, Bollinger Bands
- Real-time data from TradingView
- Trade history export to CSV
- Equity curve tracking

## Web Dashboard

![Dashboard Preview](dashboard features a modern dark theme with:
- Real-time candlestick chart with buy/sell markers
- Equity curve visualization
- Performance statistics panel
- Trade history table
- Training log console

Access at: http://localhost:5000

## Dependencies

- numpy, pandas - Data processing
- scikit-learn - ML models
- flask, flask-socketio - Web dashboard
- tvdatafeed-enhanced - TradingView data
- optuna - Hyperparameter optimization
- tensorflow - Deep learning (optional)

## License

MIT
