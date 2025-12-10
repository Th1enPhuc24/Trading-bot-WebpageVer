# Neural Network Trading Bot

AI-Powered Trading System với Neural Network và Multi-Timeframe Analysis

## 📁 Cấu Trúc Project

```
Test Prj/
├── src/                        # Source code
│   ├── core/                   # Core trading components
│   │   ├── neural_network.py   # Neural network implementation
│   │   ├── data_fetcher.py     # TradingView data fetcher
│   │   ├── data_processor.py   # Data processing & normalization
│   │   ├── signal_generator.py # Trading signal generation
│   │   ├── risk_manager.py     # Risk management
│   │   ├── trading_filters.py  # Trading filters
│   │   ├── backtest_system.py  # Backtesting engine
│   │   └── training_system.py  # Training system
│   └── utils/                  # Utility functions
│       ├── dashboard.py        # Trading dashboard visualization
│       └── multi_timeframe.py  # Multi-timeframe analyzer
├── models/                     # Trained models
│   └── weights/                # Neural network weights
├── outputs/                    # Output files
│   ├── backtests/              # Backtest result images
│   └── dashboards/             # Dashboard screenshots
├── data/                       # Data storage (optional)
├── logs/                       # Log files
├── docs/                       # Documentation
├── config.json                 # Configuration file
├── main_pipeline.py            # Complete pipeline runner
├── run_backtest_pipeline.py    # Train & backtest only
├── run_live.py                 # Live trading launcher
├── run_with_dashboard.py       # Quick dashboard launcher
└── trading_bot.py              # Main trading bot
```

## 🚀 Quick Start

### 1. Installation

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Configuration

Chỉnh sửa `config.json` theo nhu cầu:

```json
{
  "trading": {
    "symbol": "GC1!",           # Trading symbol
    "timeframe": "60"           # Primary timeframe (60 = H1)
  },
  "training": {
    "training_bars": 3500,      # Bars for training
    "epochs": 500,              # Training epochs
    "learning_rate": 0.01       # Learning rate
  },
  "risk_management": {
    "risk_percentage": 0.002,   # 0.2% risk per trade
    "stop_loss_points": 100,    # Stop loss
    "take_profit_points": 150,  # Take profit
    "max_hold_hours": 4         # Max hold time
  }
}
```

### 3. Chạy Pipeline

#### Option A: Complete Pipeline (Recommended)
Chạy toàn bộ quy trình từ thu thập dữ liệu → training → testing → live trading:

```bash
python main_pipeline.py
```

**Quy trình:**
1. 📊 Thu thập dữ liệu từ TradingView (5000 bars)
2. 🔧 Xử lý và chuẩn hóa dữ liệu (train 70% / test 30%)
3. 🧠 Train neural network trên training data
4. 🧪 Test model trên test data (backtest)
5. 📈 Xuất dashboard với kết quả test
6. 🎯 Đánh giá performance
7. 🚀 Hỏi user có muốn chạy live trading không

#### Option B: Train & Backtest Only
Chỉ train và test, không chạy live:

```bash
python run_backtest_pipeline.py
```

#### Option C: Live Trading Only
Chạy live trading với model đã train:

```bash
python run_live.py
```

**⚠️ Lưu ý:** Phải train model trước (Option A hoặc B)

#### Option D: Quick Dashboard Launch
Khởi động bot nhanh với dashboard:

```bash
python run_with_dashboard.py
```

## 📊 Dashboard Features

Dashboard hiển thị real-time:
- **Price Chart**: Giá với Buy/Close signals
- **Equity Curve**: Đường vốn và drawdown
- **Training History**: Lịch sử training loss
- **Trading Statistics**: Metrics (win rate, Sharpe, drawdown, etc.)

## 🎯 Trading Logic

### Signal Generation
- **BUY**: Neural network output > threshold (0.002)
- **Position Management**: Tối đa 1 position/symbol
- **Exit Conditions**:
  - Take Profit: +150 points
  - Stop Loss: -100 points  
  - Timeout: 4 hours maximum

### Risk Management
- Risk: 0.2% equity per trade
- Position sizing: Dynamic based on account balance
- Max drawdown protection

### Multi-Timeframe Analysis
- **Daily (D)**: Trend confirmation
- **Hourly (H1)**: Primary trading timeframe
- **5-minute (M5)**: Entry timing

## 📈 Backtest Results

Kết quả backtest gần đây:
- **Total Trades**: 510
- **Win Rate**: 53.53%
- **Total Return**: +30.31%
- **Profit Factor**: 1.65
- **Max Drawdown**: 2.36%
- **Sharpe Ratio**: 2.07

Ảnh dashboard được lưu tự động vào `outputs/backtests/`

## 🔧 Customization

### Thay đổi Symbol
```json
{
  "trading": {
    "symbol": "EURUSD"  // Change to any symbol
  }
}
```

### Điều chỉnh Risk
```json
{
  "risk_management": {
    "risk_percentage": 0.001,      // 0.1% risk (more conservative)
    "stop_loss_points": 50,        // Tighter stop loss
    "take_profit_points": 100      // Lower target
  }
}
```

### Training Parameters
```json
{
  "training": {
    "training_bars": 5000,    // More training data
    "epochs": 1000,           // More epochs
    "learning_rate": 0.005    // Lower learning rate
  }
}
```

## 📝 Logs

Logs được lưu trong `logs/` directory (tự động tạo)

## 🛠️ Development

### Running Tests
```bash
python test_tvdatafeed.py    # Test data fetching
python test_dashboard.py      # Test dashboard
```

### Training Large Dataset
```bash
python train_large_dataset.py  # Train với dataset lớn
```

## ⚠️ Disclaimer

Đây là bot trading tự động. Sử dụng với rủi ro của bạn. Luôn test kỹ trên dữ liệu historical trước khi chạy live.

## 📚 Documentation

Chi tiết xem trong `docs/`:
- `IMPLEMENTATION_SUMMARY.md`: Tổng quan implementation
- `TRAINING_REPORT.md`: Báo cáo training
- `DASHBOARD_GUIDE.md`: Hướng dẫn dashboard
- `TVDATAFEED_GUIDE.md`: Hướng dẫn data fetching

## 🤝 Support

Gặp vấn đề? Check:
1. Config.json đúng format
2. Virtual environment đã activate
3. Dependencies đã install đủ
4. TradingView data accessible

---

**Version**: 2.0  
**Last Updated**: December 2, 2025  
**Author**: Trading Bot Team
