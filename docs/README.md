# Bot Trading Mạng Neural Network

Bot trading tự động cho COMEX:GC1! (Vàng Futures) với kiến trúc mạng neural 112→7→1 và phân tích đa khung thời gian.

## Kiến Trúc

### Mạng Neural (112→7→1)
- **Lớp Input**: 112 neurons (giá đóng cửa H1 đã chuẩn hóa)
- **Lớp Hidden**: 7 neurons với hàm kích hoạt tanh
- **Lớp Output**: 1 neuron với hàm kích hoạt tanh
- **Chuẩn hóa**: Min-max scaling về [-1, 1]
- **Training**: 270 epochs, learning rate 0.0155, binary targets

### Tính Năng Chính
- ✅ Trọng số pre-trained cho 28 cặp với auto-loading
- ✅ Online retraining mỗi 20 H1 bars
- ✅ Phân tích đa khung thời gian (D/H1/M5)
- ✅ Quản lý rủi ro: 0.2% mỗi lệnh
- ✅ Tạo tín hiệu: Buy > +0.0005, Sell < -0.0005
- ✅ Chỉ 1 lệnh mỗi cặp tiền
- ✅ Thời gian giữ lệnh tối đa: 4 giờ (thường 1 giờ)
- ✅ Dashboard trực quan real-time

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Cấu Hình

Chỉnh sửa `config.json` để tùy chỉnh:
- Tham số mạng neural (input/hidden/output sizes)
- Cài đặt training (epochs, learning rate, tần suất retrain)
- Quản lý rủi ro (risk %, SL/TP points)
- Ràng buộc trading (giờ, volume filters)

## Sử Dụng

### Live Trading với Dashboard
```bash
# Chạy với dashboard trực quan real-time
python run_with_dashboard.py

# Hoặc chạy bot trực tiếp với flag dashboard
python trading_bot.py --dashboard
```

### Live Trading (Chỉ Console)
```bash
# Chạy không có dashboard (chỉ output console)
python trading_bot.py
```

### Test Dashboard
```bash
# Xem dashboard với dữ liệu mô phỏng
python test_dashboard.py
```

### Chế Độ Demo
```bash
# Chạy ví dụ các component
python examples.py
```

### Các Component Riêng Lẻ

#### Mạng Neural
```python
from neural_network import NeuralNetwork

network = NeuralNetwork(input_size=112, hidden_size=7, output_size=1)
output = network.predict(normalized_input)
```

#### Xử Lý Dữ Liệu
```python
from data_processor import DataProcessor

processor = DataProcessor(window_size=112)
normalized = processor.normalize_prices(prices, symbol='GC1!')
X, y = processor.create_training_dataset(prices, symbol='GC1!', training_bars=340)
```

#### Training
```python
from training_system import TrainingSystem

trainer = TrainingSystem(config)
network = trainer.initialize_network('GC1!', prices)
trainer.train_network(network, prices, 'GC1!')
```

#### Tạo Tín Hiệu
```python
from signal_generator import SignalGenerator

sig_gen = SignalGenerator(config)
signal = sig_gen.generate_signal(network, normalized_input, 'GC1!', current_price)
```

## Thông Số Kỹ Thuật

### Tham Số Chính Xác (theo yêu cầu)
- **TrainAfterBars**: 20 (retrain mỗi 20 H1 bars)
- **TrainingBars**: 340 (sử dụng 340 bars gần nhất)
- **Epochs**: 270 (training epochs mỗi phiên)
- **LearningRate**: 0.0155 (learning rate SGD cố định)
- **SignalThreshold**: 0.0005 (ngưỡng buy/sell)
- **RiskPercentage**: 0.002 (0.2% rủi ro mỗi lệnh)
- **StopLoss**: 50,000 points
- **TakeProfit**: 70 points

### Logic Đa Khung Thời Gian
1. **Daily (D)**: Xác định xu hướng buy/sell, dự đoán ngày xanh/đỏ
2. **Hourly (H1)**: Đo độ mạnh → xác định độ dài M5
   - H1 mạnh → M5 trades dài (2-4 giờ)
   - H1 yếu → M5 trades ngắn (30 phút-1 giờ)
3. **5-phút (M5)**: Thời điểm vào lệnh chính xác

## Cấu Trúc Dự Án

```
.
├── config.json              # File cấu hình
├── requirements.txt         # Python dependencies
├── trading_bot.py          # Bot chính điều phối
├── neural_network.py       # Mạng 112→7→1 implementation
├── data_processor.py       # Chuẩn hóa và tiền xử lý
├── training_system.py      # Logic training và retraining
├── data_fetcher.py         # Lấy dữ liệu TradingView
├── signal_generator.py     # Tạo tín hiệu (Buy/Sell/Hold)
├── multi_timeframe.py      # Phân tích D/H1/M5
├── risk_manager.py         # Quản lý tiền (0.2% risk, SL/TP)
├── trading_filters.py      # Giờ trading, volume filters
├── dashboard.py            # Dashboard trực quan real-time
├── run_with_dashboard.py   # Script chạy dashboard
├── test_dashboard.py       # Demo dashboard với dữ liệu giả
├── examples.py             # Ví dụ sử dụng
└── weights/                # Thư mục trọng số pre-trained
```

## Quy Trình Trading

1. **Khởi tạo**: Load trọng số pre-trained hoặc train từ đầu
2. **Giám sát**: Kiểm tra H1 bars mới mỗi giờ
3. **Retraining**: Tự động retrain mỗi 20 H1 bars
4. **Tạo Tín Hiệu**: 
   - Neural network forward pass trên 112 giá đã chuẩn hóa
   - Output > +0.0005 → tín hiệu BUY
   - Output < -0.0005 → tín hiệu SELL
5. **Xác Nhận Đa Khung**:
   - Kiểm tra xu hướng daily
   - Đánh giá độ mạnh H1
   - Xác nhận thời điểm vào M5
6. **Quản Lý Rủi Ro**:
   - Tính lot size (0.2% risk)
   - Đặt SL (50,000 points) và TP (70 points)
   - Áp dụng thời gian giữ lệnh tối đa 4 giờ
7. **Thực Hiện**: Mở lệnh nếu tất cả filters pass
8. **Giám Sát**: Kiểm tra điều kiện SL/TP/timeout

## Tính Năng An Toàn

- ✅ Một lệnh mỗi cặp (không grid/martingale)
- ✅ Áp dụng giờ trading (18:00-17:00 ET với break)
- ✅ Volume filters (ngưỡng tối thiểu)
- ✅ Timeout lệnh (tối đa 4 giờ)
- ✅ Risk cố định mỗi lệnh (0.2% balance)
- ✅ Không trailing stop, không breakeven (theo specs)

## Theo Dõi Hiệu Suất

Bot ghi log:
- Tất cả tín hiệu (buy/sell/hold)
- Mở/đóng lệnh với P&L
- Phiên training với loss metrics
- Kết quả phân tích đa khung thời gian
- Thống kê trading (win rate, thời gian giữ lệnh trung bình, v.v.)

### Tính Năng Dashboard Trực Tiếp
Khi chạy với flag `--dashboard`:
- **Biểu Đồ Giá**: Giá real-time với marker tín hiệu buy/sell
- **Neural Network Output**: Trực quan hóa predictions của model với ngưỡng
- **Đường Cong Vốn**: Theo dõi balance tài khoản theo thời gian
- **Training Loss**: Xem loss giảm trong các phiên retraining
- **Performance Metrics**: Win rate, Sharpe ratio, max drawdown, tổng P&L
- **Lệnh Hiện Tại**: Thông tin lệnh live với thời gian giữ
- **Lệnh Gần Đây**: 5 lệnh hoàn thành gần nhất với kết quả

## Lưu Ý

- **Dữ Liệu TradingView**: Yêu cầu `tvdatafeed` (API không chính thức, có thể có giới hạn)
- **Thông Số Cặp**: Cập nhật `risk_manager.py` với specs chính xác của broker
- **Thực Thi**: Hiện tại ở chế độ simulation - tích hợp broker API để trading thực
- **28 Cặp**: Kiến trúc hỗ trợ 28 cặp pre-trained, hiện tập trung vào GC1!

## Hướng Dẫn Chi Tiết

Xem `DASHBOARD_GUIDE.md` để biết hướng dẫn chi tiết về dashboard bằng tiếng Việt.

## Tuyên Bố Miễn Trừ Trách Nhiệm

Đây là hệ thống trading tự động. Sử dụng theo rủi ro của bạn. Luôn test kỹ trong chế độ simulation trước khi trading thực.
