# Quick Start Guide - Trading Bot Dashboard

## Hướng dẫn nhanh sử dụng Dashboard

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy Dashboard Demo (Dữ liệu mô phỏng)

Để xem dashboard hoạt động với dữ liệu giả lập:

```bash
python test_dashboard.py
```

Bạn sẽ thấy:
- 📈 Biểu đồ giá với tín hiệu mua/bán
- 🧠 Đầu ra Neural Network theo thời gian
- 💰 Đường cong vốn (Equity curve)
- 📉 Lịch sử loss khi training
- 📊 Các chỉ số hiệu suất (Win rate, Sharpe ratio, Drawdown)
- 💼 Thông tin lệnh hiện tại
- 📝 5 lệnh gần nhất

### 3. Chạy Bot với Dashboard (Live Trading)

#### Cách 1: Sử dụng launcher script
```bash
python run_with_dashboard.py
```

#### Cách 2: Chạy trực tiếp
```bash
python trading_bot.py --dashboard
```

#### Cách 3: Không dùng dashboard (chỉ console)
```bash
python trading_bot.py
```

### 4. Các thành phần Dashboard

#### Top Panel: Price Chart
- Đường giá COMEX:GC1! (Gold futures)
- Mũi tên xanh ▲: Tín hiệu BUY
- Mũi tên đỏ ▼: Tín hiệu SELL

#### Middle Left: Neural Network Output
- Đường màu vàng: Đầu ra của mạng neural
- Đường xanh nét đứt: Ngưỡng BUY (+0.0005)
- Đường đỏ nét đứt: Ngưỡng SELL (-0.0005)

#### Middle Center: Equity Curve
- Đường màu vàng: Vốn tài khoản theo thời gian
- Đường trắng nét đứt: Vốn ban đầu
- Xanh = lãi, Đỏ = lỗ

#### Middle Right: Training Loss
- Biểu đồ loss trong quá trình training
- Hiển thị 270 epochs gần nhất
- Loss giảm = model đang học tốt

#### Bottom Left: Performance Metrics
```
Total Trades:      10
Winning Trades:    6
Losing Trades:     4
Win Rate:          60.0%

Total P&L:         +125.50 pts
Current Balance:   $11,255.00
Net Profit:        +$1,255.00

Sharpe Ratio:      1.85
Max Drawdown:      -5.23%
```

#### Bottom Center: Current Position
```
Symbol:       GC1!
Direction:    BUY
Entry Price:  2655.00
Lot Size:     0.10

Stop Loss:    2605.00
Take Profit:  2662.00

Hold Time:    1.5 hours
```

#### Bottom Right: Recent Trades
```
✅ BUY   +7.00pts [TP]
❌ SELL  -2.30pts [SL]
✅ BUY   +5.50pts [TP]
✅ SELL  +6.20pts [TP]
❌ BUY   -3.10pts [timeout]
```

### 5. Keyboard Shortcuts

- **Ctrl+C**: Dừng bot
- **Close Window**: Thoát dashboard

### 6. Cập nhật Real-time

Dashboard tự động cập nhật:
- Mỗi khi có bar H1 mới
- Khi có tín hiệu mua/bán
- Khi mở/đóng lệnh
- Sau mỗi phiên training

### 7. Lưu Dashboard

Để lưu snapshot của dashboard:

```python
from dashboard import TradingDashboard

dashboard = TradingDashboard()
# ... update data ...
dashboard.save('dashboard_snapshot.png')
```

### 8. Tùy chỉnh Dashboard

Chỉnh sửa `dashboard.py`:

```python
# Thay đổi số lượng bars hiển thị
dashboard = TradingDashboard(max_bars=200)  # Mặc định 100

# Thay đổi màu sắc
plt.style.use('dark_background')  # hoặc 'default', 'ggplot', etc.

# Thay đổi kích thước figure
self.fig = plt.figure(figsize=(20, 12))  # Mặc định (16, 10)
```

### 9. Troubleshooting

#### Lỗi: "No module named 'matplotlib'"
```bash
pip install matplotlib
```

#### Dashboard không hiện
- Kiểm tra xem đã dùng flag `--dashboard` chưa
- Thử chạy `test_dashboard.py` để test riêng dashboard

#### Dashboard lag/chậm
- Giảm `max_bars` xuống 50-100
- Tăng `check_interval_minutes` lên 120 (2 giờ)

#### Không có dữ liệu
- Kiểm tra kết nối TradingView
- Xem log console để biết lỗi

### 10. Tips

✅ **Best Practices:**
- Chạy `test_dashboard.py` trước để kiểm tra dashboard hoạt động
- Dùng dashboard khi monitor bot ngắn hạn (vài giờ)
- Không dùng dashboard khi chạy 24/7 trên VPS (tốn tài nguyên)
- Lưu snapshot thường xuyên để review sau

⚠️ **Lưu ý:**
- Dashboard chỉ hiển thị, không ảnh hưởng logic trading
- Đóng window dashboard không dừng bot (dùng Ctrl+C)
- Dashboard tốn RAM/CPU, không khuyến khích chạy 24/7
- Dùng console mode (`python trading_bot.py`) cho VPS production

### 11. Demo Screenshots

Chạy `test_dashboard.py` để xem demo đầy đủ với dữ liệu giả lập!

---

**Hỗ trợ thêm?** Xem `README.md` hoặc `examples.py` để biết thêm chi tiết.
