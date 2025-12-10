# Hướng Dẫn Sử Dụng tvdatafeed-enhanced

## Cài Đặt

Bot đã được cấu hình để sử dụng **tvdatafeed-enhanced v2.2.0+** - phiên bản cải tiến của tvdatafeed với các tính năng:

- ✅ Hỗ trợ Python 3.13
- ✅ Lấy dữ liệu anonymous (không cần đăng nhập)
- ✅ Hỗ trợ tối đa 5000 bars mỗi request
- ✅ Xử lý lỗi tốt hơn
- ✅ Cache thông minh

### Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install tvdatafeed-enhanced>=2.2.0
pip install websocket-client>=1.6.0
pip install websockets>=14.0
```

## Sử Dụng

### 1. Anonymous Access (Mặc định)

Bot sử dụng anonymous access - không cần đăng nhập TradingView:

```python
from data_fetcher import TradingViewDataFetcher

fetcher = TradingViewDataFetcher(config)
# Tự động sử dụng anonymous access
```

**Ưu điểm:**
- Không cần tài khoản TradingView
- Không cần password
- Đơn giản, nhanh chóng

**Hạn chế:**
- Có thể bị rate limit
- Dữ liệu có thể bị trễ 15-20 phút

### 2. Authenticated Access (Tùy chọn)

Nếu bạn có tài khoản TradingView, có thể đăng nhập để được:
- Dữ liệu real-time
- Không bị rate limit
- Truy cập nhiều symbols hơn

**Cách 1: Sửa trực tiếp trong data_fetcher.py**

Mở file `data_fetcher.py` và sửa dòng 27:

```python
# Từ:
self.tv = TvDatafeed()

# Thành:
self.tv = TvDatafeed(username='your_username', password='your_password')
```

**Cách 2: Thêm vào config.json**

Thêm section mới vào `config.json`:

```json
{
  "tradingview": {
    "username": "your_username",
    "password": "your_password"
  }
}
```

Sau đó sửa `data_fetcher.py`:

```python
def __init__(self, config: dict):
    self.config = config
    
    # Check for credentials
    tv_config = config.get('tradingview', {})
    username = tv_config.get('username')
    password = tv_config.get('password')
    
    if username and password:
        self.tv = TvDatafeed(username=username, password=password)
        print(f"✓ Logged in as {username}")
    else:
        self.tv = TvDatafeed()
        print(f"✓ Using anonymous access")
```

## Test Kết Nối

Chạy script test để kiểm tra kết nối:

```bash
python test_tvdatafeed.py
```

Output mẫu:

```
============================================================
🧪 TVDATAFEED-ENHANCED INTEGRATION TESTS
============================================================

✓ TradingView connection initialized (tvdatafeed-enhanced v2.2.1)
  Using anonymous access - data may be limited

✓ H1 Data fetched successfully!
  Shape: (50, 6)
  Latest close: $4285.40

✅ ALL TESTS COMPLETED
```

## Các Symbols Hỗ Trợ

### Futures (COMEX)
- `GC1!` - Gold Futures (hiện tại)
- `SI1!` - Silver Futures
- `HG1!` - Copper Futures

### Forex
- `EURUSD`
- `GBPUSD`
- `USDJPY`

### Stocks
- `AAPL` - Apple
- `MSFT` - Microsoft
- `TSLA` - Tesla

Để đổi symbol, sửa trong `config.json`:

```json
{
  "trading": {
    "exchange": "COMEX",
    "symbol": "GC1!"
  }
}
```

## Timeframes Hỗ Trợ

Bot sử dụng 3 timeframes:

| Timeframe | Code | Interval |
|-----------|------|----------|
| Daily | `1D` | Interval.in_daily |
| Hourly | `60` | Interval.in_1_hour |
| 5-minute | `5` | Interval.in_5_minute |

Các timeframe khác được hỗ trợ:
- `1` - 1 minute
- `3` - 3 minutes
- `15` - 15 minutes
- `30` - 30 minutes
- `240` - 4 hours
- `1W` - Weekly
- `1M` - Monthly

## Giới Hạn

### Anonymous Access
- **Rate limit**: ~5-10 requests/phút
- **Max bars**: 5000 bars/request
- **Delay**: Dữ liệu có thể trễ 15-20 phút

### Authenticated Access
- **Rate limit**: ~20-30 requests/phút (cao hơn)
- **Max bars**: 5000 bars/request
- **Delay**: Real-time data

## Xử Lý Lỗi

### Lỗi thường gặp:

**1. "No module named 'tvdatafeed'"**
```bash
pip install tvdatafeed-enhanced
```

**2. "No module named 'websocket'"**
```bash
pip install websocket-client websockets
```

**3. "Rate limit exceeded"**
- Chờ 1-2 phút
- Hoặc login với tài khoản TradingView

**4. "Symbol not found"**
- Kiểm tra spelling của symbol
- Kiểm tra exchange (COMEX, NYSE, NASDAQ, etc.)

## Performance

Bot tự động cache dữ liệu để giảm số lượng requests:

```python
# Fetch mới
data = fetcher.fetch_data('60', 100)

# Dùng cache
cached = fetcher.get_cached_data('60')

# Check cache age
age = fetcher.get_cache_age('60')
print(f"Cache age: {age.total_seconds()} seconds")
```

## Tips

1. **Sử dụng cache**: Luôn check cache trước khi fetch mới
2. **Batch requests**: Fetch nhiều timeframes cùng lúc với `get_multi_timeframe_data()`
3. **Handle errors**: Luôn check `if data is not None`
4. **Rate limiting**: Thêm sleep giữa các requests nếu cần

## Support

- **GitHub**: https://github.com/rongardF/tvdatafeed/
- **Issues**: Báo lỗi trên GitHub Issues
- **Docs**: Đọc docstring trong code

## Version Info

- **tvdatafeed-enhanced**: 2.2.1+
- **Python**: 3.9 - 3.13
- **Dependencies**: pandas, websocket-client, websockets, requests
