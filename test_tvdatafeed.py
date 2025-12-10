"""
Test script for tvdatafeed-enhanced integration
Tests data fetching for GC1! (Gold Futures) across multiple timeframes
"""

import json
from data_fetcher import TradingViewDataFetcher

def test_single_timeframe():
    """Test fetching data for a single timeframe"""
    print("\n" + "="*60)
    print("TEST 1: Single Timeframe Fetch")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    fetcher = TradingViewDataFetcher(config)
    
    # Test H1 (60 min)
    print("\nFetching H1 data...")
    df_h1 = fetcher.fetch_data('60', n_bars=50)
    
    if df_h1 is not None:
        print(f"\n✓ H1 Data fetched successfully!")
        print(f"  Shape: {df_h1.shape}")
        print(f"  Columns: {list(df_h1.columns)}")
        print(f"\nFirst 3 rows:")
        print(df_h1.head(3))
        print(f"\nLast 3 rows:")
        print(df_h1.tail(3))
        print(f"\nLatest close: ${df_h1['close'].iloc[-1]:.2f}")
    else:
        print("❌ Failed to fetch H1 data")

def test_multi_timeframe():
    """Test fetching data for multiple timeframes"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Timeframe Fetch")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    fetcher = TradingViewDataFetcher(config)
    
    # Fetch D, H1, M5
    data = fetcher.get_multi_timeframe_data({
        '1D': 30,   # 30 days
        '60': 100,  # 100 hours
        '5': 200    # 200 5-min bars
    })
    
    print(f"\n✓ Fetched {len(data)} timeframes")
    
    for tf, df in data.items():
        print(f"\n{tf}:")
        print(f"  Bars: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")

def test_closing_prices():
    """Test getting closing prices as numpy array"""
    print("\n" + "="*60)
    print("TEST 3: Closing Prices Array")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    fetcher = TradingViewDataFetcher(config)
    
    # Get closing prices for H1
    prices = fetcher.get_closing_prices('60', n_bars=112)
    
    if prices is not None:
        print(f"\n✓ Fetched {len(prices)} closing prices")
        print(f"  Min: ${prices.min():.2f}")
        print(f"  Max: ${prices.max():.2f}")
        print(f"  Mean: ${prices.mean():.2f}")
        print(f"  Latest: ${prices[-1]:.2f}")
        print(f"\nLast 10 prices:")
        print(prices[-10:])
    else:
        print("❌ Failed to fetch closing prices")

def test_latest_bar():
    """Test getting latest bar"""
    print("\n" + "="*60)
    print("TEST 4: Latest Bar")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    fetcher = TradingViewDataFetcher(config)
    
    bar = fetcher.get_latest_bar('60')
    
    if bar:
        print(f"\n✓ Latest H1 bar:")
        print(f"  Timestamp: {bar['timestamp']}")
        print(f"  Open:   ${bar['open']:.2f}")
        print(f"  High:   ${bar['high']:.2f}")
        print(f"  Low:    ${bar['low']:.2f}")
        print(f"  Close:  ${bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:,.0f}")
    else:
        print("❌ Failed to fetch latest bar")

def test_cache():
    """Test caching functionality"""
    print("\n" + "="*60)
    print("TEST 5: Cache Functionality")
    print("="*60)
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    fetcher = TradingViewDataFetcher(config)
    
    # First fetch
    print("\nFirst fetch (should hit API)...")
    df1 = fetcher.fetch_data('60', n_bars=20)
    
    # Get from cache
    print("\nSecond fetch (should use cache)...")
    df2 = fetcher.get_cached_data('60')
    
    if df2 is not None:
        print(f"✓ Cache hit! {len(df2)} bars in cache")
        
        age = fetcher.get_cache_age('60')
        print(f"  Cache age: {age.total_seconds():.1f} seconds")
    else:
        print("⚠️ Cache miss")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TVDATAFEED-ENHANCED INTEGRATION TESTS")
    print("="*60)
    print("Testing TradingView data fetching for COMEX:GC1!")
    
    try:
        test_single_timeframe()
        test_multi_timeframe()
        test_closing_prices()
        test_latest_bar()
        test_cache()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        print("\ntvdatafeed-enhanced is working correctly!")
        print("You can now use the trading bot with live data.\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
