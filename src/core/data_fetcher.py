"""
TradingView Data Fetcher
Fetches D/H1/M5 bars from TradingView for COMEX:GC1!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from tvDatafeed.main import TvDatafeed, Interval


class TradingViewDataFetcher:
    """
    Fetches historical data from TradingView
    Supports D (Daily), H1 (60min), M5 (5min) timeframes
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.exchange = config['trading']['exchange']
        self.symbol = config['trading']['symbol']
        
        # Initialize TradingView connection (tvdatafeed-enhanced)
        try:
            # Anonymous access - no login required
            # For authenticated access: TvDatafeed(username='your_username', password='your_password')
            self.tv = TvDatafeed()
            print(f"✓ TradingView connection initialized (tvdatafeed-enhanced v2.2.1)")
            print(f"  Using anonymous access - data may be limited")
        except Exception as e:
            print(f"⚠️ TradingView connection failed: {e}")
            self.tv = None
        
        # Cache for storing data
        self.data_cache = {}
    
    def _get_interval(self, timeframe: str) -> Interval:
        """Convert timeframe string to TvDatafeed Interval"""
        timeframe_map = {
            '1D': Interval.in_daily,
            'D': Interval.in_daily,
            '60': Interval.in_1_hour,
            'H1': Interval.in_1_hour,
            '5': Interval.in_5_minute,
            'M5': Interval.in_5_minute
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return timeframe_map[timeframe]
    
    def fetch_data(self, timeframe: str = '60', n_bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from TradingView using tvdatafeed-enhanced
        
        Args:
            timeframe: Timeframe ('1D', '60', '5')
            n_bars: Number of bars to fetch (max 5000)
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if self.tv is None:
            print(f"⚠️ TradingView not initialized")
            return None
        
        try:
            interval = self._get_interval(timeframe)
            
            # Limit n_bars to reasonable amount (tvdatafeed-enhanced supports up to 5000)
            n_bars = min(n_bars, 5000)
            
            print(f"Fetching {n_bars} bars of {self.exchange}:{self.symbol} @ {timeframe}...")
            
            # tvdatafeed-enhanced uses get_hist method
            df = self.tv.get_hist(
                symbol=self.symbol,
                exchange=self.exchange,
                interval=interval,
                n_bars=n_bars,
                fut_contract=None  # For futures, can specify contract (None = continuous)
            )
            
            if df is None or len(df) == 0:
                print(f"⚠️ No data received for {self.symbol}")
                return None
            
            # Ensure data is sorted by date
            df = df.sort_index()
            
            print(f"✓ Fetched {len(df)} bars ({df.index[0]} to {df.index[-1]})")
            
            # Cache the data
            cache_key = f"{self.symbol}_{timeframe}"
            self.data_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            return df
        
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def get_closing_prices(self, timeframe: str = '60', n_bars: int = 500) -> Optional[np.ndarray]:
        """
        Get closing prices as numpy array
        
        Args:
            timeframe: Timeframe ('1D', '60', '5')
            n_bars: Number of bars to fetch
        
        Returns:
            Numpy array of closing prices or None if failed
        """
        df = self.fetch_data(timeframe, n_bars)
        
        if df is None:
            return None
        
        return df['close'].values
    
    def get_multi_timeframe_data(self, n_bars_dict: Optional[Dict[str, int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes (D/H1/M5)
        
        Args:
            n_bars_dict: Dictionary mapping timeframe to number of bars
                        Default: {'1D': 100, '60': 500, '5': 1000}
        
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if n_bars_dict is None:
            n_bars_dict = {
                '1D': 100,   # ~100 days
                '60': 500,   # ~500 hours (≈21 days)
                '5': 1000    # ~1000 5-min bars (≈3.5 days)
            }
        
        data = {}
        
        for timeframe, n_bars in n_bars_dict.items():
            df = self.fetch_data(timeframe, n_bars)
            if df is not None:
                data[timeframe] = df
        
        return data
    
    def get_latest_bar(self, timeframe: str = '60') -> Optional[Dict]:
        """
        Get the most recent bar
        
        Args:
            timeframe: Timeframe ('1D', '60', '5')
        
        Returns:
            Dictionary with OHLCV data or None
        """
        df = self.fetch_data(timeframe, n_bars=10)
        
        if df is None or len(df) == 0:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'timestamp': df.index[-1],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume']
        }
    
    def update_cache(self, timeframe: str = '60', n_bars: int = 50):
        """
        Update cached data with recent bars
        
        Args:
            timeframe: Timeframe to update
            n_bars: Number of recent bars to fetch
        """
        cache_key = f"{self.symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            # Fetch recent data
            new_data = self.fetch_data(timeframe, n_bars)
            
            if new_data is not None:
                old_data = self.data_cache[cache_key]['data']
                
                # Merge and remove duplicates
                combined = pd.concat([old_data, new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                
                self.data_cache[cache_key] = {
                    'data': combined,
                    'timestamp': datetime.now()
                }
                
                print(f"✓ Cache updated for {cache_key}: {len(combined)} bars")
        else:
            # No cache, fetch fresh data
            self.fetch_data(timeframe, n_bars)
    
    def get_cached_data(self, timeframe: str = '60') -> Optional[pd.DataFrame]:
        """Get cached data if available"""
        cache_key = f"{self.symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]['data']
        
        return None
    
    def get_cache_age(self, timeframe: str = '60') -> Optional[timedelta]:
        """Get age of cached data"""
        cache_key = f"{self.symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            cache_time = self.data_cache[cache_key]['timestamp']
            return datetime.now() - cache_time
        
        return None
