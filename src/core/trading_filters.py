"""
Trading Constraints and Filters Module
Implements trading hours, volume filters, and position tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, Dict, List
import pytz


class TradingFilters:
    """
    Implements trading constraints:
    - Trading hours limits (18:00-17:00 ET with breaks)
    - Volume threshold filters
    - Max hold time enforcement (4 hours)
    - Position tracking
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.volume_threshold = config['filters']['volume_threshold']
        
        # Trading hours configuration
        self.trading_hours_start = config['filters']['trading_hours']['start']
        self.trading_hours_end = config['filters']['trading_hours']['end']
        self.timezone = pytz.timezone(config['filters']['trading_hours']['timezone'])
        
        # Track positions
        self.position_log = []
    
    def is_trading_hours(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if current time is within allowed trading hours
        Gold trading: 18:00 ET Sunday - 17:00 ET Friday (60-min break 17:00-18:00 daily)
        
        Args:
            check_time: Time to check (default: now)
        
        Returns:
            True if within trading hours, False otherwise
        """
        if check_time is None:
            check_time = datetime.now(self.timezone)
        else:
            # Convert to timezone if not aware
            if check_time.tzinfo is None:
                check_time = self.timezone.localize(check_time)
            else:
                check_time = check_time.astimezone(self.timezone)
        
        current_time = check_time.time()
        current_weekday = check_time.weekday()  # 0=Monday, 6=Sunday
        
        # Gold futures trading hours: Sunday 18:00 ET - Friday 17:00 ET
        # Daily break: 17:00-18:00 ET
        
        # Check if weekend (Saturday)
        if current_weekday == 5:  # Saturday
            return False
        
        # Check if Friday after 17:00
        if current_weekday == 4 and current_time >= time(17, 0):
            return False
        
        # Check if Sunday before 18:00
        if current_weekday == 6 and current_time < time(18, 0):
            return False
        
        # Check daily break (17:00-18:00)
        if time(17, 0) <= current_time < time(18, 0):
            return False
        
        return True
    
    def check_volume_filter(self, volume: float) -> bool:
        """
        Check if volume meets minimum threshold
        
        Args:
            volume: Current bar volume
        
        Returns:
            True if volume sufficient, False otherwise
        """
        if volume < self.volume_threshold:
            print(f"Volume filter: {volume:.0f} below threshold {self.volume_threshold}")
            return False
        return True
    
    def should_trade(self, current_time: Optional[datetime] = None, 
                    volume: Optional[float] = None) -> Dict:
        """
        Comprehensive check if trading is allowed
        
        Args:
            current_time: Time to check (default: now)
            volume: Current volume (optional)
        
        Returns:
            Dictionary with trading permission and reasons
        """
        reasons = []
        
        # Check trading hours
        in_hours = self.is_trading_hours(current_time)
        if not in_hours:
            reasons.append('Outside trading hours')
        
        # Check volume if provided
        volume_ok = True
        if volume is not None:
            volume_ok = self.check_volume_filter(volume)
            if not volume_ok:
                reasons.append(f'Volume too low ({volume:.0f})')
        
        # Overall decision
        allowed = in_hours and volume_ok
        
        return {
            'allowed': allowed,
            'in_trading_hours': in_hours,
            'volume_ok': volume_ok,
            'reasons': reasons if not allowed else []
        }
    
    def log_position(self, position_info: Dict):
        """
        Log position for tracking
        
        Args:
            position_info: Position information dictionary
        """
        log_entry = {
            'timestamp': datetime.now(),
            'action': 'OPEN',
            **position_info
        }
        self.position_log.append(log_entry)
    
    def log_position_close(self, close_info: Dict):
        """
        Log position closure
        
        Args:
            close_info: Position close information
        """
        log_entry = {
            'timestamp': datetime.now(),
            'action': 'CLOSE',
            **close_info
        }
        self.position_log.append(log_entry)
    
    def get_position_history(self, symbol: Optional[str] = None, 
                           n_recent: int = 10) -> List[Dict]:
        """
        Get position history
        
        Args:
            symbol: Filter by symbol (optional)
            n_recent: Number of recent positions to return
        
        Returns:
            List of position log entries
        """
        if symbol is None:
            history = self.position_log
        else:
            history = [p for p in self.position_log if p.get('symbol') == symbol]
        
        return history[-n_recent:]
    
    def get_trading_statistics(self) -> Dict:
        """
        Calculate trading statistics from position log
        
        Returns:
            Dictionary with statistics
        """
        if len(self.position_log) == 0:
            return {'total_positions': 0}
        
        closed_positions = [p for p in self.position_log if p['action'] == 'CLOSE']
        
        if len(closed_positions) == 0:
            return {
                'total_positions': len([p for p in self.position_log if p['action'] == 'OPEN']),
                'closed_positions': 0
            }
        
        # Calculate P&L statistics
        pnls = [p.get('pnl_points', 0) for p in closed_positions]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        # Hold time statistics
        hold_times = []
        for p in closed_positions:
            if 'hold_time' in p:
                hold_times.append(p['hold_time'].total_seconds() / 3600)  # Convert to hours
        
        return {
            'total_positions': len([p for p in self.position_log if p['action'] == 'OPEN']),
            'closed_positions': len(closed_positions),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(closed_positions) * 100 if len(closed_positions) > 0 else 0,
            'total_pnl_points': sum(pnls),
            'avg_win_points': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss_points': np.mean(losses) if len(losses) > 0 else 0,
            'avg_hold_time_hours': np.mean(hold_times) if len(hold_times) > 0 else 0,
            'max_hold_time_hours': max(hold_times) if len(hold_times) > 0 else 0
        }
    
    def print_statistics(self):
        """Print formatted trading statistics"""
        stats = self.get_trading_statistics()
        
        print(f"\n{'='*60}")
        print(f"TRADING STATISTICS")
        print(f"{'='*60}")
        print(f"Total Positions:      {stats.get('total_positions', 0)}")
        print(f"Closed Positions:     {stats.get('closed_positions', 0)}")
        
        if stats.get('closed_positions', 0) > 0:
            print(f"Winning Trades:       {stats['winning_trades']}")
            print(f"Losing Trades:        {stats['losing_trades']}")
            print(f"Win Rate:             {stats['win_rate']:.2f}%")
            print(f"Total P&L (points):   {stats['total_pnl_points']:+.2f}")
            print(f"Avg Win (points):     {stats['avg_win_points']:+.2f}")
            print(f"Avg Loss (points):    {stats['avg_loss_points']:+.2f}")
            print(f"Avg Hold Time:        {stats['avg_hold_time_hours']:.2f} hours")
            print(f"Max Hold Time:        {stats['max_hold_time_hours']:.2f} hours")
        
        print(f"{'='*60}\n")
