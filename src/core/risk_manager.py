"""
Risk Management and Money Management Module
Implements exact specifications:
- Risk: 0.2% of balance per trade (RiskPercentage = 0.002)
- Stop Loss: 50,000 points
- Take Profit: 70 points
- No trailing stop, no breakeven
- Max hold time: 4 hours (typical 1 hour)
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta


class RiskManager:
    """
    Manages position sizing and risk parameters
    - Fixed 0.2% risk per trade
    - SL: 50,000 points (for broker compliance and lot calculation)
    - TP: 70 points (7 pips on 5-digit quotes)
    - Max hold: 4 hours
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.risk_percentage = config['risk_management']['risk_percentage']
        self.stop_loss_points = config['risk_management']['stop_loss_points']
        self.take_profit_points = config['risk_management']['take_profit_points']
        self.max_hold_hours = config['risk_management']['max_hold_hours']
        self.typical_hold_hours = config['risk_management']['typical_hold_hours']
        
        # Symbol specifications (for accurate lot calculation)
        self.symbol_specs = self._initialize_symbol_specs()
    
    def _initialize_symbol_specs(self) -> Dict:
        """
        Initialize symbol specifications for lot size calculation
        Note: These are approximate values, should be updated with actual broker specs
        """
        return {
            'GC1!': {  # Gold futures
                'point_value': 0.1,  # $0.1 per point for micro gold
                'contract_size': 10,  # 10 troy ounces for micro
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'digits': 1  # Gold quoted to 1 decimal (e.g., 2650.5)
            },
            'EURUSD': {
                'point_value': 0.00001,  # 5-digit broker
                'contract_size': 100000,
                'min_lot': 0.01,
                'max_lot': 100.0,
                'lot_step': 0.01,
                'digits': 5
            }
            # Add more symbols as needed from config.json symbol list
        }
    
    def calculate_lot_size(self, symbol: str, account_balance: float, 
                          entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate lot size based on risk percentage
        Formula: Lot = (Account Balance × Risk%) / (SL Distance × Point Value)
        
        Args:
            symbol: Trading symbol
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss_price: Stop loss price
        
        Returns:
            Calculated lot size
        """
        if symbol not in self.symbol_specs:
            print(f"Symbol {symbol} not in specs, using GC1! defaults")
            symbol = 'GC1!'
        
        specs = self.symbol_specs[symbol]
        
        # Risk amount in currency
        risk_amount = account_balance * self.risk_percentage
        
        # SL distance in points
        sl_distance = abs(entry_price - stop_loss_price)
        
        # Avoid division by zero
        if sl_distance < 1e-10:
            print(f"SL distance too small, using minimum lot")
            return specs['min_lot']
        
        # Calculate lot size
        # For gold: risk_amount / (sl_distance × point_value × contract_size)
        lot_size = risk_amount / (sl_distance * specs['point_value'] * specs['contract_size'])
        
        # Round to lot step
        lot_size = round(lot_size / specs['lot_step']) * specs['lot_step']
        
        # Apply limits
        lot_size = max(specs['min_lot'], min(lot_size, specs['max_lot']))
        
        return lot_size
    
    def calculate_lot_size_from_points(self, symbol: str, account_balance: float) -> float:
        """
        Calculate lot size using the fixed 50,000 point SL
        This is the method described in original requirements
        
        Args:
            symbol: Trading symbol
            account_balance: Current account balance
        
        Returns:
            Calculated lot size
        """
        if symbol not in self.symbol_specs:
            print(f"Symbol {symbol} not in specs, using GC1! defaults")
            symbol = 'GC1!'
        
        specs = self.symbol_specs[symbol]
        
        # Risk amount in currency
        risk_amount = account_balance * self.risk_percentage
        
        # Use fixed SL points (50,000)
        sl_points = self.stop_loss_points
        
        # Calculate lot size
        lot_size = risk_amount / (sl_points * specs['point_value'] * specs['contract_size'])
        
        # Round to lot step
        lot_size = round(lot_size / specs['lot_step']) * specs['lot_step']
        
        # Apply limits
        lot_size = max(specs['min_lot'], min(lot_size, specs['max_lot']))
        
        return lot_size
    
    def calculate_sl_tp_prices(self, symbol: str, direction: str, 
                               entry_price: float) -> Dict[str, float]:
        """
        Calculate SL and TP prices based on fixed points
        
        Args:
            symbol: Trading symbol
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
        
        Returns:
            Dictionary with 'stop_loss' and 'take_profit' prices
        """
        if symbol not in self.symbol_specs:
            symbol = 'GC1!'
        
        specs = self.symbol_specs[symbol]
        point_size = specs['point_value']
        
        # For gold: 1 point = 0.1, so 70 points = 7.0 price move
        # For forex 5-digit: 1 point = 0.00001, so 70 points = 0.0007 (7 pips)
        
        tp_distance = self.take_profit_points * point_size
        sl_distance = self.stop_loss_points * point_size
        
        if direction == 'BUY':
            take_profit = entry_price + tp_distance
            stop_loss = entry_price - sl_distance
        else:  # SELL
            take_profit = entry_price - tp_distance
            stop_loss = entry_price + sl_distance
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def check_position_timeout(self, entry_time: datetime) -> bool:
        """
        Check if position has exceeded max hold time
        
        Args:
            entry_time: When position was entered
        
        Returns:
            True if should close due to timeout, False otherwise
        """
        hold_time = datetime.now() - entry_time
        max_hold = timedelta(hours=self.max_hold_hours)
        
        return hold_time >= max_hold
    
    def get_position_info(self, symbol: str, direction: str, entry_price: float, 
                         account_balance: float) -> Dict:
        """
        Get complete position information including lot size, SL, TP
        
        Args:
            symbol: Trading symbol
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            account_balance: Current account balance
        
        Returns:
            Complete position information
        """
        # Calculate SL/TP prices
        prices = self.calculate_sl_tp_prices(symbol, direction, entry_price)
        
        # Calculate lot size (using fixed 50,000 point SL method)
        lot_size = self.calculate_lot_size_from_points(symbol, account_balance)
        
        # Calculate risk amount
        risk_amount = account_balance * self.risk_percentage
        
        position_info = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': prices['stop_loss'],
            'take_profit': prices['take_profit'],
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'risk_percentage': self.risk_percentage * 100,
            'sl_points': self.stop_loss_points,
            'tp_points': self.take_profit_points,
            'max_hold_hours': self.max_hold_hours
        }
        
        return position_info
    
    def validate_position(self, position_info: Dict) -> bool:
        """
        Validate position parameters before opening
        
        Args:
            position_info: Position information dictionary
        
        Returns:
            True if valid, False otherwise
        """
        # Check lot size
        symbol = position_info['symbol']
        lot_size = position_info['lot_size']
        
        if symbol not in self.symbol_specs:
            symbol = 'GC1!'
        
        specs = self.symbol_specs[symbol]
        
        if lot_size < specs['min_lot']:
            print(f"Lot size {lot_size} below minimum {specs['min_lot']}")
            return False
        
        if lot_size > specs['max_lot']:
            print(f"Lot size {lot_size} above maximum {specs['max_lot']}")
            return False
        
        # Check SL/TP distances
        entry = position_info['entry_price']
        sl = position_info['stop_loss']
        tp = position_info['take_profit']
        
        if abs(entry - sl) < 1e-10:
            print(f"Stop loss too close to entry")
            return False
        
        if abs(entry - tp) < 1e-10:
            print(f"Take profit too close to entry")
            return False
        
        return True
    
    def print_position_info(self, position_info: Dict):
        """Print formatted position information"""
        print(f"\n{'='*60}")
        print(f"POSITION INFORMATION: {position_info['symbol']} {position_info['direction']}")
        print(f"{'='*60}")
        print(f"Entry Price:      {position_info['entry_price']:.2f}")
        print(f"Stop Loss:        {position_info['stop_loss']:.2f} ({position_info['sl_points']} points)")
        print(f"Take Profit:      {position_info['take_profit']:.2f} ({position_info['tp_points']} points)")
        print(f"Lot Size:         {position_info['lot_size']:.2f}")
        print(f"Risk Amount:      ${position_info['risk_amount']:.2f} ({position_info['risk_percentage']:.2f}%)")
        print(f"Max Hold Time:    {position_info['max_hold_hours']} hours")
        print(f"{'='*60}\n")
