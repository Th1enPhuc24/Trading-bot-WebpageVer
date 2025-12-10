"""
Real-time TP/SL Monitoring Test
================================

Mô phỏng logic check TP/SL real-time trên M5 data
"""

from datetime import datetime, timedelta
import time

class SimulatedPosition:
    def __init__(self):
        self.symbol = "GC1!"
        self.direction = "BUY"
        self.entry_price = 4250.0
        self.stop_loss = 4240.0
        self.take_profit = 4265.0
        self.entry_time = datetime.now()
        
    def check_tp_sl(self, high, low, close):
        """Simulate M5 TP/SL check"""
        print(f"\n📊 M5 Bar Check - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   High: ${high:.2f} | Low: ${low:.2f} | Close: ${close:.2f}")
        print(f"   TP: ${self.take_profit:.2f} | SL: ${self.stop_loss:.2f}")
        
        if self.direction == "BUY":
            if high >= self.take_profit:
                print(f"✅ TAKE PROFIT HIT at ${self.take_profit:.2f}!")
                pnl = (self.take_profit - self.entry_price) * 10
                print(f"💰 P&L: ${pnl:.2f}")
                return True, "TP"
            
            elif low <= self.stop_loss:
                print(f"🛑 STOP LOSS HIT at ${self.stop_loss:.2f}!")
                pnl = (self.stop_loss - self.entry_price) * 10
                print(f"💰 P&L: ${pnl:.2f}")
                return True, "SL"
        
        print(f"   Position still active...")
        return False, None


def simulate_trading_loop():
    """Simulate the new hybrid checking strategy"""
    print("="*70)
    print("🚀 REAL-TIME TP/SL MONITORING SIMULATION")
    print("="*70)
    print("\nStrategy:")
    print("  ✓ Check TP/SL every 1 minute when position open (M5 data)")
    print("  ✓ Check H1 signals every 60 minutes for new trades")
    print("="*70)
    
    # Scenario 1: TP hit quickly
    print("\n\n📋 SCENARIO 1: Take Profit Hit After 3 Minutes")
    print("-" * 70)
    
    position = SimulatedPosition()
    print(f"\n🟢 Position opened: {position.direction} @ ${position.entry_price:.2f}")
    print(f"   TP: ${position.take_profit:.2f} | SL: ${position.stop_loss:.2f}")
    
    # Simulate M5 bars
    m5_bars = [
        {"time": "10:00", "high": 4252, "low": 4248, "close": 4251},
        {"time": "10:05", "high": 4258, "low": 4250, "close": 4256},
        {"time": "10:10", "high": 4266, "low": 4255, "close": 4265},  # TP hit!
    ]
    
    for i, bar in enumerate(m5_bars, 1):
        print(f"\n--- Minute {i} (M5 at {bar['time']}) ---")
        closed, reason = position.check_tp_sl(bar['high'], bar['low'], bar['close'])
        
        if closed:
            print(f"\n✅ Position closed - Reason: {reason}")
            print(f"⏱️ Hold time: {i} minute(s)")
            break
        
        if i < len(m5_bars):
            print(f"⏸️ Waiting 1 minute...")
            time.sleep(1)  # Simulate 1 min wait
    
    # Scenario 2: SL hit
    print("\n\n📋 SCENARIO 2: Stop Loss Hit")
    print("-" * 70)
    
    position2 = SimulatedPosition()
    position2.direction = "BUY"
    position2.entry_price = 4250.0
    position2.stop_loss = 4240.0
    position2.take_profit = 4265.0
    
    print(f"\n🟢 Position opened: {position2.direction} @ ${position2.entry_price:.2f}")
    
    # Price drops and hits SL
    m5_bars2 = [
        {"time": "11:00", "high": 4248, "low": 4242, "close": 4245},
        {"time": "11:05", "high": 4244, "low": 4238, "close": 4239},  # SL hit!
    ]
    
    for i, bar in enumerate(m5_bars2, 1):
        print(f"\n--- Minute {i} (M5 at {bar['time']}) ---")
        closed, reason = position2.check_tp_sl(bar['high'], bar['low'], bar['close'])
        
        if closed:
            print(f"\n🛑 Position closed - Reason: {reason}")
            print(f"⏱️ Hold time: {i} minute(s)")
            break
        
        if i < len(m5_bars2):
            time.sleep(1)
    
    print("\n\n" + "="*70)
    print("✅ SIMULATION COMPLETED")
    print("="*70)
    print("\nKey Benefits:")
    print("  ✓ TP/SL detected within 1-5 minutes (near real-time)")
    print("  ✓ No missed opportunities")
    print("  ✓ Minimizes slippage")
    print("  ✓ Efficient API usage (only M5 when needed)")


if __name__ == "__main__":
    simulate_trading_loop()
