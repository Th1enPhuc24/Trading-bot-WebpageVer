"""
Dashboard Auto-Save Summary
===========================

✅ IMPLEMENTED FEATURES
-----------------------

1. BACKTEST DASHBOARD
   Location: outputs/backtests/
   Filename: backtest_results_YYYYMMDD_HHMMSS.png
   Trigger: After backtest completion
   Files: run_backtest_pipeline.py, main_pipeline.py

2. LIVE TRADING DASHBOARD  
   Location: outputs/live/
   Filenames:
   - live_trading_periodic_YYYYMMDD_HHMMSS.png (every 12 hours)
   - live_trading_position_closed_YYYYMMDD_HHMMSS.png (when position closes)
   - live_trading_final_state_YYYYMMDD_HHMMSS.png (when bot stops)
   
   File: trading_bot.py
   Method: _save_live_dashboard(reason)

📁 DIRECTORY STRUCTURE
---------------------
outputs/
├── backtests/       # Backtest results
├── live/            # Live trading snapshots  
└── dashboards/      # Legacy (can cleanup)

🔧 CODE CHANGES
---------------

trading_bot.py:
- Added: self.iteration_count, self.last_dashboard_save
- Added: _save_live_dashboard() method
- Modified: run_iteration() - save periodic (every 12 hours)
- Modified: check_open_positions() - save when position closes
- Modified: stop() - save final state
- Modified: run() - increment iteration_count

No changes needed for backtest - already implemented!

📊 DASHBOARD CONTENT
-------------------
All dashboards contain:
- Price chart with signals
- Equity curve & drawdown
- Training loss history
- Trading statistics

🧪 TESTING
----------
✅ Test file created: test_dashboard_save.py
✅ Directories created: outputs/backtests, outputs/live
✅ Documentation: outputs/README.md

💡 USAGE
--------
Backtest: python run_backtest_pipeline.py
Live:     python run_with_dashboard.py

Dashboards auto-save to outputs/backtests/ and outputs/live/
"""
print(__doc__)
