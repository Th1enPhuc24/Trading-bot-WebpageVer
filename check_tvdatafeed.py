from tvDatafeed.main import TvDatafeed
import inspect

tv = TvDatafeed()
print("Available methods in TvDatafeed:")
methods = [m for m in dir(tv) if not m.startswith('_')]
for m in methods:
    print(f"  - {m}")

print("\n\nChecking for WebSocket/streaming capabilities:")
has_websocket = any('socket' in m.lower() or 'stream' in m.lower() or 'subscribe' in m.lower() 
                    for m in methods)
print(f"Has WebSocket: {has_websocket}")

# Check TvDatafeed source
print("\n\nTvDatafeed class signature:")
print(inspect.signature(TvDatafeed.__init__))
