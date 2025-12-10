"""Quick test of trained model"""
import json
from neural_network import NeuralNetwork
from data_processor import DataProcessor
from data_fetcher import TradingViewDataFetcher

# Load config
config = json.load(open('config.json'))

# Fetch recent data
print("Fetching recent data...")
fetcher = TradingViewDataFetcher(config)
prices = fetcher.get_closing_prices('60', 200)

# Load trained model
print("Loading trained model...")
network = NeuralNetwork(112, 7, 1)
if network.load_weights('weights/weights_GC1!.bin'):
    print("✓ Model loaded successfully")
else:
    print("❌ Failed to load model")
    exit(1)

# Make prediction
processor = DataProcessor(112)
window = prices[-112:]
normalized = processor.normalize_prices(window, 'GC1!')
pred = network.predict(normalized)
pred_value = pred.item() if hasattr(pred, 'item') else float(pred[0]) if len(pred.shape) > 0 else float(pred)

# Display results
print(f"\n{'='*50}")
print(f"PREDICTION RESULTS")
print(f"{'='*50}")
print(f"Latest price: ${prices[-1]:.2f}")
print(f"Prediction: {pred_value:.6f}")

if pred_value > 0.0005:
    signal = "BUY ↗"
elif pred_value < -0.0005:
    signal = "SELL ↘"
else:
    signal = "HOLD —"

print(f"Signal: {signal}")
print(f"{'='*50}\n")
