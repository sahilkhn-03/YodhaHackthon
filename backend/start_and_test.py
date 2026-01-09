"""Quick test script to start simulation and show data"""
import requests
import time
import json

API_BASE = "http://localhost:8001"

print("="*60)
print("🫀 HEARTBEAT SIMULATION TEST")
print("="*60)

# Start simulation
print("\n1. Starting simulation...")
try:
    response = requests.post(f"{API_BASE}/simulation/start")
    print(f"✅ {response.json()['message']}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Show live data
print("\n2. Streaming heartbeat data (Press Ctrl+C to stop)...\n")
print(f"{'Time':<12} {'BPM':<8} {'Variability':<12} {'Stress':<10}")
print("-" * 50)

try:
    while True:
        response = requests.get(f"{API_BASE}/heartbeat/current")
        data = response.json()
        
        timestamp = data['timestamp'].split('T')[1][:8]
        bpm = data['bpm']
        variability = f"{data['variability']:.3f}"
        stress = f"{data['stress_level']:.3f}"
        
        # Color indicators
        stress_indicator = "🔴" if data['stress_level'] > 0.5 else "🟡" if data['stress_level'] > 0.3 else "🟢"
        
        print(f"{timestamp:<12} {bpm:<8} {variability:<12} {stress:<10} {stress_indicator}")
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("\n\n3. Stopping simulation...")
    requests.post(f"{API_BASE}/simulation/stop")
    print("✅ Simulation stopped")
    print("="*60)
