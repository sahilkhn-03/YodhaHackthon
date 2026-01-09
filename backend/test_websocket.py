"""
WebSocket test client - Test the real-time stress simulation.

Run this after starting the server:
    python test_websocket.py
"""

import asyncio
import websockets
import json

async def test_simulation():
    """Connect to WebSocket and receive stress data."""
    uri = "ws://localhost:8000/ws/simulation"
    
    print("🔌 Connecting to WebSocket...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected! Receiving stress data...\n")
            
            # Receive first 20 messages
            for i in range(20):
                data_str = await ws.recv()
                data = json.loads(data_str)
                
                # Pretty print the data
                if data.get("type") == "connected":
                    print(f"📡 {data['message']}\n")
                else:
                    print(f"[{i+1}] {data['timestamp']}")
                    print(f"    Stress: {data['stress_score']:.3f}")
                    print(f"    Heart Rate: {data['heart_rate']} BPM")
                    print(f"    Facial Stress: {data['facial_stress']:.3f}\n")
            
            print("✅ Test complete! WebSocket is working.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("    python backend\\main.py")


if __name__ == "__main__":
    print("="*50)
    print("WebSocket Stress Simulation Test")
    print("="*50 + "\n")
    asyncio.run(test_simulation())
