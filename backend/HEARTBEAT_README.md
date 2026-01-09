# Virtual Heartbeat Simulation API

Complete FastAPI application that simulates realistic heartbeat data for ML model training.

## 🚀 Quick Start

### 1. Run the Server

```bash
uvicorn heartbeat_sim:app --reload
```

Server starts at: **http://localhost:8000**

### 2. Start Simulation

```bash
curl -X POST http://localhost:8000/simulation/start
```

### 3. Get Current Heartbeat

```bash
curl http://localhost:8000/heartbeat/current
```

Response:
```json
{
  "timestamp": "2026-01-09T12:34:56.789Z",
  "bpm": 75,
  "variability": 0.234,
  "stress_level": 0.156,
  "state": "running"
}
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns service status and simulation state.

### Start Simulation
```bash
POST /simulation/start
```
Starts heartbeat generation at ~2 Hz (500ms intervals).

### Stop Simulation
```bash
POST /simulation/stop
```
Stops background simulation task.

### Get Current Heartbeat
```bash
GET /heartbeat/current
```
Returns latest simulated heartbeat data.

### WebSocket Stream
```
ws://localhost:8000/ws/heartbeat
```
Real-time heartbeat streaming.

### Trigger Stress Event (Optional)
```bash
POST /simulation/trigger-stress?stress_level=0.8
```
Manually trigger stress (0.0 to 1.0).

### Set Baseline BPM (Optional)
```bash
POST /simulation/set-baseline?baseline_bpm=65
```
Set resting heart rate (40-120 BPM).

## 🧪 Testing

### Option 1: Visual Monitor (HTML)
Open `heartbeat_monitor.html` in your browser for a live dashboard.

### Option 2: Python WebSocket Client
```python
import asyncio
import websockets
import json

async def monitor_heartbeat():
    uri = "ws://localhost:8000/ws/heartbeat"
    async with websockets.connect(uri) as ws:
        while True:
            data = json.loads(await ws.recv())
            if data.get('type') == 'connected':
                print("Connected!")
                continue
            print(f"BPM: {data['bpm']}, Stress: {data['stress_level']:.3f}")

asyncio.run(monitor_heartbeat())
```

### Option 3: JavaScript Client
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/heartbeat');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`BPM: ${data.bpm}`);
    // Send to ML model here
};
```

### Option 4: Polling (HTTP)
```python
import requests
import time

while True:
    response = requests.get('http://localhost:8000/heartbeat/current')
    data = response.json()
    print(f"BPM: {data['bpm']}")
    time.sleep(0.5)
```

## 🎯 Features

### Realistic Heartbeat Patterns
- **Baseline:** 60-80 BPM at rest
- **Variability:** ±2-3 BPM natural fluctuation
- **Stress Response:** Gradual increase to 90-120 BPM
- **Recovery:** Slow return to baseline
- **Random Events:** 2% chance of stress spike per update

### Architecture
- ✅ **In-memory storage** - Fast, no database overhead
- ✅ **Async/await** - Non-blocking, efficient
- ✅ **WebSocket support** - Real-time streaming
- ✅ **Multi-client** - Multiple connections supported
- ✅ **Graceful shutdown** - Clean task cancellation
- ✅ **Easy to replace** - Swap simulation with real sensor data

### Data Structure
```python
{
    "timestamp": str,        # ISO format
    "bpm": int,             # Heart rate (60-180)
    "variability": float,   # HRV indicator (0-1)
    "stress_level": float,  # Current stress (0-1)
    "state": str           # "stopped" | "running"
}
```

## 🔧 Integration with ML Model

### Approach 1: WebSocket (Recommended)
```python
# model_input.py
import asyncio
import websockets
import json

async def feed_model():
    uri = "ws://localhost:8000/ws/heartbeat"
    async with websockets.connect(uri) as ws:
        while True:
            data = json.loads(await ws.recv())
            if data.get('type') == 'connected':
                continue
            
            # Send to your ML model
            prediction = your_model.predict([
                data['bpm'],
                data['variability'],
                data['stress_level']
            ])
            print(f"Predicted stress: {prediction}")

asyncio.run(feed_model())
```

### Approach 2: Polling (Simpler)
```python
import requests
import time

def feed_model_polling():
    while True:
        response = requests.get('http://localhost:8000/heartbeat/current')
        data = response.json()
        
        # Send to your ML model
        prediction = your_model.predict([data['bpm']])
        
        time.sleep(0.5)  # Poll every 500ms
```

## 🔄 Replacing with Real Sensor

To replace simulation with real sensor data:

1. **Keep the same API structure** - No changes needed to consumers
2. **Replace `generate_heartbeat()` method:**

```python
def generate_heartbeat(self) -> HeartbeatData:
    # OLD: Simulated data
    # actual_bpm = self.current_bpm + random.uniform(-2.5, 2.5)
    
    # NEW: Real sensor
    actual_bpm = read_from_sensor()  # Your sensor code here
    
    # Rest of the code stays the same
    data = HeartbeatData(
        timestamp=datetime.utcnow().isoformat(),
        bpm=int(round(actual_bpm)),
        ...
    )
    return data
```

3. **Adjust timing** if needed:
```python
# In simulation_loop()
await asyncio.sleep(0.5)  # Change to match sensor sampling rate
```

## 📊 Simulation Profiles

### Athlete Profile (Low Resting HR)
```bash
curl -X POST "http://localhost:8000/simulation/set-baseline?baseline_bpm=55"
```

### Average Person
```bash
curl -X POST "http://localhost:8000/simulation/set-baseline?baseline_bpm=72"
```

### Elevated / Anxious
```bash
curl -X POST "http://localhost:8000/simulation/set-baseline?baseline_bpm=85"
```

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use different port
uvicorn heartbeat_sim:app --port 8001 --reload
```

### WebSocket connection fails
- Ensure simulation is started: `POST /simulation/start`
- Check firewall settings
- Use `ws://` not `wss://` for local testing

### No data returned
- Start simulation first: `POST /simulation/start`
- Check `/health` endpoint to verify state

## 📝 API Documentation

Interactive API docs available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🎓 Architecture Decisions

### Why In-Memory?
- Fast access (no database queries)
- Simple for hackathons
- Easy to scale to database later

### Why AsyncIO?
- Non-blocking WebSocket streams
- Can handle multiple clients
- Efficient resource usage

### Why 2 Hz Update Rate?
- Realistic for heartbeat monitoring
- Not too fast (bandwidth efficient)
- Not too slow (smooth charts)

### Why Gradual Changes?
- Realistic physiological response
- Heart rate doesn't jump instantly
- Better for ML model training

## 📦 Dependencies

```bash
pip install fastapi uvicorn websockets
```

That's it! Minimal dependencies, maximum functionality.

---

**Built for hackathons. Production-ready architecture. Easy to extend.** 🚀
