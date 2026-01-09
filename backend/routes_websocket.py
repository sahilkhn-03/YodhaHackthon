"""
WebSocket routes for real-time data streaming.

Why WebSockets?
- Real-time bidirectional communication
- Lower latency than HTTP polling
- Efficient for continuous data streams like live stress monitoring
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import random
import json
from datetime import datetime

router = APIRouter()

# Connection manager to handle multiple clients
class ConnectionManager:
    """
    Manages multiple WebSocket connections.
    
    Why?
    - Multiple clinicians can watch simulations simultaneously
    - Broadcast data to all connected clients
    - Clean up disconnected clients automatically
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and store new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ Client connected. Total clients: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove disconnected client."""
        self.active_connections.remove(websocket)
        print(f"❌ Client disconnected. Total clients: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """
        Send message to all connected clients.
        
        Handles disconnected clients gracefully:
        - If send fails, mark for removal
        - Clean up after broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()


async def generate_stress_data():
    """
    Generate simulated stress monitoring data.
    
    Simulates realistic patterns:
    - Stress score: 0.0 to 1.0 (gradually changing, not random jumps)
    - Heart rate: 60-120 BPM (realistic range)
    - Facial stress: 0.0 to 1.0 (based on facial analysis simulation)
    - Timestamp: ISO format for frontend charting
    
    Why realistic simulation?
    - Test frontend charts before real AI integration
    - Demo the system without needing real patients
    - Validate data flow end-to-end
    """
    # Starting baseline values
    stress_score = random.uniform(0.3, 0.5)
    heart_rate = random.randint(65, 75)
    facial_stress = random.uniform(0.2, 0.4)
    
    while True:
        # Gradual changes (more realistic than random jumps)
        stress_score += random.uniform(-0.05, 0.08)
        stress_score = max(0.0, min(1.0, stress_score))  # Clamp 0-1
        
        heart_rate += random.randint(-3, 5)
        heart_rate = max(60, min(120, heart_rate))  # Clamp 60-120
        
        facial_stress += random.uniform(-0.04, 0.06)
        facial_stress = max(0.0, min(1.0, facial_stress))
        
        # Create data packet
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "stress_score": round(stress_score, 3),
            "heart_rate": heart_rate,
            "facial_stress": round(facial_stress, 3),
            "status": "monitoring"
        }
        
        yield data
        
        # Random interval 200-500ms for realistic variation
        await asyncio.sleep(random.uniform(0.2, 0.5))


@router.websocket("/simulation")
async def websocket_simulation(websocket: WebSocket):
    """
    WebSocket endpoint for stress data simulation.
    
    Endpoint: ws://localhost:8000/ws/simulation
    
    Data format (JSON every 200-500ms):
    {
        "timestamp": "2026-01-09T12:34:56.789Z",
        "stress_score": 0.742,
        "heart_rate": 95,
        "facial_stress": 0.631,
        "status": "monitoring"
    }
    
    Frontend usage:
        const ws = new WebSocket('ws://localhost:8000/ws/simulation');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateChart(data);
        };
    
    Why async?
    - Non-blocking: Server can handle multiple clients simultaneously
    - Each client gets their own data stream
    - No performance impact on REST API endpoints
    """
    # Connect client
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Stress simulation started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Start streaming simulated data
        async for data in generate_stress_data():
            await websocket.send_json(data)
            
    except WebSocketDisconnect:
        # Client disconnected (closed browser, lost connection, etc.)
        manager.disconnect(websocket)
        print("Client disconnected from simulation")
        
    except Exception as e:
        # Handle any other errors gracefully
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Additional endpoint: Broadcast mode (all clients see same data)
@router.websocket("/simulation/broadcast")
async def websocket_broadcast(websocket: WebSocket):
    """
    Broadcast mode: All connected clients see the same data stream.
    
    Use case: Demo mode where multiple viewers watch one simulation.
    
    Endpoint: ws://localhost:8000/ws/simulation/broadcast
    """
    await manager.connect(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "mode": "broadcast",
            "message": "Connected to shared stress simulation",
            "clients": len(manager.active_connections)
        })
        
        # Keep connection alive (actual broadcasting happens in background)
        while True:
            # Wait for client messages (keep-alive, commands, etc.)
            message = await websocket.receive_text()
            
            # Echo back for debugging
            await websocket.send_json({
                "type": "echo",
                "received": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Background task for broadcast mode (optional)
async def broadcast_simulation():
    """
    Background task that broadcasts same data to all clients.
    
    How to use:
    1. Start this as a background task when server starts
    2. All clients in broadcast mode receive same data
    
    Add to main.py startup:
        @app.on_event("startup")
        async def startup():
            asyncio.create_task(broadcast_simulation())
    """
    async for data in generate_stress_data():
        if manager.active_connections:
            await manager.broadcast(data)


# How to test:
# 
# Python client:
#   pip install websockets
#   
#   import asyncio
#   import websockets
#   import json
#   
#   async def test_client():
#       uri = "ws://localhost:8000/ws/simulation"
#       async with websockets.connect(uri) as ws:
#           while True:
#               data = await ws.recv()
#               print(json.loads(data))
#   
#   asyncio.run(test_client())
#
# JavaScript client:
#   const ws = new WebSocket('ws://localhost:8000/ws/simulation');
#   ws.onmessage = (event) => {
#       const data = JSON.parse(event.data);
#       console.log('Stress:', data.stress_score, 'HR:', data.heart_rate);
#   };
