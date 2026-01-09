"""
Virtual Person Heartbeat Simulation API
Fast API application that simulates realistic heartbeat data for ML model input.

Run with: uvicorn heartbeat_sim:app --reload
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import random
import time
from datetime import datetime
from enum import Enum

# ============= Configuration =============

class SimulationState(str, Enum):
    """Simulation states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class HeartbeatData(BaseModel):
    """Heartbeat data structure"""
    timestamp: str
    bpm: int
    systolic: int  # Systolic blood pressure (mmHg)
    diastolic: int  # Diastolic blood pressure (mmHg)
    stress_level: float  # Current stress level (0-1, internal)
    state: str
    

# ============= In-Memory Storage =============

class SimulationEngine:
    """
    Manages heartbeat simulation state and data generation.
    
    Why in-memory?
    - Fast access, no database overhead
    - Easy to replace with real sensor later
    - Perfect for hackathons
    """
    
    def __init__(self):
        self.state: SimulationState = SimulationState.STOPPED
        self.current_data: Optional[HeartbeatData] = None
        self.base_bpm: int = 72  # Resting heart rate
        self.current_bpm: float = 72.0
        self.stress_level: float = 0.0
        self.task: Optional[asyncio.Task] = None
        self.websocket_clients: List[WebSocket] = []
        self.stress_button_active: bool = False
        self.stress_button_start_time: Optional[float] = None
        
        # Simulation parameters
        self.min_bpm = 60
        self.max_bpm = 180
        self.stress_threshold = 0.3  # When stress starts affecting HR
        
    def reset(self):
        """Reset to baseline"""
        self.current_bpm = float(self.base_bpm)
        self.stress_level = 0.0
        
    def generate_heartbeat(self) -> HeartbeatData:
        """
        Generate realistic heartbeat data.
        
        Realistic patterns:
        1. Baseline: 60-80 bpm at rest
        2. Variability: Small random fluctuations (±2-3 bpm)
        3. Stress response: Gradual increase to 90-120 bpm
        4. Recovery: Slow return to baseline
        """
        
        # Check if stress button is active (5 second duration)
        if self.stress_button_active and self.stress_button_start_time:
            elapsed = time.time() - self.stress_button_start_time
            if elapsed < 5.0:
                # Keep stress high for 5 seconds with fluctuation
                self.stress_level = random.uniform(0.7, 0.95)
            else:
                # After 5 seconds, deactivate and start decay
                self.stress_button_active = False
                self.stress_button_start_time = None
        else:
            # Random stress events (simulate anxiety, exercise, etc.)
            # Variable probability: 5-8% chance per update for realistic sporadic stress
            stress_probability = random.uniform(0.05, 0.08)
            if random.random() < stress_probability:
                self.stress_level = random.uniform(0.3, 0.9)
        
        # Stress decay (return to baseline) - only when button not active
        if not self.stress_button_active:
            self.stress_level *= 0.98
            self.stress_level = max(0.0, self.stress_level)
        
        # Calculate target heart rate based on stress
        if self.stress_level > self.stress_threshold:
            # Stress increases heart rate
            stress_factor = (self.stress_level - self.stress_threshold) / (1 - self.stress_threshold)
            target_bpm = self.base_bpm + (stress_factor * 50)  # Up to +50 bpm under stress
        else:
            target_bpm = self.base_bpm
        
        # Smooth transition to target (realistic gradual change)
        self.current_bpm += (target_bpm - self.current_bpm) * 0.1
        
        # Add natural variability (breathing, minor fluctuations)
        variability = random.uniform(-2.5, 2.5)
        actual_bpm = self.current_bpm + variability
        
        # Clamp to realistic range
        actual_bpm = max(self.min_bpm, min(self.max_bpm, actual_bpm))
        
        # Calculate blood pressure (correlated with heart rate and stress)
        # Normal BP: 120/80, increases with stress
        base_systolic = 120
        base_diastolic = 80
        
        bp_increase = self.stress_level * 30  # Up to +30 mmHg under stress
        systolic = int(base_systolic + bp_increase + random.uniform(-5, 5))
        diastolic = int(base_diastolic + bp_increase * 0.5 + random.uniform(-3, 3))
        
        # Clamp to realistic ranges
        systolic = max(90, min(180, systolic))
        diastolic = max(60, min(110, diastolic))
        
        # Create data packet
        data = HeartbeatData(
            timestamp=datetime.utcnow().isoformat(),
            bpm=int(round(actual_bpm)),
            systolic=systolic,
            diastolic=diastolic,
            stress_level=round(self.stress_level, 3),
            state=self.state.value
        )
        
        self.current_data = data
        return data
    
    async def simulation_loop(self):
        """
        Background task that continuously generates heartbeat data.
        
        Runs at ~2 Hz (every 500ms) for realistic monitoring.
        """
        while self.state == SimulationState.RUNNING:
            # Generate new heartbeat data
            data = self.generate_heartbeat()
            
            # Broadcast to all WebSocket clients
            await self.broadcast_to_websockets(data)
            
            # Wait before next update (2 Hz = 500ms)
            await asyncio.sleep(0.5)
    
    async def broadcast_to_websockets(self, data: HeartbeatData):
        """Send data to all connected WebSocket clients"""
        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send_json(data.dict())
            except:
                disconnected.append(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            self.websocket_clients.remove(client)


# Global simulation engine instance
engine = SimulationEngine()


# ============= FastAPI Application =============

app = FastAPI(
    title="Virtual Heartbeat Simulation API",
    description="Simulates realistic heartbeat data for ML model training",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        - Service status
        - Simulation state
        - Uptime info
    """
    return {
        "status": "healthy",
        "service": "Heartbeat Simulation API",
        "simulation_state": engine.state.value,
        "base_bpm": engine.base_bpm,
        "websocket_clients": len(engine.websocket_clients)
    }


@app.post("/simulation/start")
async def start_simulation():
    """
    Start heartbeat simulation.
    
    Creates background task that continuously generates heartbeat data.
    Safe to call multiple times (won't create duplicate tasks).
    """
    if engine.state == SimulationState.RUNNING:
        return JSONResponse(
            status_code=200,
            content={
                "message": "Simulation already running",
                "state": engine.state.value
            }
        )
    
    # Reset to baseline
    engine.reset()
    engine.state = SimulationState.RUNNING
    
    # Start background simulation loop
    engine.task = asyncio.create_task(engine.simulation_loop())
    
    return {
        "message": "Simulation started",
        "state": engine.state.value,
        "base_bpm": engine.base_bpm
    }


@app.post("/simulation/stress-test")
async def stress_test():
    """
    Trigger manual stress test.
    
    Simulates a stress event for exactly 5 seconds, then returns to normal.
    Heart rate will fluctuate during the stress period.
    """
    if engine.state != SimulationState.RUNNING:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Simulation not running",
                "message": "Start simulation first using POST /simulation/start"
            }
        )
    
    # Activate stress button timer
    engine.stress_button_active = True
    engine.stress_button_start_time = time.time()
    
    return {
        "message": "Stress test initiated",
        "duration_seconds": 5,
        "state": engine.state.value
    }


@app.post("/simulation/stop")
async def stop_simulation():
    """
    Stop heartbeat simulation.
    
    Gracefully stops the background task and resets state.
    """
    if engine.state == SimulationState.STOPPED:
        return JSONResponse(
            status_code=200,
            content={
                "message": "Simulation already stopped",
                "state": engine.state.value
            }
        )
    
    engine.state = SimulationState.STOPPED
    
    # Cancel background task if running
    if engine.task and not engine.task.done():
        engine.task.cancel()
        try:
            await engine.task
        except asyncio.CancelledError:
            pass
    
    return {
        "message": "Simulation stopped",
        "state": engine.state.value
    }


@app.get("/heartbeat/current", response_model=HeartbeatData)
async def get_current_heartbeat():
    """
    Get latest simulated heartbeat data.
    
    Returns:
        Latest heartbeat measurement with:
        - timestamp: ISO format
        - bpm: Current heart rate
        - variability: Heart rate variability (0-1)
        - stress_level: Current stress level (0-1)
        - state: Simulation state
    
    Use case:
        - ML model polling for latest data
        - Dashboard displays
        - Health monitoring
    """
    if engine.current_data is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "No data available",
                "message": "Start simulation first using POST /simulation/start"
            }
        )
    
    return engine.current_data


@app.websocket("/ws/heartbeat")
async def websocket_heartbeat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time heartbeat streaming.
    
    Endpoint: ws://localhost:8001/ws/heartbeat
    
    Streams heartbeat data at ~2 Hz (every 500ms).
    Data is pushed from the simulation loop via broadcast_to_websockets().
    
    Client example:
        const ws = new WebSocket('ws://localhost:8001/ws/heartbeat');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(`BPM: ${data.bpm}, Stress: ${data.stress_level}`);
        };
    
    Perfect for:
        - Real-time dashboards
        - Live monitoring displays
        - Continuous model input
    """
    await websocket.accept()
    engine.websocket_clients.append(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to heartbeat stream",
            "state": engine.state.value,
            "base_bpm": engine.base_bpm
        })
        
        # Keep connection alive - wait for disconnect or close message
        # Data is pushed by broadcast_to_websockets() in simulation loop
        while True:
            try:
                # Wait for disconnection or client ping/pong
                # receive_text() will raise WebSocketDisconnect when client closes
                message = await websocket.receive_text()
                
                # Optional: Handle client commands
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except WebSocketDisconnect:
                break
            except Exception:
                # Any other error means connection is broken
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Log unexpected errors but don't crash
        print(f"WebSocket error: {e}")
    finally:
        # Clean up on disconnect
        if websocket in engine.websocket_clients:
            engine.websocket_clients.remove(websocket)


# ============= Utility Endpoints =============

@app.get("/", response_class=HTMLResponse)
async def live_monitor():
    """
    Live EKG-style heartbeat monitor dashboard.
    
    Opens in browser at: http://localhost:8001/
    
    Real-time visualization with EKG waveform display.
    API accessible at /heartbeat/current for ML models.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EKG Heartbeat Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff41;
            margin-bottom: 30px;
            font-size: 42px;
            text-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
            letter-spacing: 2px;
        }
        .monitor {
            background: #1a1f3a;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 255, 65, 0.2);
            margin-bottom: 20px;
            border: 2px solid #00ff41;
        }
        
        /* EKG Display */
        .ekg-container {
            background: #0d1117;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 2px solid #00ff41;
            position: relative;
            overflow: hidden;
        }
        .ekg-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .ekg-title {
            color: #00ff41;
            font-size: 20px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .bpm-live {
            font-size: 48px;
            font-weight: bold;
            color: #00ff41;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.8);
        }
        .bpm-unit {
            font-size: 20px;
            color: #888;
            margin-left: 10px;
        }
        
        /* EKG Canvas */
        #ekgCanvas {
            width: 100%;
            height: 200px;
            background: #000;
            border-radius: 8px;
            display: block;
        }
        
        /* Metrics Grid */
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric {
            background: linear-gradient(135deg, #1a1f3a 0%, #0d1117 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid #00ff41;
            transition: transform 0.3s ease;
        }
        .metric:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 255, 65, 0.3);
        }
        .metric-value {
            font-size: 48px;
            font-weight: bold;
            color: #00ff41;
            margin-bottom: 5px;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        }
        .metric-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Blood Pressure Display */
        .bp-display {
            display: flex;
            align-items: baseline;
            justify-content: center;
            gap: 5px;
        }
        .bp-value {
            font-size: 48px;
            font-weight: bold;
            color: #00ff41;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        }
        .bp-separator {
            font-size: 36px;
            color: #666;
        }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        button {
            flex: 1;
            padding: 18px;
            border: 2px solid #00ff41;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            background: transparent;
            color: #00ff41;
        }
        button:hover {
            background: #00ff41;
            color: #0a0e27;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
            transform: translateY(-2px);
        }
        .btn-start { border-color: #00ff41; color: #00ff41; }
        .btn-stress { border-color: #ff4136; color: #ff4136; }
        .btn-stop { border-color: #ff851b; color: #ff851b; }
        .btn-start:hover { background: #00ff41; }
        .btn-stress:hover { background: #ff4136; }
        .btn-stop:hover { background: #ff851b; }
        .btn-stress:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: rgba(255, 65, 54, 0.2);
        }
        
        /* Status */
        .status {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            margin-top: 25px;
            font-weight: 600;
            font-size: 16px;
            border: 2px solid;
        }
        .status-stopped { 
            background: rgba(255, 65, 54, 0.1);
            color: #ff4136;
            border-color: #ff4136;
        }
        .status-running { 
            background: rgba(0, 255, 65, 0.1);
            color: #00ff41;
            border-color: #00ff41;
        }
        .status-connecting { 
            background: rgba(255, 133, 27, 0.1);
            color: #ff851b;
            border-color: #ff851b;
        }
        
        /* Data Log */
        .data-log {
            max-height: 150px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background: #000;
            color: #00ff41;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border: 2px solid #00ff41;
        }
        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid #1a1f3a;
        }
        
        /* Info Panel */
        .info-panel {
            background: #1a1f3a;
            border-radius: 20px;
            padding: 30px;
            border: 2px solid #00ff41;
        }
        .info-panel h2 {
            color: #00ff41;
            margin-bottom: 15px;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        }
        .info-panel p {
            color: #888;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        .api-endpoint {
            background: #0d1117;
            padding: 10px 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: #00ff41;
            margin: 10px 0;
            border: 1px solid #00ff41;
        }
        
        /* Pulse animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulsing {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ LIVE EKG MONITOR</h1>
        
        <div class="monitor">
            <!-- EKG Waveform Display -->
            <div class="ekg-container">
                <div class="ekg-header">
                    <div class="ekg-title">Electrocardiogram</div>
                    <div>
                        <span class="bpm-live" id="bpmLive">--</span>
                        <span class="bpm-unit">BPM</span>
                    </div>
                </div>
                <canvas id="ekgCanvas"></canvas>
            </div>
            
            <!-- Vital Signs -->
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="heartRate">--</div>
                    <div class="metric-label">Heart Rate (BPM)</div>
                </div>
                <div class="metric">
                    <div class="bp-display">
                        <span class="bp-value" id="systolic">--</span>
                        <span class="bp-separator">/</span>
                        <span class="bp-value" id="diastolic">--</span>
                    </div>
                    <div class="metric-label">Blood Pressure (mmHg)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="updates">0</div>
                    <div class="metric-label">Data Points Received</div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <button class="btn-start" onclick="startSimulation()">▶ START MONITOR</button>
                <button class="btn-stress" onclick="triggerStress()" id="stressBtn">⚡ STRESS TEST (5s)</button>
                <button class="btn-stop" onclick="stopSimulation()">⏹ STOP</button>
            </div>
            
            <div id="status" class="status status-stopped">
                ⚫ SYSTEM OFFLINE - Click Start to Begin Monitoring
            </div>
            
            <div class="data-log" id="dataLog">
                <div class="log-entry">📊 Awaiting connection...</div>
            </div>
        </div>
        
        <div class="info-panel">
            <h2>📡 API ACCESS FOR ML MODELS</h2>
            <p><strong>Real-time WebSocket Stream:</strong></p>
            <div class="api-endpoint">ws://localhost:8001/ws/heartbeat</div>
            
            <p><strong>HTTP Polling Endpoint:</strong></p>
            <div class="api-endpoint">GET http://localhost:8001/heartbeat/current</div>
            
            <p><strong>Set Normal/Baseline Heart Rate:</strong></p>
            <div class="api-endpoint">POST http://localhost:8001/input/set-normal-heartrate?baseline_bpm=72&reason=profile</div>
            
            <p><strong>Get Baseline Heart Rate:</strong></p>
            <div class="api-endpoint">GET http://localhost:8001/heartbeat/baseline</div>
            
            <p><strong>Video Stress Input (Facial Analysis):</strong></p>
            <div class="api-endpoint">POST http://localhost:8001/input/video-stress?stress_detected=true&intensity=0.7</div>
            
            <p><strong>Audio Stress Input (Voice Analysis):</strong></p>
            <div class="api-endpoint">POST http://localhost:8001/input/audio-stress?stress_detected=true&intensity=0.6</div>
            
            <p><strong>Combined Multi-Modal Input:</strong></p>
            <div class="api-endpoint">POST http://localhost:8001/input/combined-stress</div>
            
            <p><strong>Returns JSON:</strong></p>
            <div class="api-endpoint">{"timestamp": "...", "bpm": 72, "systolic": 120, "diastolic": 80}</div>
            
            <p style="margin-top: 20px; font-size: 14px; color: #00ff41;">
                💡 <strong>PUBLIC API:</strong> Any ML model can access this data at any time via the endpoints above.
            </p>
            <p style="margin-top: 10px; font-size: 13px; color: #888;">
                ⚡ <strong>STRESS TRIGGERS:</strong> Stress occurs randomly (5-8% chance per update). 
                ML models analyzing video/audio can trigger additional stress via input endpoints.
            </p>
        </div>
    </div>

    <script>
        let ws = null;
        let updateCount = 0;
        const maxLogEntries = 30;
        const API_BASE = window.location.origin;
        
        // EKG Canvas Setup
        const canvas = document.getElementById('ekgCanvas');
        const ctx = canvas.getContext('2d');
        let ekgData = [];
        const maxDataPoints = 200;
        let animationId = null;
        
        // Set canvas size
        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Generate EKG waveform based on BPM
        function generateEKGPoint(bpm, index) {
            const beatsPerSecond = bpm / 60;
            const samplesPerBeat = 50; // Adjust for waveform detail
            const beatProgress = (index % samplesPerBeat) / samplesPerBeat;
            
            let amplitude = 0;
            if (beatProgress < 0.1) {
                // P wave
                amplitude = Math.sin(beatProgress * 10 * Math.PI) * 20;
            } else if (beatProgress < 0.3) {
                // QRS complex
                if (beatProgress < 0.2) {
                    amplitude = -30;
                } else if (beatProgress < 0.25) {
                    amplitude = 100; // R peak
                } else {
                    amplitude = -20;
                }
            } else if (beatProgress < 0.5) {
                // T wave
                amplitude = Math.sin((beatProgress - 0.3) * 5 * Math.PI) * 30;
            }
            
            return amplitude + Math.random() * 3; // Add noise
        }
        
        // Draw EKG waveform
        function drawEKG() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw grid
            ctx.strokeStyle = '#0a3d0a';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 20) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 20) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            // Draw waveform
            if (ekgData.length > 1) {
                ctx.strokeStyle = '#00ff41';
                ctx.lineWidth = 2;
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#00ff41';
                ctx.beginPath();
                
                const xStep = canvas.width / maxDataPoints;
                const yCenter = canvas.height / 2;
                
                for (let i = 0; i < ekgData.length; i++) {
                    const x = i * xStep;
                    const y = yCenter - ekgData[i];
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
                ctx.shadowBlur = 0;
            }
            
            animationId = requestAnimationFrame(drawEKG);
        }

        async function startSimulation() {
            try {
                updateStatus('connecting', '🔄 Initializing monitor...');
                const res = await fetch(`${API_BASE}/simulation/start`, { method: 'POST' });
                const data = await res.json();
                updateStatus('running', '🟢 LIVE MONITORING ACTIVE');
                addLog(`✅ ${data.message}`);
                connectWebSocket();
                drawEKG();
            } catch (error) {
                updateStatus('stopped', '⚫ ERROR: Connection failed');
                addLog(`❌ Error: ${error.message}`);
            }
        }

        async function stopSimulation() {
            try {
                await fetch(`${API_BASE}/simulation/stop`, { method: 'POST' });
                updateStatus('stopped', '⚫ SYSTEM OFFLINE');
                addLog('⏹ Monitor stopped');
                if (ws) ws.close();
                if (animationId) cancelAnimationFrame(animationId);
                updateCount = 0;
                document.getElementById('updates').textContent = '0';
                ekgData = [];
            } catch (error) {
                addLog(`❌ Error: ${error.message}`);
            }
        }

        async function triggerStress() {
            const btn = document.getElementById('stressBtn');
            btn.disabled = true;
            btn.textContent = '⚡ STRESS ACTIVE...';
            
            try {
                const response = await fetch(`${API_BASE}/simulation/stress-test`, { method: 'POST' });
                const data = await response.json();
                addLog('⚡ Stress test initiated (5 seconds)');
                
                // Re-enable button after 5 seconds
                setTimeout(() => {
                    btn.disabled = false;
                    btn.textContent = '⚡ STRESS TEST (5s)';
                    addLog('✅ Stress test complete');
                }, 5000);
            } catch (error) {
                addLog(`❌ Error: ${error.message}`);
                btn.disabled = false;
                btn.textContent = '⚡ STRESS TEST (5s)';
            }
        }

        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/heartbeat`);
            
            ws.onopen = () => {
                addLog('🔗 WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'connected') {
                    addLog('📡 Monitor online - receiving data');
                    return;
                }
                
                updateCount++;
                
                // Update displays
                document.getElementById('bpmLive').textContent = data.bpm;
                document.getElementById('heartRate').textContent = data.bpm;
                document.getElementById('systolic').textContent = data.systolic;
                document.getElementById('diastolic').textContent = data.diastolic;
                document.getElementById('updates').textContent = updateCount;
                
                // Update EKG waveform
                const ekgPoint = generateEKGPoint(data.bpm, updateCount);
                ekgData.push(ekgPoint);
                if (ekgData.length > maxDataPoints) {
                    ekgData.shift();
                }
                
                // Log high stress events
                if (data.stress_level > 0.5) {
                    addLog(`🔴 ELEVATED: HR=${data.bpm} BP=${data.systolic}/${data.diastolic}`);
                }
            };
            
            ws.onerror = () => {
                updateStatus('stopped', '⚫ Connection Error');
                addLog('❌ WebSocket error');
            };
            
            ws.onclose = () => {
                updateStatus('stopped', '⚫ Disconnected');
                addLog('🔌 Monitor disconnected');
            };
        }

        function updateStatus(state, text) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = text;
            statusEl.className = `status status-${state}`;
        }

        function addLog(message) {
            const logEl = document.getElementById('dataLog');
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${timestamp}] ${message}`;
            logEl.insertBefore(entry, logEl.firstChild);
            while (logEl.children.length > maxLogEntries) {
                logEl.removeChild(logEl.lastChild);
            }
        }

        // Auto-connect if already running
        window.onload = async () => {
            try {
                const res = await fetch(`${API_BASE}/health`);
                const data = await res.json();
                if (data.simulation_state === 'running') {
                    updateStatus('running', '🟢 LIVE MONITORING ACTIVE');
                    addLog('📊 Reconnecting to existing session...');
                    connectWebSocket();
                    drawEKG();
                }
            } catch (error) {
                console.log('Health check failed:', error);
            }
        };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/input/video-stress")
async def video_stress_input(stress_detected: bool, intensity: float = 0.5):
    """
    Input flag from video analysis (facial expression, body language).
    
    Args:
        stress_detected: Boolean flag indicating stress detected in video
        intensity: Stress intensity (0.0 to 1.0)
    
    Use case:
        - ML model analyzing facial expressions
        - Body language detection
        - Visual stress indicators
    """
    if intensity < 0 or intensity > 1:
        return JSONResponse(
            status_code=400,
            content={"error": "intensity must be between 0 and 1"}
        )
    
    if stress_detected:
        engine.stress_level = max(engine.stress_level, intensity)
    
    return {
        "message": "Video input processed",
        "stress_detected": stress_detected,
        "current_stress_level": engine.stress_level,
        "current_bpm": int(engine.current_bpm) if engine.current_data else None
    }


@app.post("/input/audio-stress")
async def audio_stress_input(stress_detected: bool, intensity: float = 0.5):
    """
    Input flag from audio analysis (voice tone, speech patterns).
    
    Args:
        stress_detected: Boolean flag indicating stress detected in audio
        intensity: Stress intensity (0.0 to 1.0)
    
    Use case:
        - ML model analyzing voice stress
        - Speech pattern analysis
        - Vocal stress indicators
    """
    if intensity < 0 or intensity > 1:
        return JSONResponse(
            status_code=400,
            content={"error": "intensity must be between 0 and 1"}
        )
    
    if stress_detected:
        engine.stress_level = max(engine.stress_level, intensity)
    
    return {
        "message": "Audio input processed",
        "stress_detected": stress_detected,
        "current_stress_level": engine.stress_level,
        "current_bpm": int(engine.current_bpm) if engine.current_data else None
    }


@app.post("/input/combined-stress")
async def combined_stress_input(
    video_stress: bool = False,
    audio_stress: bool = False,
    video_intensity: float = 0.5,
    audio_intensity: float = 0.5
):
    """
    Combined input from both video and audio analysis.
    Takes the maximum stress level from both sources.
    
    Args:
        video_stress: Stress detected in video
        audio_stress: Stress detected in audio
        video_intensity: Video stress intensity (0.0 to 1.0)
        audio_intensity: Audio stress intensity (0.0 to 1.0)
    
    Use case:
        - Multi-modal stress detection
        - Combined ML model outputs
        - Comprehensive stress assessment
    """
    if video_intensity < 0 or video_intensity > 1:
        return JSONResponse(
            status_code=400,
            content={"error": "video_intensity must be between 0 and 1"}
        )
    if audio_intensity < 0 or audio_intensity > 1:
        return JSONResponse(
            status_code=400,
            content={"error": "audio_intensity must be between 0 and 1"}
        )
    
    # Take maximum stress from both sources
    max_intensity = 0.0
    if video_stress:
        max_intensity = max(max_intensity, video_intensity)
    if audio_stress:
        max_intensity = max(max_intensity, audio_intensity)
    
    if max_intensity > 0:
        engine.stress_level = max(engine.stress_level, max_intensity)
    
    return {
        "message": "Combined input processed",
        "video_stress": video_stress,
        "audio_stress": audio_stress,
        "applied_intensity": max_intensity,
        "current_stress_level": engine.stress_level,
        "current_bpm": int(engine.current_bpm) if engine.current_data else None
    }


@app.post("/input/set-normal-heartrate")
async def set_normal_heartrate(baseline_bpm: int, reason: str = "manual"):
    """
    Set normal/baseline heart rate from external input.
    This is the resting heart rate when there's no stress.
    
    Args:
        baseline_bpm: Normal resting heart rate (40-120 BPM)
        reason: Source of the input ("profile", "calibration", "ml_model", "manual")
    
    Use case:
        - User profile (age, fitness level)
        - Initial calibration from actual measurement
        - ML model estimation from historical data
        - Different person simulation
    
    Examples:
        - Athlete: 45-60 BPM
        - Average adult: 60-80 BPM
        - Sedentary/anxious: 80-100 BPM
    """
    if baseline_bpm < 40 or baseline_bpm > 120:
        return JSONResponse(
            status_code=400,
            content={"error": "baseline_bpm must be between 40 and 120"}
        )
    
    old_baseline = engine.base_bpm
    engine.base_bpm = baseline_bpm
    engine.current_bpm = float(baseline_bpm)  # Reset current to new baseline
    
    # Determine profile
    if baseline_bpm < 60:
        profile = "athlete"
    elif baseline_bpm < 80:
        profile = "average"
    elif baseline_bpm < 100:
        profile = "above_average"
    else:
        profile = "elevated"
    
    return {
        "message": "Normal heart rate updated",
        "old_baseline_bpm": old_baseline,
        "new_baseline_bpm": baseline_bpm,
        "profile": profile,
        "reason": reason,
        "current_bpm": int(engine.current_bpm) if engine.current_data else baseline_bpm
    }


@app.get("/heartbeat/baseline")
async def get_baseline_heartrate():
    """
    Get current baseline/normal heart rate.
    
    Returns the resting heart rate (no stress baseline).
    Useful for ML models to understand the person's normal state.
    """
    return {
        "baseline_bpm": engine.base_bpm,
        "current_bpm": int(engine.current_bpm) if engine.current_data else engine.base_bpm,
        "current_stress_level": engine.stress_level,
        "profile": "athlete" if engine.base_bpm < 60 else "average" if engine.base_bpm < 80 else "elevated",
        "state": engine.state.value
    }


@app.post("/simulation/set-baseline")
async def set_baseline_bpm(baseline_bpm: int = 72):
    """
    Set baseline heart rate.
    
    Args:
        baseline_bpm: Resting heart rate (60-100 recommended)
    
    Use case:
        - Simulate different person profiles (athlete vs sedentary)
        - Testing different baseline scenarios
    """
    if baseline_bpm < 40 or baseline_bpm > 120:
        return JSONResponse(
            status_code=400,
            content={"error": "baseline_bpm must be between 40 and 120"}
        )
    
    engine.base_bpm = baseline_bpm
    
    return {
        "message": "Baseline heart rate updated",
        "baseline_bpm": baseline_bpm,
        "profile": "athlete" if baseline_bpm < 60 else "average" if baseline_bpm < 80 else "elevated"
    }


# ============= Startup Event =============

@app.on_event("startup")
async def startup_event():
    """Initialize simulation on startup"""
    print("="*60)
    print("🫀 Virtual Heartbeat Simulation API")
    print("="*60)
    print("Ready to simulate realistic heartbeat data!")
    print("\nQuick Start:")
    print("  1. Start simulation: POST http://localhost:8000/simulation/start")
    print("  2. Get current data: GET http://localhost:8000/heartbeat/current")
    print("  3. WebSocket stream: ws://localhost:8000/ws/heartbeat")
    print("  4. API docs: http://localhost:8000/docs")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    if engine.state == SimulationState.RUNNING:
        engine.state = SimulationState.STOPPED
        if engine.task:
            engine.task.cancel()
    print("Simulation stopped")


# ============= Run Instructions =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "heartbeat_sim:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# How to use:
#
# 1. Start server:
#    uvicorn heartbeat_sim:app --reload
#
# 2. Start simulation:
#    curl -X POST http://localhost:8000/simulation/start
#
# 3. Get current heartbeat:
#    curl http://localhost:8000/heartbeat/current
#
# 4. Test WebSocket (Python):
#    pip install websockets
#    
#    import asyncio
#    import websockets
#    import json
#    
#    async def test():
#        async with websockets.connect('ws://localhost:8000/ws/heartbeat') as ws:
#            while True:
#                data = json.loads(await ws.recv())
#                print(f"BPM: {data['bpm']}, Stress: {data['stress_level']}")
#    
#    asyncio.run(test())
#
# 5. Trigger stress (optional):
#    curl -X POST "http://localhost:8000/simulation/trigger-stress?stress_level=0.8"
#
# 6. Stop simulation:
#    curl -X POST http://localhost:8000/simulation/stop
