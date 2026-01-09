# Stress Input System - Implementation Summary

## ✅ Changes Completed

### 1. **Random Stress Generation Enhanced**
- **OLD:** 2% chance per update (too rare)
- **NEW:** 5-8% variable chance per update (more realistic, sporadic)
- **Code Location:** Lines 75-85 in heartbeat_sim.py
- **Result:** Stress now occurs randomly but frequently enough to be noticeable
- **Ensures:** Not uniform (variable probability), not absent (5-8% chance)

```python
# Variable probability: 5-8% chance per update for realistic sporadic stress
stress_probability = random.uniform(0.05, 0.08)
if random.random() < stress_probability:
    self.stress_level = random.uniform(0.3, 0.9)
```

### 2. **Stress Test Button Removed**
- ✅ Button removed from HTML interface
- ✅ JavaScript function removed
- ✅ CSS styling removed
- **Result:** No manual stress triggering from UI
- **Reason:** Stress should occur naturally/randomly or from ML model inputs

### 3. **External Input Endpoints Added**
Three new FastAPI endpoints for ML model integration:

#### a. `/input/video-stress` (POST)
**Purpose:** Accept stress detection from video/facial analysis ML models

**Parameters:**
- `stress_detected`: bool - Whether stress was detected
- `intensity`: float (0.0-1.0) - Stress intensity level

**Example:**
```bash
curl -X POST "http://localhost:8001/input/video-stress?stress_detected=true&intensity=0.7"
```

**Response:**
```json
{
  "message": "Video input processed",
  "stress_detected": true,
  "current_stress_level": 0.7,
  "current_bpm": 95
}
```

**Use Cases:**
- Facial expression analysis (frowning, tension)
- Body language detection
- Visual stress indicators
- Computer vision models

#### b. `/input/audio-stress` (POST)
**Purpose:** Accept stress detection from audio/voice analysis ML models

**Parameters:**
- `stress_detected`: bool - Whether stress was detected
- `intensity`: float (0.0-1.0) - Stress intensity level

**Example:**
```bash
curl -X POST "http://localhost:8001/input/audio-stress?stress_detected=true&intensity=0.6"
```

**Response:**
```json
{
  "message": "Audio input processed",
  "stress_detected": true,
  "current_stress_level": 0.6,
  "current_bpm": 88
}
```

**Use Cases:**
- Voice tone analysis (pitch, speed, trembling)
- Speech pattern analysis
- Vocal stress indicators
- Audio processing models

#### c. `/input/combined-stress` (POST)
**Purpose:** Multi-modal stress detection (video + audio combined)

**Parameters:**
- `video_stress`: bool - Video stress detected
- `audio_stress`: bool - Audio stress detected
- `video_intensity`: float (0.0-1.0) - Video stress intensity
- `audio_intensity`: float (0.0-1.0) - Audio stress intensity

**Logic:** Takes the MAXIMUM stress level from both sources

**Example:**
```bash
curl -X POST "http://localhost:8001/input/combined-stress?video_stress=true&audio_stress=true&video_intensity=0.8&audio_intensity=0.7"
```

**Response:**
```json
{
  "message": "Combined input processed",
  "video_stress": true,
  "audio_stress": true,
  "applied_intensity": 0.8,
  "current_stress_level": 0.8,
  "current_bpm": 105
}
```

**Use Cases:**
- Multi-modal ML models
- Comprehensive stress assessment
- Fusing multiple detection sources
- Higher confidence stress detection

## 🔍 API Verification Checklist

### ✅ Input Flag Mapping to Stress Rate
**Question:** Does the input flag map to stress rate in FastAPI?

**Answer:** YES - Fully implemented

**How it works:**
1. ML model detects stress in video/audio
2. Sends `stress_detected=true` with `intensity` value
3. FastAPI endpoint receives the flag
4. Updates `engine.stress_level` with the intensity
5. Heart rate simulation responds to `stress_level`:
   - Calculates `stress_factor` based on `stress_level`
   - Increases BPM by up to +50 beats (stress_factor * 50)
   - Increases blood pressure proportionally

**Code Flow:**
```
Video/Audio ML Model 
    → POST /input/video-stress or /input/audio-stress
        → engine.stress_level = max(current, intensity)
            → generate_heartbeat() reads stress_level
                → Calculates increased BPM and BP
                    → Broadcasts to WebSocket clients
```

### ✅ Input Flag from Video Increases Heart Rate
**Question:** Does video input flag increase heart rate in FastAPI?

**Answer:** YES - Fully functional

**Mechanism:**
```python
# In /input/video-stress endpoint:
if stress_detected:
    engine.stress_level = max(engine.stress_level, intensity)

# In generate_heartbeat():
if self.stress_level > self.stress_threshold:  # 0.3
    stress_factor = (self.stress_level - 0.3) / 0.7
    target_bpm = base_bpm + (stress_factor * 50)  # Up to +50 BPM
```

**Example:**
- Baseline: 72 BPM, 120/80 BP
- Video detects stress (intensity=0.8)
- New target: 72 + (0.714 * 50) = ~108 BPM
- BP increases: ~145/95 mmHg
- Smooth transition over several updates (realistic)

### ✅ Input Flag from Audio Increases Heart Rate
**Question:** Does audio input flag increase heart rate in FastAPI?

**Answer:** YES - Same mechanism as video

**Both endpoints use identical logic:**
- Accept boolean flag + intensity
- Update `engine.stress_level`
- Simulation responds automatically
- Heart rate increases proportionally
- Blood pressure increases proportionally

## 🧪 Testing

**Test Script:** `test_stress_inputs.py`

Run test:
```bash
python backend/test_stress_inputs.py
```

**What it tests:**
1. Baseline heart rate
2. Video stress input → BPM increase
3. Audio stress input → BPM increase
4. Combined input → Maximum stress applied
5. Gradual stress decay over time

## 📊 Stress Behavior

### Random Stress (Automatic)
- **Frequency:** 5-8% chance per 500ms update
- **Intensity:** Random 0.3-0.9
- **Duration:** Decays by 2% per update (98% retention)
- **Recovery:** Returns to baseline over ~30-60 seconds

### External Stress (ML Model Input)
- **Source:** Video/Audio analysis endpoints
- **Intensity:** 0.0-1.0 (provided by ML model)
- **Duration:** Same decay rate as random stress
- **Combination:** Takes maximum of current and new stress

### Combined Behavior
```
Time 0: Random stress triggers (0.5)
Time 5: Video detects stress (0.7) → stress_level = 0.7
Time 10: Audio detects stress (0.6) → stress_level stays 0.7 (already higher)
Time 15: Combined input (video=0.9, audio=0.8) → stress_level = 0.9 (max)
Time 60: Decayed to ~0.1 (returning to baseline)
```

## 🎯 Integration for ML Models

### Python Example
```python
import requests

# Your ML model analyzes video frame
stress_detected = your_model.detect_facial_stress(frame)
intensity = your_model.get_stress_intensity()  # 0.0-1.0

# Send to heartbeat simulation
response = requests.post(
    "http://localhost:8001/input/video-stress",
    params={"stress_detected": stress_detected, "intensity": intensity}
)

# Get updated heart rate
heartbeat = requests.get("http://localhost:8001/heartbeat/current").json()
print(f"BPM after stress: {heartbeat['bpm']}")
```

### JavaScript Example
```javascript
// Your ML model analyzes audio
const stressDetected = await audioModel.detectStress(audioData);
const intensity = audioModel.getIntensity();

// Send to heartbeat simulation
const response = await fetch(
    `http://localhost:8001/input/audio-stress?stress_detected=${stressDetected}&intensity=${intensity}`,
    { method: 'POST' }
);

const result = await response.json();
console.log(`Current BPM: ${result.current_bpm}`);
```

## 📡 Available Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/heartbeat/current` | GET | Get current heart rate, BP, stress |
| `/ws/heartbeat` | WebSocket | Real-time streaming |
| `/input/video-stress` | POST | Video stress input |
| `/input/audio-stress` | POST | Audio stress input |
| `/input/combined-stress` | POST | Multi-modal input |
| `/simulation/start` | POST | Start simulation |
| `/simulation/stop` | POST | Stop simulation |
| `/health` | GET | Service health check |

## ✅ Verification Complete

- ✅ Stress occurs randomly (5-8% chance, non-uniform)
- ✅ Stress is not absent (frequent enough to observe)
- ✅ Manual stress button removed
- ✅ Video input flag maps to stress rate
- ✅ Video input increases heart rate
- ✅ Audio input flag maps to stress rate
- ✅ Audio input increases heart rate
- ✅ Combined input takes maximum stress
- ✅ All endpoints documented in UI
- ✅ Test script provided

**System is fully operational and ready for ML model integration!**
