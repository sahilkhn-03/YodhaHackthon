# NeuroBalance AI - 16-Hour Hackathon Work Plan
### Team Mission404 | 4 Members | ~4 Hours per Person | 28 Story Points Total (7 per person)

---

## Executive Summary
**Objective**: Deploy a **minimal viable psychosomatic AI platform** using a zero-hardware approach with real-time stress detection from webcam + microphone.

**Constraints**: 
- 16 hours total (~4 hours per person)
- 1 weekend sprint
- No specialized hardware
- Focus on MVP with clinical potential

**Success Criteria**:
- Live stress detection pipeline working end-to-end
- Real-time dashboard showing metrics
- Deployable Docker container
- Clinician-readable output

---

## Work Distribution by Role

### **Person 1: AI/ML Lead** (7 points | 4 hours)
Focus: **Core stress detection engine**

1. **Facial Emotion Recognition Baseline** (2pts, 45min)
   - Deploy MediaPipe Face Mesh + pre-trained emotion classifier (TensorFlow/PyTorch)
   - Extract: Eye closure, eyebrow position, mouth openness, facial symmetry
   - Output: Real-time emotion + stress confidence scores
   - *Tool*: Use MediaPipe + lightweight MobileNet model

2. **Voice Stress Signal Extraction** (2pts, 45min)
   - Librosa-based real-time audio feature extraction
   - Extract: Pitch variations, jitter, speaking rate, energy envelope
   - Implement: Low-pass filter for noise robustness
   - Output: Voice stress index (0-100)
   - *Tool*: Librosa + PyAudio for live input

3. **Multimodal Fusion Algorithm** (2pts, 45min)
   - Weighted ensemble: 40% facial + 40% vocal + 20% temporal trend
   - Kalman filter for smoothing temporal noise
   - Output: **Final Psychosomatic Stress Score (0-100)**
   - Threshold-based alerts: >70 = HIGH stress

4. **Model Inference Optimization** (1pt, 15min)
   - Package all models in single inference function
   - Target: <500ms per frame @ 30fps
   - Export as Python service callable from backend

---

### **Person 2: Backend Engineer** (7 points | 4 hours)
Focus: **API + Real-time data pipeline**

1. **FastAPI Server Bootstrap** (1.5pts, 30min)
   - Minimal FastAPI app with CORS, logging, error handling
   - Health check endpoint
   - Docker-ready structure (requirements.txt, Dockerfile)

2. **WebSocket Real-Time Streaming** (2pts, 50min)
   - `/ws/assessment` endpoint for live stress feed
   - Broadcasting: Frame-by-frame stress scores + emotion state
   - Client-side subscription model (for frontend dashboard)
   - Connection pooling & cleanup

3. **Assessment REST APIs** (2pts, 50min)
   - `POST /assessment/start` - Begin new assessment session
   - `GET /assessment/{session_id}/summary` - Retrieve session metrics
   - `POST /assessment/{session_id}/end` - Close session, store metadata
   - SQLite schema: sessions, frames, results (denormalized for speed)

4. **Database & Session Management** (1.5pts, 30min)
   - SQLite with async SQLAlchemy (aiosqlite)
   - Session tracking: start_time, duration, avg_stress, peak_stress
   - Frame history: timestamp, emotion, stress_score (compact storage)
   - Auto-cleanup: Delete old sessions >24hrs

---

### **Person 3: Frontend Engineer** (7 points | 4 hours)
Focus: **Clinical Dashboard UI**

1. **React App Skeleton + Routing** (1pt, 20min)
   - Vite-based React setup (fast HMR)
   - React Router: /assessment (live), /results (history), /about
   - Tailwind CSS for rapid styling
   - Responsive mobile-first layout

2. **Live Assessment Interface** (2.5pts, 1hr)
   - Camera feed preview (HTML5 canvas)
   - Real-time metrics display:
     - Current stress score (large gauge)
     - Emotion state (joy/sad/anxious/neutral icons)
     - Breathing guide animation
   - WebSocket connection status indicator
   - Start/Stop assessment buttons

3. **Real-Time Stress Dashboard** (2pts, 50min)
   - Line chart: stress trend (last 5 min, Chart.js)
   - Gauge chart: current stress level (0-100)
   - Emotion pie chart: distribution of detected emotions
   - Stats panel: avg_stress, peak_stress, session_duration
   - Color-coded alerts: GREEN (<40), YELLOW (40-70), RED (>70)

4. **Results & History Page** (1.5pts, 30min)
   - Past assessment cards: date, duration, avg_stress, emotion summary
   - Export session as JSON report for clinician review
   - Simple analytics: weekly stress trend (bar chart)

---

### **Person 4: DevOps + Integration Lead** (7 points | 4 hours)
Focus: **Deployment + system glue**

1. **Docker Containerization** (1.5pts, 30min)
   - Dockerfile: Python 3.10 slim + OpenCV + Librosa + FastAPI
   - Docker Compose: Backend + Frontend (nginx reverse proxy)
   - Environment config (.env): API_URL, model paths
   - Health checks & restart policies

2. **Frontend-Backend Integration** (2pts, 50min)
   - Webpack/Vite config for production build
   - API client (Axios): base URL, error handling, request retry logic
   - WebSocket client wrapper (React hook: useWebSocket)
   - CORS configuration in FastAPI

3. **End-to-End Testing Script** (2pts, 50min)
   - Automated test suite:
     - API endpoint validation (requests library)
     - WebSocket connection test
     - Latency benchmarks (stress score generation <500ms)
     - Load test: 5 concurrent assessments
   - Docker healthcheck script
   - Integration test results logged to file

4. **Deployment & Documentation** (1.5pts, 30min)
   - Production build optimization (minify, code splitting)
   - README: Quick start, API docs (auto-generated from FastAPI), architecture diagram
   - Environment setup script (install deps, download models)
   - Deployment guide: Local Docker, basic cloud (AWS EC2 / Railway)

---

## Technical Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)                                │
│  ├─ Live Assessment UI (camera + metrics)               │
│  ├─ Stress Dashboard (charts, gauge)                    │
│  ├─ WebSocket client (real-time updates)                │
│  └─ Results History                                      │
└──────────────────┬──────────────────────────────────────┘
                   │ REST + WebSocket
                   ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend                                        │
│  ├─ POST /assessment/start (session mgmt)               │
│  ├─ WebSocket /ws/assessment (streaming)                │
│  ├─ GET /assessment/{id}/summary (results)              │
│  └─ SQLite (session storage)                            │
└──────────────────┬──────────────────────────────────────┘
                   │ Python inference calls
                   ▼
┌─────────────────────────────────────────────────────────┐
│  AI/ML Inference Engine                                 │
│  ├─ MediaPipe (facial landmarks)                        │
│  ├─ Emotion classifier (pre-trained CNN)                │
│  ├─ Librosa (audio features)                            │
│  └─ Fusion module (stress score calculation)            │
└─────────────────────────────────────────────────────────┘
```

---

## Timeline & Checkpoints

| Time | Checkpoint | Owner | Status |
|------|-----------|-------|--------|
| **0:00-0:45** | Baseline models loaded + test inference | Person 1 | Critical |
| **0:45-1:30** | FastAPI server live with /assessment endpoints | Person 2 | Critical |
| **1:30-2:15** | Frontend boots, camera preview, API connection | Person 3 | Critical |
| **2:15-3:00** | Live stress score streaming end-to-end | Person 2+3 | Sync point |
| **3:00-3:45** | Dashboard charts + real-time updates working | Person 3+4 | Feature lock |
| **3:45-4:00** | Docker build, testing, final integration | Person 4+all | Go-live |

---

## Minimum Viable Features (Must-Have)

✅ **Live stress detection** from webcam + microphone  
✅ **Real-time dashboard** with visual metrics  
✅ **Session management** (start/end assessment)  
✅ **Clinician-readable output** (JSON report)  
✅ **Docker deployment** (single `docker-compose up`)  

---

## Nice-to-Have (If Time Permits)

- 📊 Weekly stress trend analytics
- 🔐 User authentication (JWT)
- 📱 Mobile app version (React Native)
- 🧠 Emotion history per session
- 🎯 Personalized stress baselines
- 📧 Clinician email reports

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Camera/mic access denied | Use browser permissions prompt + fallback simulator mode |
| Model inference too slow | Use lighter models (MobileNet), quantization, batch size 1 |
| WebSocket latency | Use msgpack compression, frame skipping (process every 3rd frame) |
| Frontend build fails | Vite pre-config, no custom webpack tweaks |
| Database locks | Use WAL mode in SQLite, async access patterns |

---

## Success Definition

**By Hour 4: You have a LIVE, DEPLOYABLE psychosomatic AI platform that:**
- Detects real-time stress from standard hardware (webcam + mic)
- Displays clinician-ready insights in a professional dashboard
- Runs in Docker with zero manual setup
- Processes frames in <500ms (real-time capable)
- **Is technically impressive** (multimodal AI fusion, WebSocket streaming, production-grade code)

**Bonus**: Beats other hackathon teams with a **fully integrated, containerized, end-to-end system** vs. partial demos.

---

**Good luck! 🚀**
